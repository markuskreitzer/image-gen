from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
VALID_DEVICES = {"auto", "cuda", "mps", "cpu"}
VALID_DTYPES = {"auto", "bfloat16", "float16", "float32"}
DEFAULT_OUTPUT_DIR = Path("outputs")

app = typer.Typer(
    name="imagen",
    help="Generate images with FLUX.2-klein.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
console = Console(stderr=True)


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len].rstrip("_") or "image"


def generate_output_path(prompt: str, output: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output is None:
        return DEFAULT_OUTPUT_DIR / f"{timestamp}_{slugify(prompt)}.png"

    path = Path(output)
    if path.is_dir():
        return path / f"{timestamp}_{slugify(prompt)}.png"
    if path.suffix == "":
        return path.with_suffix(".png")
    if path.suffix.lower() != ".png":
        raise ValueError(f"only PNG output is supported, got {path.suffix!r}")
    return path


def validate_params(width: int, height: int, steps: int, guidance_scale: float) -> list[str]:
    errors: list[str] = []
    if width <= 0 or width % 8 != 0:
        errors.append(f"width must be a positive multiple of 8, got {width}")
    if height <= 0 or height % 8 != 0:
        errors.append(f"height must be a positive multiple of 8, got {height}")
    if width > 4096 or height > 4096:
        errors.append(f"maximum dimension is 4096, got {width}x{height}")
    if steps < 1 or steps > 100:
        errors.append(f"steps must be between 1 and 100, got {steps}")
    if guidance_scale < 0:
        errors.append(f"guidance scale must be >= 0, got {guidance_scale}")
    return errors


def emit_error(json_output: bool, error_type: str, message: str, code: int, errors: list[str] | None = None) -> None:
    payload: dict[str, object] = {
        "status": "error",
        "error_type": error_type,
        "error_message": message,
    }
    if errors:
        payload["errors"] = errors

    if json_output:
        print(json.dumps(payload, indent=2))
    else:
        console.print(f"[red]Error:[/red] {message}")
        if errors:
            for error in errors:
                console.print(f"  - {error}")
    raise typer.Exit(code=code)


def configure_quiet_output() -> None:
    os.environ["TQDM_DISABLE"] = "1"
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.randint(0, 2**32 - 1)


def resolve_device(requested: str, torch_module) -> str:
    requested = requested.lower()
    if requested not in VALID_DEVICES:
        valid = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"device must be one of: {valid}")

    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch_module.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch_module.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return requested


def resolve_dtype(requested: str, device: str, torch_module):
    requested = requested.lower()
    if requested not in VALID_DTYPES:
        valid = ", ".join(sorted(VALID_DTYPES))
        raise ValueError(f"dtype must be one of: {valid}")

    if requested == "auto":
        requested = "float32" if device == "cpu" else "bfloat16"

    return requested, {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[requested]


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Text prompt for image generation")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file or directory")] = None,
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Random seed")] = None,
    steps: Annotated[int, typer.Option("--steps", "-n", help="Inference steps")] = 4,
    width: Annotated[int, typer.Option("--width", "-W", help="Image width, multiple of 8")] = 1024,
    height: Annotated[int, typer.Option("--height", "-H", help="Image height, multiple of 8")] = 1024,
    guidance_scale: Annotated[float, typer.Option("--guidance-scale", "-g", help="Guidance scale")] = 1.0,
    model: Annotated[str, typer.Option("--model", "-m", help="Hugging Face model ID")] = DEFAULT_MODEL,
    device: Annotated[str, typer.Option("--device", "-d", help="auto, cuda, mps, or cpu")] = "auto",
    dtype: Annotated[str, typer.Option("--dtype", help="auto, bfloat16, float16, or float32")] = "auto",
    cpu_offload: Annotated[
        bool,
        typer.Option("--cpu-offload/--no-cpu-offload", help="Use sequential CPU offload on CUDA"),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Write machine-readable JSON to stdout")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-essential output")] = False,
) -> None:
    """Generate an image from a text prompt."""
    errors = validate_params(width, height, steps, guidance_scale)
    if errors:
        emit_error(json_output, "InvalidParametersError", "invalid generation parameters", 2, errors)

    if json_output or quiet:
        configure_quiet_output()

    try:
        output_path = generate_output_path(prompt, output)
    except ValueError as exc:
        emit_error(json_output, "InvalidParametersError", str(exc), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_seed = resolve_seed(seed)
    show = not json_output and not quiet

    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except Exception as exc:
        emit_error(json_output, "DependencyError", str(exc), 1)

    try:
        resolved_device = resolve_device(device, torch)
        dtype_name, torch_dtype = resolve_dtype(dtype, resolved_device, torch)
    except ValueError as exc:
        emit_error(json_output, "InvalidParametersError", str(exc), 2)

    if show:
        console.print(
            Panel(
                f"[bold]{prompt}[/bold]\n"
                f"{width}x{height} | {steps} steps | seed {actual_seed} | {resolved_device} | {dtype_name}",
                title="imagen",
            )
        )

    t0 = time.time()
    try:
        status_ctx = console.status("Loading model...", spinner="dots") if show else nullcontext()
        with status_ctx:
            pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch_dtype)
            pipe.set_progress_bar_config(disable=True)
            if resolved_device == "cuda" and cpu_offload:
                pipe.enable_sequential_cpu_offload()
                placement = "cuda with sequential CPU offload"
            else:
                pipe.to(resolved_device)
                placement = resolved_device
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as exc:
        emit_error(json_output, "ModelLoadError", str(exc), 1)

    load_time = time.time() - t0
    t1 = time.time()

    try:
        progress_ctx = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )
            if show
            else nullcontext()
        )

        with progress_ctx as progress:
            step_callback = None
            if show and progress is not None:
                task = progress.add_task("Generating", total=steps)

                def step_callback(pipe, step, timestep, kwargs):
                    progress.update(task, completed=step + 1)
                    return kwargs

            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device="cpu").manual_seed(actual_seed),
                callback_on_step_end=step_callback,
            )
            image = result.images[0]
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as exc:
        message = str(exc)
        error_type = "OutOfMemoryError" if "out of memory" in message.lower() else "GenerationError"
        if error_type == "OutOfMemoryError":
            message = (
                "out of memory. Try reducing resolution, reducing steps, "
                "or keeping sequential CPU offload enabled."
            )
        emit_error(json_output, error_type, message, 1)

    gen_time = time.time() - t1
    total_time = time.time() - t0
    image.save(output_path)
    abs_path = str(output_path.resolve())

    payload = {
        "status": "success",
        "output_path": abs_path,
        "prompt": prompt,
        "seed": actual_seed,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "model": model,
        "device": resolved_device,
        "placement": placement,
        "dtype": dtype_name,
        "stats": {
            "model_load_seconds": round(load_time, 2),
            "generation_seconds": round(gen_time, 2),
            "seconds_per_step": round(gen_time / steps, 2),
            "total_seconds": round(total_time, 2),
        },
    }

    if json_output:
        print(json.dumps(payload, indent=2))
        return

    table = Table(title="Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Model load", f"{load_time:.1f}s")
    table.add_row("Generation", f"{gen_time:.1f}s")
    table.add_row("Per step", f"{gen_time / steps:.2f}s/step")
    table.add_row("Total", f"{total_time:.1f}s")
    table.add_section()
    table.add_row("Device", resolved_device)
    table.add_row("Placement", placement)
    table.add_row("Dtype", dtype_name)
    table.add_row("Resolution", f"{width}x{height}")
    table.add_row("Steps", str(steps))
    table.add_row("Seed", str(actual_seed))
    console.print(table)
    console.print(f"Saved to [bold]{abs_path}[/bold]")


@app.command()
def info(json_output: Annotated[bool, typer.Option("--json", help="Write JSON to stdout")] = False) -> None:
    """Show environment and configuration info."""
    if json_output:
        configure_quiet_output()

    import torch

    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]

    data = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "mps_available": torch.backends.mps.is_available(),
        "default_model": DEFAULT_MODEL,
        "default_device": resolve_device("auto", torch),
        "default_dtype": "auto",
    }

    try:
        from diffusers import __version__ as diffusers_version

        data["diffusers_version"] = diffusers_version
    except ImportError:
        data["diffusers_version"] = "not installed"

    if json_output:
        print(json.dumps(data, indent=2))
        return

    table = Table(title="imagen environment")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="bold")
    for key, value in data.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


def main() -> None:
    # Preserve the old `python main.py "prompt"` workflow.
    command_names = {"generate", "info"}
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in command_names:
        sys.argv.insert(1, "generate")
    app()


if __name__ == "__main__":
    main()
