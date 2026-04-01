from __future__ import annotations

import json
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

app = typer.Typer(
    name="imagen",
    help="Generate images with FLUX.2-klein on Apple Silicon.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

console = Console()

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len].rstrip("_")


def generate_output_path(prompt: str, output: Path | None) -> Path:
    if output is not None:
        p = Path(output)
        if p.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return p / f"{timestamp}_{slugify(prompt)}.png"
        if p.suffix == "":
            return p.with_suffix(".png")
        if p.suffix != ".png":
            console.print(f"[red]Error:[/red] Only PNG output is supported, got '{p.suffix}'")
            raise typer.Exit(code=2)
        return p
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"{timestamp}_{slugify(prompt)}.png")


def validate_params(width: int, height: int, steps: int, guidance_scale: float) -> list[str]:
    errors = []
    if width <= 0 or width % 8 != 0:
        errors.append(f"Width must be a positive multiple of 8, got {width}")
    if height <= 0 or height % 8 != 0:
        errors.append(f"Height must be a positive multiple of 8, got {height}")
    if width > 4096 or height > 4096:
        errors.append(f"Maximum dimension is 4096, got {width}x{height}")
    if steps < 1 or steps > 100:
        errors.append(f"Steps must be between 1 and 100, got {steps}")
    if guidance_scale < 0:
        errors.append(f"Guidance scale must be >= 0, got {guidance_scale}")
    return errors


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.randint(0, 2**32 - 1)


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Text prompt for image generation")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", "-s", help="Random seed (default: random)")] = None,
    steps: Annotated[int, typer.Option("--steps", "-n", help="Inference steps")] = 4,
    width: Annotated[int, typer.Option("--width", "-W", help="Image width (multiple of 8)")] = 1024,
    height: Annotated[int, typer.Option("--height", "-H", help="Image height (multiple of 8)")] = 1024,
    guidance_scale: Annotated[float, typer.Option("--guidance-scale", "-g", help="Guidance scale")] = 1.0,
    model: Annotated[str, typer.Option("--model", "-m", help="HuggingFace model ID")] = DEFAULT_MODEL,
    device: Annotated[str, typer.Option("--device", "-d", help="Torch device")] = "mps",
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON for agent consumption")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-essential output")] = False,
):
    """Generate an image from a text prompt using FLUX.2-klein."""
    # Validate parameters before loading model
    errors = validate_params(width, height, steps, guidance_scale)
    if errors:
        if json_output:
            print(json.dumps({"status": "error", "error_type": "InvalidParametersError", "errors": errors}))
            raise typer.Exit(code=2)
        for e in errors:
            console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)

    actual_seed = resolve_seed(seed)
    output_path = generate_output_path(prompt, output)

    # Lazy imports so --help is instant
    import torch
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16
    show = not json_output and not quiet

    if not show:
        import os
        os.environ["TQDM_DISABLE"] = "1"
        import logging
        logging.getLogger("diffusers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import diffusers.utils.logging as dl
        dl.set_verbosity_error()
        dl.disable_progress_bar()
        import transformers.utils.logging as tl
        tl.set_verbosity_error()

    if show:
        console.print(Panel(
            f"[bold]{prompt}[/bold]\n{width}x{height} | {steps} steps | seed {actual_seed}",
            title="imagen",
        ))

    # Load model
    t0 = time.time()
    try:
        with console.status("Loading model...", spinner="dots") if show else nullcontext():
            pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype)
            pipe.to(device)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        err = {"status": "error", "error_type": "ModelLoadError", "error_message": str(e)}
        if json_output:
            print(json.dumps(err))
        else:
            console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)

    load_time = time.time() - t0

    # Generate image
    t1 = time.time()
    try:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) if show else nullcontext()

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
                generator=torch.Generator("cpu").manual_seed(actual_seed),
                callback_on_step_end=step_callback,
            )
            image = result.images[0]

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        err_type = "OutOfMemoryError" if "out of memory" in str(e).lower() else "GenerationError"
        err_msg = str(e)
        if err_type == "OutOfMemoryError":
            err_msg = f"Out of memory. Try reducing resolution (current: {width}x{height}) or steps (current: {steps})."
        if json_output:
            print(json.dumps({"status": "error", "error_type": err_type, "error_message": err_msg}))
        else:
            console.print(f"[red]Error:[/red] {err_msg}")
        raise typer.Exit(code=1)

    gen_time = time.time() - t1
    total_time = time.time() - t0

    # Save
    image.save(str(output_path))
    abs_path = str(output_path.resolve())

    # Output results
    if json_output:
        print(json.dumps({
            "status": "success",
            "output_path": abs_path,
            "prompt": prompt,
            "seed": actual_seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "model": model,
            "device": device,
            "dtype": "bfloat16",
            "stats": {
                "model_load_seconds": round(load_time, 2),
                "generation_seconds": round(gen_time, 2),
                "seconds_per_step": round(gen_time / steps, 2),
                "total_seconds": round(total_time, 2),
            },
        }, indent=2))
    else:
        table = Table(title="Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Model load", f"{load_time:.1f}s")
        table.add_row("Generation", f"{gen_time:.1f}s")
        table.add_row("Per step", f"{gen_time / steps:.2f}s/step")
        table.add_row("Total", f"{total_time:.1f}s")
        table.add_section()
        table.add_row("Device", device)
        table.add_row("Dtype", "bfloat16")
        table.add_row("Resolution", f"{width}x{height}")
        table.add_row("Steps", str(steps))
        table.add_row("Seed", str(actual_seed))
        console.print(table)
        console.print(f"Saved to [bold]{abs_path}[/bold]")


@app.command()
def info(
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
):
    """Show environment and configuration info."""
    import torch

    data = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "default_model": DEFAULT_MODEL,
        "default_device": "mps",
        "default_dtype": "bfloat16",
    }

    try:
        from diffusers import __version__ as diffusers_version
        data["diffusers_version"] = diffusers_version
    except ImportError:
        data["diffusers_version"] = "not installed"

    if json_output:
        print(json.dumps(data, indent=2))
    else:
        table = Table(title="imagen environment")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="bold")
        for k, v in data.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)


if __name__ == "__main__":
    app()
