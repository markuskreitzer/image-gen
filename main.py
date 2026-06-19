import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with FLUX.2-klein-4B")
    parser.add_argument("prompt", help="Text prompt to generate")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path (default: output.png)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--width", type=positive_int, default=1024, help="Image width in pixels (default: 1024)")
    parser.add_argument("--height", type=positive_int, default=1024, help="Image height in pixels (default: 1024)")
    parser.add_argument("--steps", type=positive_int, default=4, help="Inference steps (default: 4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help="Torch dtype for loading model weights (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device (default: auto)",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable sequential CPU offloading and place the pipeline on the selected device",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return requested


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=DTYPES[args.dtype],
    )

    if device == "cuda" and not args.no_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        print("Pipeline ready on CUDA with sequential CPU offloading.")
    else:
        pipe.to(device)
        print(f"Pipeline ready on {device}.")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating: {args.prompt!r}")
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]

    image.save(output_path)
    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
