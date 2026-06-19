# image-gen

Local image generation CLI powered by
[`FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
and Hugging Face Diffusers.

The CLI supports CUDA, Apple MPS, and CPU execution. On CUDA, it uses sequential
CPU offloading by default so the 4B model can run on smaller GPUs.

## Requirements

- Python 3.12+
- `uv`
- CUDA or Apple Silicon MPS recommended
- Hugging Face access for `black-forest-labs/FLUX.2-klein-4B`

If the model requires authentication in your environment:

```bash
huggingface-cli login
```

## Install

```bash
git clone https://github.com/markuskreitzer/image-gen.git
cd image-gen
uv sync
```

The model weights are downloaded by Diffusers on first use and cached under the
standard Hugging Face cache directory.

## Usage

Generate an image:

```bash
uv run imagen generate "a simple red cube on a clean white background" \
  --output outputs/red-cube.png \
  --width 512 \
  --height 512 \
  --steps 4 \
  --seed 42
```

The old direct-script workflow also works:

```bash
uv run python main.py "a simple red cube on a clean white background" --steps 4
```

Show environment details:

```bash
uv run imagen info
uv run imagen info --json
```

Common generation options:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `outputs/{timestamp}_{prompt}.png` | Output file or directory |
| `--seed` | `-s` | random | Random seed for reproducibility |
| `--steps` | `-n` | `4` | Inference steps |
| `--width` | `-W` | `1024` | Image width, multiple of 8 |
| `--height` | `-H` | `1024` | Image height, multiple of 8 |
| `--guidance-scale` | `-g` | `1.0` | Guidance scale |
| `--device` | `-d` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--dtype` | | `auto` | `auto`, `bfloat16`, `float16`, or `float32` |
| `--cpu-offload / --no-cpu-offload` | | on | Sequential CPU offload on CUDA |
| `--json` | | off | Write machine-readable JSON to stdout |
| `--quiet` | `-q` | off | Suppress non-essential output |

Use `--no-cpu-offload` only when the full model fits in GPU memory.

## Agent Integration

Use `--json` for structured output:

```bash
uv run imagen generate --json --seed 42 "a photorealistic sunset over mountains"
```

On success, stdout contains:

```json
{
  "status": "success",
  "output_path": "/absolute/path/to/outputs/20260618_223000_a_photorealistic_sunset.png",
  "prompt": "a photorealistic sunset over mountains",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "guidance_scale": 1.0,
  "model": "black-forest-labs/FLUX.2-klein-4B",
  "device": "cuda",
  "placement": "cuda with sequential CPU offload",
  "dtype": "bfloat16"
}
```

On failure, stdout contains `status: "error"` with an `error_type`.

## Viewing Images Over SSH With Kitty

When connected from kitty, display a generated image directly in the terminal:

```bash
kitten icat --transfer-mode=stream outputs/red-cube.png
```

Inside `tmux`, add passthrough support:

```bash
kitten icat --transfer-mode=stream --passthrough=tmux outputs/red-cube.png
```

Generated images are ignored by git so local test outputs do not bloat the
repository.
