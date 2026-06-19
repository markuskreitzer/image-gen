# image-gen

Small CLI for generating images with Black Forest Labs'
`FLUX.2-klein-4B` pipeline through Hugging Face Diffusers.

## Requirements

- Python 3.12
- `uv`
- CUDA-capable GPU recommended
- Hugging Face access for `black-forest-labs/FLUX.2-klein-4B`

If the model requires authentication in your environment, run:

```bash
huggingface-cli login
```

## Setup

```bash
uv sync
```

The model weights are downloaded by Diffusers on first use and cached under the
standard Hugging Face cache directory.

## Usage

```bash
uv run python main.py "a simple red cube on a clean white background" \
  --output outputs/red-cube.png \
  --width 512 \
  --height 512 \
  --steps 4 \
  --seed 42
```

Common options:

```bash
uv run python main.py --help
```

By default, the script uses sequential CPU offloading on CUDA to reduce VRAM
pressure. Use `--no-cpu-offload` only if the full model fits in GPU memory.

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
