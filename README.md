# imagen

Local image generation CLI powered by [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Runs on Apple Silicon via MPS. Designed for both human and AI agent use.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- ~8GB free memory (model runs in bfloat16)

## Install

```bash
git clone https://github.com/markuskreitzer/image-gen.git
cd image-gen
uv sync
```

## Commands

### `imagen generate`

Generate an image from a text prompt.

```bash
uv run imagen generate "A cat holding a sign that says hello world"
```

#### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | PATH | auto | Output file path |
| `--seed` | `-s` | INT | random | Random seed for reproducibility |
| `--steps` | `-n` | INT | 4 | Number of inference steps |
| `--width` | `-W` | INT | 1024 | Image width (must be multiple of 8, max 4096) |
| `--height` | `-H` | INT | 1024 | Image height (must be multiple of 8, max 4096) |
| `--guidance-scale` | `-g` | FLOAT | 1.0 | Classifier-free guidance scale |
| `--model` | `-m` | TEXT | `black-forest-labs/FLUX.2-klein-4B` | HuggingFace model ID |
| `--device` | `-d` | TEXT | mps | Torch device |
| `--json` | | FLAG | false | Output JSON to stdout (for agents) |
| `--quiet` | `-q` | FLAG | false | Suppress non-essential output |

Default output filename: `{YYYYMMDD_HHMMSS}_{prompt_slug}.png`

### `imagen info`

Show environment and configuration details.

```bash
uv run imagen info
uv run imagen info --json
```

## Agent Integration

Use `--json` for structured output. JSON is written to **stdout only**; progress bars go to stderr.

### Generate an image

```bash
uv run imagen generate --json --seed 42 "A photorealistic sunset over mountains"
```

**stdout:**

```json
{
  "status": "success",
  "output_path": "/absolute/path/to/20260329_143052_a_photorealistic_sunset.png",
  "prompt": "A photorealistic sunset over mountains",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "guidance_scale": 1.0,
  "model": "black-forest-labs/FLUX.2-klein-4B",
  "device": "mps",
  "dtype": "bfloat16",
  "stats": {
    "model_load_seconds": 15.3,
    "generation_seconds": 95.2,
    "seconds_per_step": 23.8,
    "total_seconds": 110.5
  }
}
```

### Check environment

```bash
uv run imagen info --json
```

```json
{
  "python_version": "3.13.11",
  "torch_version": "2.11.0",
  "mps_available": true,
  "cuda_available": false,
  "default_model": "black-forest-labs/FLUX.2-klein-4B",
  "default_device": "mps",
  "default_dtype": "bfloat16",
  "diffusers_version": "0.37.1"
}
```

### Error response

On failure, JSON output includes `status: "error"` with a classified `error_type`:

```json
{
  "status": "error",
  "error_type": "InvalidParametersError",
  "errors": ["Width must be a positive multiple of 8, got 100"]
}
```

Error types: `InvalidParametersError` (exit 2), `ModelLoadError` (exit 1), `OutOfMemoryError` (exit 1), `GenerationError` (exit 1).

### Agent workflow

1. Run `imagen info --json` to verify the environment
2. Run `imagen generate --json --seed <seed> "<prompt>"` to generate
3. Parse stdout as JSON, check `status == "success"`
4. Read `output_path` for the absolute path to the generated image
5. `seed` is always returned (even when random) for reproducibility

### Performance expectations

On Apple Silicon with cached model:

| Metric | Typical |
|--------|---------|
| Model load | 15-20s |
| Generation (4 steps, 1024x1024) | 90-100s |
| Per step | ~24s |

First run downloads ~8GB of model weights (cached in `~/.cache/huggingface/hub`).
