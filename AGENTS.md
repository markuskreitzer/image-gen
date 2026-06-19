# Repository Instructions

## Branch Policy

- Use `master` as the default branch name.
- Do not rename this repository or new repositories to `main`.

## Project Layout

- `image_gen/cli.py` contains the Typer CLI implementation.
- `main.py` is a compatibility shim for `uv run python main.py "prompt"`.
- `README.md` documents user-facing setup and commands.
- `uv.lock` is committed and should stay in sync with `pyproject.toml`.

## Development

- Prefer `uv run imagen ...` for exercising the installed CLI.
- Keep generated images in `outputs/` or as ignored image files.
- Do not commit generated PNG/JPEG/WebP outputs or local model weights.
- Preserve CUDA, MPS, and CPU compatibility unless a change explicitly narrows support.

## Validation

Run these checks after CLI or dependency changes:

```bash
uv lock
uv run python -m compileall main.py image_gen
uv run imagen --help
uv run imagen generate --help
uv run imagen info --json
```

For behavior changes, also run a small generation smoke test:

```bash
uv run imagen "a simple red cube on a clean white background" \
  --output outputs/smoke.png \
  --width 512 \
  --height 512 \
  --steps 1 \
  --seed 42
```
