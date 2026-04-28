# Contributing

## Development Setup

Install the package in editable mode:

```bash
uv pip install -e . --python ../dude.k/.venv/Scripts/python.exe
```

Or with a normal Python environment:

```bash
pip install -e ".[dev]"
```

## Local Checks

Run syntax checks:

```bash
python -m compileall custom_ballspotting
```

Run formatting and lint checks when dev dependencies are installed:

```bash
ruff check .
ruff format --check .
```

## Repository Hygiene

- Do not commit datasets, extracted frames, checkpoints, predictions, or videos.
- Keep CLI commands thin; reusable logic should live in package modules.
- Prefer config-file workflows for reproducible training and inference runs.
- Keep action labels and class-specific weights in `custom_ballspotting/actions.py`.

## Checkpoint Compatibility

When loading T-DEED or DUDEK checkpoints as pretrained weights, keep model-shape
settings aligned with the checkpoint:

- `features_model_name`
- `temporal_shift_mode`
- `clip_frames_count`
- `n_layers`
- `sgp_ks`
- `sgp_k`

The custom classifier head is intentionally reinitialized for custom labels.
