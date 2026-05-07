import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "run"


def render_checkpoint_path(
    template: str | None,
    experiment_name: str,
    timestamp: str | None = None,
) -> str:
    timestamp = timestamp or utc_timestamp()
    if template is None:
        template = "checkpoints/{experiment_name}_{timestamp}_best.pt"
    rendered = template.format(
        experiment_name=slugify(experiment_name),
        timestamp=timestamp,
    )
    if not rendered.endswith((".pt", ".pth", ".ckpt")):
        rendered = os.path.join(rendered, "{experiment_name}_{timestamp}_best.pt").format(
            experiment_name=slugify(experiment_name),
            timestamp=timestamp,
        )
    return rendered


def run_artifact_root(best_checkpoint_path: str) -> str:
    """Root directory for per-epoch weights and run-scoped metadata.

    If ``save_as`` is nested as ``…/<run_id>/<run_id>_best.pt``, epoch artifacts
    live beside the best file under ``…/<run_id>/``. If the best file is flat
    ``…/<stem>_best.pt``, use ``…/<stem>/`` so multiple runs do not share one
    ``epochs/`` folder.
    """
    parent = os.path.dirname(os.path.abspath(best_checkpoint_path))
    stem = Path(best_checkpoint_path).stem
    parent_base = os.path.basename(parent.rstrip(os.sep))
    if parent_base and stem.startswith(parent_base + "_"):
        return parent
    return os.path.join(parent, stem)


def epoch_checkpoint_dirs(best_checkpoint_path: str) -> tuple[str, str]:
    """Return ``(epochs_dir, metadata_dir)`` under :func:`run_artifact_root`."""
    root = run_artifact_root(best_checkpoint_path)
    return (
        os.path.join(root, "epochs"),
        os.path.join(root, "metadata"),
    )


def write_checkpoint_metadata(
    checkpoint_path: str,
    metadata: dict[str, Any],
    *,
    metadata_file: str | None = None,
) -> str:
    if metadata_file is None:
        metadata_file = f"{Path(checkpoint_path).with_suffix('')}.metadata.json"
    meta_parent = os.path.dirname(os.path.abspath(metadata_file))
    if meta_parent:
        os.makedirs(meta_parent, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata_file


def metadata_path_for_checkpoint(checkpoint_path: str) -> str:
    """Sibling file: ``<stem>.metadata.json`` next to ``<stem>.pt``."""
    p = Path(checkpoint_path).resolve()
    return f"{p.with_suffix('')}.metadata.json"


def read_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    """Load training metadata written beside the checkpoint, if present."""
    meta_path = metadata_path_for_checkpoint(checkpoint_path)
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)
