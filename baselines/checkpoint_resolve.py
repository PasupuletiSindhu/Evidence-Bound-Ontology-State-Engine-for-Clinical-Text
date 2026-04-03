"""Resolve trained relation checkpoints (HuggingFace Trainer layout)."""
from __future__ import annotations

from pathlib import Path


def _has_relation_weights(p: Path) -> bool:
    return (p / "pytorch_model.bin").is_file() or (p / "model.safetensors").is_file()


def relation_load_dir(rel_root: Path) -> Path:
    """
    Return a directory that BERTRelationModel.load() can use: either ``rel_root`` if it
    contains weights, or the newest ``checkpoint-*`` subdirectory that does.
    """
    rel_root = rel_root.resolve()
    if not rel_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {rel_root}")
    if _has_relation_weights(rel_root):
        return rel_root
    candidates = sorted(
        (
            p
            for p in rel_root.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-") and _has_relation_weights(p)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No pytorch_model.bin or model.safetensors under {rel_root} "
        "or in checkpoint-* subdirectories."
    )


def try_relation_load_dir(rel_root: Path) -> Path | None:
    try:
        return relation_load_dir(rel_root)
    except FileNotFoundError:
        return None
