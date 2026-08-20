"""Filesystem layout and runtime defaults.

Every path can be overridden by environment variable, so the package works
outside a checkout: CI, a notebook, or someone else's copy of the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# Roboflow-format YOLO dataset: data.yaml + {train,valid,test}/{images,labels}
DATASET_DIR = _path_env("CLIMBML_DATASET", PROJECT_ROOT / "data/climbing-holds-and-volumes")
DATA_YAML = DATASET_DIR / "data.yaml"

# Ultralytics writes run directories here; exports and reports land in artifacts.
RUNS_DIR = _path_env("CLIMBML_RUNS", PROJECT_ROOT / "runs/detect")
ARTIFACTS_DIR = _path_env("CLIMBML_ARTIFACTS", PROJECT_ROOT / "artifacts")

# Detector weights used by the route/beta pipeline when no path is passed.
DEFAULT_WEIGHTS = _path_env("CLIMBML_WEIGHTS", PROJECT_ROOT / "weights/best.pt")

SPLITS = ("train", "valid", "test")


def split_dir(split: str) -> Path:
    """Dataset directory for a split name, accepting the val/valid synonyms."""
    return DATASET_DIR / ("valid" if split == "val" else split)


def resolve_device(device: str | None = None) -> str:
    """Pick the best available torch device unless one was requested."""
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
