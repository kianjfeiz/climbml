"""Geometry types shared by detection, clustering and beta generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HOLD_CLASS = 0
VOLUME_CLASS = 1


@dataclass
class Hold:
    """One detected (or ground-truth) hold, in pixel coordinates."""

    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float
    hsv: tuple[float, float, float] | None = None
    bin: str | None = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class Cluster:
    """A colour group of holds — the candidate route."""

    color: str
    hold_ids: list[int]
    prominence: float
    start_ids: list[int] = field(default_factory=list)


def holds_from_labels(label_path: Path, width: int, height: int) -> list[Hold]:
    """Read a YOLO label file as Holds, for running the pipeline on ground truth."""
    holds = []
    for i, line in enumerate(label_path.read_text().splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = (float(v) for v in parts)
        holds.append(Hold(i,
                          (cx - w / 2) * width, (cy - h / 2) * height,
                          (cx + w / 2) * width, (cy + h / 2) * height,
                          int(c), 1.0))
    return holds
