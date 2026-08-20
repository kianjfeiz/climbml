"""Photo -> detected holds -> isolated route -> beta-engine payload.

    photo ──▶ YOLO detector ──▶ colour sampling ──▶ clustering ──▶ payload
                                                                  (SoM image
                                                                   + hold JSON)

The payload is two views of the same thing: an annotated image to read, and the
same holds as JSON so positions and sizes are exact rather than estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .. import config
from .cluster import cluster
from .color import Sampler
from .holds import Hold, holds_from_labels
from .som import draw_som

CONF = 0.30                     # detector confidence floor for route building
IMGSZ = 640
MAX_SEND_SIDE = 1568            # long edge sent to the API; caps vision token cost

_models: dict[str, object] = {}


@dataclass
class Payload:
    """Everything the beta engine is given about one route."""

    image: Image.Image          # Set-of-Mark annotated, resized for the API
    holds_json: list[dict]      # route holds, bottom-to-top
    color: str
    route: list[Hold]
    start_ids: list[int]
    scale: float                # payload pixels per original pixel


def load_image(path: Path) -> Image.Image:
    """Open a photo with EXIF rotation applied, as RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def detect(img: Image.Image, weights: Path | None = None,
           conf: float = CONF, device: str = "cpu") -> list[Hold]:
    """Run the detector over a photo. Weights are loaded once per process.

    Defaults to CPU. The model is small enough to be fast there, which leaves
    the GPU free for a training run.
    """
    weights = Path(weights or config.DEFAULT_WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(
            f"detector weights not found at {weights} — train a model first "
            "(climbml train) or set CLIMBML_WEIGHTS")
    key = str(weights)
    if key not in _models:
        from ultralytics import YOLO
        _models[key] = YOLO(key)
    result = _models[key].predict(img, conf=conf, imgsz=IMGSZ, device=device, verbose=False)[0]
    return [Hold(i, *xyxy, cls=int(c), conf=float(cf))
            for i, (xyxy, c, cf) in enumerate(zip(result.boxes.xyxy.tolist(),
                                                  result.boxes.cls.tolist(),
                                                  result.boxes.conf.tolist(),
                                                  strict=True))]


def build_payload(img: Image.Image, holds: list[Hold], route_color: str | None = None,
                  start_ids: list[int] | None = None) -> Payload | None:
    """Sample colours, isolate one route, and package it for the beta engine.

    Returns None when no colour group is large enough to be a route.
    """
    sampler = Sampler(img)
    for hold in holds:
        hold.hsv = sampler.median_hsv(hold)

    clusters = cluster(holds, img.size)
    if not clusters:
        return None
    chosen = next((c for c in clusters if c.color == route_color), clusters[0])
    starts = set(start_ids or chosen.start_ids)

    route = [h for h in holds if h.id in set(chosen.hold_ids)]
    route.sort(key=lambda h: -h.cy)                   # bottom-to-top, as climbed

    width, height = img.size
    holds_json = [{
        "id": h.id,
        "x": round(h.cx / width * 100, 1),            # % from the left edge
        "y": round(h.cy / height * 100, 1),           # % from the TOP edge
        "w": round(h.w / width * 100, 1),
        "h": round(h.h / height * 100, 1),
        "isStart": h.id in starts,
    } for h in route]

    som = draw_som(img, route, starts)
    scale = min(1.0, MAX_SEND_SIDE / max(som.size))
    if scale < 1.0:
        som = som.resize((int(som.width * scale), int(som.height * scale)), Image.LANCZOS)
    return Payload(som, holds_json, chosen.color, route, sorted(starts), scale)


def analyze(path: Path, route_color: str | None = None, use_gt: bool = False,
            weights: Path | None = None) -> Payload | None:
    """Full photo-to-payload pipeline.

    With ``use_gt`` the dataset's own labels stand in for the detector, which
    separates route-isolation errors from detection errors.
    """
    img = load_image(path)
    if use_gt:
        label = path.parent.parent / "labels" / f"{path.stem}.txt"
        holds = holds_from_labels(label, *img.size)
    else:
        holds = detect(img, weights=weights)
    return build_payload(img, holds, route_color)


def inspect_clusters(path: Path, weights: Path | None = None):
    """Colour groups found in a photo — used to curate evaluation routes."""
    img = load_image(path)
    holds = detect(img, weights=weights)
    sampler = Sampler(img)
    for hold in holds:
        hold.hsv = sampler.median_hsv(hold)
    return cluster(holds, img.size)


def sample_colors(path: Path,
                  weights: Path | None = None) -> list[tuple[int, float, float, float, str]]:
    """Per-hold (id, hue, saturation, value, bin) for tuning the colour bins."""
    from .color import bin_name

    img = load_image(path)
    holds = detect(img, weights=weights)
    sampler = Sampler(img)
    rows = []
    for hold in holds:
        if hold.cls == 1:
            continue
        hsv = sampler.median_hsv(hold)
        if hsv is not None:
            rows.append((hold.id, *hsv, bin_name(*hsv)))
    return sorted(rows, key=lambda row: (row[4], row[1]))
