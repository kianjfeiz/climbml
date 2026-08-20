"""Per-image detection metrics.

Ultralytics reports dataset-level mAP, which says whether the model is good but
not which images it is bad on. These are the per-image equivalents, used to rank
a split worst-first for error analysis.
"""

from __future__ import annotations

from pathlib import Path

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 width: float, height: float) -> Box:
    """Normalised YOLO centre-form box -> pixel corners."""
    return ((cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height)


def load_gt(label_path: Path, width: float, height: float) -> list[tuple[int, Box]]:
    """Ground-truth boxes for one image as (class, pixel corners)."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = (float(v) for v in parts)
        out.append((int(cls), yolo_to_xyxy(cx, cy, w, h, width, height)))
    return out


def per_image_pr(preds: list[tuple[int, Box, float]],
                 gts: list[tuple[int, Box]],
                 thr: float = 0.5) -> tuple[float, float]:
    """Greedy confidence-ordered IoU matching -> (precision, recall).

    An image with no predictions and no ground truth scores 1.0 on both: there
    was nothing to find and nothing was invented.
    """
    matched: set[int] = set()
    tp = 0
    for cls, box, _conf in sorted(preds, key=lambda p: -p[2]):
        best_i, best_iou = None, thr
        for i, (gt_cls, gt_box) in enumerate(gts):
            if i in matched or gt_cls != cls:
                continue
            overlap = iou(box, gt_box)
            if overlap >= best_iou:
                best_i, best_iou = i, overlap
        if best_i is not None:
            matched.add(best_i)
            tp += 1
    precision = tp / len(preds) if preds else 1.0
    recall = tp / len(gts) if gts else 1.0
    return precision, recall
