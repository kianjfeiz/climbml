"""Evaluate a trained run: per-class metrics plus worst-image error analysis.

    climbml eval runs/detect/y26n-640-v12-a
    climbml eval runs/detect/y26n-640-v12-a --split test   # one-time final check

The aggregate metrics come from ultralytics. The second half ranks the split by
per-image recall and writes out the worst images, which is how the corrupted
annotations in the dataset were found (see docs/experiments.md).
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from .metrics import load_gt, per_image_pr


def evaluate(run_dir: Path, split: str = "val", worst: int = 8, conf: float = 0.25):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(run_dir / "weights/best.pt"))

    metrics = model.val(data=str(config.DATA_YAML), split=split, plots=True)
    print(f"\n== {split} metrics ==")
    print(f"mAP50    {metrics.box.map50:.4f}")
    print(f"mAP50-95 {metrics.box.map:.4f}")
    for i, name in model.names.items():
        print(f"  {name:<8} AP50 {metrics.box.ap50[i]:.4f}   AP50-95 {metrics.box.ap[i]:.4f}")

    img_dir = config.split_dir(split) / "images"
    label_dir = config.split_dir(split) / "labels"
    scored = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        preds = [(int(b.cls), tuple(b.xyxy[0].tolist()), float(b.conf))
                 for b in result.boxes]
        height, width = result.orig_shape
        gts = load_gt(label_dir / f"{img_path.stem}.txt", width, height)
        precision, recall = per_image_pr(preds, gts)
        scored.append((recall, precision, img_path, result))
    if not scored:
        print(f"no images found in {img_dir}")
        return metrics
    scored.sort(key=lambda row: (row[0], row[1]))

    out_dir = run_dir / f"error_analysis_{split}"
    out_dir.mkdir(exist_ok=True)
    for recall, precision, img_path, result in scored[:worst]:
        cv2.imwrite(str(out_dir / f"r{recall:.2f}_p{precision:.2f}_{img_path.name}"),
                    result.plot())

    mean_p = sum(row[1] for row in scored) / len(scored)
    mean_r = sum(row[0] for row in scored) / len(scored)
    print(f"\nworst {worst} images by recall -> {out_dir}")
    print(f"per-image means @conf{conf}: precision {mean_p:.3f}, recall {mean_r:.3f}")
    return metrics


def cmd_eval(args) -> None:
    evaluate(args.run_dir, split=args.split, worst=args.worst, conf=args.conf)
