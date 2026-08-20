"""Export a trained run for deployment, with a numerical parity check.

    climbml export runs/detect/y26n-640-v12-a --format coreml

An export that loads is not an export that works. Quantisation and the runtime's
own resize can move boxes and confidences, so every export is followed by
running the same images through both the PyTorch weights and the exported model.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .. import config

FORMATS = ("coreml", "onnx", "torchscript")


def export(run_dir: Path, fmt: str = "coreml", imgsz: int = 640,
           half: bool = True, dest: Path | None = None, samples: int = 5) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(run_dir / "weights/best.pt"))
    exported = Path(model.export(format=fmt, half=half, imgsz=imgsz))

    dest = Path(dest) if dest else config.ARTIFACTS_DIR / "export" / exported.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    shutil.move(str(exported), dest)
    size_mb = (sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
               if dest.is_dir() else dest.stat().st_size) / 1e6
    print(f"exported -> {dest}  ({size_mb:.1f} MB)")

    images = sorted((config.split_dir("valid") / "images").glob("*.jpg"))[:samples]
    if not images:
        print("no validation images available — skipping parity check")
        return dest

    exported_model = YOLO(str(dest), task="detect")
    print("\nparity check (conf=0.25):")
    print(f"{'image':<38} {'torch':>6} {'export':>7} {'torch conf':>11} {'export conf':>12}")
    for image in images:
        torch_result = model.predict(str(image), conf=0.25, verbose=False)[0]
        export_result = exported_model.predict(str(image), conf=0.25, verbose=False,
                                               imgsz=imgsz)[0]
        torch_conf = float(torch_result.boxes.conf.mean()) if len(torch_result.boxes) else 0.0
        export_conf = float(export_result.boxes.conf.mean()) if len(export_result.boxes) else 0.0
        print(f"{image.name[:38]:<38} {len(torch_result.boxes):>6} "
              f"{len(export_result.boxes):>7} {torch_conf:>11.3f} {export_conf:>12.3f}")
    print("\nsmall deltas are expected (fp16 + resize); large gaps mean an export bug")
    return dest


def cmd_export(args) -> None:
    export(args.run_dir, fmt=args.format, imgsz=args.imgsz,
           half=not args.no_half, dest=args.dest)
