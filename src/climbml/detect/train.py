"""Fine-tune a YOLO detector on the climbing-holds dataset.

    climbml train --smoke        # 1-epoch pipeline check on 25% of train
    climbml train                # full run (120 epochs, early-stop patience 30)

Recipe:

* ``yolo26n.pt`` pretrained on COCO. Its NMS-free head emits final boxes
  directly, so there is no post-processing to reimplement per export target.
* Augmentation changed from the ultralytics defaults in three places:
  ``flipud=0`` because an upside-down wall does not occur, ``degrees=10``
  because handheld photos are tilted, and ``fliplr=0.5`` kept because left and
  right are symmetric. Mosaic stays on: it composites the dataset's single-hold
  catalog shots into scene-like images.
* Everything else stays at defaults. The pretrained weights already carry a
  tuned recipe, so change it only with a reason.
* Loss and gains stay at YOLO26 defaults. Class health is tracked with per-class
  AP instead of reweighting the loss.
"""

from __future__ import annotations

from .. import config

BASE = dict(
    imgsz=640,
    batch=16,
    seed=0,
    workers=4,
    exist_ok=True,
    # augmentation changed for this domain, see the module docstring
    flipud=0.0,
    fliplr=0.5,
    degrees=10.0,
)


def train(model: str = "yolo26n.pt", name: str | None = None, epochs: int = 120,
          patience: int = 30, device: str | None = None, smoke: bool = False):
    from ultralytics import YOLO

    overrides = dict(BASE,
                     data=str(config.DATA_YAML),
                     project=str(config.RUNS_DIR),
                     device=config.resolve_device(device))
    if smoke:
        overrides.update(epochs=1, fraction=0.25, plots=False,
                         name=name or "smoke")
    else:
        overrides.update(epochs=epochs, patience=patience,
                         name=name or "y26n-640-v12-a")

    results = YOLO(model).train(**overrides)
    print(f"\nrun dir: {results.save_dir}")
    return results


def cmd_train(args) -> None:
    train(model=args.model, name=args.name, epochs=args.epochs,
          patience=args.patience, device=args.device, smoke=args.smoke)
