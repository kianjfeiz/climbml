"""Command-line entry point: ``climbml <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .detect.export import FORMATS


def cmd_colors(args) -> None:
    from .route.pipeline import sample_colors

    for hold_id, hue, sat, val, name in sample_colors(args.image):
        print(f"  #{hold_id:3d}  h={hue:5.1f} s={sat:.2f} v={val:.2f}  -> {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="climbml", description=__doc__)
    parser.add_argument("--version", action="version", version=f"climbml {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    explore = sub.add_parser("explore", help="dataset composition and sample grids")
    explore.add_argument("--no-plots", action="store_true")

    repair = sub.add_parser("repair-labels", help="find rotation-corrupted labels")
    repair.add_argument("run_dir", type=Path)
    repair.add_argument("--apply", action="store_true", help="rewrite the label files")
    repair.add_argument("--conf", type=float, default=0.25)

    train = sub.add_parser("train", help="fine-tune the detector")
    train.add_argument("--smoke", action="store_true", help="1-epoch pipeline check")
    train.add_argument("--model", default="yolo26n.pt")
    train.add_argument("--name")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--device", help="mps, cpu, 0 (default: best available)")

    evaluate = sub.add_parser("eval", help="metrics and worst-image error analysis")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--split", default="val", choices=["val", "test"])
    evaluate.add_argument("--worst", type=int, default=8)
    evaluate.add_argument("--conf", type=float, default=0.25)

    export = sub.add_parser("export", help="export a run with a parity check")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--format", default=FORMATS[0], choices=FORMATS)
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--no-half", action="store_true", help="export at fp32")
    export.add_argument("--dest", type=Path)

    colors = sub.add_parser("colors", help="per-hold HSV and colour bin for one photo")
    colors.add_argument("image", type=Path)

    candidates = sub.add_parser("beta-candidates", help="dense wall images to curate")
    candidates.add_argument("--min-holds", type=int, default=22)
    candidates.add_argument("--limit", type=int, default=40)

    prep = sub.add_parser("beta-prep", help="build payloads without calling the API")
    prep.add_argument("--sweep", action="store_true", help="try the top 3 colours per image")

    run = sub.add_parser("beta-run", help="generate beta for the evaluation routes")
    run.add_argument("--variant", default="low",
                     choices=("thinking", "medium", "low", "fast"))
    run.add_argument("--model", default=None,
                     help="OpenRouter slug, e.g. <provider>/<model>; "
                          "defaults to $CLIMBML_BETA_MODEL")
    run.add_argument("--only", help="substring of the image stem")

    sub.add_parser("beta-report", help="scoreboard across saved runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "explore":
        from .data.annotations import cmd_explore
        return cmd_explore(args)
    if args.command == "repair-labels":
        from .data.repair import cmd_repair_labels
        return cmd_repair_labels(args)
    if args.command == "train":
        from .detect.train import cmd_train
        return cmd_train(args)
    if args.command == "eval":
        from .detect.evaluate import cmd_eval
        return cmd_eval(args)
    if args.command == "export":
        from .detect.export import cmd_export
        return cmd_export(args)
    if args.command == "colors":
        return cmd_colors(args)

    from .beta import harness
    if args.command == "beta-candidates":
        return harness.cmd_candidates(args)
    if args.command == "beta-prep":
        return harness.cmd_prep(args)
    if args.command == "beta-run":
        return harness.cmd_run(args)
    if args.command == "beta-report":
        return harness.cmd_report(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
