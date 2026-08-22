"""Evaluation harness for the beta engine.

A boulder problem has several valid sequences, so there is no ground truth to
score against. Two checks stand in: geometric ones that flag plans no climber
could execute (see :func:`quality_report`), and rendered images for the calls
only a person can make. Runs are written to disk so variants stay comparable.

    climbml beta-candidates          # dense wall images worth curating
    climbml beta-prep                # payloads + Set-of-Mark renders, no API calls
    climbml beta-run --variant low   # generate, score, render
    climbml beta-report              # scoreboard across saved runs
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .. import config
from ..route.pipeline import analyze, inspect_clusters
from .engine import VARIANTS
from .render import render
from .schema import PROMPT_VERSION, quality_report

ROUTES_YAML = config.PROJECT_ROOT / "configs/eval_routes.yaml"
OUT_DIR = config.ARTIFACTS_DIR / "beta"

#: scoreboard thresholds; a plan crossing one gets flagged
MAX_DESCENTS = 1
MAX_SAME_HAND = 2
MAX_REACH_PCT = 40


def load_routes(path: Path = ROUTES_YAML) -> list[dict]:
    return yaml.safe_load(path.read_text())


def wall_path(route: dict) -> Path:
    return config.split_dir(route["split"]) / "images" / f"{route['stem']}.jpg"


def route_key(route: dict, color: str) -> str:
    return f"{route['stem'][:36]}.{color}"


def dense_images(min_holds: int = 22) -> list[tuple[str, str, int]]:
    """Dataset images with enough labelled holds to be a real wall, densest first."""
    rows = []
    for split in ("test", "valid"):
        for label in sorted((config.split_dir(split) / "labels").glob("*.txt")):
            count = len(label.read_text().splitlines())
            if count >= min_holds:
                rows.append((split, label.stem, count))
    return sorted(rows, key=lambda r: -r[2])


def cmd_candidates(args) -> None:
    rows = dense_images(args.min_holds)
    print(f"{len(rows)} images with >= {args.min_holds} labelled holds")
    for split, stem, count in rows[: args.limit]:
        print(f"  {count:4d}  {split:5s}  {stem}")


def cmd_prep(args) -> None:
    """Build payloads without calling the API — cheap way to check grounding."""
    out = OUT_DIR / "prep"
    out.mkdir(parents=True, exist_ok=True)
    for route in load_routes():
        path = wall_path(route)
        if args.sweep:
            clusters = inspect_clusters(path)
            table = "  ".join(f"{c.color}:{len(c.hold_ids)}@{c.prominence:.2f}"
                              for c in clusters[:5])
            print(f"  {route['stem'][:40]}: {table}")
            colors = [c.color for c in clusters[:3]]
        else:
            colors = [route.get("color")]

        for color in colors:
            payload = analyze(path, route_color=color)
            if payload is None:
                print(f"  {route['stem'][:40]}: no colour cluster found")
                continue
            key = route_key(route, payload.color)
            payload.image.save(out / f"{key}.som.jpg", quality=85)
            (out / f"{key}.payload.json").write_text(json.dumps({
                "color": payload.color,
                "starts": payload.start_ids,
                "holds": payload.holds_json,
            }, indent=1))
            if not args.sweep:
                print(f"  {route['stem'][:40]}: {payload.color} route, "
                      f"{len(payload.holds_json)} holds")
    print(f"-> {out}")


def run_dir_name(model: str, variant: str) -> str:
    """One directory per model+variant, so comparison runs never overwrite."""
    return f"{model.replace('/', '-')}.{variant}"


def cmd_run(args) -> None:
    from .engine import generate, resolve_model  # keep the API client import lazy

    model = resolve_model(args.model)
    variant = VARIANTS[args.variant]
    out = OUT_DIR / run_dir_name(model, args.variant)
    out.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0

    for route in load_routes():
        if args.only and args.only not in route["stem"]:
            continue
        payload = analyze(wall_path(route), route_color=route.get("color"))
        if payload is None:
            continue

        result = generate(payload, model=model, **variant)
        total_cost += result.cost
        holds_by_id = {h["id"]: h for h in payload.holds_json}
        key = route_key(route, payload.color)
        (out / f"{key}.json").write_text(json.dumps({
            "stem": route["stem"], "color": payload.color,
            "model": result.model, "variant": args.variant,
            "prompt_version": PROMPT_VERSION,
            "errors": result.errors, "repaired": result.repaired,
            "latency_s": round(result.latency_s, 1),
            "tokens": [result.input_tokens, result.output_tokens],
            "cost": round(result.cost, 4),
            "report": quality_report(result.plan, holds_by_id) if result.plan else None,
            "plan": result.plan,
        }, indent=1))
        if result.plan:
            render(payload, result.plan, out / f"{key}.beta.jpg")

        status = "ok" if not result.errors else f"errors {result.errors}"
        print(f"  {key}: {result.latency_s:5.1f}s ${result.cost:.3f} "
              f"{'repaired ' if result.repaired else ''}{status}")
    print(f"-> {out}  total ${total_cost:.2f}")


def flags(record: dict) -> list[str]:
    """What is worth checking about one generated plan."""
    if record["errors"] or not record.get("report"):
        return ["invalid"]          # nothing usable came back; the rest is moot
    report = record["report"]
    out = []
    if report.get("hand_descents", 0) > MAX_DESCENTS:
        out.append(f"descents={report['hand_descents']}")
    if not report.get("ends_at_top"):
        out.append("no-top")
    if report.get("same_hand_consecutive", 0) > MAX_SAME_HAND:
        out.append("same-hand")
    if report.get("feet_above_hands", 0):
        out.append("feet-high")
    if report.get("max_reach_pct", 0) > MAX_REACH_PCT:
        out.append(f"reach={report['max_reach_pct']}%")
    return out


def cmd_report(args) -> None:
    if not OUT_DIR.exists():
        print(f"no runs in {OUT_DIR}")
        return
    for run_dir in sorted(OUT_DIR.iterdir()):
        if run_dir.name == "prep" or not run_dir.is_dir():
            continue
        records = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]
        if not records:
            continue
        clean = sum(1 for r in records if not flags(r))
        print(f"\n== {run_dir.name} ({len(records)} routes, {clean} clean) ==")
        for record in records:
            report = record.get("report") or {}
            print(f"  {record['stem'][:36]:38s} {record['color']:7s} "
                  f"{report.get('hand_moves', 0):2d}h/{report.get('foot_moves', 0):2d}f "
                  f"{report.get('grade') or 'V?':6s} {record['latency_s']:5.1f}s "
                  f"${record['cost']:.3f}  {' '.join(flags(record)) or 'clean'}")
        print(f"  total ${sum(r['cost'] for r in records):.2f}, "
              f"median {sorted(r['latency_s'] for r in records)[len(records) // 2]:.1f}s")
