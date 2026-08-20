"""Draw a generated beta over the wall photo for review.

The geometric checks catch impossible plans. Whether a plan is the sequence a
climber would pick is answered by looking. Each limb gets a coloured track
through the holds it uses, nodes are numbered in move order, and the panel
lists the plan as text.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from ..route.pipeline import Payload
from ..route.som import load_font

LH = (92, 133, 204)
RH = (28, 140, 115)
FOOT = (108, 117, 125)
PANEL_W = 560
INK = (20, 20, 25)
MUTED = (120, 120, 130)


def limb_color(limb: str) -> tuple[int, int, int]:
    return LH if limb == "LH" else RH if limb == "RH" else FOOT


def render(payload: Payload, plan: dict, out_path) -> None:
    """Write an annotated wall photo plus a move-list panel to `out_path`."""
    img = payload.image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    pos = {h.id: (h.cx * payload.scale, h.cy * payload.scale) for h in payload.route}

    radius = max(11, int(min(width, height) * 0.016))
    font = load_font(int(radius * 1.1), bold=True)

    tracks: dict[str, list[tuple[float, float]]] = {}
    for move in plan["moves"]:
        if move["hold"] in pos:
            tracks.setdefault(move["limb"], []).append(pos[move["hold"]])
    for limb, points in tracks.items():
        for a, b in zip(points, points[1:], strict=False):
            draw.line([a, b], fill=(255, 255, 255), width=6)   # halo for contrast
            draw.line([a, b], fill=limb_color(limb), width=3)

    for i, move in enumerate(plan["moves"], 1):
        if move["hold"] not in pos:
            continue
        x, y = pos[move["hold"]]
        color = limb_color(move["limb"])
        if move["limb"] in ("LF", "RF"):
            # feet offset down-left so a hold shared with a hand stays readable
            x, y = x - radius * 1.4, y + radius * 1.4
            draw.rectangle([x - radius, y - radius, x + radius, y + radius],
                           fill=color, outline=(255, 255, 255), width=2)
        else:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=color, outline=(255, 255, 255), width=2)
        label = str(i)
        draw.text((x - draw.textlength(label, font=font) / 2, y - radius * 0.62),
                  label, font=font, fill=(255, 255, 255))

    panel = _move_panel(plan, height, payload.color)
    combo = Image.new("RGB", (width + PANEL_W, height), (255, 255, 255))
    combo.paste(img, (0, 0))
    combo.paste(panel, (width, 0))
    combo.save(out_path, quality=88)


def _move_panel(plan: dict, height: int, color: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_W, height), (250, 250, 252))
    draw = ImageDraw.Draw(panel)
    body, bold, small = load_font(19), load_font(19, bold=True), load_font(15)

    y = 18
    draw.text((20, y), f"{color} - {plan.get('grade') or 'V?'}   "
                       f"conf {plan.get('confidence', 0):.2f}", font=bold, fill=INK)
    y += 30
    for line in _wrap(plan.get("overview", ""), small, PANEL_W - 40, draw):
        draw.text((20, y), line, font=small, fill=(90, 90, 100))
        y += 20
    if tags := ", ".join(plan.get("style", [])):
        draw.text((20, y), tags, font=small, fill=LH)
        y += 22
    y += 8

    for i, move in enumerate(plan["moves"], 1):
        crux = " CRUX" if move.get("isCrux") else ""
        draw.text((20, y), f"{i:>2} {move['limb']} -> {move['hold']}{crux}",
                  font=bold, fill=limb_color(move["limb"]))
        y += 24
        for line in _wrap(move["action"], body, PANEL_W - 60, draw):
            draw.text((40, y), line, font=body, fill=INK)
            y += 24
        for line in _wrap(move.get("detail") or "", small, PANEL_W - 60, draw):
            draw.text((40, y), line, font=small, fill=MUTED)
            y += 20
        y += 6
        if y > height - 40:
            draw.text((20, y), "...", font=bold, fill=MUTED)
            break
    return panel


def _wrap(text: str, font, width: int, draw) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
