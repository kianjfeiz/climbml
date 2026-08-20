"""A synthetic wall, so the vision tests need no dataset and no weights.

A plain wall colour with coloured rectangles pasted on it, so the expected
colour and position of every hold is known exactly.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from climbml.route.holds import HOLD_CLASS, Hold

WALL = (150, 130, 110)          # warm plywood
BLUE = (40, 90, 200)
RED = (200, 50, 50)
WALL_SIZE = (400, 600)

#: blue holds run up the centre; red holds sit off to one side
BLUE_CENTERS = ((200, 500), (200, 350), (200, 200))
RED_CENTERS = ((60, 520), (60, 400), (60, 280))
HOLD_SIZE = 40


def make_hold(hold_id: int, cx: float, cy: float, size: float = HOLD_SIZE,
              cls: int = HOLD_CLASS) -> Hold:
    return Hold(hold_id, cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2,
                cls, 0.9)


def build_wall() -> tuple[Image.Image, list[Hold]]:
    img = Image.new("RGB", WALL_SIZE, WALL)
    draw = ImageDraw.Draw(img)
    holds = []
    for i, ((cx, cy), color) in enumerate(
            [(c, BLUE) for c in BLUE_CENTERS] + [(c, RED) for c in RED_CENTERS]):
        holds.append(make_hold(i, cx, cy))
        half = HOLD_SIZE / 2
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)
    return img, holds
