from PIL import Image, ImageDraw

from climbml.beta.render import LH, PANEL_W, _wrap, limb_color, render
from climbml.route.pipeline import build_payload


def test_limb_colors_are_distinct():
    assert len({limb_color(limb) for limb in ("LH", "RH", "LF")}) == 3
    assert limb_color("LF") == limb_color("RF")


def test_wrap_breaks_on_width():
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    from climbml.route.som import load_font
    lines = _wrap("bump to the sloper rail and stand up", load_font(20), 80, draw)
    assert len(lines) > 1
    assert " ".join(lines) == "bump to the sloper rail and stand up"


def test_render_writes_photo_plus_panel(tmp_path, wall_photo):
    img, holds = wall_photo
    payload = build_payload(img, holds, route_color="Blue")
    ids = [h["id"] for h in payload.holds_json]
    plan = {
        "overview": "Straight up the middle.", "grade": "V2",
        "style": ["juggy"], "confidence": 0.8,
        "moves": [
            {"limb": "LH", "hold": ids[0], "action": "start on the low jug",
             "detail": None, "isCrux": False, "confidence": 0.9},
            {"limb": "RH", "hold": ids[1], "action": "reach the middle edge",
             "detail": "keep your hips close", "isCrux": True, "confidence": 0.7},
            {"limb": "LF", "hold": ids[0], "action": "step through",
             "detail": None, "isCrux": False, "confidence": 0.6},
            {"limb": "LH", "hold": ids[-1], "action": "match the top jug",
             "detail": None, "isCrux": False, "confidence": 0.8},
        ],
    }
    out = tmp_path / "beta.jpg"
    render(payload, plan, out)

    rendered = Image.open(out)
    assert rendered.size == (payload.image.width + PANEL_W, payload.image.height)
    # the panel carries the accent colour used for left-hand moves
    assert any(abs(sum(rgb) - sum(LH)) < 30 for _, rgb in
               rendered.crop((payload.image.width, 0, rendered.width, rendered.height))
               .getcolors(maxcolors=1 << 20))
