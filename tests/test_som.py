from climbml.route.pipeline import MAX_SEND_SIDE, build_payload
from climbml.route.som import draw_som

from synthetic import BLUE, WALL


def _pixel_brightness(img, xy):
    return sum(img.getpixel(xy))


def test_off_route_pixels_are_dimmed(wall_photo):
    img, holds = wall_photo
    som = draw_som(img, holds[:3], starts={0})
    assert _pixel_brightness(som, (350, 100)) < sum(WALL)       # bare wall, dimmed
    assert _pixel_brightness(som, (200, 350)) >= sum(BLUE) - 3  # route hold, untouched


def test_every_route_hold_gets_a_tag(wall_photo):
    img, holds = wall_photo
    som = draw_som(img, holds, starts={0})
    # Start tags are yellow; the other chips are white. Both must appear.
    colors = som.getcolors(maxcolors=1 << 20)
    assert any(count > 50 and rgb == (255, 214, 10) for count, rgb in colors)
    assert any(count > 50 and rgb == (255, 255, 255) for count, rgb in colors)


def test_payload_orders_holds_bottom_to_top_and_marks_starts(wall_photo):
    img, holds = wall_photo
    payload = build_payload(img, holds, route_color="Blue")
    ys = [h["y"] for h in payload.holds_json]
    assert ys == sorted(ys, reverse=True)       # y is % from the top
    assert payload.color == "Blue"
    assert [h["id"] for h in payload.holds_json if h["isStart"]] == payload.start_ids


def test_payload_positions_are_percentages(wall_photo):
    img, holds = wall_photo
    payload = build_payload(img, holds, route_color="Blue")
    lowest = payload.holds_json[0]
    assert lowest["x"] == 50.0                  # centred on a 400px-wide wall
    assert 0 < lowest["y"] < 100
    assert lowest["w"] == 10.0


def test_payload_is_capped_to_the_send_size(wall_photo):
    img, holds = wall_photo
    big = img.resize((img.width * 6, img.height * 6))
    scaled = [type(h)(h.id, *(v * 6 for v in (h.x1, h.y1, h.x2, h.y2)), h.cls, h.conf)
              for h in holds]
    payload = build_payload(big, scaled, route_color="Blue")
    assert max(payload.image.size) == MAX_SEND_SIDE
    assert payload.scale < 1.0


def test_no_route_returns_none(wall_photo):
    img, holds = wall_photo
    assert build_payload(img, holds[:1]) is None
