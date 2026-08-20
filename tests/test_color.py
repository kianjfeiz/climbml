from climbml.route.color import SAT_MIN, Sampler, bin_name, rgb_to_hsv


def test_rgb_to_hsv_primaries():
    assert rgb_to_hsv((255, 0, 0)) == (0.0, 1.0, 1.0)
    assert rgb_to_hsv((0, 255, 0))[0] == 120.0
    assert rgb_to_hsv((0, 0, 255))[0] == 240.0
    assert rgb_to_hsv((0, 0, 0)) == (0.0, 0.0, 0.0)


def test_hue_bins_cover_the_wheel():
    seen = {bin_name(hue, 0.9, 0.9) for hue in range(0, 360)}
    assert "Gray" not in seen          # every saturated hue lands in a colour
    assert {"Red", "Blue", "Purple", "Pink"} <= seen


def test_red_bin_wraps_through_zero():
    assert bin_name(350, 0.9, 0.9) == "Red"
    assert bin_name(5, 0.9, 0.9) == "Red"


def test_blue_purple_boundary():
    assert bin_name(239, 0.9, 0.9) == "Blue"
    assert bin_name(241, 0.9, 0.9) == "Purple"


def test_dark_desaturated_holds_read_black():
    assert bin_name(210, 0.9, 0.15) == "Black"      # too dark to have a colour
    assert bin_name(210, 0.35, 0.30) == "Black"     # black rubber in shadow


def test_bright_faint_tint_reads_white_not_blue():
    # A white hold under cool light keeps a little blue; the cutoff ramps with value.
    assert bin_name(210, SAT_MIN + 0.05, 1.0) == "White"
    assert bin_name(210, 0.35, 1.0) == "Blue"       # a genuine pastel still bins


def test_low_value_neutral_reads_gray():
    assert bin_name(210, 0.05, 0.40) == "Gray"


def test_sampler_recovers_hold_colour_not_wall(wall_photo):
    img, holds = wall_photo
    sampler = Sampler(img)
    blue_hue = sampler.median_hsv(holds[0])[0]
    red_hue = sampler.median_hsv(holds[3])[0]
    assert bin_name(*sampler.median_hsv(holds[0])) == "Blue"
    assert bin_name(*sampler.median_hsv(holds[3])) == "Red"
    assert 195 <= blue_hue < 240
    assert red_hue < 15 or red_hue >= 345


def test_sampler_rejects_degenerate_box(wall_photo):
    img, holds = wall_photo
    hold = holds[0]
    hold.x2 = hold.x1 - 1
    assert Sampler(img).median_hsv(hold) is None
