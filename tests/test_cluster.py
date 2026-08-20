from climbml.route.cluster import MIN_CLUSTER, cluster, start_holds
from climbml.route.color import Sampler
from climbml.route.holds import VOLUME_CLASS

from synthetic import make_hold


def _sampled(img, holds):
    sampler = Sampler(img)
    for hold in holds:
        hold.hsv = sampler.median_hsv(hold)
    return holds


def test_clusters_split_by_colour(wall_photo):
    img, holds = wall_photo
    clusters = cluster(_sampled(img, holds), img.size)
    assert {c.color for c in clusters} == {"Blue", "Red"}
    assert all(len(c.hold_ids) == 3 for c in clusters)


def test_centred_route_is_most_prominent(wall_photo):
    img, holds = wall_photo
    clusters = cluster(_sampled(img, holds), img.size)
    assert clusters[0].color == "Blue"          # the blue line runs up the middle
    assert clusters[0].prominence > clusters[1].prominence


def test_groups_below_minimum_are_dropped(wall_photo):
    img, holds = wall_photo
    kept = holds[: MIN_CLUSTER] + holds[MIN_CLUSTER:MIN_CLUSTER + 1]
    clusters = cluster(_sampled(img, kept), img.size)
    assert [c.color for c in clusters] == ["Blue"]


def test_volumes_are_excluded(wall_photo):
    img, holds = wall_photo
    for hold in holds[:3]:
        hold.cls = VOLUME_CLASS
    clusters = cluster(_sampled(img, holds), img.size)
    assert [c.color for c in clusters] == ["Red"]


def test_starts_are_the_lowest_pair_within_reach():
    width = 1000
    low = make_hold(0, 500, 900)
    near = make_hold(1, 600, 880)
    far = make_hold(2, 50, 890)                 # lower than `near`, but across the wall
    assert start_holds([low, near, far], width) == [0, 1]


def test_starts_fall_back_to_one_hold_when_nothing_is_in_reach():
    width = 1000
    assert start_holds([make_hold(0, 500, 900), make_hold(1, 50, 880)], width) == [0]
