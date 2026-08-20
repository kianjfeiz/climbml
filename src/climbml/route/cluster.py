"""Group detected holds into routes by colour, and rank the candidates.

The detector finds every hold on the wall. A route is the subset sharing a
colour, so grouping is a bin assignment (see :mod:`climbml.route.color`).
Choosing which group the photographer meant is the harder half: a wall photo
usually centres the route, that route spans the frame vertically, and chalk
dust and black footholds are never the subject.
"""

from __future__ import annotations

from .color import NEUTRALS, bin_name
from .holds import VOLUME_CLASS, Cluster, Hold

MIN_CLUSTER = 3                 # fewer holds than this is noise, not a route
NEUTRAL_PENALTY = 0.55
START_REACH = 0.25              # lateral reach for the second start hold, as % of width

# Prominence weights: centred, many holds, tall spread, large holds.
W_CENTRALITY, W_COUNT, W_SPREAD, W_AREA = 0.30, 0.25, 0.20, 0.25


def start_holds(group: list[Hold], width: float) -> list[int]:
    """Guess the start holds: the lowest hold, plus the next-lowest within reach.

    Taking the two absolute lowest holds instead picks wall-spanning pairs on
    spray walls, which the language model then follows.
    """
    by_low = sorted(group, key=lambda h: -h.y2)
    starts = [by_low[0].id]
    for h in by_low[1:]:
        if abs(h.cx - by_low[0].cx) < width * START_REACH:
            starts.append(h.id)
            break
    return starts


def cluster(holds: list[Hold], img_size: tuple[int, int]) -> list[Cluster]:
    """Colour-group holds into candidate routes, most prominent first.

    Holds need a sampled ``hsv``; volumes are excluded (they are shared
    features of the wall, not part of one route's colour).
    """
    width, height = img_size
    bins: dict[str, list[Hold]] = {}
    for hold in holds:
        if hold.cls == VOLUME_CLASS or hold.hsv is None:
            continue
        hold.bin = bin_name(*hold.hsv)
        bins.setdefault(hold.bin, []).append(hold)

    groups = {name: g for name, g in bins.items() if len(g) >= MIN_CLUSTER}
    if not groups:
        return []

    cx0, cy0 = width / 2, height / 2
    max_dist = (cx0 ** 2 + cy0 ** 2) ** 0.5
    max_count = max(len(g) for g in groups.values())
    mean_areas = {name: sum(h.w * h.h for h in g) / len(g) for name, g in groups.items()}
    max_area = max(mean_areas.values())

    out = []
    for name, group in groups.items():
        gx = sum(h.cx for h in group) / len(group)
        gy = sum(h.cy for h in group) / len(group)
        centrality = 1 - min(1, ((gx - cx0) ** 2 + (gy - cy0) ** 2) ** 0.5 / max_dist)
        ys = [h.cy for h in group]
        spread = (max(ys) - min(ys)) / height

        prominence = (W_CENTRALITY * centrality
                      + W_COUNT * len(group) / max_count
                      + W_SPREAD * min(1, spread * 1.6)
                      + W_AREA * mean_areas[name] / max_area)
        if name in NEUTRALS:
            prominence *= NEUTRAL_PENALTY

        out.append(Cluster(name, [h.id for h in group], prominence,
                           start_holds(group, width)))
    out.sort(key=lambda c: -c.prominence)
    return out
