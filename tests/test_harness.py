from climbml.beta.engine import VARIANTS, Result
from climbml.beta.harness import flags, load_routes, route_key


def record(errors=(), **report):
    base = dict(hand_descents=0, ends_at_top=True, same_hand_consecutive=0,
                feet_above_hands=0, max_reach_pct=20)
    return {"errors": list(errors), "report": {**base, **report}}


def test_a_sound_plan_has_no_flags():
    assert flags(record()) == []


def test_flags_name_what_went_wrong():
    assert "invalid" in flags(record(errors=["unknown hold ids [99]"]))
    assert "no-top" in flags(record(ends_at_top=False))
    assert "feet-high" in flags(record(feet_above_hands=2))
    assert "same-hand" in flags(record(same_hand_consecutive=4))
    assert any(f.startswith("reach=") for f in flags(record(max_reach_pct=70)))
    assert any(f.startswith("descents=") for f in flags(record(hand_descents=3)))


def test_a_plan_without_a_report_is_still_flagged():
    assert flags({"errors": ["stop_reason=max_tokens"], "report": None}) == ["invalid"]


def test_eval_routes_config_is_usable():
    routes = load_routes()
    assert routes, "the curated evaluation set should not be empty"
    for route in routes:
        assert set(route) == {"split", "stem", "color"}
        assert route["split"] in ("train", "valid", "test")
    keys = [route_key(r, r["color"] or "auto") for r in routes]
    assert len(set(keys)) == len(keys), "route keys must be unique per run directory"


def test_variants_cover_the_effort_ladder():
    assert set(VARIANTS) == {"thinking", "medium", "low", "fast"}
    assert VARIANTS["fast"]["thinking"] is False


def test_result_cost_uses_model_pricing():
    result = Result(plan={}, errors=[], latency_s=1.0, input_tokens=1_000_000,
                    output_tokens=100_000, repaired=False, model="anthropic/claude-sonnet-5")
    assert result.cost == 3.00 + 1.5


def test_unknown_model_costs_nothing_rather_than_guessing():
    result = Result({}, [], 1.0, 1000, 1000, False, model="some-other-model")
    assert result.cost == 0.0
