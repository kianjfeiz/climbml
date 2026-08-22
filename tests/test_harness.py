import pytest

from climbml.beta.engine import VARIANTS, Result, resolve_model
from climbml.beta.harness import flags, load_routes, route_key, run_dir_name


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


def test_result_carries_the_cost_openrouter_reported():
    result = Result(plan={}, errors=[], latency_s=1.0, input_tokens=1_000_000,
                    output_tokens=100_000, repaired=False,
                    model="some-provider/some-model", cost=0.0451)
    assert result.cost == 0.0451


def test_cost_defaults_to_zero_when_the_router_reports_none():
    result = Result({}, [], 1.0, 1000, 1000, False, model="some-provider/some-model")
    assert result.cost == 0.0


def test_model_comes_from_the_argument_or_the_environment(monkeypatch):
    monkeypatch.delenv("CLIMBML_BETA_MODEL", raising=False)
    assert resolve_model("some-provider/some-model") == "some-provider/some-model"
    monkeypatch.setenv("CLIMBML_BETA_MODEL", "other/model")
    assert resolve_model(None) == "other/model"
    assert resolve_model("explicit/wins") == "explicit/wins"


def test_no_model_is_an_error_rather_than_a_baked_in_default(monkeypatch):
    monkeypatch.delenv("CLIMBML_BETA_MODEL", raising=False)
    with pytest.raises(SystemExit):
        resolve_model(None)


def test_run_dirs_keep_models_apart():
    assert run_dir_name("a/one", "low") != run_dir_name("b/two", "low")
    assert "/" not in run_dir_name("a/one", "low")
