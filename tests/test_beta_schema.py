import json

from climbml.beta.schema import BETA_SCHEMA, quality_report, user_prompt, validate

VALID = {1, 2, 3, 4}
STARTS = {1, 2}


def move(limb, hold, **kw):
    return dict(limb=limb, hold=hold, action="pull on", detail=None,
                isCrux=False, confidence=0.8, **kw)


def test_schema_requires_every_move_field():
    move_schema = BETA_SCHEMA["properties"]["moves"]["items"]
    assert set(move_schema["required"]) == set(move_schema["properties"])
    assert move_schema["additionalProperties"] is False


def test_valid_plan_passes():
    plan = {"moves": [move("LH", 1), move("RH", 2), move("LF", 3)]}
    assert validate(plan, VALID, STARTS) == []


def test_unknown_hold_ids_are_rejected():
    plan = {"moves": [move("LH", 1), move("RH", 99)]}
    assert "unknown hold ids [99]" in validate(plan, VALID, STARTS)[0]


def test_plan_must_have_at_least_two_moves():
    assert validate({"moves": [move("LH", 1)]}, VALID, STARTS)[0] == "only 1 moves"


def test_sequence_must_begin_on_a_start_hold():
    plan = {"moves": [move("LH", 3), move("RH", 4)]}
    errors = validate(plan, VALID, STARTS)
    assert any("miss start holds" in e for e in errors)


def test_foot_moves_do_not_count_as_the_start():
    plan = {"moves": [move("LF", 3), move("LH", 1), move("RH", 2)]}
    assert validate(plan, VALID, STARTS) == []


def test_user_prompt_carries_the_holds_as_json():
    holds = [{"id": 1, "x": 10.0, "y": 90.0, "w": 5.0, "h": 5.0, "isStart": True}]
    prompt = user_prompt("Blue", holds)
    assert "blue" in prompt and "1 tagged holds" in prompt
    assert json.dumps(holds, separators=(",", ":")) in prompt


# ------------------------------------------------------------ quality report

HOLDS = {1: {"x": 40, "y": 90}, 2: {"x": 60, "y": 88}, 3: {"x": 50, "y": 60},
         4: {"x": 50, "y": 20}}


def test_report_counts_limbs_and_detects_a_topout():
    plan = {"moves": [move("LH", 1), move("RH", 2), move("LF", 1),
                      move("LH", 3), move("RH", 4)]}
    report = quality_report(plan, HOLDS)
    assert (report["hand_moves"], report["foot_moves"]) == (4, 1)
    assert report["ends_at_top"] is True
    assert report["feet_above_hands"] == 0


def test_report_flags_a_foot_above_the_hands():
    plan = {"moves": [move("LH", 1), move("RH", 2), move("LF", 4)]}
    assert quality_report(plan, HOLDS)["feet_above_hands"] == 1


def test_report_flags_hands_moving_back_down():
    plan = {"moves": [move("LH", 1), move("RH", 4), move("LH", 3)]}
    assert quality_report(plan, HOLDS)["hand_descents"] == 1


def test_report_flags_the_same_hand_twice_in_a_row():
    plan = {"moves": [move("LH", 1), move("LH", 3), move("RH", 4)]}
    assert quality_report(plan, HOLDS)["same_hand_consecutive"] == 1


def test_matching_a_hold_is_not_a_same_hand_run():
    plan = {"moves": [move("LH", 1), move("RH", 1), move("LH", 4)]}
    assert quality_report(plan, HOLDS)["same_hand_consecutive"] == 0


def test_report_ignores_unknown_holds():
    plan = {"moves": [move("LH", 1), move("RH", 99)]}
    assert quality_report(plan, HOLDS)["moves"] == 2


def test_real_plan_and_route_are_consistent(example_route, example_plan):
    ids = {h["id"] for h in example_route["holds"]}
    assert validate(example_plan, ids, set(example_route["starts"])) == []
    report = quality_report(example_plan, {h["id"]: h for h in example_route["holds"]})
    assert report["hand_moves"] >= 2
    assert report["feet_above_hands"] == 0
