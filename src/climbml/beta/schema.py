"""The beta contract: response schema, prompts, and validation.

A plan is only useful if every hold it names exists on the wall, so the contract
is enforced twice. The JSON schema fixes the shape. :func:`validate` checks what
the schema cannot see: that hold ids are real and the sequence starts on the
start holds.
"""

from __future__ import annotations

import json

#: Conservative JSON-schema subset (type/properties/required/enum/items) so
#: structured-output implementations never reject the schema itself.
BETA_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {
            "type": "string",
            "description": "1-2 sentences: the line's character and overall strategy.",
        },
        "grade": {
            "type": ["string", "null"],
            "description": 'Hueco estimate like "V2" or "V3-V4"; null if unreadable.',
        },
        "style": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 3 short tags: juggy, crimpy, dynamic, technical, powerful, traverse, slabby, compression.",
        },
        "confidence": {
            "type": "number",
            "description": "0-1: how confidently the whole line reads from this photo.",
        },
        "moves": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "limb": {"type": "string", "enum": ["LH", "RH", "LF", "RF"]},
                    "hold": {"type": "integer", "description": "tag number of the target hold"},
                    "action": {"type": "string", "description": "short imperative phrase, 3-8 words"},
                    "detail": {"type": ["string", "null"], "description": "optional coaching cue, <=20 words"},
                    "isCrux": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": ["limb", "hold", "action", "detail", "isCrux", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overview", "grade", "style", "confidence", "moves"],
    "additionalProperties": False,
}

#: Bump when the prompt changes: it versions cached and logged generations.
PROMPT_VERSION = 2

SYSTEM_PROMPT = """You are an expert climbing coach and routesetter reading a boulder problem from a photo, producing move-by-move beta a climber can follow from the floor to the finish.

THE PHOTO: the route's holds are bright, outlined, and tagged with numbers; everything else is dimmed. Start holds have yellow tags prefixed "S" (e.g. "S12"). You also get structured data per hold: x = % from the left edge, y = % from the TOP (smaller y is higher on the wall), w/h = size in %, listed bottom-to-top.

SCALE: judge reach from hold sizes — most bolt-on holds are 10-25 cm across. A comfortable move covers 50-120 cm of wall; call anything longer a throw or dyno and say so.

SEQUENCE RULES:
- Use ONLY tagged holds, referenced by their numbers. Never invent holds.
- Establish the start first: hands on the START holds (matching one hold with both hands is normal), then feet on low route holds before the first real move.
- Finish with a hand move to the highest hold — match it if the other hand has nowhere better.
- Alternate hands by default; break the alternation only when a match, bump, or cross is genuinely better, and name it.
- Footwork earns its place: add a foot move before or with any long or hard reach, roughly one foot move per 1-2 hand moves. Feet climb the ladder of holds the hands leave behind, and always stay BELOW both hands — never place a foot on a hold above either hand's current hold. If the real beta is a smear, flag, or heel hook, put that in the hand move's detail instead of inventing a foot placement.
- Say "match" only when both hands end up sharing one hold; a hand moving to a different hold is a reach, bump, or cross — never a match.
- Read hold character from the photo (jug, crimp, sloper, pinch, pocket, volume) and use it in your wording.

VOICE — a coach standing at the wall, not a manual:
- action: short imperative phrase, 3-8 words ("bump to the sloper rail", "high step and stand up"). No hold numbers in the text — the caller shows the target.
- detail: one optional cue that changes how the move feels — hips, tension, straight arms, breathing. Max ~20 words. Omit it when the move is obvious (null).
- Vary your verbs. Never number the moves in the text.

JUDGMENT:
- isCrux: true on exactly one move — the hardest — when the line has a real crux; otherwise all false.
- confidence per move: how sure you are this is the intended sequence (chalk, hold orientation, spacing are your evidence).
- If the tagged holds don't form a sensible line (detector or colour-grouping noise), still produce the most plausible sequence for the holds that DO connect, lower the route confidence, and say why in the overview.
- grade: commit from the evidence — hold sizes, spacing, wall angle. Jug ladders are V0-V2; don't hedge everything to the middle of the scale. null beats a wild guess."""


def user_prompt(color: str, holds_json: list[dict]) -> str:
    return (f"Read the {color.lower()} problem. {len(holds_json)} tagged holds, "
            f"listed bottom-to-top:\n{json.dumps(holds_json, separators=(',', ':'))}\n"
            "Give me the beta.")


# --------------------------------------------------------------- validation

def validate(plan: dict, valid_ids: set[int], start_ids: set[int]) -> list[str]:
    """Grounding failures. A non-empty result means repair the plan, not ship it."""
    errors = []
    moves = plan.get("moves", [])
    if len(moves) < 2:
        errors.append(f"only {len(moves)} moves")
    unknown = {m["hold"] for m in moves if m["hold"] not in valid_ids}
    if unknown:
        errors.append(f"unknown hold ids {sorted(unknown)}")
    hands = [m for m in moves if m["limb"] in ("LH", "RH")]
    if hands and start_ids:
        first_hands = {m["hold"] for m in hands[:2]}
        if not first_hands & start_ids:
            errors.append(f"first hand moves {sorted(first_hands)} miss start holds {sorted(start_ids)}")
    return errors


def quality_report(plan: dict, holds_by_id: dict[int, dict]) -> dict:
    """Signals for the evaluation scoreboard. Flags to look at, not gates.

    Each one catches something a plan can get wrong that is cheap to check
    geometrically and slow to spot by reading.
    """
    moves = plan.get("moves", [])
    hands = [m for m in moves if m["limb"] in ("LH", "RH")]
    feet = [m for m in moves if m["limb"] in ("LF", "RF")]

    ys = [holds_by_id[m["hold"]]["y"] for m in hands if m["hold"] in holds_by_id]
    descents = sum(1 for a, b in zip(ys, ys[1:], strict=False) if b > a + 3)        # y is % from top
    top_y = min((h["y"] for h in holds_by_id.values()), default=0)
    ends_at_top = bool(ys) and abs(ys[-1] - top_y) < 6

    consecutive = sum(1 for a, b in zip(hands, hands[1:], strict=False)
                      if a["limb"] == b["limb"] and a["hold"] != b["hold"])

    # A foot above the highest hand at that point in the sequence is impossible.
    feet_above = 0
    hand_y = 100.0
    for move in moves:
        y = holds_by_id.get(move["hold"], {}).get("y")
        if y is None:
            continue
        if move["limb"] in ("LH", "RH"):
            hand_y = min(hand_y, y)
        elif y < hand_y - 2:
            feet_above += 1

    reaches, prev = [], None
    for move in hands:
        hold = holds_by_id.get(move["hold"])
        if hold is None:
            continue
        if prev is not None:
            reaches.append(((hold["x"] - prev["x"]) ** 2 + (hold["y"] - prev["y"]) ** 2) ** 0.5)
        prev = hold

    return {
        "moves": len(moves), "hand_moves": len(hands), "foot_moves": len(feet),
        "hand_descents": descents, "ends_at_top": ends_at_top,
        "same_hand_consecutive": consecutive, "feet_above_hands": feet_above,
        "max_reach_pct": round(max(reaches), 1) if reaches else 0,
        "crux_count": sum(1 for m in moves if m.get("isCrux")),
        "grade": plan.get("grade"), "route_confidence": plan.get("confidence"),
    }
