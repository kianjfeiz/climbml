"""Beta generation against a vision-language model.

Structured outputs guarantee the response shape. :func:`validate` checks the
grounding the schema cannot express. One repair round-trip re-prompts with the
exact errors, which recovers most failures without a full retry.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass

from ..route.pipeline import Payload
from .schema import BETA_SCHEMA, SYSTEM_PROMPT, user_prompt, validate

MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000

#: $ per million tokens (input, output), for the cost column in reports.
#: Sonnet 5 list price; introductory pricing (2.00/10.00) applies through
#: 2026-08-31, so historical figures in docs/experiments.md read lower.
PRICING = {"claude-sonnet-5": (3.00, 15.00)}

#: effort/thinking presets compared in the evaluation harness
VARIANTS = {
    "thinking": dict(effort=None, thinking=True),     # adaptive thinking, default effort
    "medium": dict(effort="medium", thinking=True),
    "low": dict(effort="low", thinking=True),
    "fast": dict(effort=None, thinking=False),
}


@dataclass
class Result:
    plan: dict | None
    errors: list[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    repaired: bool
    model: str = MODEL

    @property
    def cost(self) -> float:
        price_in, price_out = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6


def _image_block(payload: Payload) -> dict:
    buf = io.BytesIO()
    payload.image.save(buf, format="JPEG", quality=82)
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def generate(payload: Payload, client=None, model: str = MODEL,
             effort: str | None = None, thinking: bool = True) -> Result:
    """Generate one move plan for an isolated route.

    ``client`` is an ``anthropic.Anthropic`` instance; one is constructed from
    the environment when omitted.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    messages = [{"role": "user", "content": [
        _image_block(payload),
        {"type": "text", "text": user_prompt(payload.color, payload.holds_json)},
    ]}]
    kwargs: dict = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": BETA_SCHEMA}},
    )
    if effort:
        kwargs["output_config"]["effort"] = effort
    kwargs["thinking"] = {"type": "adaptive"} if thinking else {"type": "disabled"}

    valid_ids = {h["id"] for h in payload.holds_json}
    start_ids = {h["id"] for h in payload.holds_json if h["isStart"]}

    started = time.time()
    in_tokens = out_tokens = 0
    repaired = False
    plan, errors = None, []

    for attempt in range(2):
        response = client.messages.create(**kwargs, messages=messages)
        in_tokens += response.usage.input_tokens
        out_tokens += response.usage.output_tokens
        if response.stop_reason in ("refusal", "max_tokens"):
            errors = [f"stop_reason={response.stop_reason}"]
            break

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as exc:
            errors = [f"bad json: {exc}"]
            break

        errors = validate(plan, valid_ids, start_ids)
        if not errors or attempt == 1:
            break
        repaired = True
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Your beta has grounding errors: "
             + "; ".join(errors)
             + ". Re-read the tags and produce a corrected full plan."},
        ]

    return Result(plan, errors, time.time() - started, in_tokens, out_tokens, repaired, model)
