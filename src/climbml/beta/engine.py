"""Beta generation against a vision-language model, served through OpenRouter.

Structured outputs guarantee the response shape. :func:`validate` checks the
grounding the schema cannot express. One repair round-trip re-prompts with the
exact errors, which recovers most failures without a full retry.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass

from ..route.pipeline import Payload
from .schema import BETA_SCHEMA, SYSTEM_PROMPT, user_prompt, validate

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

#: Which model to read beta with, as an OpenRouter ``<provider>/<model>`` slug.
#: No provider is baked in: pass ``--model`` or set this, so the same prompt,
#: schema and scoreboard can be pointed at any model OpenRouter serves.
MODEL_ENV = "CLIMBML_BETA_MODEL"
MAX_TOKENS = 8000

#: effort/thinking presets compared in the evaluation harness
VARIANTS = {
    "thinking": dict(effort=None, thinking=True),     # reasoning on, provider default effort
    "medium": dict(effort="medium", thinking=True),
    "low": dict(effort="low", thinking=True),
    "fast": dict(effort=None, thinking=False),
}


def resolve_model(model: str | None = None) -> str:
    """The model slug to run, from the argument or the environment."""
    slug = model or os.environ.get(MODEL_ENV)
    if not slug:
        raise SystemExit(
            f"no model given: pass --model or set {MODEL_ENV} to an OpenRouter "
            "slug like <provider>/<model> (see https://openrouter.ai/models)")
    return slug


@dataclass
class Result:
    plan: dict | None
    errors: list[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    repaired: bool
    model: str
    #: USD actually charged, as reported by OpenRouter. Asking the router what
    #: the call cost keeps the scoreboard honest across models rather than
    #: carrying a price table that goes stale every time a provider re-prices.
    cost: float = 0.0


def _client():
    """An OpenAI-compatible client pointed at OpenRouter."""
    import openai

    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(f"{API_KEY_ENV} is not set")
    return openai.OpenAI(base_url=BASE_URL, api_key=key)


def _usage(response) -> tuple[int, int, float]:
    """(input tokens, output tokens, USD) from an OpenRouter response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0.0
    extra = getattr(usage, "model_extra", None) or {}
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = extra.get("cost", 0.0)
    return (usage.prompt_tokens or 0, usage.completion_tokens or 0, float(cost or 0.0))


def _image_block(payload: Payload) -> dict:
    buf = io.BytesIO()
    payload.image.save(buf, format="JPEG", quality=82)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def generate(payload: Payload, client=None, model: str | None = None,
             effort: str | None = None, thinking: bool = True) -> Result:
    """Generate one move plan for an isolated route.

    ``model`` is any OpenRouter slug; ``client`` is an OpenAI-compatible client
    bound to OpenRouter, constructed from the environment when omitted.
    """
    model = resolve_model(model)
    if client is None:
        client = _client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            _image_block(payload),
            {"type": "text", "text": user_prompt(payload.color, payload.holds_json)},
        ]},
    ]
    #: ``require_parameters`` keeps OpenRouter from falling back to a provider
    #: that would silently drop the schema or the reasoning setting.
    reasoning: dict = {"enabled": True} if thinking else {"enabled": False}
    if thinking and effort:
        reasoning = {"effort": effort}
    kwargs: dict = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_schema", "json_schema": {
            "name": "beta_plan", "strict": True, "schema": BETA_SCHEMA}},
        extra_body={"reasoning": reasoning, "provider": {"require_parameters": True},
                    "usage": {"include": True}},
    )

    valid_ids = {h["id"] for h in payload.holds_json}
    start_ids = {h["id"] for h in payload.holds_json if h["isStart"]}

    started = time.time()
    in_tokens = out_tokens = 0
    cost = 0.0
    repaired = False
    plan, errors = None, []

    for attempt in range(2):
        response = client.chat.completions.create(**kwargs, messages=messages)
        used_in, used_out, used_cost = _usage(response)
        in_tokens += used_in
        out_tokens += used_out
        cost += used_cost
        choice = response.choices[0]
        if choice.message.refusal:
            errors = [f"refusal: {choice.message.refusal}"]
            break
        if choice.finish_reason in ("content_filter", "length"):
            errors = [f"finish_reason={choice.finish_reason}"]
            break

        text = choice.message.content or ""
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

    return Result(plan, errors, time.time() - started, in_tokens, out_tokens,
                  repaired, model, cost)
