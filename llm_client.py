"""Thin wrapper around the Anthropic Python SDK.

Two entry points, matching the two ways stages call the model:

- call_structured(): a single `messages.parse()` call with no tools —
  used by every simulated stage (transportation/accommodation/activities),
  by itinerary/guide/replanning generation, and as the second half of the
  real-search two-call pattern.
- call_with_web_search(): a plain `messages.create()` call with the
  `web_search` server tool attached — used only by the four real-search
  stages (inspiration/dining/attractions/shopping) as the first half of
  their two-call pattern. Returns the raw response so its content blocks
  (including encrypted_content for search results) can be replayed back
  into the follow-up structured call.
- call_with_web_fetch(): same shape, but with the `web_fetch` server tool —
  used by the planning orchestrator to pull the full text of 1-n already-
  known article URLs (chosen ahead of time by a local title-index lookup)
  instead of running a fresh web_search. The web_fetch tool can only fetch
  URLs that already appear in the conversation, so callers must embed the
  target URLs as plain text in user_content.

No booking/payment tool is defined anywhere in this module or the codebase.

Both entry points also time the call and read `response.usage` to build a
CallMetrics object (stage left blank here — the caller in
agents/base_agent.py fills it in, since this module doesn't know which
stage it's serving) for the token/cost/time telemetry shown in the CLI and
the interactive web demo.
"""
import time
from typing import List, Optional, Tuple, Type, TypeVar

import anthropic
from pydantic import BaseModel

from pricing import compute_cost_usd
from schemas import CallMetrics

T = TypeVar("T", bound=BaseModel)

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _build_metrics(call_type: str, model: str, usage, duration_ms: float) -> CallMetrics:
    web_search_requests = (
        usage.server_tool_use.web_search_requests
        if getattr(usage, "server_tool_use", None)
        else 0
    )
    return CallMetrics(
        stage="",
        call_type=call_type,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        web_search_requests=web_search_requests,
        duration_ms=duration_ms,
        cost_usd=compute_cost_usd(model, usage.input_tokens, usage.output_tokens, web_search_requests),
    )


def call_structured(
    model: str,
    system: str,
    user_content: str,
    output_format: Type[T],
    extra_messages: Optional[List[dict]] = None,
    max_tokens: int = 4096,
) -> Tuple[T, CallMetrics]:
    """Single structured-output call. No tools — this is the "hard boundary"
    call path used by every stage except the research half of real-search
    stages: nothing here can invoke a booking/payment tool because none is
    ever passed."""
    messages = [{"role": "user", "content": user_content}]
    if extra_messages:
        messages = extra_messages + messages
    start = time.monotonic()
    response = get_client().messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_format=output_format,
    )
    duration_ms = (time.monotonic() - start) * 1000
    metrics = _build_metrics("structured", model, response.usage, duration_ms)
    if response.parsed_output is None:
        raise RuntimeError(
            f"Structured output parsing failed for {output_format.__name__} "
            f"(stop_reason={response.stop_reason}, output_tokens={response.usage.output_tokens}, "
            f"max_tokens={max_tokens}). This usually means the response was cut off before "
            "completing valid JSON — try raising max_tokens for this call."
        )
    return response.parsed_output, metrics


def call_with_web_search(
    model: str,
    system: str,
    user_content: str,
    allowed_domains: Optional[List[str]] = None,
    max_uses: int = 5,
    max_tokens: int = 4000,
) -> Tuple[anthropic.types.Message, CallMetrics]:
    """First half of the real-search two-call pattern. Returns the raw
    anthropic Message so its content blocks can be replayed into a
    follow-up call.messages.parse() for structured synthesis."""
    tool: dict = {"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    start = time.monotonic()
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        messages=[{"role": "user", "content": user_content}],
    )
    duration_ms = (time.monotonic() - start) * 1000
    metrics = _build_metrics("web_search", model, response.usage, duration_ms)
    return response, metrics


def call_with_web_fetch(
    model: str,
    system: str,
    user_content: str,
    allowed_domains: Optional[List[str]] = None,
    max_uses: int = 3,
    max_tokens: int = 4000,
    max_content_tokens: int = 8000,
) -> Tuple[anthropic.types.Message, CallMetrics]:
    """Fetches the full text of specific known URLs (see module docstring).
    Uses the basic web_fetch_20250910 variant — no beta header required, and
    unlike the dynamic-filtering _20260209+ variants it works on every model
    this project offers, including claude-haiku-4-5."""
    tool: dict = {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": max_uses,
        "citations": {"enabled": True},
        "max_content_tokens": max_content_tokens,
    }
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    start = time.monotonic()
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        messages=[{"role": "user", "content": user_content}],
    )
    duration_ms = (time.monotonic() - start) * 1000
    metrics = _build_metrics("web_fetch", model, response.usage, duration_ms)
    return response, metrics
