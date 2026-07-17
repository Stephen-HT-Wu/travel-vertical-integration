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

No booking/payment tool is defined anywhere in this module or the codebase.
"""
from typing import List, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def call_structured(
    model: str,
    system: str,
    user_content: str,
    output_format: Type[T],
    extra_messages: Optional[List[dict]] = None,
    max_tokens: int = 4096,
) -> T:
    """Single structured-output call. No tools — this is the "hard boundary"
    call path used by every stage except the research half of real-search
    stages: nothing here can invoke a booking/payment tool because none is
    ever passed."""
    messages = [{"role": "user", "content": user_content}]
    if extra_messages:
        messages = extra_messages + messages
    response = get_client().messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_format=output_format,
    )
    return response.parsed_output


def call_with_web_search(
    model: str,
    system: str,
    user_content: str,
    allowed_domains: Optional[List[str]] = None,
    max_uses: int = 5,
    max_tokens: int = 4000,
):
    """First half of the real-search two-call pattern. Returns the raw
    anthropic Message so its content blocks can be replayed into a
    follow-up call.messages.parse() for structured synthesis."""
    tool: dict = {"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        messages=[{"role": "user", "content": user_content}],
    )
    return response
