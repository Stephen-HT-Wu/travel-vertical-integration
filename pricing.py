"""Approximate list pricing for cost estimation, used only to give a rough
feasibility-assessment number alongside token counts — not exact billing
(no prompt-caching discounts, volume discounts, etc. are modeled).

Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-07).
"""

PRICING_PER_MTOK = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}
DEFAULT_PRICING = PRICING_PER_MTOK["claude-opus-4-8"]  # conservative fallback for unlisted models
WEB_SEARCH_PRICE_PER_SEARCH = 0.01  # $10 / 1,000 searches


def compute_cost_usd(
    model: str, input_tokens: int, output_tokens: int, web_search_requests: int = 0
) -> float:
    rates = PRICING_PER_MTOK.get(model, DEFAULT_PRICING)
    return (
        input_tokens / 1_000_000 * rates["input"]
        + output_tokens / 1_000_000 * rates["output"]
        + web_search_requests * WEB_SEARCH_PRICE_PER_SEARCH
    )
