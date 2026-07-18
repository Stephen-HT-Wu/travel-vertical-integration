"""Deterministic, no-LLM-call deep-link construction for the three
transaction-candidate stages (transportation/accommodation/activities).

The LLM only ever supplies free-text search terms (`deep_link_query`) —
never a URL. Every URL here is built from a fixed per-stage template with
the query text passed through `urlencode()`, then the resulting host is
asserted against a hardcoded allowlist before being handed back to the
caller. This guarantees the link can only ever point at the intended
vendor's real domain; it does NOT guarantee the page loads or returns good
results (see README/plan for per-vendor confidence notes — all three were
live-verified working as of 2026-07-18).

No booking/payment happens here or anywhere downstream of here — this
module's only job is "open a real search page with real trip context
pre-filled," never "complete a transaction."
"""
from urllib.parse import urlencode, urlsplit

from pydantic import BaseModel


class DeepLink(BaseModel):
    vendor: str
    url: str


# stage -> (display vendor name, allowed host)
_STAGE_VENDORS = {
    "transportation": ("Google 地圖", "www.google.com"),
    "accommodation": ("Booking.com", "www.booking.com"),
    "activities": ("KKday", "www.kkday.com"),
}


def _build_url(stage_name: str, deep_link_query: str, origin: str, party_size: int) -> str:
    query = deep_link_query.strip() or origin

    if stage_name == "transportation":
        params = {"api": "1", "origin": origin, "destination": query, "travelmode": "transit"}
        return "https://www.google.com/maps/dir/?" + urlencode(params)

    if stage_name == "accommodation":
        params = {"ss": query, "group_adults": party_size, "no_rooms": 1}
        return "https://www.booking.com/searchresults.html?" + urlencode(params)

    if stage_name == "activities":
        params = {"keyword": query}
        return "https://www.kkday.com/zh-tw/product/productlist?" + urlencode(params)

    raise ValueError(f"no deep-link template for stage '{stage_name}'")


def build_deep_link(stage_name: str, deep_link_query: str, origin: str, party_size: int = 1) -> DeepLink:
    if stage_name not in _STAGE_VENDORS:
        raise ValueError(f"'{stage_name}' has no deep-link vendor mapping")
    vendor, allowed_host = _STAGE_VENDORS[stage_name]

    url = _build_url(stage_name, deep_link_query, origin, party_size)

    actual_host = urlsplit(url).netloc
    if actual_host != allowed_host:
        # Only reachable via a bug in this module (templates are literals) —
        # never by anything the LLM supplies — but fail loudly rather than
        # ever hand back an unexpected host.
        raise RuntimeError(f"deep link host mismatch: got {actual_host!r}, expected {allowed_host!r}")

    return DeepLink(vendor=vendor, url=url)
