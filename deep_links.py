"""Deterministic, no-LLM-call deep-link construction for real vendor referral
links (transportation/accommodation/activities), used once a user picks one
of the two full-trip plan options (see chat_session.select_plan()).

The LLM only ever supplies free-text search terms (`deep_link_query`, now
always a real name pulled from live search results, not a fabricated one or
a category/style string) — never a URL. Every URL here is built from a
fixed per-vendor template with the query text passed through `urlencode()`,
then the resulting host is asserted against a hardcoded allowlist before
being handed back to the caller. This guarantees the link can only ever
point at the intended vendor's real domain; it does NOT guarantee the page
loads or returns good results.

No booking/payment happens here or anywhere downstream of here — this
module's only job is "open a real search page with real trip context
pre-filled," never "complete a transaction."
"""
from urllib.parse import urlencode, urlsplit

from pydantic import BaseModel


class DeepLink(BaseModel):
    vendor: str
    url: str


# vendor key -> (display vendor name, allowed host)
_VENDOR_HOSTS = {
    "google_maps": ("Google 地圖", "www.google.com"),
    "google_search": ("Google 搜尋", "www.google.com"),
    "kkday": ("KKday", "www.kkday.com"),
}


def _assert_host(vendor_key: str, url: str) -> DeepLink:
    vendor, allowed_host = _VENDOR_HOSTS[vendor_key]
    actual_host = urlsplit(url).netloc
    if actual_host != allowed_host:
        # Only reachable via a bug in this module (templates are literals) —
        # never by anything the LLM supplies — but fail loudly rather than
        # ever hand back an unexpected host.
        raise RuntimeError(f"deep link host mismatch: got {actual_host!r}, expected {allowed_host!r}")
    return DeepLink(vendor=vendor, url=url)


def build_transportation_link(origin: str, destination: str) -> DeepLink:
    """Google Maps transit directions — live-verified working (2026-07-18),
    high confidence (official public URL-scheme documentation)."""
    params = {"api": "1", "origin": origin, "destination": destination, "travelmode": "transit"}
    return _assert_host("google_maps", "https://www.google.com/maps/dir/?" + urlencode(params))


def build_accommodation_link(real_name: str, destination: str) -> DeepLink:
    """A Google search for the real accommodation name + destination — not a
    specific OTA's page. Two earlier designs were tried and both failed a
    live-verification check (2026-07-22):

    1. Booking.com/Agoda search-results pages worked mechanically, but
       committing to ONE OTA per accommodation risked the property simply
       not being listed there.
    2. Trying to link directly to "this one specific property" page (e.g. a
       Booking.com `/hotel/{country}/{slug}.html` canonical URL) by grabbing
       whatever booking.com/agoda.com link a search happened to surface was
       tested and rejected: searching a real small Taiwanese B&B's name
       plus "booking.com" surfaced a Booking.com link for a DIFFERENT,
       similarly-named property in a different city; searching a known
       hotel's name plus "agoda.com" surfaced no agoda.com page at all.
       Guessing "this is the same place" from search-result text is exactly
       the kind of confident-but-wrong claim this codebase avoids elsewhere
       (see e.g. the Booking/Agoda decision this replaced).

    A plain Google search has no such failure mode: it never commits to a
    specific (possibly wrong) property page, has the highest coverage of
    any single mechanism (works regardless of which OTA(s), if any, list
    the property), and lets the user compare results and pick the right one
    themselves."""
    params = {"q": f"{real_name} {destination} 訂房"}
    return _assert_host("google_search", "https://www.google.com/search?" + urlencode(params))


def build_activities_link(real_name_or_keyword: str) -> DeepLink:
    """KKday product search — medium confidence (WAF blocks direct fetch
    verification; route/params confirmed real only via Google's indexed
    results, live-verified 2026-07-18). Now optionally fed a real
    activity/venue name instead of a generic category keyword."""
    params = {"keyword": real_name_or_keyword}
    return _assert_host("kkday", "https://www.kkday.com/zh-tw/product/productlist?" + urlencode(params))
