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
    "booking": ("Booking.com", "www.booking.com"),
    "agoda": ("Agoda", "www.agoda.com"),
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


def build_accommodation_links(real_name: str, party_size: int = 1) -> "list[DeepLink]":
    """Returns [Booking.com, Agoda] for a REAL hotel/lodging name pulled
    from live search results.

    Booking.com: `ss={real_name}`, no dates — live-verified working
    (2026-07-18), high confidence. Feeding it the real name (rather than the
    old category/style string) actually resolves a past honesty tension:
    the old version deliberately avoided a specific hotel name because that
    name was LLM-fabricated; now that it's a real, search-sourced name,
    using it directly is the more honest choice, not a compromise.

    Agoda: live-checked TWICE in a read-only browser session
    (2026-07-22, no purchase/booking action taken) — including a targeted
    re-check after an initial design draft assumed a "homepage prefill"
    tier would work. It does not:
      - A genuine *results-list* URL requires a `lastSearchedCity=<id>&
        area=<id>` pair resolved by Agoda's own undocumented autocomplete
        endpoint (e.g. searching "礁溪" resolved to
        `lastSearchedCity=12080&area=503393`) — not derivable from a plain
        name, and out of scope to reverse-engineer.
      - A URL built from only human-constructable params (`textToSearch`/
        `searchText`/`checkIn`/`checkOut`/`adults`/`los`/`locale`, no
        resolved ID) does NOT prefill anything: Agoda strips the entire
        query string client-side and redirects to a bare homepage
        (`window.location.search` came back `""` after landing, confirmed
        via direct inspection, for two different param-name variants).
      - Submitting the on-page search form without first picking a real
        autocomplete suggestion also does not navigate anywhere — Agoda
        requires the resolved ID even to proceed.
    Given none of that is derivable without calling Agoda's private
    autocomplete API (out of scope — undocumented, ToS-risky), the honest
    choice is a plain homepage link with no false pretense of pre-filling
    anything. The UI must tell the user to search for the real name
    themselves once there (see webapp_static/index.html referral panel
    copy) rather than implying the link does more than it does."""
    booking = _assert_host(
        "booking",
        "https://www.booking.com/searchresults.html?"
        + urlencode({"ss": real_name, "group_adults": party_size, "no_rooms": 1}),
    )
    agoda = _assert_host("agoda", "https://www.agoda.com/zh-tw/")
    return [booking, agoda]


def build_activities_link(real_name_or_keyword: str) -> DeepLink:
    """KKday product search — medium confidence (WAF blocks direct fetch
    verification; route/params confirmed real only via Google's indexed
    results, live-verified 2026-07-18). Now optionally fed a real
    activity/venue name instead of a generic category keyword."""
    params = {"keyword": real_name_or_keyword}
    return _assert_host("kkday", "https://www.kkday.com/zh-tw/product/productlist?" + urlencode(params))
