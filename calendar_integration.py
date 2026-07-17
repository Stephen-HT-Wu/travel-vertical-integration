"""Google Calendar OAuth + event creation for the 'add my confirmed
itinerary to my own Google Calendar' step.

This module never ships or holds real Google credentials — the user must
create their own OAuth 2.0 client in Google Cloud Console first (see
README's "Google Calendar 設定" section) and place the downloaded JSON at
google_oauth_credentials.json in the project root (gitignored). Every
calendar write only happens after the user completes Google's own consent
screen; nothing here can create events without that real, user-driven
authorization step.

Scope is deliberately narrow: calendar.events only (create/manage events on
the user's calendar), not full calendar read access.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from schemas import ItineraryOutput

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CREDENTIALS_PATH = Path(__file__).parent / "google_oauth_credentials.json"


class CalendarNotConfigured(Exception):
    pass


def _require_credentials_file() -> Path:
    if not CREDENTIALS_PATH.exists():
        raise CalendarNotConfigured(
            f"找不到 {CREDENTIALS_PATH.name}。請先依照 README「Google Calendar 設定」的步驟，"
            "在 Google Cloud Console 建立 OAuth 用戶端憑證並下載 JSON 檔案，"
            f"存成專案根目錄的 {CREDENTIALS_PATH.name}。"
        )
    return CREDENTIALS_PATH


def build_flow(redirect_uri: str) -> Flow:
    return Flow.from_client_secrets_file(
        str(_require_credentials_file()), scopes=SCOPES, redirect_uri=redirect_uri
    )


def build_auth_url(redirect_uri: str, state: str) -> str:
    flow = build_flow(redirect_uri)
    auth_url, _ = flow.authorization_url(
        access_type="online", include_granted_scopes="true", prompt="consent", state=state
    )
    return auth_url


_TIME_BLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def _parse_time_block(time_block: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    m = _TIME_BLOCK_RE.match(time_block.strip())
    if not m:
        return None
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    return (h1, m1), (h2, m2)


def create_events_from_itinerary(
    code: str,
    redirect_uri: str,
    itinerary: ItineraryOutput,
    start_date: datetime,
    timezone_name: str = "Asia/Taipei",
) -> List[dict]:
    """Exchanges the OAuth code for credentials (this is the point where the
    user's real consent, already granted on Google's screen, takes effect),
    then creates one calendar event per itinerary time block starting from
    start_date. Time blocks that don't parse as "HH:MM-HH:MM" are skipped
    rather than guessed at. Returns the created event records."""
    flow = build_flow(redirect_uri)
    flow.fetch_token(code=code)
    service = build("calendar", "v3", credentials=flow.credentials)

    created: List[dict] = []
    for day in itinerary.days:
        day_date = start_date + timedelta(days=day.day_number - 1)
        for block in day.blocks:
            parsed = _parse_time_block(block.time_block)
            if not parsed:
                continue
            (h1, m1), (h2, m2) = parsed
            start_dt = day_date.replace(hour=h1, minute=m1, second=0, microsecond=0)
            end_dt = day_date.replace(hour=h2, minute=m2, second=0, microsecond=0)
            if end_dt <= start_dt:
                end_dt = end_dt + timedelta(days=1)  # time block crosses midnight
            event_body = {
                "summary": block.theme,
                "location": block.location_hint,
                "description": block.notes or "由旅遊產業垂直整合 Agent Demo 建立",
                "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone_name},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": timezone_name},
            }
            result = service.events().insert(calendarId="primary", body=event_body).execute()
            created.append({
                "id": result.get("id"),
                "html_link": result.get("htmlLink"),
                "summary": block.theme,
            })
    return created
