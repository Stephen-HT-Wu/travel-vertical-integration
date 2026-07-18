"""Interactive local web demo — chat-driven front end.

Real multi-turn text chat for inspiration / itinerary / transaction
candidates (transportation, accommodation, activities), each transaction
ending in a real deep-link referral to a real vendor's search results (see
deep_links.py) instead of a simulated order; then — unless disabled via
local_settings.json — automatic hand-off to the existing
UserSimulatorAgent-driven tail (dining/attractions/shopping/guide/
replanning/review); then an optional, explicitly user-authorized Google
Calendar sync (also toggleable).

Run with: python webapp.py   (then open http://127.0.0.1:8000)
"""
import json
import queue
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

import calendar_integration
from dashboard.render_dashboard import render as render_dashboard_html
from chat_session import ChatSession
from local_settings import load_local_settings
from persona import Persona
from schemas import CalendarSyncResult, RunConfig, TripLog

load_dotenv()

app = FastAPI(title="旅遊產業垂直整合 Agent Demo")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "webapp_static"
OUTPUT_DIR = BASE_DIR / "output" / "chat_runs"
CALENDAR_REDIRECT_URI = "http://127.0.0.1:8000/api/calendar/oauth/callback"

LOCAL_SETTINGS = load_local_settings()

CHAT_SESSIONS: Dict[str, ChatSession] = {}
TAIL_QUEUES: Dict[str, "queue.Queue"] = {}


class RunRequest(BaseModel):
    age_group: str = "26-35"
    gender: str = "unspecified"
    home_location: str = "台北"
    destination_location: str
    trip_length_type: str = "one_day"
    days: int = 1
    party_size: int = 2
    site_mode: str = "unrestricted"
    model: str = "claude-opus-4-8"


class MessageRequest(BaseModel):
    message: str = ""


class ConfirmInspirationRequest(BaseModel):
    option_id: str


class ConfirmCandidateRequest(BaseModel):
    stage: str
    candidate_id: str


def _get_session(run_id: str) -> ChatSession:
    session = CHAT_SESSIONS.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="找不到這個對話（run_id 不存在或伺服器已重啟）")
    return session


def _require_tail_enabled() -> None:
    if not LOCAL_SETTINGS.enable_tail_pipeline:
        raise HTTPException(status_code=403, detail="尾段自動流程（餐飲/景點/購物/導覽/重排/評價）目前透過 local_settings.json 停用")


def _require_calendar_enabled() -> None:
    if not LOCAL_SETTINGS.enable_calendar_sync:
        raise HTTPException(status_code=403, detail="Google Calendar 同步目前透過 local_settings.json 停用")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return {
        "default_inspiration_domains": LOCAL_SETTINGS.default_inspiration_domains,
        "enable_tail_pipeline": LOCAL_SETTINGS.enable_tail_pipeline,
        "enable_calendar_sync": LOCAL_SETTINGS.enable_calendar_sync,
    }


# -- chat lifecycle -------------------------------------------------------
@app.post("/api/chat/start")
def chat_start(req: RunRequest):
    persona = Persona(
        age_group=req.age_group, gender=req.gender, home_location=req.home_location,
        destination_location=req.destination_location,
        trip_length_type=req.trip_length_type, days=req.days, party_size=req.party_size,
    )
    allowed_domains = []
    if req.site_mode == "allowlist":
        if LOCAL_SETTINGS.default_inspiration_domains:
            allowed_domains = LOCAL_SETTINGS.default_inspiration_domains
        else:
            data = json.loads((BASE_DIR / "config" / "trusted_domains.json").read_text(encoding="utf-8"))
            allowed_domains = data["domains"]
    run_config = RunConfig(
        site_mode=req.site_mode, allowed_domains=allowed_domains, model=req.model, language="zh-TW"
    )
    run_id = uuid.uuid4().hex[:12]
    CHAT_SESSIONS[run_id] = ChatSession(run_id, persona, run_config, OUTPUT_DIR / run_id, local_settings=LOCAL_SETTINGS)
    return {"run_id": run_id}


@app.post("/api/chat/{run_id}/message")
def chat_message(run_id: str, req: MessageRequest):
    session = _get_session(run_id)
    try:
        return session.send_message(req.message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/chat/{run_id}/confirm-inspiration")
def chat_confirm_inspiration(run_id: str, req: ConfirmInspirationRequest):
    session = _get_session(run_id)
    try:
        return session.confirm_inspiration(req.option_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/chat/{run_id}/confirm-itinerary")
def chat_confirm_itinerary(run_id: str):
    session = _get_session(run_id)
    try:
        return session.confirm_itinerary()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/chat/{run_id}/refer")
def chat_refer(run_id: str, req: ConfirmCandidateRequest):
    """Confirms the picked candidate and immediately generates its deep-link
    referral in one round trip, so the frontend's 'pick a candidate' click
    only needs a single request."""
    session = _get_session(run_id)
    try:
        confirmed = session.confirm_candidate(req.stage, req.candidate_id)
        referral = session.generate_referral(req.stage)
        return {**referral, "candidate": confirmed["candidate"]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


# -- tail auto-run (SSE), same event shapes as the pipeline stages above --
def _execute_tail(run_id: str) -> None:
    session = CHAT_SESSIONS[run_id]
    q = TAIL_QUEUES[run_id]
    try:
        for event in session.run_tail():
            q.put(event)
    except Exception:
        pass  # already surfaced as an "error" event by run_tail_streaming
    finally:
        q.put(None)


@app.get("/api/chat/{run_id}/tail-events")
def chat_tail_events(run_id: str):
    _require_tail_enabled()
    _get_session(run_id)
    q: "queue.Queue" = queue.Queue()
    TAIL_QUEUES[run_id] = q
    thread = threading.Thread(target=_execute_tail, args=(run_id,), daemon=True)
    thread.start()

    def gen():
        while True:
            event = q.get()
            if event is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# -- dashboard / raw JSON for a chat run -----------------------------------
@app.get("/api/chat/{run_id}/dashboard", response_class=HTMLResponse)
def chat_dashboard(run_id: str):
    session = _get_session(run_id)
    return HTMLResponse(render_dashboard_html(session.trip_log))


@app.get("/api/chat/{run_id}/trip_log.json")
def chat_trip_log_json(run_id: str):
    session = _get_session(run_id)
    path = session.output_dir / "trip_log.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="trip_log.json 尚未產生")
    return FileResponse(path, media_type="application/json", filename="trip_log.json")


# -- Google Calendar (real OAuth; nothing here writes a calendar event ----
# without the user completing Google's own consent screen first) ---------
@app.get("/api/calendar/auth-url")
def calendar_auth_url(run_id: str, start_date: str):
    _require_calendar_enabled()
    _get_session(run_id)  # 404s early if the run doesn't exist
    state = json.dumps({"run_id": run_id, "start_date": start_date})
    try:
        url = calendar_integration.build_auth_url(CALENDAR_REDIRECT_URI, state=state)
    except calendar_integration.CalendarNotConfigured as exc:
        return {"error": str(exc)}
    return {"auth_url": url}


@app.get("/api/calendar/oauth/callback", response_class=HTMLResponse)
def calendar_oauth_callback(code: str, state: str):
    _require_calendar_enabled()
    try:
        parsed_state = json.loads(state)
        run_id = parsed_state["run_id"]
        start_date_str = parsed_state.get("start_date")
    except (json.JSONDecodeError, KeyError):
        return HTMLResponse("<p>無效的 state 參數。</p>", status_code=400)

    session = CHAT_SESSIONS.get(run_id)
    if session is not None:
        trip_log, output_dir = session.trip_log, session.output_dir
    else:
        # in-memory session lost (e.g. server restarted) -- fall back to disk
        trip_log_path = OUTPUT_DIR / run_id / "trip_log.json"
        if not trip_log_path.exists():
            return HTMLResponse("<p>找不到這次執行的紀錄。</p>", status_code=404)
        trip_log = TripLog.model_validate_json(trip_log_path.read_text(encoding="utf-8"))
        output_dir = trip_log_path.parent

    if not trip_log.stages.itinerary:
        return HTMLResponse("<p>行程尚未確認，無法加入行事曆。</p>", status_code=400)

    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        start_dt = datetime.now() + timedelta(days=1)

    try:
        created = calendar_integration.create_events_from_itinerary(
            code, CALENDAR_REDIRECT_URI, trip_log.stages.itinerary, start_dt
        )
    except Exception as exc:  # noqa: BLE001
        return HTMLResponse(f"<p>加入 Google Calendar 失敗：{exc}</p><p><a href='/'>返回</a></p>", status_code=500)

    trip_log.calendar_sync = CalendarSyncResult(
        synced_at=datetime.now(timezone.utc).isoformat(),
        calendar_event_count=len(created),
        calendar_id="primary",
        event_links=[e["html_link"] for e in created if e.get("html_link")],
    )
    (output_dir / "trip_log.json").write_text(trip_log.model_dump_json(indent=2), encoding="utf-8")

    links_html = "".join(
        f'<li><a href="{link}" target="_blank" rel="noopener">{link}</a></li>'
        for link in trip_log.calendar_sync.event_links[:10]
    )
    return HTMLResponse(f"""
        <html><body style="font-family: -apple-system, sans-serif; max-width: 640px; margin: 3rem auto; padding: 0 1rem;">
        <h2>✅ 已加入 {len(created)} 個行程到你的 Google Calendar</h2>
        <ul>{links_html}</ul>
        <p><a href="/">返回 demo</a></p>
        </body></html>
    """)


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
