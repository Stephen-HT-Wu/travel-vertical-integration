"""Real-user multi-turn chat session for the front three layers (inspiration,
itinerary, transaction candidates), ending each transaction candidate with a
real deep-link referral to a real vendor's search results (see deep_links.py)
instead of pretending a booking happened, then handing off to the existing
auto-pipeline (orchestrator.run_tail_streaming) for the content-recommendation
tail (dining/attractions/shopping/guide/replanning/review) — unless disabled
via local_settings.enable_tail_pipeline, in which case the session ends right
after the last referral. That tail, when it does run, stays
UserSimulatorAgent-driven exactly as before.

Unlike the old fully-automatic flow, confirmations here come from an actual
user action (a button click reaching confirm_inspiration/confirm_itinerary/
confirm_candidate) — never inferred from free text and never simulated.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from agents.accommodation_agent import AccommodationAgent
from agents.activities_agent import ActivitiesAgent
from agents.inspiration_agent import InspirationAgent
from agents.itinerary_agent import ItineraryAgent
from agents.transportation_agent import TransportationAgent
from deep_links import build_deep_link
from local_settings import LocalSettings
from orchestrator import TripOrchestrator
from persona import Persona
from schemas import (
    CandidateConfirmation,
    CandidateStageOutput,
    ChatMessage,
    HITLLogEntry,
    InspirationDestinationOption,
    InspirationOutput,
    ItineraryConfirmation,
    ItineraryOutput,
    ReferralEvent,
    RunConfig,
    StageMetrics,
    TripLog,
    summarize_itinerary,
)

CANDIDATE_PHASE_NEXT = {"transportation": "accommodation", "accommodation": "activities", "activities": "tail"}
INSPIRATION_TO_TRANSPORT_PHASES = ["inspiration", "itinerary", "transportation", "accommodation", "activities"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatSession:
    def __init__(
        self,
        run_id: str,
        persona: Persona,
        run_config: RunConfig,
        output_dir: Path,
        local_settings: Optional[LocalSettings] = None,
    ):
        self.run_id = run_id
        self.persona = persona
        self.run_config = run_config
        self.output_dir = output_dir
        self.local_settings = local_settings or LocalSettings()
        self.phase = "inspiration"  # inspiration -> itinerary -> transportation -> accommodation -> activities -> tail -> done
        self.trip_log = TripLog(run_id=run_id, created_at=_now(), persona=persona, run_config=run_config)

        self.history: Dict[str, List[dict]] = {}
        self.last_turn: Dict[str, object] = {}
        self.selected_inspiration: Optional[InspirationDestinationOption] = None

        model = run_config.model
        self.inspiration_agent = InspirationAgent(model)
        self.itinerary_agent = ItineraryAgent(model)
        self.transportation_agent = TransportationAgent(model)
        self.accommodation_agent = AccommodationAgent(model)
        self.activities_agent = ActivitiesAgent(model)

    def _next_phase_after(self, stage_name: str) -> str:
        next_phase = CANDIDATE_PHASE_NEXT[stage_name]
        if next_phase == "tail" and not self.local_settings.enable_tail_pipeline:
            return "done"
        return next_phase

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trip_log.json").write_text(self.trip_log.model_dump_json(indent=2), encoding="utf-8")

    def _log_hitl(self, stage: str, checkpoint_type: str, summary: str, status: str, reasoning: str) -> None:
        self.trip_log.hitl_log.append(
            HITLLogEntry(
                stage=stage, checkpoint_type=checkpoint_type, presented_summary=summary,
                decision_status=status, user_simulator_reasoning=reasoning, timestamp=_now(),
            )
        )

    def _record_transcript(self, phase: str, user_message: str, reply_message: str) -> None:
        transcript = self.trip_log.chat_transcripts.setdefault(phase, [])
        transcript.append(ChatMessage(role="user", content=user_message, timestamp=_now()))
        transcript.append(ChatMessage(role="assistant", content=reply_message, timestamp=_now()))

    # -- chat turns -------------------------------------------------------
    def send_message(self, user_message: str) -> dict:
        phase = self.phase
        history = self.history.get(phase, [])

        if phase == "inspiration":
            agent = self.inspiration_agent
            call = lambda: agent.chat(self.persona, history, user_message, self.run_config)  # noqa: E731
        elif phase == "itinerary":
            agent = self.itinerary_agent
            call = lambda: agent.chat(  # noqa: E731
                self.persona, history, user_message, self.selected_inspiration, self.run_config
            )
        elif phase == "transportation":
            agent = self.transportation_agent
            call = lambda: agent.chat(self.persona, history, user_message, self.trip_log.stages.itinerary)  # noqa: E731
        elif phase == "accommodation":
            agent = self.accommodation_agent
            call = lambda: agent.chat(  # noqa: E731
                self.persona, history, user_message, self.trip_log.stages.itinerary, self.trip_log.stages.transportation
            )
        elif phase == "activities":
            agent = self.activities_agent
            call = lambda: agent.chat(self.persona, history, user_message, self.trip_log.stages.itinerary)  # noqa: E731
        else:
            raise ValueError(f"send_message is not valid in phase '{phase}'")

        try:
            turn, new_history = call()
        finally:
            # Record metrics for whatever API call happened, even if a
            # downstream check (e.g. validate_candidate_turn) rejects the
            # result — the call still cost real tokens and should count
            # toward the feasibility-assessment totals.
            if agent.last_call_metrics:
                stage_metrics = StageMetrics.from_calls(phase, agent.last_call_metrics)
                self.trip_log.metrics.add_stage(stage_metrics)
                self._save()

        self.history[phase] = new_history
        self.last_turn[phase] = turn
        self._record_transcript(phase, user_message, turn.reply_message)
        self._save()

        return {
            "type": "chat_reply",
            "phase": phase,
            "reply_message": turn.reply_message,
            "proposal": turn.model_dump(mode="json"),
            "metrics": stage_metrics.model_dump(),
            "totals": self.trip_log.metrics.model_dump(),
        }

    # -- confirmations ------------------------------------------------------
    def confirm_inspiration(self, option_id: str) -> dict:
        turn = self.last_turn.get("inspiration")
        if turn is None:
            raise ValueError("尚未有靈感提案可以確認")
        option = next((o for o in turn.destination_options if o.id == option_id), None)
        if option is None:
            raise ValueError(f"找不到靈感選項 {option_id}")

        self.selected_inspiration = option
        self.trip_log.stages.inspiration = InspirationOutput(
            destination_options=turn.destination_options, queries_used=[]
        )
        self._log_hitl("inspiration", "candidate_confirmation", option.name, "confirmed", f"使用者選定：{option.name}")
        self.phase = "itinerary"
        self._save()
        return {
            "type": "phase_advanced", "from_phase": "inspiration", "to_phase": "itinerary",
            "selection": option.model_dump(mode="json"),
        }

    def confirm_itinerary(self) -> dict:
        turn = self.last_turn.get("itinerary")
        if turn is None:
            raise ValueError("尚未有行程草案可以確認")

        itinerary = ItineraryOutput(
            itinerary_id=str(uuid.uuid4()),
            days=turn.days,
            based_on_inspiration_ids=[self.selected_inspiration.id] if self.selected_inspiration else [],
            confirmation=ItineraryConfirmation(status="confirmed", feedback="使用者於對話中確認行程"),
        )
        self.trip_log.stages.itinerary = itinerary
        self._log_hitl(
            "itinerary", "itinerary_confirmation", summarize_itinerary(itinerary), "confirmed", "使用者於對話中確認行程"
        )
        self.phase = "transportation"
        self._save()
        return {
            "type": "phase_advanced", "from_phase": "itinerary", "to_phase": "transportation",
            "itinerary": itinerary.model_dump(mode="json"),
        }

    def confirm_candidate(self, stage_name: str, candidate_id: str) -> dict:
        if stage_name not in CANDIDATE_PHASE_NEXT:
            raise ValueError(f"'{stage_name}' 不是可對話確認的交易候選階段")
        turn = self.last_turn.get(stage_name)
        if turn is None:
            raise ValueError(f"尚未有 {stage_name} 候選方案可以確認")
        candidate = next((c for c in turn.candidates if c.id == candidate_id), None)
        if candidate is None:
            raise ValueError(f"找不到候選方案 {candidate_id}")

        status = "confirmed" if candidate_id == turn.agent_selected_candidate_id else "swapped"
        stage_output = CandidateStageOutput(
            candidates=turn.candidates,
            agent_selected_candidate_id=turn.agent_selected_candidate_id,
            agent_selection_rationale=turn.agent_selection_rationale,
            confirmation=CandidateConfirmation(
                status=status, final_candidate_id=candidate_id, feedback="使用者於對話中選定"
            ),
        )
        setattr(self.trip_log.stages, stage_name, stage_output)
        self._log_hitl(
            stage_name, "candidate_confirmation", turn.agent_selection_rationale, status, f"使用者選定：{candidate.name}"
        )
        self._save()
        return {
            "type": "candidate_confirmed", "phase": stage_name,
            "candidate": candidate.model_dump(mode="json"),
        }

    def generate_referral(self, stage_name: str) -> dict:
        """Purely local, deterministic — no LLM call, no real booking/payment
        system contacted. Builds a real deep link to a real vendor's search
        results (see deep_links.py) and records that we referred the user
        out; we do not track what happens after they click through."""
        stage_output: Optional[CandidateStageOutput] = getattr(self.trip_log.stages, stage_name, None)
        if stage_output is None or stage_output.confirmation is None:
            raise ValueError(f"{stage_name} 尚未確認候選方案，無法導流")

        candidate = next(c for c in stage_output.candidates if c.id == stage_output.confirmation.final_candidate_id)
        if stage_name == "transportation":
            # The real target is already a known, fixed fact (the persona's
            # own destination_location) — more reliable than trusting the
            # model to have echoed it back correctly per-candidate.
            query = self.persona.destination_location.strip() or self.persona.home_location
        else:
            query = (candidate.deep_link_query or "").strip() or self.persona.home_location

        deep_link = build_deep_link(
            stage_name, deep_link_query=query, origin=self.persona.home_location, party_size=self.persona.party_size
        )
        referral = ReferralEvent(
            stage=stage_name, vendor=deep_link.vendor, url=deep_link.url, deep_link_query=query,
            candidate_id=candidate.id, candidate_name=candidate.name, referred_at=_now(),
        )
        stage_output.referral = referral
        setattr(self.trip_log.stages, stage_name, stage_output)
        self._log_hitl(
            stage_name, "referral", f"導流至 {deep_link.vendor}：{query}",
            "referred", f"使用者前往 {deep_link.vendor} 查看真實報價與庫存",
        )

        next_phase = self._next_phase_after(stage_name)
        self.phase = next_phase
        self._save()
        return {
            "type": "referral_created", "phase": stage_name,
            "referral": referral.model_dump(mode="json"), "next_phase": next_phase,
        }

    # -- tail handoff ---------------------------------------------------
    def run_tail(self) -> Iterator[dict]:
        if not self.local_settings.enable_tail_pipeline:
            self.phase = "done"
            return
        orchestrator = TripOrchestrator(self.run_config, self.output_dir)
        yield from orchestrator.run_tail_streaming(self.trip_log)
        self.phase = "done"
