"""Real-user multi-turn chat session for the unified planning phase, ending
with the user picking one of two complete, real (not simulated) trip plan
options — each bundling itinerary + transportation + accommodation (if
overnight) + activities, all with real vendor deep links (see deep_links.py)
— instead of separate per-stage confirm+referral steps. Then hands off to
the existing auto-pipeline (orchestrator.run_tail_streaming) for the
content-recommendation tail (dining/attractions/shopping/guide/replanning/
review) — unless disabled via local_settings.enable_tail_pipeline, in which
case the session ends right after the plan pick. That tail, when it does
run, stays UserSimulatorAgent-driven exactly as before.

Persona is no longer collected via a structured form: the session starts
with a placeholder Persona and PlanningAgent.generate_plan_bundle() resolves
the real one from the user's free-text description (see
agents/planning_agent.py's IntakeDecision), reusing the same clarification-
round mechanism that used to handle style intent alone.

Unlike the old fully-automatic flow, the plan pick here comes from an actual
user action (a button click reaching select_plan) — never inferred from
free text and never simulated.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import deep_links
from agents.planning_agent import PlanningAgent
from local_settings import LocalSettings
from orchestrator import TripOrchestrator
from persona import Persona
from schemas import (
    CandidateConfirmation,
    CandidateStageOutput,
    ChatMessage,
    HITLLogEntry,
    ItineraryConfirmation,
    ItineraryOutput,
    ReferralEvent,
    RunConfig,
    StageMetrics,
    TripLog,
    TripPlanBundleRecord,
)
from site_index import SiteIndex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatSession:
    def __init__(
        self,
        run_id: str,
        run_config: RunConfig,
        output_dir: Path,
        local_settings: Optional[LocalSettings] = None,
        site_index: Optional[SiteIndex] = None,
    ):
        self.run_id = run_id
        # Placeholder — overwritten by PlanningAgent.generate_plan_bundle()'s
        # resolved_persona as soon as the free-text intake resolves. Both
        # required str fields, so an empty Persona() alone would fail
        # validation; empty strings are valid placeholders.
        self.persona = Persona(home_location="", destination_location="")
        self.run_config = run_config
        self.output_dir = output_dir
        self.local_settings = local_settings or LocalSettings()
        self.site_index = site_index
        self.phase = "itinerary"  # itinerary -> tail -> done
        self.trip_log = TripLog(run_id=run_id, created_at=_now(), persona=self.persona, run_config=run_config)

        self.history: Dict[str, List[dict]] = {}
        self.last_turn: Dict[str, object] = {}
        self.clarification_rounds = 0
        self.plan_ready = False

        self.planning_agent = PlanningAgent(run_config.model)

    def _next_after_plan_pick(self) -> str:
        return "tail" if self.local_settings.enable_tail_pipeline else "done"

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trip_log.json").write_text(self.trip_log.model_dump_json(indent=2), encoding="utf-8")

    def _log_hitl(self, stage: str, checkpoint_type: str, summary: str, status: str, reasoning: str) -> None:
        self.trip_log.hitl_log.append(
            HITLLogEntry(
                stage=stage, checkpoint_type=checkpoint_type, presented_summary=summary,
                decision_status=status, reasoning=reasoning, timestamp=_now(),
            )
        )

    def _record_transcript(self, phase: str, user_message: str, reply_message: str) -> None:
        transcript = self.trip_log.chat_transcripts.setdefault(phase, [])
        transcript.append(ChatMessage(role="user", content=user_message, timestamp=_now()))
        transcript.append(ChatMessage(role="assistant", content=reply_message, timestamp=_now()))

    # -- chat turns -------------------------------------------------------
    def send_message(self, user_message: str) -> dict:
        phase = self.phase
        if phase != "itinerary":
            raise ValueError(f"send_message is not valid in phase '{phase}'")
        history = self.history.get(phase, [])

        agent = self.planning_agent
        if not self.plan_ready:
            call = lambda: agent.generate_plan_bundle(  # noqa: E731
                self.persona, history, user_message, self.run_config, self.site_index,
                self.local_settings.rag_top_n, self.clarification_rounds,
                self.local_settings.rag_max_clarifying_questions,
            )
        else:
            call = lambda: agent.refine_bundle(history, user_message)  # noqa: E731

        try:
            turn, new_history, resolved_persona = call()
        finally:
            # Record metrics for whatever API call happened, even if a
            # downstream check rejects the result — the call still cost
            # real tokens and should count toward the feasibility-
            # assessment totals.
            if agent.last_call_metrics:
                stage_metrics = StageMetrics.from_calls(phase, agent.last_call_metrics)
                self.trip_log.metrics.add_stage(stage_metrics)
                self._save()

        if resolved_persona is not None:
            self.persona = resolved_persona
            self.trip_log.persona = resolved_persona  # dashboard/tail/calendar all read trip_log.persona directly

        if not self.plan_ready:
            if turn.needs_clarification:
                self.clarification_rounds += 1
            else:
                self.plan_ready = True

        if turn.plan_ready:
            # Overwritten on every ready turn (including refinements) so the
            # bundle available to select_plan() always reflects the latest
            # edits, not just the first draft.
            self.trip_log.plan_bundle = TripPlanBundleRecord(
                reply_message=turn.reply_message, options=turn.options,
                agent_recommended_option_id=turn.agent_recommended_option_id,
            )

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

    # -- plan selection ---------------------------------------------------
    def select_plan(self, option_id: str) -> dict:
        """Purely local, deterministic — no LLM call. Writes the chosen
        TripPlanOption's data into the existing stages.itinerary/
        transportation/accommodation/activities shapes (unchanged from
        before this pivot) so orchestrator.run_tail_streaming/
        calendar_integration/dashboard all keep working without knowing
        about the two-option bundle. Builds real vendor deep links (see
        deep_links.py) for every real transportation/accommodation/activity
        candidate the option carries, and records exactly one HITL entry
        for the whole pick — replacing what used to be up to 7 entries
        (1 itinerary confirm + 3x(candidate confirm + referral))."""
        bundle = self.trip_log.plan_bundle
        if bundle is None or not self.plan_ready:
            raise ValueError("尚未有完整方案可以選擇")
        option = next((o for o in bundle.options if o.option_id == option_id), None)
        if option is None:
            raise ValueError(f"找不到方案 {option_id}")

        itinerary = ItineraryOutput(
            itinerary_id=str(uuid.uuid4()),
            days=option.days,
            based_on_inspiration_ids=[],
            sources=option.primary_sources + option.corroboration_sources,
            confirmation=ItineraryConfirmation(
                status="confirmed", feedback=f"使用者選擇方案 {option.label}：{option.why_recommended}"
            ),
        )
        self.trip_log.stages.itinerary = itinerary

        summary_parts: List[str] = []
        stage_referrals: Dict[str, List[dict]] = {}

        if option.transportation:
            candidate = option.transportation[0]
            link = deep_links.build_transportation_link(
                self.persona.home_location, self.persona.destination_location
            )
            referral = ReferralEvent(
                stage="transportation", vendor=link.vendor, url=link.url,
                deep_link_query=self.persona.destination_location, candidate_id=candidate.id,
                candidate_name=candidate.name, referred_at=_now(),
            )
            self.trip_log.stages.transportation = CandidateStageOutput(
                candidates=option.transportation,
                agent_selected_candidate_id=candidate.id,
                agent_selection_rationale="使用者選擇了包含此交通建議的整體行程方案",
                confirmation=CandidateConfirmation(
                    status="confirmed", final_candidate_id=candidate.id, feedback="使用者選擇了包含此方案的整體行程"
                ),
                referral=referral, referrals=[referral],
            )
            stage_referrals["transportation"] = [referral.model_dump(mode="json")]
            summary_parts.append(f"交通：{candidate.name}")

        if option.accommodation:
            candidate = option.accommodation[0]
            real_name = (candidate.deep_link_query or candidate.name).strip()
            links = deep_links.build_accommodation_links(real_name, self.persona.party_size)
            referrals = [
                ReferralEvent(
                    stage="accommodation", vendor=link.vendor, url=link.url, deep_link_query=real_name,
                    candidate_id=candidate.id, candidate_name=candidate.name, referred_at=_now(),
                )
                for link in links
            ]
            self.trip_log.stages.accommodation = CandidateStageOutput(
                candidates=option.accommodation,
                agent_selected_candidate_id=candidate.id,
                agent_selection_rationale="使用者選擇了包含此住宿建議的整體行程方案",
                confirmation=CandidateConfirmation(
                    status="confirmed", final_candidate_id=candidate.id, feedback="使用者選擇了包含此方案的整體行程"
                ),
                referral=referrals[0], referrals=referrals,
            )
            stage_referrals["accommodation"] = [r.model_dump(mode="json") for r in referrals]
            summary_parts.append(f"住宿：{candidate.name}")

        if option.activities:
            referrals = []
            for candidate in option.activities:
                real_name = (candidate.deep_link_query or candidate.name).strip()
                link = deep_links.build_activities_link(real_name)
                referrals.append(
                    ReferralEvent(
                        stage="activities", vendor=link.vendor, url=link.url, deep_link_query=real_name,
                        candidate_id=candidate.id, candidate_name=candidate.name, referred_at=_now(),
                    )
                )
            self.trip_log.stages.activities = CandidateStageOutput(
                candidates=option.activities,
                agent_selected_candidate_id=option.activities[0].id,
                agent_selection_rationale="使用者選擇了包含這些活動建議的整體行程方案",
                confirmation=CandidateConfirmation(
                    status="confirmed", final_candidate_id=option.activities[0].id, feedback="使用者選擇了包含此方案的整體行程"
                ),
                referral=referrals[0], referrals=referrals,
            )
            stage_referrals["activities"] = [r.model_dump(mode="json") for r in referrals]
            summary_parts.append("活動：" + "、".join(c.name for c in option.activities))

        bundle.selected_option_id = option_id
        bundle.selected_at = _now()

        total_links = sum(len(v) for v in stage_referrals.values())
        self._log_hitl(
            "itinerary", "plan_selection",
            f"方案 {option.label}：{option.why_recommended}" + ("｜" + "｜".join(summary_parts) if summary_parts else ""),
            "confirmed" if option_id == bundle.agent_recommended_option_id else "swapped",
            f"使用者選擇方案 {option.label}，共產生 {total_links} 個真實 vendor 導流連結",
        )

        next_phase = self._next_after_plan_pick()
        self.phase = next_phase
        self._save()
        return {
            "type": "plan_selected", "option": option.model_dump(mode="json"),
            "referrals": stage_referrals, "next_phase": next_phase,
        }

    # -- tail handoff ---------------------------------------------------
    def run_tail(self) -> Iterator[dict]:
        if not self.local_settings.enable_tail_pipeline:
            self.phase = "done"
            return
        orchestrator = TripOrchestrator(self.run_config, self.output_dir)
        yield from orchestrator.run_tail_streaming(self.trip_log)
        self.phase = "done"
