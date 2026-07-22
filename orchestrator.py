"""Sequences every stage, wires the HITL checkpoints to UserSimulatorAgent,
and owns the hard boundary: no booking/payment client is ever constructed
here, so there is nothing for any stage — or any adversarial prompt — to
call for a real transaction.

run_streaming() is a generator that yields progress events (stage_started /
stage_completed / hitl_checkpoint / run_completed / error) with per-stage
CallMetrics attached, so both the CLI (run()) and the interactive web demo
(webapp.py) can show live token/cost/time telemetry as each step runs.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from agents.accommodation_agent import AccommodationAgent
from agents.activities_agent import ActivitiesAgent
from agents.attractions_agent import AttractionsAgent
from agents.base_agent import StageAgent
from agents.dining_agent import DiningAgent
from agents.in_trip_guide_agent import InTripGuideAgent
from agents.inspiration_agent import InspirationAgent
from agents.itinerary_agent import ItineraryAgent
from agents.shopping_agent import ShoppingAgent
from agents.transportation_agent import TransportationAgent
from agents.user_simulator_agent import UserSimulatorAgent
from persona import Persona
from schemas import (
    CallMetrics,
    CandidateStageOutput,
    HITLLogEntry,
    ItineraryOutput,
    ReplanningOutput,
    RunConfig,
    RunMetrics,
    StageMetrics,
    TripLog,
    summarize_itinerary,
)

STAGE_LABELS = {
    "inspiration": "靈感探索",
    "itinerary": "行程規劃",
    "transportation": "交通",
    "accommodation": "住宿",
    "dining": "餐飲",
    "attractions": "景點",
    "activities": "活動",
    "shopping": "購物",
    "in_trip_guide": "行中導覽",
    "replanning": "動態重新排程",
    "review": "評價與分享",
}

DOWNSTREAM_STAGE_ORDER = [
    "transportation",
    "accommodation",
    "activities",
    "dining",
    "attractions",
    "shopping",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TripOrchestrator:
    def __init__(self, run_config: RunConfig, output_dir: Path):
        self.run_config = run_config
        self.output_dir = output_dir
        model = run_config.model
        self.inspiration_agent = InspirationAgent(model)
        self.itinerary_agent = ItineraryAgent(model)
        self.transportation_agent = TransportationAgent(model)
        self.accommodation_agent = AccommodationAgent(model)
        self.dining_agent = DiningAgent(model)
        self.attractions_agent = AttractionsAgent(model)
        self.activities_agent = ActivitiesAgent(model)
        self.shopping_agent = ShoppingAgent(model)
        self.in_trip_guide_agent = InTripGuideAgent(model)
        self.user_simulator: Optional[UserSimulatorAgent] = None
        self.trip_log: Optional[TripLog] = None

    # -- event helpers ------------------------------------------------
    def _event(self, event_type: str, stage: Optional[str] = None, label: Optional[str] = None, data: Optional[dict] = None) -> dict:
        return {"type": event_type, "stage": stage, "label": label, "data": data or {}, "timestamp": _now()}

    def _log_hitl(
        self,
        checkpoint_type: str,
        stage: str,
        presented_summary: str,
        decision_status: str,
        reasoning: str,
    ) -> HITLLogEntry:
        entry = HITLLogEntry(
            stage=stage,
            checkpoint_type=checkpoint_type,
            presented_summary=presented_summary,
            decision_status=decision_status,
            reasoning=reasoning,
            timestamp=_now(),
        )
        self.trip_log.hitl_log.append(entry)
        return entry

    def _finalize_candidate_stage(
        self,
        run_metrics: RunMetrics,
        stage_name: str,
        agent: StageAgent,
        stage_output: CandidateStageOutput,
    ) -> Iterator[dict]:
        """Shared HITL-confirm + metrics-collect + save + event logic used by
        all six downstream candidate stages, after their .run() already
        produced stage_output."""
        calls: List[CallMetrics] = list(agent.last_call_metrics)

        confirmation = self.user_simulator.confirm_candidate(stage_name, stage_output)
        calls.extend(self.user_simulator.last_call_metrics)
        self._log_hitl(
            "candidate_confirmation", stage_name, stage_output.agent_selection_rationale,
            confirmation.status, confirmation.feedback,
        )
        yield self._event(
            "hitl_checkpoint", stage=stage_name, label=STAGE_LABELS[stage_name],
            data=self.trip_log.hitl_log[-1].model_dump(mode="json"),
        )

        stage_output.confirmation = confirmation
        setattr(self.trip_log.stages, stage_name, stage_output)
        stage_metrics = StageMetrics.from_calls(stage_name, calls)
        run_metrics.add_stage(stage_metrics)
        self._save(self.trip_log)
        yield self._event(
            "stage_completed", stage=stage_name, label=STAGE_LABELS[stage_name],
            data={
                "output": stage_output.model_dump(mode="json"),
                "metrics": stage_metrics.model_dump(),
                "totals": run_metrics.model_dump(),
            },
        )

    # -- main pipeline --------------------------------------------------
    def run_streaming(self, persona: Persona) -> Iterator[dict]:
        trip_log = TripLog(
            run_id=str(uuid.uuid4()), created_at=_now(), persona=persona, run_config=self.run_config,
        )
        self.trip_log = trip_log
        self.user_simulator = UserSimulatorAgent(persona, self.run_config.model)
        run_metrics = trip_log.metrics

        try:
            yield self._event(
                "run_started",
                data={"persona": persona.model_dump(mode="json"), "run_config": self.run_config.model_dump(mode="json")},
            )

            # 1. Inspiration
            yield self._event("stage_started", stage="inspiration", label=STAGE_LABELS["inspiration"])
            inspiration = self.inspiration_agent.run(persona, self.run_config)
            trip_log.stages.inspiration = inspiration
            stage_metrics = StageMetrics.from_calls("inspiration", self.inspiration_agent.last_call_metrics)
            run_metrics.add_stage(stage_metrics)
            self._save(trip_log)
            yield self._event(
                "stage_completed", stage="inspiration", label=STAGE_LABELS["inspiration"],
                data={
                    "output": inspiration.model_dump(mode="json"),
                    "metrics": stage_metrics.model_dump(),
                    "totals": run_metrics.model_dump(),
                },
            )

            # 2. Itinerary + confirm (max 1 revision loop)
            yield self._event("stage_started", stage="itinerary", label=STAGE_LABELS["itinerary"])
            itinerary_calls: List[CallMetrics] = []

            itinerary = self.itinerary_agent.run(persona, inspiration)
            itinerary_calls.extend(self.itinerary_agent.last_call_metrics)

            confirmation = self.user_simulator.confirm_itinerary(itinerary)
            itinerary_calls.extend(self.user_simulator.last_call_metrics)
            self._log_hitl("itinerary_confirmation", "itinerary", summarize_itinerary(itinerary), confirmation.status, confirmation.feedback)
            yield self._event("hitl_checkpoint", stage="itinerary", label=STAGE_LABELS["itinerary"], data=trip_log.hitl_log[-1].model_dump(mode="json"))

            if confirmation.status == "revised":
                itinerary = self.itinerary_agent.run(persona, inspiration, revision_notes=confirmation.revision_notes)
                itinerary_calls.extend(self.itinerary_agent.last_call_metrics)
                confirmation = self.user_simulator.confirm_itinerary(itinerary)
                itinerary_calls.extend(self.user_simulator.last_call_metrics)
                self._log_hitl("itinerary_confirmation", "itinerary", summarize_itinerary(itinerary), confirmation.status, confirmation.feedback)
                yield self._event("hitl_checkpoint", stage="itinerary", label=STAGE_LABELS["itinerary"], data=trip_log.hitl_log[-1].model_dump(mode="json"))

            itinerary.confirmation = confirmation
            trip_log.stages.itinerary = itinerary
            stage_metrics = StageMetrics.from_calls("itinerary", itinerary_calls)
            run_metrics.add_stage(stage_metrics)
            self._save(trip_log)
            yield self._event(
                "stage_completed", stage="itinerary", label=STAGE_LABELS["itinerary"],
                data={
                    "output": itinerary.model_dump(mode="json"),
                    "metrics": stage_metrics.model_dump(),
                    "totals": run_metrics.model_dump(),
                },
            )

            # 3. Transaction-candidate stages (Layer 3 — transportation/accommodation/activities;
            # grouped together and ahead of the Layer 7 content stages so this matches the order
            # the chat-driven web flow uses: run_tail_streaming() picks up right after this point).
            yield self._event("stage_started", stage="transportation", label=STAGE_LABELS["transportation"])
            transportation = self.transportation_agent.run(persona, itinerary)
            yield from self._finalize_candidate_stage(run_metrics, "transportation", self.transportation_agent, transportation)

            yield self._event("stage_started", stage="accommodation", label=STAGE_LABELS["accommodation"])
            accommodation = self.accommodation_agent.run(persona, itinerary, transportation)
            yield from self._finalize_candidate_stage(run_metrics, "accommodation", self.accommodation_agent, accommodation)

            yield self._event("stage_started", stage="activities", label=STAGE_LABELS["activities"])
            activities = self.activities_agent.run(persona, itinerary)
            yield from self._finalize_candidate_stage(run_metrics, "activities", self.activities_agent, activities)

            yield from self._run_tail(trip_log)
        except Exception as exc:  # noqa: BLE001 — surfaced to CLI/web caller as an event, then re-raised
            yield self._event("error", data={"message": str(exc)})
            raise

    def run_tail_streaming(self, trip_log: TripLog) -> Iterator[dict]:
        """Chat-mode handoff entry point: continues an existing trip_log
        (persona/inspiration/itinerary/transportation/accommodation/
        activities already populated by ChatSession) through dining ->
        attractions -> shopping -> in_trip_guide -> replanning -> review,
        exactly like the tail of run_streaming() above."""
        self.trip_log = trip_log
        self.user_simulator = UserSimulatorAgent(trip_log.persona, self.run_config.model)
        try:
            yield from self._run_tail(trip_log)
        except Exception as exc:  # noqa: BLE001
            yield self._event("error", data={"message": str(exc)})
            raise

    def _run_tail(self, trip_log: TripLog) -> Iterator[dict]:
        persona = trip_log.persona
        itinerary = trip_log.stages.itinerary
        run_metrics = trip_log.metrics

        yield self._event("stage_started", stage="dining", label=STAGE_LABELS["dining"])
        dining = self.dining_agent.run(persona, itinerary, self.run_config)
        yield from self._finalize_candidate_stage(run_metrics, "dining", self.dining_agent, dining)

        yield self._event("stage_started", stage="attractions", label=STAGE_LABELS["attractions"])
        attractions = self.attractions_agent.run(persona, itinerary, self.run_config)
        yield from self._finalize_candidate_stage(run_metrics, "attractions", self.attractions_agent, attractions)

        yield self._event("stage_started", stage="shopping", label=STAGE_LABELS["shopping"])
        shopping = self.shopping_agent.run(persona, itinerary, self.run_config)
        yield from self._finalize_candidate_stage(run_metrics, "shopping", self.shopping_agent, shopping)

        # In-trip guide
        yield self._event("stage_started", stage="in_trip_guide", label=STAGE_LABELS["in_trip_guide"])
        guide = self.in_trip_guide_agent.run(persona, trip_log.stages)
        trip_log.stages.in_trip_guide = guide
        stage_metrics = StageMetrics.from_calls("in_trip_guide", self.in_trip_guide_agent.last_call_metrics)
        run_metrics.add_stage(stage_metrics)
        self._save(trip_log)
        yield self._event(
            "stage_completed", stage="in_trip_guide", label=STAGE_LABELS["in_trip_guide"],
            data={
                "output": guide.model_dump(mode="json"),
                "metrics": stage_metrics.model_dump(),
                "totals": run_metrics.model_dump(),
            },
        )

        # Disruption & dynamic replanning bridge
        yield self._event("stage_started", stage="replanning", label=STAGE_LABELS["replanning"])
        replanning_calls: List[CallMetrics] = []

        self.itinerary_agent.stage_name = "replanning"  # tag these calls correctly; restored below
        disruption = self.itinerary_agent.generate_disruption(itinerary, trip_log.stages)
        replanning_calls.extend(self.itinerary_agent.last_call_metrics)

        replanning = self.itinerary_agent.replan(itinerary, disruption, trip_log.stages)
        replanning_calls.extend(self.itinerary_agent.last_call_metrics)
        self.itinerary_agent.stage_name = "itinerary"

        confirmation = self.user_simulator.confirm_replanning(replanning)
        replanning_calls.extend(self.user_simulator.last_call_metrics)
        self._log_hitl("replanning_confirmation", "replanning", replanning.change_summary, confirmation.status, confirmation.feedback)
        yield self._event("hitl_checkpoint", stage="replanning", label=STAGE_LABELS["replanning"], data=trip_log.hitl_log[-1].model_dump(mode="json"))

        replanning.confirmation = confirmation
        trip_log.disruption_event = disruption
        trip_log.replanning = replanning
        stage_metrics = StageMetrics.from_calls("replanning", replanning_calls)
        run_metrics.add_stage(stage_metrics)
        self._save(trip_log)
        yield self._event(
            "stage_completed", stage="replanning", label=STAGE_LABELS["replanning"],
            data={
                "output": replanning.model_dump(mode="json"),
                "metrics": stage_metrics.model_dump(),
                "totals": run_metrics.model_dump(),
            },
        )

        # Final review
        yield self._event("stage_started", stage="review", label=STAGE_LABELS["review"])
        review = self.user_simulator.final_review(trip_log)
        trip_log.stages.review = review
        stage_metrics = StageMetrics.from_calls("review", self.user_simulator.last_call_metrics)
        run_metrics.add_stage(stage_metrics)
        self._save(trip_log)
        yield self._event(
            "stage_completed", stage="review", label=STAGE_LABELS["review"],
            data={
                "output": review.model_dump(mode="json"),
                "metrics": stage_metrics.model_dump(),
                "totals": run_metrics.model_dump(),
            },
        )

        yield self._event(
            "run_completed",
            data={"trip_log": trip_log.model_dump(mode="json"), "totals": run_metrics.model_dump()},
        )

    def run(self, persona: Persona) -> TripLog:
        """CLI-friendly wrapper: drains run_streaming(), printing live
        per-stage token/cost/time as each step completes."""
        for event in self.run_streaming(persona):
            _print_cli_event(event)
        return self.trip_log

    def _save(self, trip_log: TripLog) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "trip_log.json"
        path.write_text(trip_log.model_dump_json(indent=2), encoding="utf-8")


def _print_cli_event(event: dict) -> None:
    t = event["type"]
    if t == "run_started":
        return  # run_demo.py already prints the persona summary before starting
    elif t == "stage_started":
        print(f"\n▶ {event['label']}（{event['stage']}）執行中...")
    elif t == "stage_completed":
        m = event["data"]["metrics"]
        search_note = f" · {m['total_web_search_requests']} 次搜尋" if m["total_web_search_requests"] else ""
        print(
            f"  ✓ 完成 — {m['total_duration_ms'] / 1000:.1f}s · "
            f"{m['total_input_tokens']}+{m['total_output_tokens']} tokens{search_note} · "
            f"${m['total_cost_usd']:.4f}"
        )
    elif t == "hitl_checkpoint":
        d = event["data"]
        print(f"  🧑 虛擬使用者：{d['decision_status']} — {d['reasoning']}")
    elif t == "run_completed":
        tot = event["data"]["totals"]
        print("\n=== 總計（可行性評估用，約略金額，非精確帳單）===")
        print(f"總時間：{tot['total_duration_ms'] / 1000:.1f}s")
        print(f"總 tokens：輸入 {tot['total_input_tokens']} + 輸出 {tot['total_output_tokens']}")
        if tot["total_web_search_requests"]:
            print(f"總搜尋次數：{tot['total_web_search_requests']}")
        print(f"總花費：${tot['total_cost_usd']:.4f}")
    elif t == "error":
        print(f"\n❌ 錯誤：{event['data']['message']}")
