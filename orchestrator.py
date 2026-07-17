"""Sequences every stage, wires the HITL checkpoints to UserSimulatorAgent,
and owns the hard boundary: no booking/payment client is ever constructed
here, so there is nothing for any stage — or any adversarial prompt — to
call for a real transaction."""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.accommodation_agent import AccommodationAgent
from agents.activities_agent import ActivitiesAgent
from agents.attractions_agent import AttractionsAgent
from agents.dining_agent import DiningAgent
from agents.in_trip_guide_agent import InTripGuideAgent
from agents.inspiration_agent import InspirationAgent
from agents.itinerary_agent import ItineraryAgent
from agents.shopping_agent import ShoppingAgent
from agents.transportation_agent import TransportationAgent
from agents.user_simulator_agent import UserSimulatorAgent
from persona import Persona
from schemas import (
    CandidateStageOutput,
    HITLLogEntry,
    ItineraryOutput,
    ReplanningOutput,
    RunConfig,
    TripLog,
    summarize_itinerary,
)


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
        self.user_simulator: Optional[UserSimulatorAgent] = None  # built once persona is known

    def run(self, persona: Persona) -> TripLog:
        trip_log = TripLog(
            run_id=str(uuid.uuid4()),
            created_at=_now(),
            persona=persona,
            run_config=self.run_config,
        )
        self.user_simulator = UserSimulatorAgent(persona, self.run_config.model)

        print(f"[1/6] 靈感探索（真實網路搜尋）...")
        trip_log.stages.inspiration = self.inspiration_agent.run(persona, self.run_config)
        self._save(trip_log)

        print(f"[2/6] 行程規劃 + 使用者確認...")
        itinerary = self.itinerary_agent.run(persona, trip_log.stages.inspiration)
        confirmation = self._confirm_itinerary(trip_log, itinerary)
        if confirmation.status == "revised":
            print("      使用者要求修改，重新規劃一次...")
            itinerary = self.itinerary_agent.run(
                persona, trip_log.stages.inspiration, revision_notes=confirmation.revision_notes
            )
            confirmation = self._confirm_itinerary(trip_log, itinerary)
        itinerary.confirmation = confirmation
        trip_log.stages.itinerary = itinerary
        self._save(trip_log)

        print(f"[3/6] 下游候選方案（交通/住宿/餐飲/景點/活動/購物）...")
        transportation = self.transportation_agent.run(persona, itinerary)
        transportation.confirmation = self._confirm_candidate(trip_log, "transportation", transportation)
        trip_log.stages.transportation = transportation
        self._save(trip_log)

        accommodation = self.accommodation_agent.run(persona, itinerary, transportation)
        accommodation.confirmation = self._confirm_candidate(trip_log, "accommodation", accommodation)
        trip_log.stages.accommodation = accommodation
        self._save(trip_log)

        dining = self.dining_agent.run(persona, itinerary, self.run_config)
        dining.confirmation = self._confirm_candidate(trip_log, "dining", dining)
        trip_log.stages.dining = dining
        self._save(trip_log)

        attractions = self.attractions_agent.run(persona, itinerary, self.run_config)
        attractions.confirmation = self._confirm_candidate(trip_log, "attractions", attractions)
        trip_log.stages.attractions = attractions
        self._save(trip_log)

        activities = self.activities_agent.run(persona, itinerary)
        activities.confirmation = self._confirm_candidate(trip_log, "activities", activities)
        trip_log.stages.activities = activities
        self._save(trip_log)

        shopping = self.shopping_agent.run(persona, itinerary, self.run_config)
        shopping.confirmation = self._confirm_candidate(trip_log, "shopping", shopping)
        trip_log.stages.shopping = shopping
        self._save(trip_log)

        print(f"[4/6] 行中導覽...")
        trip_log.stages.in_trip_guide = self.in_trip_guide_agent.run(persona, trip_log.stages)
        self._save(trip_log)

        print(f"[5/6] 模擬突發狀況 + 動態重新排程...")
        disruption = self.itinerary_agent.generate_disruption(itinerary, trip_log.stages)
        replanning = self.itinerary_agent.replan(itinerary, disruption, trip_log.stages)
        replanning.confirmation = self._confirm_replanning(trip_log, replanning)
        trip_log.disruption_event = disruption
        trip_log.replanning = replanning
        self._save(trip_log)

        print(f"[6/6] 最終評價與分享...")
        trip_log.stages.review = self.user_simulator.final_review(trip_log)
        self._save(trip_log)

        return trip_log

    def _confirm_itinerary(self, trip_log: TripLog, itinerary: ItineraryOutput):
        confirmation = self.user_simulator.confirm_itinerary(itinerary)
        trip_log.hitl_log.append(
            HITLLogEntry(
                stage="itinerary",
                checkpoint_type="itinerary_confirmation",
                presented_summary=summarize_itinerary(itinerary),
                decision_status=confirmation.status,
                user_simulator_reasoning=confirmation.feedback,
                timestamp=_now(),
            )
        )
        return confirmation

    def _confirm_candidate(
        self, trip_log: TripLog, stage_name: str, stage_output: CandidateStageOutput
    ):
        confirmation = self.user_simulator.confirm_candidate(stage_name, stage_output)
        trip_log.hitl_log.append(
            HITLLogEntry(
                stage=stage_name,
                checkpoint_type="candidate_confirmation",
                presented_summary=stage_output.agent_selection_rationale,
                decision_status=confirmation.status,
                user_simulator_reasoning=confirmation.feedback,
                timestamp=_now(),
            )
        )
        return confirmation

    def _confirm_replanning(self, trip_log: TripLog, replanning: ReplanningOutput):
        confirmation = self.user_simulator.confirm_replanning(replanning)
        trip_log.hitl_log.append(
            HITLLogEntry(
                stage="replanning",
                checkpoint_type="replanning_confirmation",
                presented_summary=replanning.change_summary,
                decision_status=confirmation.status,
                user_simulator_reasoning=confirmation.feedback,
                timestamp=_now(),
            )
        )
        return confirmation

    def _save(self, trip_log: TripLog) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "trip_log.json"
        path.write_text(trip_log.model_dump_json(indent=2), encoding="utf-8")
