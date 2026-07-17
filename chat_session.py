"""Real-user multi-turn chat session for the front three layers (inspiration,
itinerary, transaction candidates), ending each transaction candidate in a
simulated "redirect out to book -> order confirmed" checkout, then handing
off to the existing auto-pipeline (orchestrator.run_tail_streaming) for the
content-recommendation tail (dining/attractions/shopping/guide/replanning/
review), which stays UserSimulatorAgent-driven exactly as before.

Unlike the old fully-automatic flow, confirmations here come from an actual
user action (a button click reaching confirm_inspiration/confirm_itinerary/
confirm_candidate) — never inferred from free text and never simulated.
"""
import random
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from agents.accommodation_agent import AccommodationAgent
from agents.activities_agent import ActivitiesAgent
from agents.inspiration_agent import InspirationAgent
from agents.itinerary_agent import ItineraryAgent
from agents.transportation_agent import TransportationAgent
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
    RunConfig,
    SimulatedOrder,
    StageMetrics,
    TripLog,
    summarize_itinerary,
)

CANDIDATE_PHASE_NEXT = {"transportation": "accommodation", "accommodation": "activities", "activities": "tail"}
INSPIRATION_TO_TRANSPORT_PHASES = ["inspiration", "itinerary", "transportation", "accommodation", "activities"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_order_id() -> str:
    return "SIM-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


class ChatSession:
    def __init__(self, run_id: str, persona: Persona, run_config: RunConfig, output_dir: Path):
        self.run_id = run_id
        self.persona = persona
        self.run_config = run_config
        self.output_dir = output_dir
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
            turn, new_history = self.inspiration_agent.chat(self.persona, history, user_message, self.run_config)
            agent = self.inspiration_agent
        elif phase == "itinerary":
            turn, new_history = self.itinerary_agent.chat(self.persona, history, user_message, self.selected_inspiration)
            agent = self.itinerary_agent
        elif phase == "transportation":
            turn, new_history = self.transportation_agent.chat(
                self.persona, history, user_message, self.trip_log.stages.itinerary
            )
            agent = self.transportation_agent
        elif phase == "accommodation":
            turn, new_history = self.accommodation_agent.chat(
                self.persona, history, user_message, self.trip_log.stages.itinerary, self.trip_log.stages.transportation
            )
            agent = self.accommodation_agent
        elif phase == "activities":
            turn, new_history = self.activities_agent.chat(
                self.persona, history, user_message, self.trip_log.stages.itinerary
            )
            agent = self.activities_agent
        else:
            raise ValueError(f"send_message is not valid in phase '{phase}'")

        self.history[phase] = new_history
        self.last_turn[phase] = turn
        stage_metrics = StageMetrics.from_calls(phase, agent.last_call_metrics)
        self.trip_log.metrics.add_stage(stage_metrics)
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
                status=status, final_candidate_id=candidate_id, feedback="使用者於對話中選定並前往訂購"
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

    def place_order(self, stage_name: str) -> dict:
        """Purely local mock — no LLM call, no real booking/payment system
        contacted. order_id is a random token, never a real confirmation
        number. This is the explicit 'redirect out to book -> order
        confirmed' step the user asked to see, kept honest about being fake."""
        stage_output: Optional[CandidateStageOutput] = getattr(self.trip_log.stages, stage_name, None)
        if stage_output is None or stage_output.confirmation is None:
            raise ValueError(f"{stage_name} 尚未確認候選方案，無法下單")

        candidate = next(c for c in stage_output.candidates if c.id == stage_output.confirmation.final_candidate_id)
        order = SimulatedOrder(
            order_id=_generate_order_id(), stage=stage_name, candidate_id=candidate.id,
            candidate_name=candidate.name, price_range=candidate.price_range, confirmed_at=_now(),
        )
        stage_output.order = order
        setattr(self.trip_log.stages, stage_name, stage_output)

        next_phase = CANDIDATE_PHASE_NEXT[stage_name]
        self.phase = next_phase
        self._save()
        return {
            "type": "order_confirmed", "phase": stage_name,
            "order": order.model_dump(mode="json"), "next_phase": next_phase,
        }

    # -- tail handoff ---------------------------------------------------
    def run_tail(self) -> Iterator[dict]:
        orchestrator = TripOrchestrator(self.run_config, self.output_dir)
        yield from orchestrator.run_tail_streaming(self.trip_log)
        self.phase = "done"
