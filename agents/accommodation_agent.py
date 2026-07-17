"""Layer 3 (simulated) — lodging candidates."""
from agents.base_agent import StageAgent, mock_candidate_system_prompt
from persona import Persona
from schemas import CandidateStageOutput, ItineraryOutput, summarize_itinerary

CATEGORY = "住宿"
VENDOR_HINT = "飯店、民宿、青年旅館等住宿業者"


class AccommodationAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("accommodation", model)

    def run(
        self,
        persona: Persona,
        itinerary: ItineraryOutput,
        transportation: CandidateStageOutput,
    ) -> CandidateStageOutput:
        system = mock_candidate_system_prompt(CATEGORY, VENDOR_HINT)
        chosen_transport_id = (
            transportation.confirmation.final_candidate_id
            if transportation.confirmation
            else transportation.agent_selected_candidate_id
        )
        chosen_transport = next(
            (c for c in transportation.candidates if c.id == chosen_transport_id), None
        )
        transport_note = f"已選定交通方案：{chosen_transport.name}" if chosen_transport else ""
        user_content = (
            f"人物設定：{persona.summary_zh()}\n"
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n"
            f"{transport_note}\n\n"
            f"請依人數（{persona.party_size} 人）與行程地點，提出住宿候選方案"
            f"（若天數為 half_day/one_day 可視情況判斷是否真的需要住宿，若不需要仍請提出候選但在 "
            f"agent_selection_rationale 說明可能不需入住）。"
        )
        return self.run_mock(system, user_content, CandidateStageOutput)
