"""Layer 3 (simulated) — flight / THSR / TRA style transportation candidates."""
from typing import List, Tuple

from agents.base_agent import (
    StageAgent,
    chat_candidate_system_prompts,
    mock_candidate_system_prompt,
    validate_candidate_turn,
)
from persona import Persona
from schemas import CandidateChatTurn, CandidateStageOutput, ItineraryOutput, summarize_itinerary

CATEGORY = "交通"
VENDOR_HINT = "航空公司、台灣高鐵、台鐵等交通業者的班次"
DEEP_LINK_QUERY_HINT = "這趟行程實際要去的目的地地名（例如「貓空」「九份」），會拿去查 Google 地圖大眾運輸路線，出發地會自動帶入使用者的 home_location，不用重複填"


class TransportationAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("transportation", model)

    def run(self, persona: Persona, itinerary: ItineraryOutput) -> CandidateStageOutput:
        system = mock_candidate_system_prompt(CATEGORY, VENDOR_HINT)
        user_content = (
            f"人物設定：{persona.summary_zh()}\n"
            f"出發地：{persona.home_location}\n\n"
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            "請提出往返目的地的交通候選方案（去程與回程，適合這個人數與行程天數的班次類型）。"
        )
        return self.run_mock(system, user_content, CandidateStageOutput)

    def chat(
        self, persona: Persona, history: List[dict], user_message: str, itinerary: ItineraryOutput
    ) -> Tuple[CandidateChatTurn, List[dict]]:
        start_system, refine_system = chat_candidate_system_prompts(CATEGORY, VENDOR_HINT, DEEP_LINK_QUERY_HINT)
        if not history:
            user_content = (
                f"人物設定：{persona.summary_zh()}\n出發地：{persona.home_location}\n\n"
                f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
                f"{user_message or '請提出交通候選方案。'}"
            )
            turn, new_history = self.start_chat(start_system, user_content, CandidateChatTurn)
        else:
            turn, new_history = self.continue_chat(refine_system, history, user_message, CandidateChatTurn)
        validate_candidate_turn(CATEGORY, turn)
        return turn, new_history
