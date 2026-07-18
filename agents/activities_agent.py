"""Layer 3 (simulated) — KKday/Klook-style bookable activity candidates."""
from typing import List, Tuple

from agents.base_agent import (
    StageAgent,
    chat_candidate_system_prompts,
    mock_candidate_system_prompt,
    validate_candidate_turn,
)
from persona import Persona
from schemas import CandidateChatTurn, CandidateStageOutput, ItineraryOutput, summarize_itinerary

CATEGORY = "活動"
VENDOR_HINT = "KKday、Klook 風格的線上活動/體驗平台商品"
DEEP_LINK_QUERY_HINT = (
    "適合拿去 KKday 搜尋的簡短關鍵字，反映這個方案的主題（例如「貓空纜車」「貓空 茶園體驗」），"
    "會拿去查 KKday 的商品搜尋結果，同一輪的 3 個方案應該各自對應自己的活動關鍵字"
)


class ActivitiesAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("activities", model)

    def run(self, persona: Persona, itinerary: ItineraryOutput) -> CandidateStageOutput:
        system = mock_candidate_system_prompt(CATEGORY, VENDOR_HINT)
        user_content = (
            f"人物設定：{persona.summary_zh()}\n\n"
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            "請提出可預訂的活動/體驗候選方案，適合這個人物設定與行程節奏。"
        )
        return self.run_mock(system, user_content, CandidateStageOutput)

    def chat(
        self, persona: Persona, history: List[dict], user_message: str, itinerary: ItineraryOutput
    ) -> Tuple[CandidateChatTurn, List[dict]]:
        start_system, refine_system = chat_candidate_system_prompts(CATEGORY, VENDOR_HINT, DEEP_LINK_QUERY_HINT)
        if not history:
            user_content = (
                f"人物設定：{persona.summary_zh()}\n\n"
                f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
                f"{user_message or '請提出活動候選方案。'}"
            )
            turn, new_history = self.start_chat(start_system, user_content, CandidateChatTurn)
        else:
            turn, new_history = self.continue_chat(refine_system, history, user_message, CandidateChatTurn)
        validate_candidate_turn(CATEGORY, turn)
        return turn, new_history
