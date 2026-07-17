"""Layer 3 (simulated) — KKday/Klook-style bookable activity candidates."""
from agents.base_agent import StageAgent, mock_candidate_system_prompt
from persona import Persona
from schemas import CandidateStageOutput, ItineraryOutput, summarize_itinerary

CATEGORY = "活動"
VENDOR_HINT = "KKday、Klook 風格的線上活動/體驗平台商品"


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
