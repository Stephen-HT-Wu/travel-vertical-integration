"""Layer 7 (real search) — shopping recommendations grounded in trusted content."""
from agents.base_agent import StageAgent, search_candidate_prompts
from persona import Persona
from schemas import CandidateStageOutput, ItineraryOutput, RunConfig, summarize_itinerary

CATEGORY = "購物"
VENDOR_HINT = "符合行程地點的真實商圈、市集、特色店家"


class ShoppingAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("shopping", model)

    def run(
        self, persona: Persona, itinerary: ItineraryOutput, run_config: RunConfig
    ) -> CandidateStageOutput:
        research_system, synth_system = search_candidate_prompts(CATEGORY, VENDOR_HINT)
        user_content = (
            f"人物設定：{persona.summary_zh()}\n\n"
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            "請搜尋符合行程地點的真實購物選項。"
        )
        return self.run_with_search(
            research_system, synth_system, user_content, CandidateStageOutput, run_config
        )
