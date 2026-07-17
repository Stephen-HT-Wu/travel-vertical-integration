"""Layer 7 (real search) — attraction recommendations grounded in trusted content."""
from agents.base_agent import StageAgent, search_candidate_prompts
from persona import Persona
from schemas import CandidateStageOutput, ItineraryOutput, RunConfig, summarize_itinerary

CATEGORY = "景點"
VENDOR_HINT = "符合行程地點與時段的真實景點、展覽、地標"


class AttractionsAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("attractions", model)

    def run(
        self, persona: Persona, itinerary: ItineraryOutput, run_config: RunConfig
    ) -> CandidateStageOutput:
        research_system, synth_system = search_candidate_prompts(CATEGORY, VENDOR_HINT)
        user_content = (
            f"人物設定：{persona.summary_zh()}\n\n"
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            "請搜尋符合行程地點與時段的真實景點選項。"
        )
        return self.run_with_search(
            research_system, synth_system, user_content, CandidateStageOutput, run_config
        )
