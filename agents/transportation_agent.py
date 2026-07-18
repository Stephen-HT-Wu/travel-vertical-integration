"""Layer 3 (simulated) — long-distance point-to-point transportation candidates
between persona.home_location and persona.destination_location (flight / THSR
/ TRA / long-distance bus style)."""
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
VENDOR_HINT = "航空公司、台灣高鐵、台鐵、國道客運等長途交通業者的班次"
DEEP_LINK_QUERY_HINT = "通常填目的地地名即可，實際 deep link 的目的地固定使用使用者已設定的到達地，這欄只是備用參考、不影響實際連結"
DURATION_SCHEDULE_HINT = (
    "這是交通類別，每個候選方案務必在 description 中清楚說明「經費」「預估時間（車程/航程）」"
    "「班次/頻率（如果有的話，例如『約每小時 1-2 班』『每日 3 班對開』；沒有固定班次的類型如包車可以說明『可彈性預約』）」"
    "三項資訊；並把預估時間額外填入 duration 欄位（例如「約 1 小時 10 分」）、班次資訊額外填入 schedule_note 欄位"
    "（例如「約每小時 1-2 班」，沒有固定班次就填「可彈性預約」，不要留空字串）。"
)


class TransportationAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("transportation", model)

    def run(self, persona: Persona, itinerary: ItineraryOutput) -> CandidateStageOutput:
        system = mock_candidate_system_prompt(CATEGORY, VENDOR_HINT) + "\n\n" + DURATION_SCHEDULE_HINT
        user_content = (
            f"人物設定：{persona.summary_zh()}\n"
            f"出發地：{persona.home_location}　到達地：{persona.destination_location}\n\n"
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            f"請提出從 {persona.home_location} 到 {persona.destination_location} 往返的長途交通候選方案"
            "（去程與回程，適合這個人數與行程天數的班次類型）。"
        )
        return self.run_mock(system, user_content, CandidateStageOutput)

    def chat(
        self, persona: Persona, history: List[dict], user_message: str, itinerary: ItineraryOutput
    ) -> Tuple[CandidateChatTurn, List[dict]]:
        start_system, refine_system = chat_candidate_system_prompts(
            CATEGORY, VENDOR_HINT, DEEP_LINK_QUERY_HINT, DURATION_SCHEDULE_HINT
        )
        if not history:
            user_content = (
                f"人物設定：{persona.summary_zh()}\n"
                f"出發地：{persona.home_location}　到達地：{persona.destination_location}\n\n"
                f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
                f"{user_message or f'請提出從 {persona.home_location} 到 {persona.destination_location} 的長途交通候選方案。'}"
            )
            turn, new_history = self.start_chat(start_system, user_content, CandidateChatTurn)
        else:
            turn, new_history = self.continue_chat(refine_system, history, user_message, CandidateChatTurn)
        validate_candidate_turn(CATEGORY, turn)
        return turn, new_history
