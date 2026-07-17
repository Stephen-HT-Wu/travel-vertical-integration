"""Layer 2 — itinerary drafting, plus the dynamic-replanning bridge
(Layer 2 dynamic scheduling + Layer 5 exception handling)."""
from typing import Optional

from agents.base_agent import StageAgent
from persona import Persona
from schemas import (
    DisruptionEvent,
    InspirationOutput,
    ItineraryOutput,
    ReplanningOutput,
    StageResults,
    summarize_itinerary,
    summarize_stage_results,
)

RUN_SYSTEM = """你是一位行程規劃 agent。根據使用者的人物設定與選定的旅遊靈感，規劃一份具體的、分時段的行程。

行程的天數與步調要符合 trip_length_type（half_day/one_day/multi_day，multi_day 時要對應 days 天數）與 party_size（人數多寡會影響行程的彈性與體力安排）。每一天用數個 time_block（例如 "09:00-10:30"）劃分，每個時段填入 theme（這個時段做什麼）、location_hint（大概地點）、notes（實用提醒，例如交通銜接、備案）。based_on_inspiration_ids 填入你參考的靈感選項 id。"""

REVISE_SYSTEM = RUN_SYSTEM + """

使用者對前一版行程提出了修改意見（見輸入中的 revision_notes），請據此調整行程，其餘部分盡量維持不變。"""

DISRUPTION_SYSTEM = """你是一位情境模擬 agent。根據已確認的行程與各階段候選方案，挑選「一個」情境合理的行中突發狀況（例如某天下午安排了戶外景點，就模擬臨時降雨；或某個景點模擬臨時公休/客滿）。

這是為了展示 agent 動態應變能力而刻意模擬的情境，不是真實天氣或營業資料，請將 is_simulated 設為 true。只挑一個最能影響當天行程、最適合展示「動態重新排程」能力的狀況。"""

REPLAN_SYSTEM = """你是一位行程規劃 agent，現在需要因應一個行中突發狀況（disruption_event）重新安排受影響那一天的行程。

只調整受影響當天（day_number 與 disruption_event 相同）的行程，其餘時段盡量維持原樣，只在必要時微調銜接。revised_day 是調整後的完整當日行程。change_summary 用人類看得懂的話說明「改了什麼、為什麼」。concierge_notification 模擬一則「數位禮賓」會發給旅客的簡短通知訊息（例如：偵測到狀況、已經怎麼調整、需不需要旅客確認）。"""


class ItineraryAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("itinerary", model)

    def run(
        self,
        persona: Persona,
        inspiration: InspirationOutput,
        revision_notes: Optional[str] = None,
    ) -> ItineraryOutput:
        inspiration_summary = "\n".join(
            f"- [{opt.id}] {opt.name}：{opt.summary}" for opt in inspiration.destination_options
        )
        user_content = (
            f"人物設定：{persona.summary_zh()}\n"
            f"trip_length_type={persona.trip_length_type}, days={persona.days}, "
            f"party_size={persona.party_size}\n\n"
            f"可參考的靈感選項：\n{inspiration_summary}\n\n"
            "請選擇最適合的靈感選項並規劃出完整行程。"
        )
        system = RUN_SYSTEM
        if revision_notes:
            user_content += f"\n\n使用者的修改意見（revision_notes）：{revision_notes}"
            system = REVISE_SYSTEM
        return self.run_mock(system, user_content, ItineraryOutput)

    def generate_disruption(
        self, itinerary: ItineraryOutput, confirmed_stages: StageResults
    ) -> DisruptionEvent:
        user_content = (
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            f"已確認的候選方案：\n{summarize_stage_results(confirmed_stages)}\n\n"
            "請挑選一個情境合理的突發狀況。"
        )
        return self.run_mock(DISRUPTION_SYSTEM, user_content, DisruptionEvent)

    def replan(
        self,
        itinerary: ItineraryOutput,
        disruption: DisruptionEvent,
        confirmed_stages: StageResults,
    ) -> ReplanningOutput:
        user_content = (
            f"已確認行程：\n{summarize_itinerary(itinerary)}\n\n"
            f"已確認的候選方案：\n{summarize_stage_results(confirmed_stages)}\n\n"
            f"突發狀況：第 {disruption.day_number} 天 {disruption.affected_time_block}，"
            f"{disruption.disruption_type}：{disruption.description}\n\n"
            "請重新安排受影響當天的行程。"
        )
        return self.run_mock(REPLAN_SYSTEM, user_content, ReplanningOutput)
