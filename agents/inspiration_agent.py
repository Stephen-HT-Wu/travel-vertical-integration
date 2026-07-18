"""Layer 1 — real web search, destination/theme inspiration options."""
from typing import List, Optional, Tuple

from agents.base_agent import StageAgent
from persona import Persona
from schemas import InspirationChatTurn, InspirationOutput, RunConfig

RESEARCH_SYSTEM = """你是一位旅遊靈感研究員。使用者已經確定要從出發地前往某個目的地旅遊（見人物設定中的 destination_location）——你的任務不是幫他挑選要去哪個城市/地區，而是針對這個已確定的目的地，使用 web_search 工具搜尋真實、時效性高、來源可信的「在當地可以怎麼玩」主題與行程靈感（例如當地知名的景點組合、路線、體驗、季節限定活動）。

搜尋時請考慮這個人物設定適合的旅遊步調、預算感、體力負荷。至少搜尋 2-3 次不同關鍵字組合，確保有多個真實來源可以引用。"""

SYNTH_SYSTEM = """根據上一輪的真實搜尋結果，整理出 3-5 個「在目的地當地可以怎麼玩」的主題選項，輸出結構化 JSON。

每個選項的 source_url 與 source_title 必須是搜尋結果中真實存在的值，絕對不可捏造。citation_snippet 使用搜尋結果中的原文片段（簡短摘錄）。queries_used 記錄你在研究階段實際使用過的搜尋關鍵字。"""

CHAT_RESEARCH_SYSTEM = """你是一位旅遊靈感顧問，正在跟真實旅客對話。使用者已經確定要從出發地前往某個目的地旅遊（見訊息中的人物設定），你的任務不是幫他挑選要去哪裡，而是針對這個已確定的目的地，使用 web_search 工具搜尋真實、時效性高、來源可信的「在當地可以怎麼玩」主題與行程靈感，準備提供 3-5 個選項。"""

CHAT_SYNTH_SYSTEM = """根據上一輪的真實搜尋結果，整理出 3-5 個「在目的地當地可以怎麼玩」的主題選項，輸出結構化 JSON。source_url/source_title 必須是搜尋結果中的真實值，絕不可捏造。reply_message 用聊天口吻簡短介紹這些選項，邀請使用者挑一個喜歡的、或提出調整意見（例如想要更悠閒/更聚焦某個主題/預算更低）。"""

CHAT_REFINE_SYSTEM = """你是一位旅遊靈感顧問，正在跟真實旅客進行多輪對話。根據對話紀錄中已經搜尋到的真實選項與使用者最新的訊息，調整/篩選/補充 destination_options（通常維持 3-5 個），不需要再重新搜尋——除非使用者明確要求完全不同的地區或主題，這種情況下請在 reply_message 誠實說明「這需要重新搜尋，這一輪先根據目前的資訊調整」，並盡量在既有選項中找出最接近的替代方案。reply_message 用聊天口吻自然回覆使用者的訊息。"""


class InspirationAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("inspiration", model)

    def run(self, persona: Persona, run_config: RunConfig) -> InspirationOutput:
        user_content = (
            f"人物設定：{persona.summary_zh()}\n"
            f"目的地：{persona.destination_location}（已確定，不需要協助選擇目的地）\n\n"
            f"請搜尋 3-5 個在 {persona.destination_location} 當地可以怎麼玩的真實靈感主題，"
            "並說明每個選項為什麼適合這個人物設定。"
        )
        return self.run_with_search(
            RESEARCH_SYSTEM, SYNTH_SYSTEM, user_content, InspirationOutput, run_config
        )

    def chat(
        self, persona: Persona, history: List[dict], user_message: str, run_config: RunConfig
    ) -> Tuple[InspirationChatTurn, List[dict]]:
        """First turn (history empty) grounds in one real web search;
        later turns refine within that same conversation (see
        base_agent.start_search_chat's docstring for why)."""
        if not history:
            user_content = (
                f"人物設定：{persona.summary_zh()}\n"
                f"目的地：{persona.destination_location}（已確定，不需要協助選擇目的地）\n\n"
                f"{user_message or f'請給我一些在 {persona.destination_location} 當地怎麼玩的靈感。'}"
            )
            return self.start_search_chat(
                CHAT_RESEARCH_SYSTEM, CHAT_SYNTH_SYSTEM, user_content, InspirationChatTurn, run_config
            )
        return self.continue_chat(CHAT_REFINE_SYSTEM, history, user_message, InspirationChatTurn)
