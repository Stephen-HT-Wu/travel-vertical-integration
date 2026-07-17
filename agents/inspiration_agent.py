"""Layer 1 — real web search, destination/theme inspiration options."""
from typing import List, Optional, Tuple

from agents.base_agent import StageAgent
from persona import Persona
from schemas import InspirationChatTurn, InspirationOutput, RunConfig

RESEARCH_SYSTEM = """你是一位旅遊靈感研究員。使用 web_search 工具，針對使用者的人物設定（年齡層、性別、出發地、天數、同行人數），搜尋真實、時效性高、來源可信的旅遊靈感與目的地主題。

搜尋時請考慮：這個人物設定適合的旅遊步調、預算感、體力負荷，以及出發地到目的地的合理距離（半日遊/一日遊應偏向鄰近地區，多日遊可以考慮較遠的目的地）。至少搜尋 2-3 次不同關鍵字組合，確保有多個真實來源可以引用。"""

SYNTH_SYSTEM = """根據上一輪的真實搜尋結果，整理出 3-5 個旅遊靈感/目的地選項，輸出結構化 JSON。

每個選項的 source_url 與 source_title 必須是搜尋結果中真實存在的值，絕對不可捏造。citation_snippet 使用搜尋結果中的原文片段（簡短摘錄）。queries_used 記錄你在研究階段實際使用過的搜尋關鍵字。"""

CHAT_RESEARCH_SYSTEM = """你是一位旅遊靈感顧問，正在跟真實旅客對話。使用 web_search 工具，根據使用者的人物設定與訊息，搜尋真實、時效性高、來源可信的旅遊靈感與目的地主題，準備提供 3-5 個選項。"""

CHAT_SYNTH_SYSTEM = """根據上一輪的真實搜尋結果，整理出 3-5 個旅遊靈感選項，輸出結構化 JSON。source_url/source_title 必須是搜尋結果中的真實值，絕不可捏造。reply_message 用聊天口吻簡短介紹這些選項，邀請使用者挑一個喜歡的、或提出調整意見（例如想要更悠閒/更遠/預算更低）。"""

CHAT_REFINE_SYSTEM = """你是一位旅遊靈感顧問，正在跟真實旅客進行多輪對話。根據對話紀錄中已經搜尋到的真實選項與使用者最新的訊息，調整/篩選/補充 destination_options（通常維持 3-5 個），不需要再重新搜尋——除非使用者明確要求完全不同的地區或主題，這種情況下請在 reply_message 誠實說明「這需要重新搜尋，這一輪先根據目前的資訊調整」，並盡量在既有選項中找出最接近的替代方案。reply_message 用聊天口吻自然回覆使用者的訊息。"""


class InspirationAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("inspiration", model)

    def run(self, persona: Persona, run_config: RunConfig) -> InspirationOutput:
        user_content = (
            f"人物設定：{persona.summary_zh()}\n\n"
            "請搜尋 3-5 個符合這個人物設定的真實旅遊靈感或目的地主題，"
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
            user_content = f"人物設定：{persona.summary_zh()}\n\n{user_message or '請給我一些旅遊靈感。'}"
            return self.start_search_chat(
                CHAT_RESEARCH_SYSTEM, CHAT_SYNTH_SYSTEM, user_content, InspirationChatTurn, run_config
            )
        return self.continue_chat(CHAT_REFINE_SYSTEM, history, user_message, InspirationChatTurn)
