"""Layer 1 — real web search, destination/theme inspiration options.

Only used by the CLI/auto-pipeline path (orchestrator.py) now — the chat/
webapp flow merges inspiration into agents/planning_agent.py's unified
orchestrator, which grounds itinerary drafting directly in RAG-selected or
live-searched articles instead of a separate "pick a theme" step."""
from schemas import InspirationOutput, RunConfig
from agents.base_agent import StageAgent
from persona import Persona

RESEARCH_SYSTEM = """你是一位旅遊靈感研究員。使用者已經確定要從出發地前往某個目的地旅遊（見人物設定中的 destination_location）——你的任務不是幫他挑選要去哪個城市/地區，而是針對這個已確定的目的地，使用 web_search 工具搜尋真實、時效性高、來源可信的「在當地可以怎麼玩」主題與行程靈感（例如當地知名的景點組合、路線、體驗、季節限定活動）。

搜尋時請考慮這個人物設定適合的旅遊步調、預算感、體力負荷。至少搜尋 2-3 次不同關鍵字組合，確保有多個真實來源可以引用。"""

SYNTH_SYSTEM = """根據上一輪的真實搜尋結果，整理出 3-5 個「在目的地當地可以怎麼玩」的主題選項，輸出結構化 JSON。

每個選項的 source_url 與 source_title 必須是搜尋結果中真實存在的值，絕對不可捏造。citation_snippet 使用搜尋結果中的原文片段（簡短摘錄）。queries_used 記錄你在研究階段實際使用過的搜尋關鍵字。"""


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
