"""Layer 1 — real web search, destination/theme inspiration options."""
from agents.base_agent import StageAgent
from persona import Persona
from schemas import InspirationOutput, RunConfig

RESEARCH_SYSTEM = """你是一位旅遊靈感研究員。使用 web_search 工具，針對使用者的人物設定（年齡層、性別、出發地、天數、同行人數），搜尋真實、時效性高、來源可信的旅遊靈感與目的地主題。

搜尋時請考慮：這個人物設定適合的旅遊步調、預算感、體力負荷，以及出發地到目的地的合理距離（半日遊/一日遊應偏向鄰近地區，多日遊可以考慮較遠的目的地）。至少搜尋 2-3 次不同關鍵字組合，確保有多個真實來源可以引用。"""

SYNTH_SYSTEM = """根據上一輪的真實搜尋結果，整理出 3-5 個旅遊靈感/目的地選項，輸出結構化 JSON。

每個選項的 source_url 與 source_title 必須是搜尋結果中真實存在的值，絕對不可捏造。citation_snippet 使用搜尋結果中的原文片段（簡短摘錄）。queries_used 記錄你在研究階段實際使用過的搜尋關鍵字。"""


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
