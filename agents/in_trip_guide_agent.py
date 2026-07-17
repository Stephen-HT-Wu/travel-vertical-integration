"""Layer 4 — digital concierge in-trip guide, consolidating confirmed stages."""
from agents.base_agent import StageAgent
from persona import Persona
from schemas import InTripGuideOutput, StageResults, summarize_stage_results

SYSTEM = """你是一位「數位禮賓」行中導覽 agent。根據使用者人物設定與整趟已確認的行程/候選方案，為每一天產出實用的行中提醒。

每天的 tips 是幾條具體、可執行的提醒（交通銜接、排隊訣竅、衣著建議等）；emergency_info 是簡短的緊急聯絡資訊範本（例如當地報案電話、旅遊平安險提醒——用通用範本即可，不需查證真實電話）；local_phrases 是幾句對這個行程有幫助的當地常用短語。"""


class InTripGuideAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("in_trip_guide", model)

    def run(self, persona: Persona, stages: StageResults) -> InTripGuideOutput:
        user_content = (
            f"人物設定：{persona.summary_zh()}\n\n"
            f"已確認的行程與候選方案：\n{summarize_stage_results(stages)}\n\n"
            "請產出逐日的行中導覽內容。"
        )
        return self.run_mock(SYSTEM, user_content, InTripGuideOutput)
