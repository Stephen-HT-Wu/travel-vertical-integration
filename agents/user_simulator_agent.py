"""Independent virtual-user agent. Plays the human at every HITL checkpoint
and produces the final review — never folded into the orchestrator, so its
decisions are a genuine "second opinion" call, the same way a real human's
would be in a non-demo build."""
from agents.base_agent import StageAgent
from persona import Persona
from schemas import (
    CandidateConfirmation,
    CandidateStageOutput,
    ItineraryConfirmation,
    ItineraryOutput,
    ReplanningConfirmation,
    ReplanningOutput,
    ReviewOutput,
    TripLog,
    summarize_itinerary,
    summarize_stage_results,
)

PERSONA_ROLE_PREFIX = """你正在扮演一位真實旅客，人物設定為：{persona_summary}。

在每個決策點，依照這個人物設定合理地做決定：通常會確認合理的提案，但遇到明顯跟人物設定不合的狀況（超出合理預算感、對這個年齡層不適合、對這個人數不實際）時，就要提出意見或更換選項。要果斷、有自己的判斷，不要對每個提案都無條件說好。"""

ITINERARY_INSTRUCTION = """請針對這份行程草稿做出決定：確認（confirmed）或要求修改（revised，並在 revision_notes 寫出具體修改方向）。feedback 用旅客口吻簡短說明你的想法。"""

CANDIDATE_INSTRUCTION = """agent 針對「{stage_name}」提出了 3 個候選方案，並推薦了其中一個。請你做出決定：確認 agent 的推薦（confirmed，final_candidate_id 等於推薦的方案）、換成另一個候選方案（swapped，final_candidate_id 填你選的那個）、或都不要（declined）。feedback 用旅客口吻簡短說明理由。"""

REPLANNING_INSTRUCTION = """行程中發生了一個突發狀況，agent 提出了重新排程的方案。請你決定是否接受這個調整：confirmed 或 declined。feedback 用旅客口吻簡短說明理由。"""

FINAL_REVIEW_INSTRUCTION = """整趟行程（含突發狀況與重新排程的處理）都已經跑完了。請你以這個人物設定的口吻，給出整體評價：overall_rating（1-5 分）、每個類別（交通、住宿、餐飲、景點、活動、購物、行中導覽、異常處理）各自的 category_ratings、review_text（幾句真實的評論語氣文字）、would_recommend、persona_alignment_notes（這趟行程規劃是否真的貼合你的人物設定）、以及 share_caption（一段簡短、適合放在社群貼文分享的文字，第一人稱、有記憶點）。"""


class UserSimulatorAgent(StageAgent):
    def __init__(self, persona: Persona, model: str = "claude-opus-4-8"):
        super().__init__("user_simulator", model)
        self.persona = persona
        self._role_prefix = PERSONA_ROLE_PREFIX.format(persona_summary=persona.summary_zh())

    def confirm_itinerary(self, itinerary: ItineraryOutput) -> ItineraryConfirmation:
        system = f"{self._role_prefix}\n\n{ITINERARY_INSTRUCTION}"
        user_content = f"行程草稿：\n{summarize_itinerary(itinerary)}"
        return self.run_mock(system, user_content, ItineraryConfirmation)

    def confirm_candidate(
        self, stage_name: str, stage_output: CandidateStageOutput
    ) -> CandidateConfirmation:
        system = f"{self._role_prefix}\n\n{CANDIDATE_INSTRUCTION.format(stage_name=stage_name)}"
        candidates_summary = "\n".join(
            f"- [{c.id}] {c.name}（{c.vendor}，{c.price_range}）{'← agent 推薦' if c.id == stage_output.agent_selected_candidate_id else ''}"
            for c in stage_output.candidates
        )
        user_content = (
            f"候選方案：\n{candidates_summary}\n\n"
            f"agent 推薦理由：{stage_output.agent_selection_rationale}"
        )
        return self.run_mock(system, user_content, CandidateConfirmation)

    def confirm_replanning(self, replanning: ReplanningOutput) -> ReplanningConfirmation:
        system = f"{self._role_prefix}\n\n{REPLANNING_INSTRUCTION}"
        user_content = (
            f"突發狀況：{replanning.trigger.description}\n\n"
            f"調整摘要：{replanning.change_summary}\n\n"
            f"數位禮賓通知：{replanning.concierge_notification}"
        )
        return self.run_mock(system, user_content, ReplanningConfirmation)

    def final_review(self, trip_log: TripLog) -> ReviewOutput:
        system = f"{self._role_prefix}\n\n{FINAL_REVIEW_INSTRUCTION}"
        user_content = f"完整行程摘要：\n{summarize_stage_results(trip_log.stages)}"
        if trip_log.replanning:
            user_content += f"\n\n行中突發狀況與應對：{trip_log.replanning.change_summary}"
        return self.run_mock(system, user_content, ReviewOutput)
