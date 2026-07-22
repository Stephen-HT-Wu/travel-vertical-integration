"""Unified planning orchestrator for the chat/webapp flow — replaces the old
two-phase "pick an inspiration theme, then separately draft an itinerary"
dance with one flow: understand the user's style preference for the trip
(asking up to a configurable number of clarifying rounds if it isn't clear
yet), then ground the itinerary directly in either a locally-cached RAG
title-selection + web_fetch (fast path) or a live web_search (fallback path
when RAG is disabled or has no good match).

This is chat-only. The CLI/auto-pipeline path (orchestrator.py) still uses
InspirationAgent.run()/ItineraryAgent.run() unchanged — this class doesn't
touch or replace those."""
from typing import List, Optional, Tuple

from agents.base_agent import StageAgent
from llm_client import call_structured
from persona import Persona
from schemas import CallMetrics, PlanningChatTurn, PlanningDecision, PlanningSynthesis, RagSelection, RunConfig
from site_index import SiteIndex

DECIDE_SYSTEM = """你是一位行程規劃顧問，正在跟真實旅客對話。使用者已經確定要從出發地前往某個目的地旅遊（人物設定裡的 destination_location 已經定案，不需要協助選擇目的地，也不需要再次確認目的地）。

你的任務是先了解這趟行程的「風格偏好」，才能規劃出真正貼合需求的行程——例如步調要悠閒還是緊湊、主題偏好（美食／自然／文青／秘境／親子……）、預算感、有沒有特別想排或想避開的活動類型。

如果使用者到目前為止的訊息已經透露出足夠的風格線索，就把 needs_clarification 設為 false，並在 style_summary 用一兩句話具體整理出你理解的風格偏好（這段之後會直接拿去檢索文章、規劃行程，要具體可用，不要寫「使用者想要不錯的行程」這種空話）。

如果訊息是空的、或內容還不足以判斷風格，就把 needs_clarification 設為 true，reply_message 用聊天口吻自然地問一個具體的風格偏好問題，一輪只問一個重點，不要一次問一大堆。needs_clarification 為 true 時 style_summary 留空字串即可。"""

RAG_SELECT_SYSTEM = """你是一位內容檢索助理。你會拿到一份真實文章標題列表（每行格式為「- [網址] 標題」），任務是從中挑出最符合這趟行程人物設定、目的地與風格偏好的文章。

只能挑選列表中真實出現過的網址，絕對不可以自己編網址或修改網址。如果列表裡沒有任何一篇跟目的地或風格真的相關（例如列表主題完全是別的城市、別的主題），selected_urls 就回傳空陣列，並在 selection_rationale 誠實說明原因，不要勉強湊數。"""

FETCH_SYSTEM = """你是一位行程規劃顧問，正在跟真實旅客對話。使用 web_fetch 工具抓取使用者訊息中列出的文章網址，取得規劃行程所需的實際內容（景點順序、交通銜接、營業時間、當地建議路線、體驗心得）。"""

FETCH_SYNTH_SYSTEM = """根據上一輪抓取到的真實文章內容，並參考使用者的行程風格偏好，規劃一份具體的、分時段的行程。

天數與步調要符合 trip_length_type（half_day/one_day/multi_day，multi_day 對應 days 天數）與 party_size。每一天用數個 time_block 劃分，每個時段填入 theme、location_hint、notes，內容應該整合上述文章中的實際資訊。sources 欄位可以留空（後續會用抓取結果自動覆蓋，不用擔心這個欄位）。reply_message 用聊天口吻簡短介紹這份草案的安排邏輯與參考來源，邀請使用者提出調整意見或確認。"""

SEARCH_RESEARCH_SYSTEM = """你是一位行程規劃顧問，正在跟真實旅客對話。使用 web_search 工具，針對使用者的人物設定、目的地與行程風格偏好，搜尋最多 3 篇真實、時效性高、來源可信的文章，取得規劃具體行程所需的細節。"""

SEARCH_SYNTH_SYSTEM = """根據上一輪最多 3 篇真實文章的搜尋結果，並參考使用者的行程風格偏好，規劃一份具體的、分時段的行程。

天數與步調要符合 trip_length_type（half_day/one_day/multi_day，multi_day 對應 days 天數）與 party_size。每一天用數個 time_block 劃分，每個時段填入 theme、location_hint、notes，內容應該整合上述文章中的實際資訊。sources 欄位填入你實際參考過的文章網址與標題（最多 3 個，必須是搜尋結果中真實存在的值，不可捏造）。reply_message 用聊天口吻簡短介紹這份草案的安排邏輯與參考來源，邀請使用者提出調整意見或確認。"""

SEARCH_SYNTH_SYSTEM_AFTER_RAG_MISS = SEARCH_SYNTH_SYSTEM + """

重要：這一輪是因為本機常用資料庫裡沒有找到夠合適的文章，才臨時改用即時網路搜尋。reply_message 開頭務必誠實告知使用者這件事（例如：「我們常用的資料庫裡沒有找到很符合的內容，讓我搜尋一下我的網站……」這種自然的口吻），再接著介紹搜尋到的行程安排，不要隱瞞這是備援搜尋。"""

REFINE_SYSTEM = """你是一位行程規劃顧問，根據對話紀錄與使用者最新的訊息調整行程。只調整使用者提到的部分，其餘時段盡量維持不變（除非改動會牽動銜接的時段）。sources 欄位維持原本的內容（除非這輪真的需要更新）。reply_message 用聊天口吻回覆調整了什麼，或回答使用者的問題。"""


class PlanningAgent(StageAgent):
    def __init__(self, model: str = "claude-opus-4-8"):
        super().__init__("itinerary", model)

    def _select_from_rag(
        self, persona: Persona, style_summary: str, site_index: SiteIndex, top_n: int
    ) -> Tuple[RagSelection, CallMetrics]:
        titles_block = "\n".join(f"- [{a.url}] {a.title}" for a in site_index.articles)
        user_content = (
            f"人物設定：{persona.summary_zh()}\n"
            f"目的地：{persona.destination_location}\n"
            f"使用者的行程風格偏好：{style_summary}\n\n"
            f"以下是可以參考的真實文章標題列表：\n{titles_block}\n\n"
            f"請從上面的列表中，挑出最多 {top_n} 篇跟這個人物設定、目的地、風格偏好最相關的文章。"
        )
        output, metrics = call_structured(
            model=self.model, system=RAG_SELECT_SYSTEM, user_content=user_content,
            output_format=RagSelection, max_tokens=1024,
        )
        return output, self._tag(metrics)

    def start_or_continue(
        self,
        persona: Persona,
        history: List[dict],
        user_message: str,
        run_config: RunConfig,
        site_index: Optional[SiteIndex],
        rag_top_n: int,
        round_number: int,
        max_rounds: int,
    ) -> Tuple[PlanningChatTurn, List[dict]]:
        accumulated: List[CallMetrics] = []

        if not history:
            user_content = (
                f"人物設定：{persona.summary_zh()}\n"
                f"目的地：{persona.destination_location}（已確定）\n\n"
                f"{user_message or '（使用者尚未輸入訊息，請主動邀請他分享這趟行程的風格偏好。）'}"
            )
            decision, decide_history = self.start_chat(DECIDE_SYSTEM, user_content, PlanningDecision)
        else:
            decision, decide_history = self.continue_chat(DECIDE_SYSTEM, history, user_message, PlanningDecision)
        accumulated += self.last_call_metrics

        if round_number >= max_rounds:
            decision.needs_clarification = False  # hard cap — don't trust the model to stop asking on its own

        if decision.needs_clarification:
            self.last_call_metrics = accumulated
            turn = PlanningChatTurn(reply_message=decision.reply_message, needs_clarification=True, itinerary_ready=False)
            return turn, decide_history

        style_summary = decision.style_summary.strip() or "使用者未明確表達風格偏好，請依人物設定規劃一份步調適中、涵蓋當地代表性體驗的行程。"

        selected_urls: List[str] = []
        if site_index and site_index.articles:
            selection, select_metrics = self._select_from_rag(persona, style_summary, site_index, rag_top_n)
            accumulated.append(select_metrics)
            known_urls = {a.url for a in site_index.articles}
            selected_urls = [u for u in selection.selected_urls if u in known_urls][:rag_top_n]

        base_context = (
            f"人物設定：{persona.summary_zh()}\n"
            f"trip_length_type={persona.trip_length_type}, days={persona.days}, party_size={persona.party_size}\n"
            f"使用者的行程風格偏好：{style_summary}\n\n"
        )

        if selected_urls:
            url_list_block = "\n".join(selected_urls)
            user_content = base_context + f"請抓取以下文章的內容，並整合成一份具體的行程：\n{url_list_block}"
            output, new_history, fetched_sources = self.start_fetch_chat(
                FETCH_SYSTEM, FETCH_SYNTH_SYSTEM, user_content, PlanningSynthesis,
                run_config, max_uses=len(selected_urls),
            )
            if fetched_sources:
                output.sources = fetched_sources
        else:
            synth_system = SEARCH_SYNTH_SYSTEM_AFTER_RAG_MISS if site_index is not None else SEARCH_SYNTH_SYSTEM
            user_content = base_context + "請搜尋並規劃一份符合這個風格偏好的行程。"
            output, new_history = self.start_search_chat(
                SEARCH_RESEARCH_SYSTEM, synth_system, user_content, PlanningSynthesis, run_config, max_uses=3,
            )
        accumulated += self.last_call_metrics
        self.last_call_metrics = accumulated

        turn = PlanningChatTurn(
            reply_message=output.reply_message, needs_clarification=False, itinerary_ready=True,
            days=output.days, sources=output.sources,
        )
        return turn, new_history

    def refine(self, history: List[dict], user_message: str) -> Tuple[PlanningChatTurn, List[dict]]:
        output, new_history = self.continue_chat(REFINE_SYSTEM, history, user_message, PlanningSynthesis)
        turn = PlanningChatTurn(
            reply_message=output.reply_message, needs_clarification=False, itinerary_ready=True,
            days=output.days, sources=output.sources,
        )
        return turn, new_history
