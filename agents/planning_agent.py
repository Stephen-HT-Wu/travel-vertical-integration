"""Unified planning orchestrator for the chat/webapp flow.

Replaces the old single-itinerary flow (start_or_continue/refine) with a
two-option full-trip bundle: one structured call now extracts BOTH the
user's free-text persona description AND their style/pacing preference
(IntakeDecision), reusing the same round_number/max_rounds clarification
loop that used to handle style intent alone; once resolved, generates
exactly two complete trip packages — each with real (not simulated)
transportation/accommodation/activities baked in, grounded in the
locally-cached RAG title-selection (primary axis) plus a live web_search
pass for real vendor names and supplementary corroboration (see
start_dual_ground_chat in agents/base_agent.py).

This is chat-only. The CLI/auto-pipeline path (orchestrator.py) still uses
InspirationAgent.run()/ItineraryAgent.run() unchanged — this class doesn't
touch or replace those."""
from typing import List, Optional, Tuple

from agents.base_agent import StageAgent
from llm_client import call_structured
from persona import Persona
from schemas import (
    CallMetrics,
    IntakeDecision,
    PlanBundleSynthesis,
    PlanBundleTurn,
    RagSelection,
    RunConfig,
)
from site_index import SiteIndex

INTAKE_SYSTEM = """你是一位行程規劃顧問，正在跟真實旅客對話。使用者會用自己的話描述這趟旅行，可能一次講很多資訊，也可能講得很簡略。

你的任務是從對話內容（包含這一輪與之前所有輪次）中，盡可能擷取出以下資訊：
- home_location：出發地（城市/地區）
- destination_location：目的地（城市/地區）——這是規劃行程的必要資訊，沒有這個完全無法規劃
- age_group：年齡層，只能是 "18-25"、"26-35"、"36-50"、"51+" 其中之一，沒提到就留空
- gender：性別，只能是 "male"、"female"、"unspecified"，沒提到就留 "unspecified"
- trip_length_type：只能是 "half_day"、"one_day"、"multi_day"，沒提到、但看起來是一日遊/半日遊/多日遊要判斷出來
- days：天數（僅 trip_length_type 是 multi_day 時有意義），沒提到就留空
- party_size：同行人數，沒提到就留空
- companion_notes：同行者的特殊需求，例如「有長輩」「帶小小孩」「有寵物同行」「行動不便」等——這點特別重要，要盡量從使用者的敘述裡挑出來，不要漏掉
- style_summary：這趟旅行的風格偏好（步調、主題、預算感），例如「悠閒不要走太多路」「想吃美食」「想帶長輩慢慢逛」

**每一輪都要重新輸出你目前為止「累積理解到的全部資訊」，不是只回報這一輪使用者新講的內容**——因為你看到的是完整對話歷史，請把之前輪次已經確定的資訊也一併帶入這次的輸出，不要因為使用者這輪沒有重複講就把已經確定的欄位清空。

判斷是否需要追問（needs_clarification）：
- 如果 destination_location 還不知道，一定要追問（沒有目的地完全無法規劃行程），reply_message 用聊天口吻自然地問目的地是哪裡。
- 如果目的地已知、但 home_location 還不知道，追問出發地。
- 如果出發地跟目的地都已知，但完全沒有任何風格/步調/主題偏好的線索，可以追問一個具體的風格偏好問題（這個不用像目的地那麼堅持——如果使用者已經被追問過還是講不清楚風格，之後系統會自動套用一份預設風格繼續規劃）。
- 每一輪只問一個最關鍵的問題，不要一次問一大堆。needs_clarification 為 true 時，其他已經確定的欄位還是要照樣填，只有還不知道的欄位留空。"""

RAG_SELECT_SYSTEM = """你是一位內容檢索助理。你會拿到一份真實文章標題列表（每行格式為「- [網址] 標題」），任務是從中挑出最符合這趟行程人物設定、目的地與風格偏好的文章，作為「兩個方案共用」的主軸參考來源。

只能挑選列表中真實出現過的網址，絕對不可以自己編網址或修改網址。如果列表裡沒有任何一篇跟目的地或風格真的相關（例如列表主題完全是別的城市、別的主題），selected_urls 就回傳空陣列，並在 selection_rationale 誠實說明原因，不要勉強湊數。"""

BUNDLE_FETCH_SYSTEM = """你是一位行程規劃顧問，正在跟真實旅客對話。使用 web_fetch 工具抓取使用者訊息中列出的文章網址，取得規劃行程所需的實際內容（景點順序、交通銜接、營業時間、當地建議路線、體驗心得）。"""

CORROBORATION_SEARCH_SYSTEM = """你是一位行程規劃研究員。使用 web_search 工具，針對使用者的人物設定、目的地與行程風格偏好，搜尋以下真實資訊：

1. 真實可查詢的交通方式（例如台鐵/客運/國道路線名稱、大致時刻或班距）。
2. 若行程需要過夜：真實存在、可查詢到的旅宿名稱（絕對不可捏造），要符合這趟行程的風格與同行需求（例如親子/長輩/寵物友善）。
3. 真實存在的活動/景點/體驗名稱，適合安排進行程。

同時也順便搜尋 1-2 篇可以作為「佐證來源」的其他真實文章（不必是同一個網站），之後會列在方案旁邊供使用者自行點擊查證。"""

SEARCH_SYSTEM_AFTER_RAG_MISS = """你是一位行程規劃顧問，正在跟真實旅客對話。使用 web_search 工具，針對使用者的人物設定、目的地與行程風格偏好，搜尋規劃這趟行程所需的完整真實資訊，包括：

1. 目的地本身的行程內容建議（景點、美食、體驗，可參考最多 3 篇真實文章）。
2. 真實可查詢的交通方式。
3. 若行程需要過夜：真實存在、可查詢到的旅宿名稱，要符合這趟行程的風格與同行需求。
4. 真實存在的活動/景點/體驗名稱。

搜尋範圍不限定在特定網站（因為本機常用資料庫這次沒有找到夠合適的內容）。"""

_BUNDLE_SYNTH_COMMON = """根據上一輪的真實研究結果（{primary_source_desc}），規劃「恰好兩個」風格明顯不同的完整旅行方案（例如一個步調悠閒、一個主題更聚焦；或依人物設定的同行需求做出不同取捨），每個方案都要包含：

1. days：分時段的行程時間軸。天數與步調要符合 trip_length_type（half_day/one_day/multi_day，multi_day 對應 days 天數）、party_size 與 companion_notes（同行需求，例如帶長輩要步調放慢、有寵物同行要避開不友善的場所）。每個時段除了 theme/location_hint/notes 之外，如果這個時段的內容真的能追溯到你讀過的某篇文章的具體段落，就填入 source_url/source_title/citation_quote（citation_quote 是一段簡短的真實引用文字，不可捏造）；如果無法明確追溯到具體來源，這三個欄位就全部留空（null），絕對不要為了填滿欄位而編造引用。

2. transportation：恰好 1 筆真實交通建議（CandidateOption），data_source 固定為 "real_search"，name/vendor/description 必須是搜尋結果中真實存在的資訊，deep_link_query 填「真實地名/路線名稱」（之後會用來組真實的 Google 地圖連結）。

3. accommodation：如果 trip_length_type 是 half_day 或 one_day（不過夜），這裡留空陣列 []；如果是 multi_day（過夜），提供 1 筆真實存在的旅宿（CandidateOption），data_source 固定為 "real_search"，name 必須是搜尋結果中真實存在的旅宿名稱（絕對不可捏造），deep_link_query 也填這個真實旅宿名稱（之後會用來查 Booking.com／Agoda 真實報價）。

4. activities：1-3 筆真實存在的活動/體驗/景點（CandidateOption），data_source 固定為 "real_search"，deep_link_query 填真實的活動/景點名稱或關鍵字。

5. primary_sources / corroboration_sources：分別列出這個方案實際引用到的主軸來源文章、佐證來源文章（都必須是研究結果中真實出現過的網址與標題，不可捏造；兩個方案通常共用同一組來源，不需要各自搜尋一次）。

6. option_id 設為與 label 相同的字串（"A" 或 "B"）；is_agent_recommended 恰好一個方案是 true；why_recommended 用一兩句話說明為什麼推薦（或為什麼是另一個備選）。agent_recommended_option_id 填被推薦方案的 option_id。

reply_message 用聊天口吻簡短介紹這兩個方案的差異，邀請使用者選擇其中一個或提出調整意見。"""

BUNDLE_FETCH_SYNTH_SYSTEM = _BUNDLE_SYNTH_COMMON.format(
    primary_source_desc="第一段 web_fetch 抓取到的文章全文作為行程主軸，第二段 web_search 找到的真實交通/住宿/活動資訊"
)

BUNDLE_SEARCH_SYNTH_SYSTEM_AFTER_RAG_MISS = (
    _BUNDLE_SYNTH_COMMON.format(
        primary_source_desc="web_search 搜尋到的真實資訊（含目的地行程內容與交通/住宿/活動）"
    )
    + """

重要：這一輪是因為本機常用資料庫裡沒有找到夠合適的文章，才臨時改用即時網路搜尋。reply_message 開頭務必誠實告知使用者這件事（例如：「我們常用的資料庫裡沒有找到很符合的內容，讓我搜尋一下我的網站……」這種自然的口吻），再接著介紹這兩個方案，不要隱瞞這是備援搜尋。"""
)

REFINE_BUNDLE_SYSTEM = """你是一位行程規劃顧問，根據對話紀錄與使用者最新的訊息調整這兩個方案（例如換掉某個行程時段、交通/住宿/活動選項，或使用者只是在問問題就正常回答並維持方案不變）。只調整使用者提到的部分，其餘盡量維持不變。來源/引用相關欄位的誠實原則維持不變（無法追溯來源就留空，不可捏造）。reply_message 用聊天口吻回覆調整了什麼，或回答使用者的問題。"""


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
            f"請從上面的列表中，挑出最多 {top_n} 篇跟這個人物設定、目的地、風格偏好最相關的文章，"
            f"作為兩個方案共用的主軸參考來源。"
        )
        output, metrics = call_structured(
            model=self.model, system=RAG_SELECT_SYSTEM, user_content=user_content,
            output_format=RagSelection, max_tokens=1024,
        )
        return output, self._tag(metrics)

    @staticmethod
    def _resolve_persona(decision: IntakeDecision) -> Persona:
        trip_length_type = decision.trip_length_type or "one_day"
        if decision.days is not None:
            days = decision.days
        elif trip_length_type == "multi_day":
            days = 3
        else:
            days = 1
        return Persona(
            age_group=decision.age_group or "26-35",
            gender=decision.gender or "unspecified",
            home_location=(decision.home_location or "").strip(),
            destination_location=(decision.destination_location or "").strip(),
            trip_length_type=trip_length_type,
            days=days,
            party_size=decision.party_size or 1,
            companion_notes=decision.companion_notes.strip(),
        )

    def generate_plan_bundle(
        self,
        persona: Persona,
        history: List[dict],
        user_message: str,
        run_config: RunConfig,
        site_index: Optional[SiteIndex],
        rag_top_n: int,
        round_number: int,
        max_rounds: int,
    ) -> Tuple[PlanBundleTurn, List[dict], Optional[Persona]]:
        accumulated: List[CallMetrics] = []

        if not history:
            user_content = user_message or (
                "（使用者尚未輸入訊息，請主動邀請他用一段話描述這趟旅行：出發地、目的地、天數/人數，"
                "以及是否有長輩/幼兒/寵物等特殊同行需求。）"
            )
            decision, decide_history = self.start_chat(INTAKE_SYSTEM, user_content, IntakeDecision)
        else:
            decision, decide_history = self.continue_chat(INTAKE_SYSTEM, history, user_message, IntakeDecision)
        accumulated += self.last_call_metrics

        have_destination = bool((decision.destination_location or "").strip())
        have_home = bool((decision.home_location or "").strip())
        if not have_destination or not have_home:
            # Never force past a missing destination/home location, no matter
            # the round cap — a plan literally cannot be generated without
            # them, and silently defaulting would be dishonest, not efficient.
            needs_clarification = True
        elif round_number >= max_rounds:
            needs_clarification = False  # cap only bites once the essentials are known
        else:
            needs_clarification = decision.needs_clarification

        if needs_clarification:
            self.last_call_metrics = accumulated
            turn = PlanBundleTurn(reply_message=decision.reply_message, needs_clarification=True)
            return turn, decide_history, None

        resolved_persona = self._resolve_persona(decision)
        style_summary = decision.style_summary.strip() or (
            "使用者未明確表達風格偏好，請依人物設定規劃步調適中、涵蓋當地代表性體驗的行程。"
        )

        selected_urls: List[str] = []
        if site_index and site_index.articles:
            selection, select_metrics = self._select_from_rag(resolved_persona, style_summary, site_index, rag_top_n)
            accumulated.append(select_metrics)
            known_urls = {a.url for a in site_index.articles}
            selected_urls = [u for u in selection.selected_urls if u in known_urls][:rag_top_n]

        base_context = (
            f"人物設定：{resolved_persona.summary_zh()}\n"
            f"trip_length_type={resolved_persona.trip_length_type}, days={resolved_persona.days}, "
            f"party_size={resolved_persona.party_size}\n"
            f"使用者的行程風格偏好：{style_summary}\n\n"
        )

        if selected_urls:
            url_list_block = "\n".join(selected_urls)
            fetch_user_content = base_context + f"請抓取以下文章的內容，作為行程主軸：\n{url_list_block}"
            search_user_content = (
                base_context + "請搜尋規劃這趟行程所需的真實交通/住宿（若過夜）/活動資訊，以及 1-2 篇可作佐證的其他文章。"
            )
            output, new_history, fetched_sources = self.start_dual_ground_chat(
                fetch_system_prompt=BUNDLE_FETCH_SYSTEM,
                fetch_user_content=fetch_user_content,
                search_system_prompt=CORROBORATION_SEARCH_SYSTEM,
                search_user_content=search_user_content,
                synth_system_prompt=BUNDLE_FETCH_SYNTH_SYSTEM,
                synth_user_content="請根據以上研究結果，產出兩個完整方案。",
                output_format=PlanBundleSynthesis,
                run_config=run_config,
                fetch_max_uses=len(selected_urls),
            )
        else:
            search_user_content = base_context + "請搜尋規劃這趟行程所需的完整真實資訊（目的地行程內容、交通、住宿（若過夜）、活動）。"
            output, new_history, fetched_sources = self.start_dual_ground_chat(
                fetch_system_prompt=None,
                fetch_user_content=None,
                search_system_prompt=SEARCH_SYSTEM_AFTER_RAG_MISS,
                search_user_content=search_user_content,
                synth_system_prompt=BUNDLE_SEARCH_SYNTH_SYSTEM_AFTER_RAG_MISS,
                synth_user_content="請根據以上研究結果，產出兩個完整方案。",
                output_format=PlanBundleSynthesis,
                run_config=run_config,
                search_max_uses=6,
            )
        accumulated += self.last_call_metrics
        self.last_call_metrics = accumulated

        if fetched_sources:
            # Ground truth titles from the actual fetch, same trust pattern
            # as extract_fetched_sources elsewhere — prefer the real fetched
            # title over whatever the synthesis call copied into JSON.
            url_title_map = {s.url: s.title for s in fetched_sources}
            for option in output.options:
                for src in option.primary_sources:
                    if src.url in url_title_map:
                        src.title = url_title_map[src.url]

        turn = PlanBundleTurn(
            reply_message=output.reply_message,
            needs_clarification=False,
            plan_ready=True,
            options=output.options,
            agent_recommended_option_id=output.agent_recommended_option_id,
        )
        return turn, new_history, resolved_persona

    def refine_bundle(self, history: List[dict], user_message: str) -> Tuple[PlanBundleTurn, List[dict], None]:
        output, new_history = self.continue_chat(REFINE_BUNDLE_SYSTEM, history, user_message, PlanBundleSynthesis)
        turn = PlanBundleTurn(
            reply_message=output.reply_message,
            needs_clarification=False,
            plan_ready=True,
            options=output.options,
            agent_recommended_option_id=output.agent_recommended_option_id,
        )
        return turn, new_history, None
