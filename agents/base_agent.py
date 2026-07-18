"""Shared call plumbing for every stage agent.

Two call patterns, matching stage_metadata.REAL_SEARCH_STAGES vs.
SIMULATED_CANDIDATE_STAGES:

- run_mock(): a single structured call, tools=[] (implicitly — llm_client
  never attaches tools to a structured call). Used by transportation,
  accommodation, activities, itinerary, in_trip_guide, replanning.
- run_with_search(): the two-call pattern (web_search research, then a
  structured synthesis call over the same conversation). Used only by
  inspiration, dining, attractions, shopping.

Only run_with_search ever touches a tool, and the only tool it ever attaches
is web_search — there is no booking/payment tool anywhere for any stage to
reach for, under any prompt.

Each call also records a CallMetrics entry (tokens/cost/duration) onto
self.last_call_metrics, reset at the start of every run_mock/run_with_search
call. The orchestrator reads it immediately after invoking a stage method,
so this stays simple mutable state rather than changing every stage agent's
return signature.
"""
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

from llm_client import call_structured, call_with_web_search
from schemas import CallMetrics, RunConfig

T = TypeVar("T", bound=BaseModel)

SYNTH_INSTRUCTION = (
    "Now emit the final structured output. Use the real source_url and "
    "source_title values from the search results above for every item — "
    "never invent a URL."
)

_MOCK_CANDIDATE_TEMPLATE = """你是一個旅遊產業「{category}」候選方案 agent，角色定位是提出 {vendor_hint} 這類選項。

你的任務：根據使用者人物設定與已確認的行程，提出恰好 3 個合理、在地合理的候選方案。這些是模擬資料（simulated），不是真實的即時庫存或價格——你沒有連接任何真實訂位或付款系統，也沒有能力執行真實交易，只負責提案供使用者確認。

每個候選方案都要把 data_source 設為 "simulated"，source_url/source_title 留空（null）。從三個方案中選出一個你認為最適合這個人物設定的，填入 agent_selected_candidate_id，並在 agent_selection_rationale 說明理由。deep_link_query 這個情境下不需要，留空（null）即可。"""

_SEARCH_CANDIDATE_RESEARCH_TEMPLATE = """你是一個旅遊產業「{category}」內容推薦研究員，負責尋找 {vendor_hint} 這類真實選項。

使用 web_search 工具，針對使用者的人物設定與已確認的行程，搜尋 3 個真實、符合情境的推薦選項，並在研究過程中記下每個選項明確的來源網址與標題，供之後整理成結構化輸出。"""

_SEARCH_CANDIDATE_SYNTH_TEMPLATE = """根據上一輪的真實搜尋結果，整理出其中最相關的 3 個「{category}」候選方案，輸出結構化 JSON。

每個候選方案都要把 data_source 設為 "real_search"，source_url/source_title 必須使用搜尋結果中的真實值，絕對不可捏造或留空。從三個方案中選出一個你認為最適合這個人物設定的，填入 agent_selected_candidate_id，並在 agent_selection_rationale 說明理由。deep_link_query 這個情境下不需要，留空（null）即可。"""


def mock_candidate_system_prompt(category: str, vendor_hint: str) -> str:
    return _MOCK_CANDIDATE_TEMPLATE.format(category=category, vendor_hint=vendor_hint)


def search_candidate_prompts(category: str, vendor_hint: str) -> "tuple[str, str]":
    research = _SEARCH_CANDIDATE_RESEARCH_TEMPLATE.format(category=category, vendor_hint=vendor_hint)
    synth = _SEARCH_CANDIDATE_SYNTH_TEMPLATE.format(category=category)
    return research, synth


_CHAT_CANDIDATE_START_TEMPLATE = """你是一個旅遊產業「{category}」候選方案顧問，正在跟真實旅客對話，角色定位是提出 {vendor_hint} 這類選項。

根據使用者人物設定與已確認的行程，提出恰好 3 個合理、在地合理的候選方案。這些是模擬資料（simulated）——你沒有連接任何真實訂位或付款系統，不能執行真實交易，只負責提案。data_source 固定為 "simulated"，source_url/source_title 留空（null）。從中選一個你認為最適合這個人物設定的，填入 agent_selected_candidate_id，並在 agent_selection_rationale 說明理由。reply_message 用聊天口吻簡短介紹這些方案，邀請使用者選擇其中一個、或提出調整意見（例如想要更便宜/更快/更豪華的選項）。

重要：candidates 這個欄位一定要包含 3 個完整填寫的方案（每個都要有真實具體的 name/vendor/price_range/description，不可以是空陣列），而且內容要跟你在 reply_message 裡描述的方案完全對應——不要只把方案寫在 reply_message 裡就把 candidates 留空或用佔位文字帶過。agent_selected_candidate_id 必須是這 3 個 candidates 之一的真實 id，不可以是 "placeholder" 或任何虛構值。

方案名稱請用「類型/風格描述」而非聽起來像真實品牌的專屬名稱（例如「近台北車站的平價商務旅館」而非「台北車站前設計青年旅館」這種容易被誤認為真實房源的名字），避免使用者以為這是某個可以指名搜尋到的真實地方。

{extra_field_hint}

deep_link_query：每個候選方案（每個 candidate）都要各自填寫自己的 deep_link_query，不是整輪共用一個——{deep_link_query_hint}。這段文字之後會被拿去組一個真實網站的搜尋網址，讓使用者可以自己去該網站查看真實報價/庫存——只要填「真實世界看得懂的地名、地區或風格關鍵字」，不要填候選方案的虛構名稱、任何網址，或 "placeholder" 這類佔位文字；同一輪的 3 個候選方案，deep_link_query 應該反映各自的風格/地區/價位差異，不要三個都填一樣的內容。"""

_CHAT_CANDIDATE_REFINE_TEMPLATE = """你是「{category}」候選方案顧問，根據對話紀錄與使用者最新的訊息調整候選方案（例如換掉某個選項、找更便宜/更快/更符合需求的、或使用者只是在問問題就正常回答並維持方案不變）。data_source 固定為 "simulated"。方案名稱維持用「類型/風格描述」而非聽起來像真實品牌的專屬名稱。reply_message 用聊天口吻自然回覆使用者的訊息。

重要：candidates 這個欄位一定要包含 3 個完整填寫的方案（不可以是空陣列或佔位文字），agent_selected_candidate_id 必須是這 3 個 candidates 之一的真實 id。

{extra_field_hint}

deep_link_query：每個候選方案都要各自重新填寫自己的 deep_link_query（{deep_link_query_hint}），若使用者的調整意見影響了地點/風格/關鍵字，記得更新，不可以留空或寫 "placeholder"。"""


def chat_candidate_system_prompts(
    category: str, vendor_hint: str, deep_link_query_hint: str, extra_field_hint: str = ""
) -> "tuple[str, str]":
    start = _CHAT_CANDIDATE_START_TEMPLATE.format(
        category=category, vendor_hint=vendor_hint, deep_link_query_hint=deep_link_query_hint,
        extra_field_hint=extra_field_hint,
    )
    refine = _CHAT_CANDIDATE_REFINE_TEMPLATE.format(
        category=category, deep_link_query_hint=deep_link_query_hint, extra_field_hint=extra_field_hint,
    )
    return start, refine


def validate_candidate_turn(category: str, turn) -> None:
    """Defense-in-depth against a real, observed failure mode: a structured
    output that's technically valid JSON but semantically degenerate (e.g.
    empty candidates list, agent_selected_candidate_id left as the literal
    string "placeholder") — the schema constraint alone doesn't stop the
    model from doing this. Raises a clear, actionable error instead of
    silently handing the caller unusable data.

    deep_link_query is validated per-candidate (it now lives on
    CandidateOption, not the turn) except for "交通": that stage's deep link
    always uses persona.destination_location, set deterministically by
    chat_session.generate_referral, so the model's own deep_link_query there
    is unused and not worth hard-failing a turn over."""
    if not turn.candidates:
        raise RuntimeError(f"「{category}」候選方案回傳空陣列，請重新輸入訊息再試一次")
    candidate_ids = {c.id for c in turn.candidates}
    if turn.agent_selected_candidate_id not in candidate_ids:
        raise RuntimeError(
            f"「{category}」agent_selected_candidate_id ({turn.agent_selected_candidate_id!r}) "
            f"不在候選方案 id 清單中，請重新輸入訊息再試一次"
        )
    if category != "交通":
        for c in turn.candidates:
            query = (c.deep_link_query or "").strip()
            if not query or query.lower() == "placeholder":
                raise RuntimeError(
                    f"「{category}」候選方案「{c.name}」的 deep_link_query 是空值或佔位文字，請重新輸入訊息再試一次"
                )


class StageAgent:
    def __init__(self, stage_name: str, model: str = "claude-opus-4-8"):
        self.stage_name = stage_name
        self.model = model
        self.last_call_metrics: List[CallMetrics] = []

    def _tag(self, metrics: CallMetrics) -> CallMetrics:
        return metrics.model_copy(update={"stage": self.stage_name})

    def run_mock(
        self,
        system_prompt: str,
        user_content: str,
        output_format: Type[T],
        max_tokens: int = 4096,
    ) -> T:
        self.last_call_metrics = []
        output, metrics = call_structured(
            model=self.model,
            system=system_prompt,
            user_content=user_content,
            output_format=output_format,
            max_tokens=max_tokens,
        )
        self.last_call_metrics.append(self._tag(metrics))
        return output

    def run_with_search(
        self,
        research_system_prompt: str,
        synth_system_prompt: str,
        user_content: str,
        output_format: Type[T],
        run_config: RunConfig,
        max_tokens: int = 4000,
        synth_max_tokens: int = 8000,
        max_uses: int = 5,
    ) -> T:
        """synth_max_tokens defaults higher than the research call's budget:
        the synthesis turn has to fit the full replayed search-result context
        *and* a complete JSON payload in its output, and heavy search results
        (observed: 90k+ input tokens on a single inspiration call) can push a
        4000-token synthesis output budget to truncate mid-JSON, which makes
        response.parsed_output silently come back None. See llm_client.call_structured."""
        self.last_call_metrics = []
        allowed_domains: Optional[List[str]] = (
            run_config.allowed_domains if run_config.site_mode == "allowlist" else None
        )
        research, research_metrics = call_with_web_search(
            model=self.model,
            system=research_system_prompt,
            user_content=user_content,
            allowed_domains=allowed_domains,
            max_tokens=max_tokens,
            max_uses=max_uses,
        )
        self.last_call_metrics.append(self._tag(research_metrics))
        extra_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": research.content},
        ]
        output, synth_metrics = call_structured(
            model=self.model,
            system=synth_system_prompt,
            user_content=SYNTH_INSTRUCTION,
            output_format=output_format,
            extra_messages=extra_messages,
            max_tokens=synth_max_tokens,
        )
        self.last_call_metrics.append(self._tag(synth_metrics))
        return output

    # -- real-user multi-turn chat primitives ---------------------------
    # Used by the chat-driven stages (inspiration / itinerary / transaction
    # candidates). Unlike run_mock/run_with_search (single-shot, then
    # auto-confirmed by UserSimulatorAgent), these keep a plain Anthropic
    # message-history list that the caller (chat_session.py) threads through
    # turn after turn until the real user is satisfied.

    def start_chat(
        self, system_prompt: str, user_content: str, output_format: Type[T], max_tokens: int = 4096
    ) -> "tuple[T, List[dict]]":
        self.last_call_metrics = []
        output, metrics = call_structured(
            model=self.model, system=system_prompt, user_content=user_content, output_format=output_format,
            max_tokens=max_tokens,
        )
        self.last_call_metrics.append(self._tag(metrics))
        history = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.model_dump_json()},
        ]
        return output, history

    def start_search_chat(
        self,
        research_system_prompt: str,
        synth_system_prompt: str,
        user_content: str,
        output_format: Type[T],
        run_config: RunConfig,
        max_tokens: int = 4000,
        synth_max_tokens: int = 8000,
        max_uses: int = 5,
    ) -> "tuple[T, List[dict]]":
        """Like start_chat, but the first turn is grounded in one real
        web_search pass (see run_with_search's docstring for why
        synth_max_tokens defaults higher). Later turns in the same chat
        reuse this grounding via continue_chat rather than searching again —
        keeps a multi-turn inspiration conversation from re-paying the ~90k+
        input-token cost of a fresh search on every refinement."""
        self.last_call_metrics = []
        allowed_domains: Optional[List[str]] = (
            run_config.allowed_domains if run_config.site_mode == "allowlist" else None
        )
        research, research_metrics = call_with_web_search(
            model=self.model, system=research_system_prompt, user_content=user_content,
            allowed_domains=allowed_domains, max_tokens=max_tokens, max_uses=max_uses,
        )
        self.last_call_metrics.append(self._tag(research_metrics))
        history = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": research.content},
        ]
        output, synth_metrics = call_structured(
            model=self.model, system=synth_system_prompt, user_content=SYNTH_INSTRUCTION,
            output_format=output_format, extra_messages=history, max_tokens=synth_max_tokens,
        )
        self.last_call_metrics.append(self._tag(synth_metrics))
        history = history + [
            {"role": "user", "content": SYNTH_INSTRUCTION},
            {"role": "assistant", "content": output.model_dump_json()},
        ]
        return output, history

    def continue_chat(
        self, system_prompt: str, history: List[dict], user_message: str, output_format: Type[T], max_tokens: int = 4096
    ) -> "tuple[T, List[dict]]":
        self.last_call_metrics = []
        output, metrics = call_structured(
            model=self.model, system=system_prompt, user_content=user_message, output_format=output_format,
            extra_messages=history, max_tokens=max_tokens,
        )
        self.last_call_metrics.append(self._tag(metrics))
        new_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output.model_dump_json()},
        ]
        return output, new_history
