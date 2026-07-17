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

每個候選方案都要把 data_source 設為 "simulated"，source_url/source_title 留空（null）。從三個方案中選出一個你認為最適合這個人物設定的，填入 agent_selected_candidate_id，並在 agent_selection_rationale 說明理由。"""

_SEARCH_CANDIDATE_RESEARCH_TEMPLATE = """你是一個旅遊產業「{category}」內容推薦研究員，負責尋找 {vendor_hint} 這類真實選項。

使用 web_search 工具，針對使用者的人物設定與已確認的行程，搜尋 3 個真實、符合情境的推薦選項，並在研究過程中記下每個選項明確的來源網址與標題，供之後整理成結構化輸出。"""

_SEARCH_CANDIDATE_SYNTH_TEMPLATE = """根據上一輪的真實搜尋結果，整理出其中最相關的 3 個「{category}」候選方案，輸出結構化 JSON。

每個候選方案都要把 data_source 設為 "real_search"，source_url/source_title 必須使用搜尋結果中的真實值，絕對不可捏造或留空。從三個方案中選出一個你認為最適合這個人物設定的，填入 agent_selected_candidate_id，並在 agent_selection_rationale 說明理由。"""


def mock_candidate_system_prompt(category: str, vendor_hint: str) -> str:
    return _MOCK_CANDIDATE_TEMPLATE.format(category=category, vendor_hint=vendor_hint)


def search_candidate_prompts(category: str, vendor_hint: str) -> "tuple[str, str]":
    research = _SEARCH_CANDIDATE_RESEARCH_TEMPLATE.format(category=category, vendor_hint=vendor_hint)
    synth = _SEARCH_CANDIDATE_SYNTH_TEMPLATE.format(category=category)
    return research, synth


_CHAT_CANDIDATE_START_TEMPLATE = """你是一個旅遊產業「{category}」候選方案顧問，正在跟真實旅客對話，角色定位是提出 {vendor_hint} 這類選項。

根據使用者人物設定與已確認的行程，提出恰好 3 個合理、在地合理的候選方案。這些是模擬資料（simulated）——你沒有連接任何真實訂位或付款系統，不能執行真實交易，只負責提案。data_source 固定為 "simulated"，source_url/source_title 留空（null）。從中選一個你認為最適合這個人物設定的，填入 agent_selected_candidate_id，並在 agent_selection_rationale 說明理由。reply_message 用聊天口吻簡短介紹這些方案，邀請使用者選擇其中一個、或提出調整意見（例如想要更便宜/更快/更豪華的選項）。"""

_CHAT_CANDIDATE_REFINE_TEMPLATE = """你是「{category}」候選方案顧問，根據對話紀錄與使用者最新的訊息調整候選方案（例如換掉某個選項、找更便宜/更快/更符合需求的、或使用者只是在問問題就正常回答並維持方案不變）。data_source 固定為 "simulated"。reply_message 用聊天口吻自然回覆使用者的訊息。"""


def chat_candidate_system_prompts(category: str, vendor_hint: str) -> "tuple[str, str]":
    start = _CHAT_CANDIDATE_START_TEMPLATE.format(category=category, vendor_hint=vendor_hint)
    refine = _CHAT_CANDIDATE_REFINE_TEMPLATE.format(category=category)
    return start, refine


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
            allowed_domains=allowed_domains, max_tokens=max_tokens,
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
