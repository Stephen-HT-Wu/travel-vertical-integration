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
"""
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

from llm_client import call_structured, call_with_web_search
from schemas import RunConfig

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


class StageAgent:
    def __init__(self, stage_name: str, model: str = "claude-opus-4-8"):
        self.stage_name = stage_name
        self.model = model

    def run_mock(
        self,
        system_prompt: str,
        user_content: str,
        output_format: Type[T],
        max_tokens: int = 4096,
    ) -> T:
        return call_structured(
            model=self.model,
            system=system_prompt,
            user_content=user_content,
            output_format=output_format,
            max_tokens=max_tokens,
        )

    def run_with_search(
        self,
        research_system_prompt: str,
        synth_system_prompt: str,
        user_content: str,
        output_format: Type[T],
        run_config: RunConfig,
        max_tokens: int = 4000,
    ) -> T:
        allowed_domains: Optional[List[str]] = (
            run_config.allowed_domains if run_config.site_mode == "allowlist" else None
        )
        research = call_with_web_search(
            model=self.model,
            system=research_system_prompt,
            user_content=user_content,
            allowed_domains=allowed_domains,
            max_tokens=max_tokens,
        )
        extra_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": research.content},
        ]
        return call_structured(
            model=self.model,
            system=synth_system_prompt,
            user_content=SYNTH_INSTRUCTION,
            output_format=output_format,
            extra_messages=extra_messages,
            max_tokens=max_tokens,
        )
