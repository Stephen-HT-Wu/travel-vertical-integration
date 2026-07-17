"""All structured-output and trip-log models for the pipeline.

Kept flat (no dicts, no recursive types) because these are passed as
Pydantic models to `client.messages.parse(output_format=...)`, whose
JSON-schema translation does not support `additionalProperties`-style
free-form dicts or recursion.
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from persona import Persona

CheckpointType = Literal[
    "itinerary_confirmation",
    "candidate_confirmation",
    "replanning_confirmation",
    "final_review",
]


class RunConfig(BaseModel):
    site_mode: Literal["unrestricted", "allowlist"] = "unrestricted"
    allowed_domains: List[str] = []
    model: str = "claude-opus-4-8"
    language: str = "zh-TW"


# ---------------------------------------------------------------------------
# Cost/token/time telemetry — for feasibility assessment, not exact billing.
# ---------------------------------------------------------------------------
class CallMetrics(BaseModel):
    stage: str
    call_type: Literal["structured", "web_search"]
    model: str
    input_tokens: int
    output_tokens: int
    web_search_requests: int = 0
    duration_ms: float
    cost_usd: float


class StageMetrics(BaseModel):
    stage: str
    calls: List[CallMetrics] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_web_search_requests: int = 0
    total_duration_ms: float = 0
    total_cost_usd: float = 0

    @classmethod
    def from_calls(cls, stage: str, calls: List[CallMetrics]) -> "StageMetrics":
        return cls(
            stage=stage,
            calls=calls,
            total_input_tokens=sum(c.input_tokens for c in calls),
            total_output_tokens=sum(c.output_tokens for c in calls),
            total_web_search_requests=sum(c.web_search_requests for c in calls),
            total_duration_ms=sum(c.duration_ms for c in calls),
            total_cost_usd=sum(c.cost_usd for c in calls),
        )


class RunMetrics(BaseModel):
    stages: List[StageMetrics] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_web_search_requests: int = 0
    total_duration_ms: float = 0
    total_cost_usd: float = 0

    def add_stage(self, stage_metrics: StageMetrics) -> None:
        self.stages.append(stage_metrics)
        self.total_input_tokens += stage_metrics.total_input_tokens
        self.total_output_tokens += stage_metrics.total_output_tokens
        self.total_web_search_requests += stage_metrics.total_web_search_requests
        self.total_duration_ms += stage_metrics.total_duration_ms
        self.total_cost_usd += stage_metrics.total_cost_usd


class HITLLogEntry(BaseModel):
    stage: str
    checkpoint_type: CheckpointType
    presented_summary: str
    decision_status: str
    user_simulator_reasoning: str
    timestamp: str


# ---------------------------------------------------------------------------
# Inspiration (Layer 1 — real search, destination/theme options, not candidates)
# ---------------------------------------------------------------------------
class InspirationDestinationOption(BaseModel):
    id: str
    name: str
    summary: str
    why_recommended: str
    source_url: str
    source_title: str
    citation_snippet: str


class InspirationOutput(BaseModel):
    destination_options: List[InspirationDestinationOption]
    queries_used: List[str]


class InspirationChatTurn(BaseModel):
    """One turn of the real-user chat conversation for inspiration
    exploration. reply_message is shown as a chat bubble; destination_options
    is the current (possibly refined) proposal, redrawn each turn."""
    reply_message: str
    destination_options: List[InspirationDestinationOption]


# ---------------------------------------------------------------------------
# Itinerary (Layer 2)
# ---------------------------------------------------------------------------
class ItineraryBlock(BaseModel):
    time_block: str
    theme: str
    location_hint: str
    notes: str


class ItineraryDay(BaseModel):
    day_number: int
    blocks: List[ItineraryBlock]


class ItineraryConfirmation(BaseModel):
    status: Literal["confirmed", "revised"]
    feedback: str
    revision_notes: Optional[str] = None


class ItineraryOutput(BaseModel):
    itinerary_id: str
    days: List[ItineraryDay]
    based_on_inspiration_ids: List[str]
    confirmation: Optional[ItineraryConfirmation] = None


class ItineraryChatTurn(BaseModel):
    """One turn of the real-user chat conversation for itinerary planning."""
    reply_message: str
    days: List[ItineraryDay]


# ---------------------------------------------------------------------------
# Generic candidate stage output — shared by transportation, accommodation,
# dining, attractions, activities, shopping. data_source distinguishes the
# real-search recommendation stages from the simulated commerce stages.
# ---------------------------------------------------------------------------
class CandidateOption(BaseModel):
    id: str
    vendor: str
    name: str
    price_range: str
    rating: float
    description: str
    highlights: List[str]
    data_source: Literal["real_search", "simulated"]
    source_url: Optional[str] = None
    source_title: Optional[str] = None


class CandidateConfirmation(BaseModel):
    status: Literal["confirmed", "swapped", "declined"]
    final_candidate_id: str
    feedback: str


class SimulatedOrder(BaseModel):
    """A mocked 'redirected out to book, order confirmed' record. This is
    always fabricated locally (no LLM call, no real payment/booking system
    contacted) — order_id is a random token, not a real confirmation number."""
    order_id: str
    stage: str
    candidate_id: str
    candidate_name: str
    price_range: str
    confirmed_at: str
    is_simulated: bool = True


class CandidateStageOutput(BaseModel):
    day_number: Optional[int] = None
    time_block: Optional[str] = None
    candidates: List[CandidateOption]
    agent_selected_candidate_id: str
    agent_selection_rationale: str
    confirmation: Optional[CandidateConfirmation] = None
    order: Optional[SimulatedOrder] = None


class CandidateChatTurn(BaseModel):
    """One turn of the real-user chat conversation for a transaction-candidate
    stage (transportation/accommodation/activities)."""
    reply_message: str
    candidates: List[CandidateOption]
    agent_selected_candidate_id: str
    agent_selection_rationale: str


# ---------------------------------------------------------------------------
# In-trip guide (Layer 4)
# ---------------------------------------------------------------------------
class GuideSection(BaseModel):
    day_number: int
    tips: List[str]
    emergency_info: str
    local_phrases: List[str]


class InTripGuideOutput(BaseModel):
    guide_sections: List[GuideSection]


# ---------------------------------------------------------------------------
# Disruption & dynamic replanning bridge (Layer 2 dynamic scheduling + Layer 5)
# ---------------------------------------------------------------------------
class DisruptionEvent(BaseModel):
    id: str
    day_number: int
    affected_time_block: str
    disruption_type: Literal["weather", "venue_closure", "other"]
    description: str
    is_simulated: bool = True


class ReplanningConfirmation(BaseModel):
    status: Literal["confirmed", "declined"]
    feedback: str


class ReplanningOutput(BaseModel):
    trigger: DisruptionEvent
    revised_day: ItineraryDay
    change_summary: str
    concierge_notification: str
    confirmation: Optional[ReplanningConfirmation] = None


# ---------------------------------------------------------------------------
# Review & sharing (first-party feedback loop)
# ---------------------------------------------------------------------------
class CategoryRating(BaseModel):
    category: str
    rating: float


class ReviewOutput(BaseModel):
    overall_rating: float
    category_ratings: List[CategoryRating]
    review_text: str
    would_recommend: bool
    persona_alignment_notes: str
    share_caption: str


# ---------------------------------------------------------------------------
# Trip log — the shared artifact flowing through the pipeline into the dashboard
# ---------------------------------------------------------------------------
class StageResults(BaseModel):
    inspiration: Optional[InspirationOutput] = None
    itinerary: Optional[ItineraryOutput] = None
    transportation: Optional[CandidateStageOutput] = None
    accommodation: Optional[CandidateStageOutput] = None
    dining: Optional[CandidateStageOutput] = None
    attractions: Optional[CandidateStageOutput] = None
    activities: Optional[CandidateStageOutput] = None
    shopping: Optional[CandidateStageOutput] = None
    in_trip_guide: Optional[InTripGuideOutput] = None
    review: Optional[ReviewOutput] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class CalendarSyncResult(BaseModel):
    synced_at: str
    calendar_event_count: int
    calendar_id: str
    event_links: List[str] = []


class TripLog(BaseModel):
    run_id: str
    created_at: str
    persona: Persona
    run_config: RunConfig
    stages: StageResults = StageResults()
    disruption_event: Optional[DisruptionEvent] = None
    replanning: Optional[ReplanningOutput] = None
    hitl_log: List[HITLLogEntry] = []
    metrics: RunMetrics = RunMetrics()
    chat_transcripts: Dict[str, List[ChatMessage]] = {}
    calendar_sync: Optional[CalendarSyncResult] = None


# ---------------------------------------------------------------------------
# Context-summarization helpers shared by agents that need to ground a call
# in earlier stages (itinerary regeneration, in-trip guide, replanning,
# final review).
# ---------------------------------------------------------------------------
_CANDIDATE_STAGE_NAMES = [
    "transportation",
    "accommodation",
    "dining",
    "attractions",
    "activities",
    "shopping",
]


def summarize_itinerary(itinerary: ItineraryOutput) -> str:
    lines = []
    for day in itinerary.days:
        lines.append(f"第 {day.day_number} 天：")
        for block in day.blocks:
            lines.append(f"  {block.time_block} {block.theme}（{block.location_hint}）")
    return "\n".join(lines)


def summarize_stage_results(stages: StageResults) -> str:
    lines = []
    if stages.itinerary:
        lines.append(summarize_itinerary(stages.itinerary))
    for name in _CANDIDATE_STAGE_NAMES:
        stage_output: Optional[CandidateStageOutput] = getattr(stages, name)
        if not stage_output:
            continue
        chosen_id = (
            stage_output.confirmation.final_candidate_id
            if stage_output.confirmation
            else stage_output.agent_selected_candidate_id
        )
        chosen = next((c for c in stage_output.candidates if c.id == chosen_id), None)
        if chosen:
            lines.append(f"{name}：已選定 {chosen.name}（{chosen.vendor}）")
    return "\n".join(lines)
