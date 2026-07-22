"""Offline regression test for the streaming generator plumbing added to
agents/base_agent.py, agents/planning_agent.py, chat_session.py — no real
API calls. Monkeypatches the three llm_client call_* functions (as imported
into agents.base_agent's namespace) to return canned (result, CallMetrics)
tuples instantly, then drains the streaming generators and asserts:
  - the event sequence (labels, order) matches the expected branch,
  - self.last_call_metrics ends up with the right count,
  - the accumulated cost matches the sum of the canned per-call costs
    (this is the specific check that catches a future edit accidentally
    dropping an `accumulated += ...` line or the final
    `self.last_call_metrics = accumulated` reassignment).

Run with: python tests/test_streaming_generators.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.base_agent as base_agent  # noqa: E402
import agents.planning_agent as planning_agent_module  # noqa: E402
from agents.planning_agent import PlanningAgent  # noqa: E402
from chat_session import ChatSession  # noqa: E402
from local_settings import LocalSettings  # noqa: E402
from persona import Persona  # noqa: E402
from schemas import (  # noqa: E402
    CallMetrics,
    IntakeDecision,
    ItineraryDay,
    PlanBundleSynthesis,
    RagSelection,
    RunConfig,
    StageMetrics,
    TripPlanOption,
)
from site_index import SiteArticle, SiteIndex  # noqa: E402

FAILURES = []


def check(label, condition):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        FAILURES.append(label)


def _metrics(cost_usd, call_type="structured"):
    return CallMetrics(
        stage="itinerary", call_type=call_type, model="fake-model",
        input_tokens=100, output_tokens=50, web_search_requests=1 if call_type == "web_search" else 0,
        duration_ms=10.0, cost_usd=cost_usd,
    )


def _fake_option(option_id, label):
    return TripPlanOption(
        option_id=option_id, label=label, is_agent_recommended=(label == "A"), why_recommended="因為好",
        days=[ItineraryDay(day_number=1, blocks=[])], transportation=[], accommodation=[], activities=[],
        primary_sources=[], corroboration_sources=[],
    )


def install_fakes(intake_decision, rag_selection=None, synth_cost=0.5):
    """Patches agents.base_agent's call_structured/call_with_web_search/
    call_with_web_fetch (module-level names, since base_agent calls them
    bare) to return canned results with distinct, known costs."""
    call_log = []

    def fake_call_structured(output_format, **kwargs):
        if output_format is IntakeDecision:
            call_log.append("intake")
            return intake_decision, _metrics(0.01)
        if output_format is RagSelection:
            call_log.append("rag_select")
            return rag_selection, _metrics(0.001)
        if output_format is PlanBundleSynthesis:
            call_log.append("synth")
            synthesis = PlanBundleSynthesis(
                reply_message="這是兩個方案", options=[_fake_option("A", "A"), _fake_option("B", "B")],
                agent_recommended_option_id="A",
            )
            return synthesis, _metrics(synth_cost)
        raise AssertionError(f"unexpected output_format {output_format}")

    def fake_call_with_web_fetch(**kwargs):
        call_log.append("fetch")
        return SimpleNamespace(content="fake fetch content"), _metrics(0.02, call_type="web_fetch")

    def fake_call_with_web_search(**kwargs):
        call_log.append("search")
        return SimpleNamespace(content="fake search content"), _metrics(0.03, call_type="web_search")

    base_agent.call_structured = fake_call_structured
    base_agent.call_with_web_fetch = fake_call_with_web_fetch
    base_agent.call_with_web_search = fake_call_with_web_search
    # agents/planning_agent.py imports call_structured into its OWN module
    # namespace separately (used directly by _select_from_rag_streaming, not
    # via a StageAgent method) — must be patched independently.
    planning_agent_module.call_structured = fake_call_structured
    return call_log


def drain(gen):
    events = []
    result = None
    for event in gen:
        if event["type"] == "result":
            result = event["value"]
        else:
            events.append(event)
    return events, result


def run_config():
    return RunConfig(site_mode="unrestricted", model="fake-model")


def persona():
    return Persona(home_location="台北", destination_location="礁溪")


def site_index_with_articles():
    return SiteIndex(
        source_sitemap="https://example.com/sitemap.xml", built_at="2026-01-01T00:00:00Z",
        articles=[SiteArticle(url="https://example.com/a", title="A 文章")],
    )


print("== generate_plan_bundle_streaming: RAG-hit branch (5 steps) ==")
install_fakes(
    IntakeDecision(
        needs_clarification=False, reply_message="", style_summary="悠閒", home_location="台北",
        destination_location="礁溪", trip_length_type="one_day", party_size=2,
    ),
    rag_selection=RagSelection(selected_urls=["https://example.com/a"], selection_rationale="相關"),
)
agent = PlanningAgent("fake-model")
events, result = drain(agent.generate_plan_bundle_streaming(
    persona(), [], "我想去礁溪玩", run_config(), site_index_with_articles(), rag_top_n=3, round_number=0, max_rounds=3,
))
labels = [e["label"] for e in events if e["type"] == "step_started"]
check("5 step_started events in order", labels == ["了解你的需求", "從常用資料庫挑選文章", "抓取主軸文章全文", "搜尋真實交通/住宿/活動資訊", "整合成兩個完整方案"])
check("matching step_completed count", len([e for e in events if e["type"] == "step_completed"]) == 5)
check("all step_completed carry stage='itinerary'", all(e["metrics"]["stage"] == "itinerary" for e in events if e["type"] == "step_completed"))
check("result unpacks to (turn, history, persona)", result is not None and len(result) == 3)
turn, history, resolved_persona = result
check("turn.plan_ready is True", turn.plan_ready is True)
check("resolved_persona is not None", resolved_persona is not None)
check("agent.last_call_metrics has 5 entries", len(agent.last_call_metrics) == 5)
expected_total = 0.01 + 0.001 + 0.02 + 0.03 + 0.5
actual_total = StageMetrics.from_calls("itinerary", agent.last_call_metrics).total_cost_usd
check(f"total_cost_usd == {expected_total} (got {actual_total})", abs(actual_total - expected_total) < 1e-9)

print("\n== generate_plan_bundle_streaming: RAG-miss branch (3 steps, no site_index) ==")
install_fakes(
    IntakeDecision(
        needs_clarification=False, reply_message="", style_summary="悠閒", home_location="台北",
        destination_location="礁溪", trip_length_type="one_day", party_size=2,
    ),
)
agent2 = PlanningAgent("fake-model")
events2, result2 = drain(agent2.generate_plan_bundle_streaming(
    persona(), [], "我想去礁溪玩", run_config(), None, rag_top_n=3, round_number=0, max_rounds=3,
))
labels2 = [e["label"] for e in events2 if e["type"] == "step_started"]
check("3 step_started events in order", labels2 == ["了解你的需求", "搜尋行程內容與交通/住宿/活動資訊", "整合成兩個完整方案"])
check("agent2.last_call_metrics has 3 entries", len(agent2.last_call_metrics) == 3)
expected_total2 = 0.01 + 0.03 + 0.5
actual_total2 = StageMetrics.from_calls("itinerary", agent2.last_call_metrics).total_cost_usd
check(f"total_cost_usd == {expected_total2} (got {actual_total2})", abs(actual_total2 - expected_total2) < 1e-9)

print("\n== generate_plan_bundle_streaming: needs-clarification branch (1 step) ==")
install_fakes(
    IntakeDecision(needs_clarification=True, reply_message="請問你想從哪裡出發？", destination_location="礁溪", home_location=None),
)
agent3 = PlanningAgent("fake-model")
events3, result3 = drain(agent3.generate_plan_bundle_streaming(
    persona(), [], "我想去礁溪玩", run_config(), None, rag_top_n=3, round_number=0, max_rounds=3,
))
labels3 = [e["label"] for e in events3 if e["type"] == "step_started"]
check("1 step_started event", labels3 == ["了解你的需求"])
check("result[2] (persona) is None", result3[2] is None)
check("turn.needs_clarification is True", result3[0].needs_clarification is True)

print("\n== chat_session.send_message_streaming relays events + builds message_done ==")
install_fakes(
    IntakeDecision(
        needs_clarification=False, reply_message="", style_summary="悠閒", home_location="台北",
        destination_location="礁溪", trip_length_type="one_day", party_size=2,
    ),
)
session = ChatSession("offline_test2", run_config(), Path("/tmp/offline_stream_test"), local_settings=LocalSettings(enable_tail_pipeline=False))
events4 = list(session.send_message_streaming("我想去礁溪玩，兩個人"))
outer_labels = [e["label"] for e in events4 if e["type"] == "step_started"]
check("outer relay preserves 3-step RAG-miss sequence", outer_labels == ["了解你的需求", "搜尋行程內容與交通/住宿/活動資訊", "整合成兩個完整方案"])
check("every step event carries stage='itinerary'", all(e.get("stage") == "itinerary" for e in events4 if e["type"] in ("step_started", "step_completed")))
message_done = [e for e in events4 if e["type"] == "message_done"]
check("exactly one message_done event", len(message_done) == 1)
data = message_done[0]["data"]
check("message_done.data has the 6 expected keys", set(data.keys()) == {"type", "phase", "reply_message", "proposal", "metrics", "totals"})
check("session.trip_log.metrics.total_cost_usd matches", abs(session.trip_log.metrics.total_cost_usd - (0.01 + 0.03 + 0.5)) < 1e-9)
check("session.plan_ready is True after a ready turn", session.plan_ready is True)

if FAILURES:
    print(f"\n{len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll streaming generator checks passed.")
