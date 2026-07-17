"""Structural sanity check on a generated trip_log.json — not a full test
suite, just enough to catch an obviously broken run.

Usage:
    python tests/test_sanity.py output/trip_log.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import TripLog  # noqa: E402
from stage_metadata import REAL_SEARCH_STAGES, SIMULATED_CANDIDATE_STAGES  # noqa: E402

CANDIDATE_STAGE_NAMES = ["transportation", "accommodation", "dining", "attractions", "activities", "shopping"]


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)


def run_checks(trip_log: TripLog) -> list:
    failures: list = []

    check(trip_log.stages.inspiration is not None, "inspiration stage missing", failures)
    if trip_log.stages.inspiration:
        check(
            len(trip_log.stages.inspiration.destination_options) > 0,
            "inspiration produced no destination options",
            failures,
        )
        for opt in trip_log.stages.inspiration.destination_options:
            check(bool(opt.source_url), f"inspiration option {opt.id} has empty source_url", failures)

    check(trip_log.stages.itinerary is not None, "itinerary stage missing", failures)
    if trip_log.stages.itinerary:
        check(
            trip_log.stages.itinerary.confirmation is not None,
            "itinerary was never confirmed",
            failures,
        )

    for name in CANDIDATE_STAGE_NAMES:
        stage = getattr(trip_log.stages, name)
        check(stage is not None, f"{name} stage missing", failures)
        if not stage:
            continue
        check(len(stage.candidates) > 0, f"{name} produced no candidates", failures)
        check(stage.confirmation is not None, f"{name} was never confirmed", failures)
        expected_source = "real_search" if name in REAL_SEARCH_STAGES else "simulated"
        for c in stage.candidates:
            check(
                c.data_source == expected_source,
                f"{name} candidate {c.id} has data_source={c.data_source}, expected {expected_source}",
                failures,
            )
            if expected_source == "real_search":
                check(bool(c.source_url), f"{name} candidate {c.id} missing source_url", failures)

    check(trip_log.stages.in_trip_guide is not None, "in_trip_guide stage missing", failures)
    check(trip_log.disruption_event is not None, "disruption_event missing", failures)
    check(trip_log.replanning is not None, "replanning missing", failures)
    if trip_log.replanning:
        check(trip_log.replanning.confirmation is not None, "replanning was never confirmed", failures)

    check(trip_log.stages.review is not None, "review stage missing", failures)
    if trip_log.stages.review:
        check(
            1 <= trip_log.stages.review.overall_rating <= 5,
            f"overall_rating {trip_log.stages.review.overall_rating} out of [1,5]",
            failures,
        )
        check(bool(trip_log.stages.review.share_caption), "review has empty share_caption", failures)

    expected_checkpoint_count = 1 + len(CANDIDATE_STAGE_NAMES) + 1  # itinerary + 6 candidates + replanning
    check(
        len(trip_log.hitl_log) >= expected_checkpoint_count,
        f"hitl_log has {len(trip_log.hitl_log)} entries, expected >= {expected_checkpoint_count}",
        failures,
    )

    return failures


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python tests/test_sanity.py <trip_log.json>")
        sys.exit(1)
    trip_log = TripLog.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    failures = run_checks(trip_log)
    if failures:
        print(f"FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("OK — trip_log.json passed all structural sanity checks.")


if __name__ == "__main__":
    main()
