"""Renders a trip_log.json into a single self-contained dashboard.html.

Usage:
    python dashboard/render_dashboard.py output/trip_log.json output/dashboard.html
"""
import sys
from pathlib import Path

# Allow running this file directly (`python dashboard/render_dashboard.py ...`)
# by putting the repo root — where schemas.py / stage_metadata.py live — on
# sys.path, since Python otherwise only adds this script's own directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from schemas import TripLog  # noqa: E402
from stage_metadata import STAGE_METADATA  # noqa: E402

TEMPLATE_DIR = Path(__file__).resolve().parent

CANDIDATE_STAGE_LABELS = [
    ("transportation", "交通"),
    ("accommodation", "住宿"),
    ("dining", "餐飲"),
    ("attractions", "景點"),
    ("activities", "活動"),
    ("shopping", "購物"),
]


def render(trip_log: TripLog) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("template.html.j2")
    candidate_stages = [
        (key, label, getattr(trip_log.stages, key)) for key, label in CANDIDATE_STAGE_LABELS
    ]
    # chat_transcripts is only ever populated by the chat webapp (chat_session.py) —
    # the CLI/auto-pipeline never writes it — so its presence reliably tells us
    # whether hitl_log entries came from a real user or from UserSimulatorAgent.
    who_label = "使用者" if trip_log.chat_transcripts else "虛擬使用者"
    return template.render(
        trip_log=trip_log,
        stage_metadata=STAGE_METADATA,
        candidate_stages=candidate_stages,
        who_label=who_label,
    )


def render_to_file(trip_log: TripLog, output_path: Path) -> None:
    html = render(trip_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python dashboard/render_dashboard.py <trip_log.json> <dashboard.html>")
        sys.exit(1)
    trip_log_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    trip_log = TripLog.model_validate_json(trip_log_path.read_text(encoding="utf-8"))
    render_to_file(trip_log, output_path)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
