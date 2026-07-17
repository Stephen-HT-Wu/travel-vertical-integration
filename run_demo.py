"""CLI entrypoint: runs the full pipeline for one persona and produces both
output/trip_log.json and output/dashboard.html."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import TripOrchestrator
from persona import add_persona_cli_args, build_persona_from_args
from schemas import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="旅遊產業垂直整合 Agent Demo")
    add_persona_cli_args(parser)
    parser.add_argument(
        "--site-mode",
        choices=["unrestricted", "allowlist"],
        default="unrestricted",
        help="真實搜尋階段（靈感/餐飲/景點/購物）是否限定在可信網域清單內",
    )
    parser.add_argument(
        "--site-list",
        default="config/trusted_domains.json",
        help="--site-mode allowlist 時使用的網域清單檔案（JSON，含 'domains' 陣列）",
    )
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--language", default="zh-TW")
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def load_allowed_domains(site_list_path: str) -> list:
    data = json.loads(Path(site_list_path).read_text(encoding="utf-8"))
    return data["domains"]


def main() -> None:
    load_dotenv()
    args = parse_args()
    persona = build_persona_from_args(args)

    allowed_domains = load_allowed_domains(args.site_list) if args.site_mode == "allowlist" else []
    run_config = RunConfig(
        site_mode=args.site_mode,
        allowed_domains=allowed_domains,
        model=args.model,
        language=args.language,
    )

    output_dir = Path(args.output_dir)
    print(f"開始規劃行程 — {persona.summary_zh()}")
    print(f"搜尋模式：{run_config.site_mode}" + (f"（{len(allowed_domains)} 個可信網域）" if allowed_domains else ""))

    orchestrator = TripOrchestrator(run_config, output_dir)
    trip_log = orchestrator.run(persona)

    trip_log_path = output_dir / "trip_log.json"
    print(f"\n行程紀錄已儲存：{trip_log_path}")

    from dashboard.render_dashboard import render_to_file

    dashboard_path = output_dir / "dashboard.html"
    render_to_file(trip_log, dashboard_path)
    print(f"Dashboard 已產生：{dashboard_path}")


if __name__ == "__main__":
    main()
