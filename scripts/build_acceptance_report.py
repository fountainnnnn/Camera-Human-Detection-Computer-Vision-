from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import common  # noqa: F401

from src.acceptance import (
    AcceptanceThresholds,
    build_acceptance_report,
    write_acceptance_report_json,
    write_acceptance_report_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an acceptance and tuning report from validation and runtime evidence.")
    parser.add_argument("--clip-results", default="reports/alert_clip_results.csv")
    parser.add_argument("--runtime-summary", default="reports/runtime_monitor/runtime_monitor_summary.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--min-alert-precision", type=float, default=0.9)
    parser.add_argument("--min-clip-recall", type=float, default=0.8)
    parser.add_argument("--max-false-positives-per-hour", type=float, default=0.5)
    parser.add_argument("--max-average-cpu-percent", type=float, default=80.0)
    parser.add_argument("--max-peak-cpu-percent", type=float, default=95.0)
    parser.add_argument("--min-average-fps", type=float, default=1.0)
    parser.add_argument("--max-disconnects", type=int, default=0)
    parser.add_argument("--max-stale-samples", type=int, default=2)
    parser.add_argument("--min-runtime-seconds", type=float, default=1800.0)
    parser.add_argument("--allow-missing-runtime-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    clip_results_path = Path(args.clip_results)
    clip_rows = []
    if clip_results_path.exists():
        with clip_results_path.open("r", encoding="utf-8", newline="") as handle:
            clip_rows = list(csv.DictReader(handle))

    runtime_payload = None
    runtime_summary_path = Path(args.runtime_summary)
    if runtime_summary_path.exists():
        runtime_payload = json.loads(runtime_summary_path.read_text(encoding="utf-8"))

    thresholds = AcceptanceThresholds(
        min_alert_precision=args.min_alert_precision,
        min_clip_recall=args.min_clip_recall,
        max_false_positives_per_hour=args.max_false_positives_per_hour,
        max_average_cpu_percent=args.max_average_cpu_percent,
        max_peak_cpu_percent=args.max_peak_cpu_percent,
        min_average_fps=args.min_average_fps,
        max_disconnects=args.max_disconnects,
        max_stale_samples=args.max_stale_samples,
        min_runtime_seconds=args.min_runtime_seconds,
        require_runtime_monitor=not args.allow_missing_runtime_summary,
    )
    report = build_acceptance_report(
        clip_rows=clip_rows,
        runtime_monitor_payload=runtime_payload,
        thresholds=thresholds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_acceptance_report_json(output_dir / "acceptance_report.json", report, thresholds)
    write_acceptance_report_markdown(output_dir / "acceptance_report.md", report, thresholds)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
