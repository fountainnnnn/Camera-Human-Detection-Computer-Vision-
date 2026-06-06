from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import common  # noqa: F401

from src.runtime_validation import (
    RuntimeThresholds,
    read_status_sample,
    summarize_runtime_samples,
    evaluate_runtime_summary,
    write_runtime_samples_csv,
    write_runtime_summary_json,
    write_runtime_summary_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor Security AI runtime status and write evidence reports.")
    parser.add_argument("--status-path", default="logs/status.json")
    parser.add_argument("--output-dir", default="reports/runtime_monitor")
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-average-cpu-percent", type=float, default=None)
    parser.add_argument("--max-peak-cpu-percent", type=float, default=None)
    parser.add_argument("--max-ram-percent", type=float, default=None)
    parser.add_argument("--min-average-fps", type=float, default=None)
    parser.add_argument("--max-disconnects", type=int, default=None)
    parser.add_argument("--max-stale-samples", type=int, default=None)
    parser.add_argument("--min-runtime-seconds", type=float, default=None)
    parser.add_argument("--require-zero-exit", action="store_true")
    return parser


def collect_runtime_samples(
    status_path: str | Path,
    duration_seconds: float,
    poll_interval_seconds: float,
) -> tuple[list, float]:
    path = Path(status_path)
    started = time.monotonic()
    samples = []

    while True:
        elapsed = time.monotonic() - started
        if elapsed >= duration_seconds:
            break

        if path.exists():
            observed_at = datetime.now(timezone.utc).isoformat()
            try:
                samples.append(read_status_sample(path, observed_at=observed_at))
            except Exception:  # noqa: BLE001
                pass

        remaining = duration_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(poll_interval_seconds, remaining))

    return samples, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, elapsed_seconds = collect_runtime_samples(
        status_path=args.status_path,
        duration_seconds=args.duration_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )

    if not samples:
        message = (
            "# Runtime Monitor Summary\n\n"
            "Overall: FAIL\n\n"
            "No runtime samples were collected from the configured status path.\n"
        )
        (output_dir / "runtime_monitor_summary.md").write_text(message, encoding="utf-8")
        (output_dir / "runtime_monitor_summary.json").write_text(
            '{"summary": null, "evaluation": {"passed": false, "checks": []}, "thresholds": {}}',
            encoding="utf-8",
        )
        return 1

    thresholds = RuntimeThresholds(
        max_average_cpu_percent=args.max_average_cpu_percent,
        max_peak_cpu_percent=args.max_peak_cpu_percent,
        max_ram_percent=args.max_ram_percent,
        min_average_fps=args.min_average_fps,
        max_disconnects=args.max_disconnects,
        max_stale_samples=args.max_stale_samples,
        min_runtime_seconds=args.min_runtime_seconds,
        require_zero_exit=args.require_zero_exit,
    )
    summary = summarize_runtime_samples(samples, elapsed_seconds=elapsed_seconds, exit_code=None)
    evaluation = evaluate_runtime_summary(summary, thresholds)

    write_runtime_samples_csv(output_dir / "runtime_monitor_samples.csv", samples)
    write_runtime_summary_json(output_dir / "runtime_monitor_summary.json", summary, evaluation, thresholds)
    write_runtime_summary_markdown(output_dir / "runtime_monitor_summary.md", summary, evaluation, thresholds)

    return 0 if evaluation.passed else 1


if __name__ == "__main__":
    sys.exit(main())
