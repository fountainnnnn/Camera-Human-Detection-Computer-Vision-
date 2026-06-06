from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import common  # noqa: F401


def run_review_bundle(reports_dir: str | Path, runtime_summary: str | Path) -> int:
    from scripts.review_validation_run import main as review_main

    return review_main(
        [
            "--reports-dir",
            str(reports_dir),
            "--runtime-summary",
            str(runtime_summary),
        ]
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_demo_inputs(output_dir: Path) -> Path:
    _write_csv(
        output_dir / "alert_clip_results.csv",
        [
            "clip_id",
            "subset",
            "alert_count",
            "true_alert_count",
            "false_alert_count",
            "had_human_gt",
            "alerted_on_positive_clip",
            "missed_positive_clip",
            "false_positive_clip",
            "duration_seconds",
        ],
        [
            {
                "clip_id": "demo-day-positive",
                "subset": "day",
                "alert_count": 1,
                "true_alert_count": 1,
                "false_alert_count": 0,
                "had_human_gt": True,
                "alerted_on_positive_clip": True,
                "missed_positive_clip": False,
                "false_positive_clip": False,
                "duration_seconds": 20,
            },
            {
                "clip_id": "demo-night-fp",
                "subset": "night_vision",
                "alert_count": 1,
                "true_alert_count": 0,
                "false_alert_count": 1,
                "had_human_gt": False,
                "alerted_on_positive_clip": False,
                "missed_positive_clip": False,
                "false_positive_clip": True,
                "duration_seconds": 120,
            },
        ],
    )
    _write_csv(
        output_dir / "alert_frame_predictions.csv",
        [
            "clip_id",
            "subset",
            "timestamp_seconds",
            "has_human_gt",
            "detected_confidence",
            "detection_count",
            "image_name",
        ],
        [
            {
                "clip_id": "demo-day-positive",
                "subset": "day",
                "timestamp_seconds": 0.0,
                "has_human_gt": True,
                "detected_confidence": 0.82,
                "detection_count": 1,
                "image_name": "demo-day-positive-001.jpg",
            },
            {
                "clip_id": "demo-day-positive",
                "subset": "day",
                "timestamp_seconds": 1.0,
                "has_human_gt": True,
                "detected_confidence": 0.86,
                "detection_count": 1,
                "image_name": "demo-day-positive-002.jpg",
            },
            {
                "clip_id": "demo-night-fp",
                "subset": "night_vision",
                "timestamp_seconds": 0.0,
                "has_human_gt": False,
                "detected_confidence": 0.74,
                "detection_count": 1,
                "image_name": "demo-night-fp-001.jpg",
            },
        ],
    )
    runtime_dir = output_dir / "runtime_monitor"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_summary = runtime_dir / "runtime_monitor_summary.json"
    runtime_summary.write_text(
        json.dumps(
            {
                "summary": {
                    "average_cpu_percent": 42.0,
                    "peak_cpu_percent": 57.0,
                    "average_fps": 2.0,
                    "disconnect_count": 0,
                    "stale_sample_count": 0,
                    "elapsed_seconds": 1800.0,
                },
                "evaluation": {"passed": True, "checks": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return runtime_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic validation artifacts and run the review bundle.")
    parser.add_argument("--output-dir", default="reports/demo_validation_review")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_summary = _write_demo_inputs(output_dir)
    return run_review_bundle(output_dir, runtime_summary)


if __name__ == "__main__":
    sys.exit(main())
