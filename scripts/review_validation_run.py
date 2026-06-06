from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import common  # noqa: F401


def run_analysis(reports_root: str | Path) -> int:
    module = importlib.import_module("scripts.analyze_alert_failures")
    return module.main(
        [
            "--clip-results",
            str(Path(reports_root) / "alert_clip_results.csv"),
            "--frame-predictions",
            str(Path(reports_root) / "alert_frame_predictions.csv"),
            "--output-dir",
            str(reports_root),
        ]
    )


def run_tuning(reports_root: str | Path) -> int:
    module = importlib.import_module("scripts.tune_alert_thresholds")
    return module.main(
        [
            "--frame-predictions",
            str(Path(reports_root) / "alert_frame_predictions.csv"),
            "--output-dir",
            str(reports_root),
        ]
    )


def run_acceptance(reports_root: str | Path, runtime_summary: str | Path) -> int:
    module = importlib.import_module("scripts.build_acceptance_report")
    return module.main(
        [
            "--clip-results",
            str(Path(reports_root) / "alert_clip_results.csv"),
            "--runtime-summary",
            str(runtime_summary),
            "--output-dir",
            str(reports_root),
        ]
    )


def _write_index(reports_root: Path) -> Path:
    index_path = reports_root / "validation_review_index.md"
    index_path.write_text(
        "\n".join(
            [
                "# Validation Review Index",
                "",
                "## Primary outputs",
                "",
                "- `alert_failure_analysis.md`",
                "- `false_positive_frames.csv`",
                "- `missed_positive_clips.csv`",
                "- `alert_threshold_sweep.csv`",
                "- `alert_threshold_recommendations.md`",
                "- `acceptance_report.md`",
                "- `acceptance_report.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return index_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-validation review steps over existing reports.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--runtime-summary", default="reports/runtime_monitor/runtime_monitor_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reports_root = Path(args.reports_dir)
    runtime_summary = Path(args.runtime_summary)

    for required in [
        reports_root / "alert_clip_results.csv",
        reports_root / "alert_frame_predictions.csv",
        runtime_summary,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    for runner in (
        lambda: run_analysis(reports_root),
        lambda: run_tuning(reports_root),
        lambda: run_acceptance(reports_root, runtime_summary),
    ):
        exit_code = runner()
        if exit_code != 0:
            return exit_code

    _write_index(reports_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
