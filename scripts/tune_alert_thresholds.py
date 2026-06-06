from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import common  # noqa: F401

from src.alert_evaluation import FramePredictionRecord
from src.alert_tuning import recommend_best_setting, sweep_alert_thresholds


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep alert thresholds against saved frame predictions.")
    parser.add_argument("--frame-predictions", default="reports/alert_frame_predictions.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--confidence-thresholds", default="0.5,0.6,0.65,0.7,0.75")
    parser.add_argument("--rolling-window-sizes", default="6")
    parser.add_argument("--min-positive-frames", default="2,3,4")
    parser.add_argument("--min-detection-duration-seconds", default="1.0,2.0,3.0")
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    return parser


def _load_records(path: Path) -> list[FramePredictionRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        records.append(
            FramePredictionRecord(
                clip_id=str(row["clip_id"]),
                subset=str(row.get("subset", "unspecified")),
                timestamp_seconds=float(row["timestamp_seconds"]),
                has_human_gt=str(row["has_human_gt"]).strip().lower() in {"1", "true", "yes"},
                detected_confidence=float(row["detected_confidence"]),
                detection_count=int(row["detection_count"]),
                image_name=str(row.get("image_name", "")),
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    frame_predictions_path = Path(args.frame_predictions)
    if not frame_predictions_path.exists():
        raise FileNotFoundError(frame_predictions_path)

    records = _load_records(frame_predictions_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = sweep_alert_thresholds(
        records,
        confidence_thresholds=_parse_float_list(args.confidence_thresholds),
        rolling_window_sizes=_parse_int_list(args.rolling_window_sizes),
        min_positive_frames_options=_parse_int_list(args.min_positive_frames),
        min_detection_duration_seconds_options=_parse_float_list(args.min_detection_duration_seconds),
        cooldown_seconds=args.cooldown_seconds,
    )
    best = recommend_best_setting(results)

    sweep_path = output_dir / "alert_threshold_sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    report_path = output_dir / "alert_threshold_recommendations.md"
    report_path.write_text(
        "\n".join(
            [
                "# Alert Threshold Recommendations",
                "",
                "## Recommended setting",
                "",
                f"- confidence threshold: `{best.confidence_threshold}`",
                f"- rolling window size: `{best.rolling_window_size}`",
                f"- min positive frames: `{best.min_positive_frames}`",
                f"- min detection duration seconds: `{best.min_detection_duration_seconds}`",
                f"- cooldown seconds: `{best.cooldown_seconds}`",
                "",
                "## Result",
                "",
                f"- alert precision: `{best.alert_precision}`",
                f"- clip recall: `{best.clip_recall}`",
                f"- false positives per hour: `{best.false_positives_per_hour}`",
                f"- total alerts: `{best.total_alerts}`",
                f"- true alerts: `{best.true_alerts}`",
                f"- false alerts: `{best.false_alerts}`",
                "",
                "## Outputs",
                "",
                f"- sweep CSV: `{sweep_path}`",
            ]
        ),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
