from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
import common  # noqa: F401

from src.alert_evaluation import FramePredictionRecord, evaluate_alert_predictions
from src.config import load_config
from src.detector import YOLOPersonDetector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate alert-level behavior on extracted frames plus a manifest.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-path", help="Optional model checkpoint override for this evaluation run.")
    parser.add_argument("--confidence-threshold", type=float, help="Optional confidence threshold override.")
    args = parser.parse_args(argv)

    image_dir = Path(args.images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    required = {"image", "clip_id", "timestamp_seconds", "has_human_gt"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if "subset" not in manifest.columns:
        manifest["subset"] = "unspecified"

    config = load_config(args.config)
    if args.model_path:
        config.model.path = args.model_path
    if args.confidence_threshold is not None:
        config.model.confidence_threshold = args.confidence_threshold
    detector = YOLOPersonDetector(config)
    records: list[FramePredictionRecord] = []

    for row in manifest.to_dict(orient="records"):
        image_path = image_dir / str(row["image"])
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        detections = detector.detect(frame)
        detected_confidence = max((item.confidence for item in detections), default=0.0)
        records.append(
            FramePredictionRecord(
                clip_id=str(row["clip_id"]),
                subset=str(row.get("subset", "unspecified")),
                timestamp_seconds=float(row["timestamp_seconds"]),
                has_human_gt=bool(row["has_human_gt"]),
                detected_confidence=float(detected_confidence),
                detection_count=len(detections),
                image_name=str(row["image"]),
            )
        )

    clip_results, summary = evaluate_alert_predictions(
        records,
        confidence_threshold=config.model.confidence_threshold,
        rolling_window_size=config.detection.rolling_window_size,
        min_positive_frames=config.detection.min_positive_frames,
        min_detection_duration_seconds=config.detection.min_detection_duration_seconds,
        cooldown_seconds=config.detection.alert_cooldown_seconds,
    )

    clip_df = pd.DataFrame([result.__dict__ for result in clip_results])
    clip_path = output_dir / "alert_clip_results.csv"
    clip_df.to_csv(clip_path, index=False)
    frame_predictions_path = output_dir / "alert_frame_predictions.csv"
    with frame_predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "subset",
                "timestamp_seconds",
                "has_human_gt",
                "detected_confidence",
                "detection_count",
                "image_name",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    subset_path = output_dir / "alert_subset_metrics.csv"
    if not clip_df.empty:
        grouped = (
            clip_df.groupby("subset", dropna=False)[
                ["alert_count", "true_alert_count", "false_alert_count", "had_human_gt", "alerted_on_positive_clip", "missed_positive_clip"]
            ]
            .sum()
            .reset_index()
        )
        grouped.to_csv(subset_path, index=False)

    summary_path = output_dir / "alert_evaluation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Alert Evaluation Summary",
                "",
                f"- clips: {len(clip_results)}",
                f"- total alerts: {summary.total_alerts}",
                f"- true alerts: {summary.true_alerts}",
                f"- false alerts: {summary.false_alerts}",
                f"- alert precision: {summary.alert_precision:.4f}",
                f"- positive clips: {summary.positive_clips}",
                f"- alerted positive clips: {summary.alerted_positive_clips}",
                f"- missed positive clips: {summary.missed_positive_clips}",
                f"- clip recall: {summary.clip_recall:.4f}",
                f"- false positives per hour: {summary.false_positives_per_hour:.4f}",
                f"- average detection latency seconds: {summary.average_detection_latency_seconds if summary.average_detection_latency_seconds is not None else 'n/a'}",
                "- clip results: reports/alert_clip_results.csv",
                "- frame predictions: reports/alert_frame_predictions.csv",
                "- subset metrics: reports/alert_subset_metrics.csv",
            ]
        ),
        encoding="utf-8",
    )

    print(f"summary={summary_path}")
    print(f"clip_results={clip_path}")
    print(f"frame_predictions={frame_predictions_path}")
    if not clip_df.empty:
        print(f"subset_metrics={subset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
