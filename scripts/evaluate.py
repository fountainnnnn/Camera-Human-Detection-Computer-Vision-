from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import common  # noqa: F401

from src.config import load_config
from src.detector import YOLOPersonDetector
from src.evaluation import Counts, match_detections, precision_recall_f1, read_yolo_labels
from src.utils import annotate_frame


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ThresholdArtifacts:
    rows: list[dict]
    counts: Counts
    matched_confidences: list[float]
    false_positive_confidences: list[float]


def load_evaluation_config(config_path: str, model_path: str | None = None):
    config = load_config(config_path)
    if model_path:
        config.model.path = model_path
    return config


def parse_thresholds(raw: str | None) -> list[float]:
    if not raw:
        return [0.5, 0.6, 0.65, 0.7, 0.75]
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds must include at least one numeric value")
    for threshold in thresholds:
        if threshold < 0 or threshold > 1:
            raise ValueError("--thresholds values must be between 0 and 1")
    return thresholds


def _draw_ground_truth(frame, gt_boxes: list[tuple[int, int, int, int]]):
    annotated = frame.copy()
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return annotated


def evaluate_threshold(
    image_paths: list[Path],
    labels_dir: Path,
    detector: YOLOPersonDetector,
    threshold: float,
    output_dir: Path,
    camera_name: str,
    image_metadata: dict[str, dict[str, str]] | None = None,
) -> ThresholdArtifacts:
    original = detector.config.model.confidence_threshold
    detector.config.model.confidence_threshold = threshold
    counts = Counts()
    rows: list[dict] = []
    matched_confidences: list[float] = []
    false_positive_confidences: list[float] = []
    false_positive_dir = output_dir / "false_positives" / f"thr_{threshold:.2f}"
    false_negative_dir = output_dir / "false_negatives" / f"thr_{threshold:.2f}"
    false_positive_dir.mkdir(parents=True, exist_ok=True)
    false_negative_dir.mkdir(parents=True, exist_ok=True)

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            label_path = labels_dir / f"{image_path.stem}.txt"
            gt_boxes = read_yolo_labels(
                label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else [],
                width,
                height,
            )
            detections = detector.detect(frame)
            match = match_detections(gt_boxes, detections)
            counts.tp += match.tp
            counts.fp += match.fp
            counts.fn += match.fn
            matched_confidences.extend(match.matched_confidences)
            false_positive_confidences.extend(match.false_positive_confidences)

            rows.append(
                {
                    "image": image_path.name,
                    "subset": (image_metadata or {}).get(image_path.name, {}).get("subset", "unspecified"),
                    "clip_id": (image_metadata or {}).get(image_path.name, {}).get("clip_id", ""),
                    "threshold": threshold,
                    "gt_boxes": len(gt_boxes),
                    "detections": len(detections),
                    "tp": match.tp,
                    "fp": match.fp,
                    "fn": match.fn,
                    "max_confidence": round(max((item.confidence for item in detections), default=0.0), 4),
                }
            )

            if match.fp > 0:
                annotated = annotate_frame(frame, detections, camera_name, f"thr={threshold:.2f}")
                cv2.imwrite(str(false_positive_dir / image_path.name), annotated)
            if match.fn > 0:
                annotated = _draw_ground_truth(frame, gt_boxes)
                cv2.imwrite(str(false_negative_dir / image_path.name), annotated)
    finally:
        detector.config.model.confidence_threshold = original

    return ThresholdArtifacts(
        rows=rows,
        counts=counts,
        matched_confidences=matched_confidences,
        false_positive_confidences=false_positive_confidences,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate person detection on labeled images.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-path", help="Optional model checkpoint override for this evaluation run.")
    parser.add_argument(
        "--thresholds",
        help="Comma-separated confidence thresholds to sweep. Defaults to 0.5,0.6,0.65,0.7,0.75.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional CSV with per-image metadata. Supported columns: image, subset, clip_id.",
    )
    args = parser.parse_args()

    image_paths = [
        path
        for path in sorted(Path(args.images).glob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels_dir = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_metadata: dict[str, dict[str, str]] = {}
    if args.manifest:
        manifest_df = pd.read_csv(args.manifest)
        if "image" not in manifest_df.columns:
            raise ValueError("--manifest must include an 'image' column")
        for row in manifest_df.to_dict(orient="records"):
            image_metadata[str(row["image"])] = {
                "subset": str(row.get("subset", "unspecified")),
                "clip_id": str(row.get("clip_id", "")),
            }

    config = load_evaluation_config(args.config, args.model_path)
    detector = YOLOPersonDetector(config)
    thresholds = parse_thresholds(args.thresholds)
    rows = []
    per_image_rows: list[dict] = []
    confidence_rows: list[dict] = []

    for threshold in thresholds:
        artifacts = evaluate_threshold(
            image_paths=image_paths,
            labels_dir=labels_dir,
            detector=detector,
            threshold=threshold,
            output_dir=output_dir,
            camera_name=config.camera.name,
            image_metadata=image_metadata,
        )
        precision, recall, f1 = precision_recall_f1(artifacts.counts)
        rows.append(
            {
                "threshold": threshold,
                "tp": artifacts.counts.tp,
                "fp": artifacts.counts.fp,
                "fn": artifacts.counts.fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
        per_image_rows.extend(artifacts.rows)
        confidence_rows.extend(
            {"threshold": threshold, "kind": "matched", "confidence": round(value, 4)}
            for value in artifacts.matched_confidences
        )
        confidence_rows.extend(
            {"threshold": threshold, "kind": "false_positive", "confidence": round(value, 4)}
            for value in artifacts.false_positive_confidences
        )

    df = pd.DataFrame(rows)
    threshold_csv = output_dir / "threshold_sweep.csv"
    df.to_csv(threshold_csv, index=False)
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(output_dir / "per_image_results.csv", index=False)
    pd.DataFrame(confidence_rows).to_csv(output_dir / "confidence_distribution.csv", index=False)
    subset_metrics_path = output_dir / "subset_metrics.csv"
    if not per_image_df.empty:
        subset_summary = (
            per_image_df.groupby(["threshold", "subset"], dropna=False)[["tp", "fp", "fn"]]
            .sum()
            .reset_index()
        )
        subset_summary["precision"] = subset_summary["tp"] / (subset_summary["tp"] + subset_summary["fp"]).clip(lower=1)
        subset_summary["recall"] = subset_summary["tp"] / (subset_summary["tp"] + subset_summary["fn"]).clip(lower=1)
        subset_summary["f1"] = (
            2 * subset_summary["precision"] * subset_summary["recall"]
        ) / (subset_summary["precision"] + subset_summary["recall"]).clip(lower=1e-9)
        subset_summary.to_csv(subset_metrics_path, index=False)

    best = max(rows, key=lambda row: row["precision"])
    summary = output_dir / "evaluation_summary.md"
    subset_note = "- subset metrics: reports/subset_metrics.csv" if not per_image_df.empty else "- subset metrics: n/a"
    summary.write_text(
        "\n".join(
            [
                "# Evaluation Summary",
                "",
                f"- images: {len(image_paths)}",
                f"- best threshold: {best['threshold']}",
                f"- precision: {best['precision']}",
                f"- recall: {best['recall']}",
                f"- f1: {best['f1']}",
                f"- false positives: {best['fp']}",
                f"- false negatives: {best['fn']}",
                "- confusion matrix: person detections are counted box-wise as TP/FP/FN",
                "- false_positives_per_hour: n/a for still-image evaluation",
                "- false_negatives directory: reports/false_negatives/",
                "- false_positives directory: reports/false_positives/",
                "- confidence distribution: reports/confidence_distribution.csv",
                subset_note,
            ]
        ),
        encoding="utf-8",
    )

    print(f"summary={summary}")
    print(f"threshold_sweep={threshold_csv}")
    if not per_image_df.empty:
        print(f"subset_metrics={subset_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
