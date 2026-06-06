from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .alert_evaluation import FramePredictionRecord, evaluate_alert_predictions


@dataclass
class AlertTuningResult:
    confidence_threshold: float
    rolling_window_size: int
    min_positive_frames: int
    min_detection_duration_seconds: float
    cooldown_seconds: float
    total_alerts: int
    true_alerts: int
    false_alerts: int
    alert_precision: float
    clip_recall: float
    false_positives_per_hour: float
    positive_clips: int
    missed_positive_clips: int


def sweep_alert_thresholds(
    records: Iterable[FramePredictionRecord],
    *,
    confidence_thresholds: list[float],
    rolling_window_sizes: list[int],
    min_positive_frames_options: list[int],
    min_detection_duration_seconds_options: list[float],
    cooldown_seconds: float,
) -> list[AlertTuningResult]:
    record_list = list(records)
    results: list[AlertTuningResult] = []

    for confidence_threshold in confidence_thresholds:
        for rolling_window_size in rolling_window_sizes:
            for min_positive_frames in min_positive_frames_options:
                for min_duration in min_detection_duration_seconds_options:
                    _, summary = evaluate_alert_predictions(
                        record_list,
                        confidence_threshold=confidence_threshold,
                        rolling_window_size=rolling_window_size,
                        min_positive_frames=min_positive_frames,
                        min_detection_duration_seconds=min_duration,
                        cooldown_seconds=cooldown_seconds,
                    )
                    results.append(
                        AlertTuningResult(
                            confidence_threshold=confidence_threshold,
                            rolling_window_size=rolling_window_size,
                            min_positive_frames=min_positive_frames,
                            min_detection_duration_seconds=min_duration,
                            cooldown_seconds=cooldown_seconds,
                            total_alerts=summary.total_alerts,
                            true_alerts=summary.true_alerts,
                            false_alerts=summary.false_alerts,
                            alert_precision=round(summary.alert_precision, 4),
                            clip_recall=round(summary.clip_recall, 4),
                            false_positives_per_hour=round(summary.false_positives_per_hour, 4),
                            positive_clips=summary.positive_clips,
                            missed_positive_clips=summary.missed_positive_clips,
                        )
                    )
    return results


def recommend_best_setting(results: list[AlertTuningResult]) -> AlertTuningResult:
    if not results:
        raise ValueError("At least one tuning result is required")

    return sorted(
        results,
        key=lambda item: (
            -item.alert_precision,
            -item.clip_recall,
            item.false_positives_per_hour,
            -item.true_alerts,
            item.confidence_threshold,
            item.min_positive_frames,
            item.min_detection_duration_seconds,
        ),
    )[0]
