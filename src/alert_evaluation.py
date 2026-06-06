from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .alert_logic import TemporalAlertFilter


@dataclass
class FramePredictionRecord:
    clip_id: str
    subset: str
    timestamp_seconds: float
    has_human_gt: bool
    detected_confidence: float
    detection_count: int
    image_name: str = ""


@dataclass
class ClipAlertResult:
    clip_id: str
    subset: str
    alert_count: int
    true_alert_count: int
    false_alert_count: int
    had_human_gt: bool
    alerted_on_positive_clip: bool
    missed_positive_clip: bool
    false_positive_clip: bool
    first_human_timestamp_seconds: float | None
    first_alert_timestamp_seconds: float | None
    detection_latency_seconds: float | None
    duration_seconds: float


@dataclass
class AlertEvaluationSummary:
    total_alerts: int
    true_alerts: int
    false_alerts: int
    positive_clips: int
    alerted_positive_clips: int
    missed_positive_clips: int
    false_positive_clips: int
    negative_duration_hours: float
    false_positives_per_hour: float
    alert_precision: float
    clip_recall: float
    average_detection_latency_seconds: float | None


def _timestamp_from_seconds(timestamp_seconds: float) -> datetime:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(seconds=timestamp_seconds)


def evaluate_alert_predictions(
    records: Iterable[FramePredictionRecord],
    *,
    confidence_threshold: float,
    rolling_window_size: int,
    min_positive_frames: int,
    min_detection_duration_seconds: float,
    cooldown_seconds: float,
) -> tuple[list[ClipAlertResult], AlertEvaluationSummary]:
    grouped: dict[str, list[FramePredictionRecord]] = {}
    for record in records:
        grouped.setdefault(record.clip_id, []).append(record)

    clip_results: list[ClipAlertResult] = []
    total_alerts = 0
    true_alerts = 0
    false_alerts = 0
    positive_clips = 0
    alerted_positive_clips = 0
    missed_positive_clips = 0
    false_positive_clips = 0
    negative_duration_hours = 0.0
    latencies: list[float] = []

    for clip_id, clip_records in grouped.items():
        clip_records.sort(key=lambda item: item.timestamp_seconds)
        filter_ = TemporalAlertFilter(
            rolling_window_size=rolling_window_size,
            min_positive_frames=min_positive_frames,
            min_detection_duration_seconds=min_detection_duration_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        had_human_gt = any(record.has_human_gt for record in clip_records)
        subset = clip_records[0].subset if clip_records else "unspecified"
        first_human = next((record.timestamp_seconds for record in clip_records if record.has_human_gt), None)
        first_alert = None
        alert_count = 0
        true_alert_count = 0
        false_alert_count = 0

        for record in clip_records:
            detected = record.detection_count > 0 and record.detected_confidence >= confidence_threshold
            decision = filter_.evaluate(_timestamp_from_seconds(record.timestamp_seconds), detected)
            if decision.triggered:
                alert_count += 1
                if record.has_human_gt:
                    true_alert_count += 1
                    if first_alert is None:
                        first_alert = record.timestamp_seconds
                else:
                    false_alert_count += 1

        duration_seconds = 0.0
        if clip_records:
            duration_seconds = clip_records[-1].timestamp_seconds - clip_records[0].timestamp_seconds
        if not had_human_gt:
            negative_duration_hours += max(duration_seconds, 0.0) / 3600.0

        alerted_on_positive_clip = had_human_gt and true_alert_count > 0
        missed_positive_clip = had_human_gt and true_alert_count == 0
        false_positive_clip = (not had_human_gt) and false_alert_count > 0
        detection_latency = None
        if first_human is not None and first_alert is not None:
            detection_latency = max(0.0, first_alert - first_human)
            latencies.append(detection_latency)

        total_alerts += alert_count
        true_alerts += true_alert_count
        false_alerts += false_alert_count
        if had_human_gt:
            positive_clips += 1
            if alerted_on_positive_clip:
                alerted_positive_clips += 1
            if missed_positive_clip:
                missed_positive_clips += 1
        if false_positive_clip:
            false_positive_clips += 1

        clip_results.append(
            ClipAlertResult(
                clip_id=clip_id,
                subset=subset,
                alert_count=alert_count,
                true_alert_count=true_alert_count,
                false_alert_count=false_alert_count,
                had_human_gt=had_human_gt,
                alerted_on_positive_clip=alerted_on_positive_clip,
                missed_positive_clip=missed_positive_clip,
                false_positive_clip=false_positive_clip,
                first_human_timestamp_seconds=first_human,
                first_alert_timestamp_seconds=first_alert,
                detection_latency_seconds=detection_latency,
                duration_seconds=duration_seconds,
            )
        )

    alert_precision = true_alerts / max(total_alerts, 1)
    clip_recall = alerted_positive_clips / max(positive_clips, 1)
    false_positives_per_hour = false_alerts / max(negative_duration_hours, 1e-9) if negative_duration_hours > 0 else 0.0
    average_detection_latency_seconds = sum(latencies) / len(latencies) if latencies else None

    summary = AlertEvaluationSummary(
        total_alerts=total_alerts,
        true_alerts=true_alerts,
        false_alerts=false_alerts,
        positive_clips=positive_clips,
        alerted_positive_clips=alerted_positive_clips,
        missed_positive_clips=missed_positive_clips,
        false_positive_clips=false_positive_clips,
        negative_duration_hours=negative_duration_hours,
        false_positives_per_hour=false_positives_per_hour,
        alert_precision=alert_precision,
        clip_recall=clip_recall,
        average_detection_latency_seconds=average_detection_latency_seconds,
    )
    return clip_results, summary
