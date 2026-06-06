from src.alert_evaluation import FramePredictionRecord, evaluate_alert_predictions


def test_alert_evaluation_computes_precision_and_recall() -> None:
    records = [
        FramePredictionRecord("clip-positive", "night", 0.0, True, 0.9, 1),
        FramePredictionRecord("clip-positive", "night", 1.0, True, 0.92, 1),
        FramePredictionRecord("clip-positive", "night", 2.0, True, 0.95, 1),
        FramePredictionRecord("clip-negative", "night", 0.0, False, 0.0, 0),
        FramePredictionRecord("clip-negative", "night", 1.0, False, 0.85, 1),
        FramePredictionRecord("clip-negative", "night", 2.0, False, 0.88, 1),
        FramePredictionRecord("clip-negative", "night", 3.0, False, 0.9, 1),
    ]

    clip_results, summary = evaluate_alert_predictions(
        records,
        confidence_threshold=0.8,
        rolling_window_size=3,
        min_positive_frames=2,
        min_detection_duration_seconds=1.0,
        cooldown_seconds=60.0,
    )

    assert len(clip_results) == 2
    assert summary.total_alerts == 2
    assert summary.true_alerts == 1
    assert summary.false_alerts == 1
    assert summary.alert_precision == 0.5
    assert summary.positive_clips == 1
    assert summary.alerted_positive_clips == 1
    assert summary.missed_positive_clips == 0
    assert summary.clip_recall == 1.0


def test_alert_evaluation_marks_missed_positive_clip() -> None:
    records = [
        FramePredictionRecord("clip-positive", "ir", 0.0, True, 0.4, 1),
        FramePredictionRecord("clip-positive", "ir", 1.0, True, 0.45, 1),
        FramePredictionRecord("clip-positive", "ir", 2.0, True, 0.42, 1),
    ]

    _, summary = evaluate_alert_predictions(
        records,
        confidence_threshold=0.8,
        rolling_window_size=3,
        min_positive_frames=2,
        min_detection_duration_seconds=1.0,
        cooldown_seconds=60.0,
    )

    assert summary.total_alerts == 0
    assert summary.positive_clips == 1
    assert summary.alerted_positive_clips == 0
    assert summary.missed_positive_clips == 1
    assert summary.clip_recall == 0.0
