from src.alert_evaluation import FramePredictionRecord
from src.alert_tuning import recommend_best_setting, sweep_alert_thresholds


def test_sweep_alert_thresholds_reports_candidate_metrics() -> None:
    records = [
        FramePredictionRecord("clip-positive", "day", 0.0, True, 0.55, 1),
        FramePredictionRecord("clip-positive", "day", 1.0, True, 0.65, 1),
        FramePredictionRecord("clip-positive", "day", 2.0, True, 0.75, 1),
        FramePredictionRecord("clip-negative", "day", 0.0, False, 0.0, 0),
        FramePredictionRecord("clip-negative", "day", 1.0, False, 0.7, 1),
        FramePredictionRecord("clip-negative", "day", 2.0, False, 0.72, 1),
        FramePredictionRecord("clip-negative", "day", 3.0, False, 0.74, 1),
    ]

    results = sweep_alert_thresholds(
        records,
        confidence_thresholds=[0.5, 0.7],
        rolling_window_sizes=[3],
        min_positive_frames_options=[2, 3],
        min_detection_duration_seconds_options=[1.0],
        cooldown_seconds=60.0,
    )

    assert len(results) == 4
    by_conf = {(row.confidence_threshold, row.min_positive_frames): row for row in results}
    assert by_conf[(0.5, 2)].alert_precision == 0.5
    assert by_conf[(0.7, 3)].clip_recall == 0.0
    assert by_conf[(0.7, 2)].false_alerts == 1


def test_recommend_best_setting_prefers_precision_then_recall_then_false_positives() -> None:
    records = [
        FramePredictionRecord("clip-positive", "ir", 0.0, True, 0.82, 1),
        FramePredictionRecord("clip-positive", "ir", 1.0, True, 0.84, 1),
        FramePredictionRecord("clip-positive", "ir", 2.0, True, 0.86, 1),
        FramePredictionRecord("clip-negative", "ir", 0.0, False, 0.0, 0),
        FramePredictionRecord("clip-negative", "ir", 1.0, False, 0.7, 1),
        FramePredictionRecord("clip-negative", "ir", 2.0, False, 0.72, 1),
    ]
    results = sweep_alert_thresholds(
        records,
        confidence_thresholds=[0.65, 0.8],
        rolling_window_sizes=[3],
        min_positive_frames_options=[2],
        min_detection_duration_seconds_options=[1.0],
        cooldown_seconds=60.0,
    )

    best = recommend_best_setting(results)

    assert best.confidence_threshold == 0.8
    assert best.alert_precision >= 1.0
