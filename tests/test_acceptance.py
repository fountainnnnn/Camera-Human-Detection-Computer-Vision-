from src.acceptance import (
    AcceptanceThresholds,
    build_acceptance_report,
)


def test_build_acceptance_report_flags_metric_failures_and_suggests_tuning() -> None:
    clip_rows = [
        {
            "clip_id": "day-positive",
            "subset": "day",
            "alert_count": 1,
            "true_alert_count": 1,
            "false_alert_count": 0,
            "had_human_gt": True,
            "alerted_on_positive_clip": True,
            "missed_positive_clip": False,
            "false_positive_clip": False,
            "duration_seconds": 30.0,
        },
        {
            "clip_id": "ir-positive",
            "subset": "ir",
            "alert_count": 0,
            "true_alert_count": 0,
            "false_alert_count": 0,
            "had_human_gt": True,
            "alerted_on_positive_clip": False,
            "missed_positive_clip": True,
            "false_positive_clip": False,
            "duration_seconds": 20.0,
        },
        {
            "clip_id": "night-negative",
            "subset": "night_vision",
            "alert_count": 1,
            "true_alert_count": 0,
            "false_alert_count": 1,
            "had_human_gt": False,
            "alerted_on_positive_clip": False,
            "missed_positive_clip": False,
            "false_positive_clip": True,
            "duration_seconds": 3600.0,
        },
    ]
    runtime_summary = {
        "summary": {
            "average_cpu_percent": 88.0,
            "peak_cpu_percent": 96.0,
            "average_fps": 0.8,
            "disconnect_count": 1,
            "stale_sample_count": 3,
            "elapsed_seconds": 900.0,
        },
        "evaluation": {
            "passed": False,
            "checks": [],
        },
    }
    thresholds = AcceptanceThresholds(
        min_alert_precision=0.9,
        min_clip_recall=0.8,
        max_false_positives_per_hour=0.5,
        max_average_cpu_percent=80.0,
        max_peak_cpu_percent=95.0,
        min_average_fps=1.0,
        max_disconnects=0,
        max_stale_samples=2,
        min_runtime_seconds=1800.0,
    )

    report = build_acceptance_report(
        clip_rows=clip_rows,
        runtime_monitor_payload=runtime_summary,
        thresholds=thresholds,
    )

    checks = {check.name: check for check in report.checks}
    recommendation_text = "\n".join(item.summary for item in report.recommendations)

    assert report.passed is False
    assert checks["alert_precision"].passed is False
    assert checks["clip_recall"].passed is False
    assert checks["false_positives_per_hour"].passed is False
    assert checks["average_cpu_percent"].passed is False
    assert checks["peak_cpu_percent"].passed is False
    assert checks["average_fps"].passed is False
    assert checks["disconnect_count"].passed is False
    assert checks["stale_sample_count"].passed is False
    assert "Increase model.confidence_threshold" in recommendation_text
    assert "Lower detection.sample_fps or model.inference_size" in recommendation_text
    assert "Fine-tune the model on IR/night footage" in recommendation_text


def test_build_acceptance_report_requires_validation_data() -> None:
    report = build_acceptance_report(
        clip_rows=[],
        runtime_monitor_payload=None,
        thresholds=AcceptanceThresholds(),
    )

    checks = {check.name: check for check in report.checks}

    assert report.passed is False
    assert checks["validation_data_present"].passed is False
    assert checks["runtime_monitor_present"].passed is False
