from src.error_analysis import analyze_alert_failures


def test_analyze_alert_failures_summarizes_false_positives_and_missed_clips() -> None:
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
            "duration_seconds": 15.0,
        },
        {
            "clip_id": "ir-missed",
            "subset": "ir",
            "alert_count": 0,
            "true_alert_count": 0,
            "false_alert_count": 0,
            "had_human_gt": True,
            "alerted_on_positive_clip": False,
            "missed_positive_clip": True,
            "false_positive_clip": False,
            "duration_seconds": 22.0,
        },
        {
            "clip_id": "night-fp",
            "subset": "night_vision",
            "alert_count": 2,
            "true_alert_count": 0,
            "false_alert_count": 2,
            "had_human_gt": False,
            "alerted_on_positive_clip": False,
            "missed_positive_clip": False,
            "false_positive_clip": True,
            "duration_seconds": 60.0,
        },
    ]
    frame_rows = [
        {
            "clip_id": "night-fp",
            "subset": "night_vision",
            "timestamp_seconds": 11.0,
            "has_human_gt": False,
            "detected_confidence": 0.91,
            "detection_count": 1,
            "image_name": "night-fp-001.jpg",
        },
        {
            "clip_id": "night-fp",
            "subset": "night_vision",
            "timestamp_seconds": 12.0,
            "has_human_gt": False,
            "detected_confidence": 0.87,
            "detection_count": 1,
            "image_name": "night-fp-002.jpg",
        },
        {
            "clip_id": "ir-missed",
            "subset": "ir",
            "timestamp_seconds": 5.0,
            "has_human_gt": True,
            "detected_confidence": 0.34,
            "detection_count": 1,
            "image_name": "ir-missed-001.jpg",
        },
        {
            "clip_id": "ir-missed",
            "subset": "ir",
            "timestamp_seconds": 6.0,
            "has_human_gt": True,
            "detected_confidence": 0.29,
            "detection_count": 1,
            "image_name": "ir-missed-002.jpg",
        },
    ]

    report = analyze_alert_failures(clip_rows=clip_rows, frame_rows=frame_rows)

    assert report.false_positive_clip_count == 1
    assert report.missed_positive_clip_count == 1
    assert report.false_positive_subsets[0]["subset"] == "night_vision"
    assert report.missed_positive_subsets[0]["subset"] == "ir"
    assert report.top_false_positive_frames[0]["image_name"] == "night-fp-001.jpg"
    assert report.low_confidence_positive_frames[0]["image_name"] == "ir-missed-002.jpg"
