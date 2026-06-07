from datetime import datetime, timedelta, timezone

from src.alert_logic import TemporalAlertFilter


def test_alert_requires_enough_positive_frames_and_duration() -> None:
    filter_ = TemporalAlertFilter(
        rolling_window_size=5,
        min_positive_frames=3,
        min_detection_duration_seconds=2.0,
        cooldown_seconds=60.0,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert not filter_.evaluate(base, True).triggered
    assert not filter_.evaluate(base + timedelta(seconds=1), True).triggered

    decision = filter_.evaluate(base + timedelta(seconds=2), True)
    assert decision.triggered
    assert decision.reason == "confirmed"


def test_alert_respects_cooldown() -> None:
    filter_ = TemporalAlertFilter(
        rolling_window_size=5,
        min_positive_frames=1,
        min_detection_duration_seconds=0.0,
        cooldown_seconds=10.0,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert filter_.evaluate(base, True).triggered
    second = filter_.evaluate(base + timedelta(seconds=5), True)
    assert not second.triggered
    assert second.reason == "cooldown_active"


def test_alert_requires_current_detection_before_starting_cooldown() -> None:
    filter_ = TemporalAlertFilter(
        rolling_window_size=3,
        min_positive_frames=2,
        min_detection_duration_seconds=0.0,
        cooldown_seconds=10.0,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    filter_.history.append((base, True))
    filter_.history.append((base + timedelta(seconds=1), True))
    filter_.current_positive_start = base

    no_current_detection = filter_.evaluate(base + timedelta(seconds=2), False)
    assert not no_current_detection.triggered
    assert no_current_detection.reason == "current_frame_no_detection"

    alert = filter_.evaluate(base + timedelta(seconds=3), True)
    assert alert.triggered
    assert alert.reason == "confirmed"


def test_alert_filter_adds_results_to_rolling_history() -> None:
    filter_ = TemporalAlertFilter(
        rolling_window_size=2,
        min_positive_frames=2,
        min_detection_duration_seconds=0.0,
        cooldown_seconds=0.0,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    filter_.evaluate(base, True)
    filter_.evaluate(base + timedelta(seconds=1), False)
    filter_.evaluate(base + timedelta(seconds=2), True)

    assert list(filter_.history) == [
        (base + timedelta(seconds=1), False),
        (base + timedelta(seconds=2), True),
    ]
