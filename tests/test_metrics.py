from datetime import datetime, timedelta, timezone
import importlib
from types import SimpleNamespace
from unittest.mock import patch


def test_runtime_metrics_snapshot_reports_state_and_fps(monkeypatch) -> None:
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 0.0,
        virtual_memory=lambda: SimpleNamespace(percent=0.0),
    )
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        metrics_module = importlib.import_module("src.metrics")
        metrics = metrics_module.RuntimeMetrics()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    metrics.note_frame(base)
    metrics.note_frame(base + timedelta(seconds=0.5))
    metrics.note_frame(base + timedelta(seconds=1.0))
    metrics.note_detection(base + timedelta(seconds=1.0))
    metrics.note_alert(base + timedelta(seconds=1.5))

    monkeypatch.setattr("src.metrics.psutil.cpu_percent", lambda interval=None: 12.5)
    monkeypatch.setattr(
        "src.metrics.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=48.0),
    )

    snapshot = metrics.snapshot(input_mode="rtsp", camera_connected=True)

    assert snapshot.status == "ok"
    assert snapshot.input_mode == "rtsp"
    assert snapshot.camera_connected is True
    assert snapshot.last_frame_time == (base + timedelta(seconds=1.0)).isoformat()
    assert snapshot.last_detection_time == (base + timedelta(seconds=1.0)).isoformat()
    assert snapshot.last_alert_time == (base + timedelta(seconds=1.5)).isoformat()
    assert snapshot.cpu_percent == 12.5
    assert snapshot.ram_percent == 48.0
    assert snapshot.fps == 2.0
    assert snapshot.total_frames == 3
    assert snapshot.total_alerts == 1


def test_runtime_metrics_fps_is_zero_without_two_frames() -> None:
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 0.0,
        virtual_memory=lambda: SimpleNamespace(percent=0.0),
    )
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        metrics_module = importlib.import_module("src.metrics")
        metrics = metrics_module.RuntimeMetrics()
    metrics.note_frame(datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert metrics.fps() == 0.0
