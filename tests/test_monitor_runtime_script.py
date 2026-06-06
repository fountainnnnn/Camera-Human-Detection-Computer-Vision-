import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.runtime_validation import RuntimeSample


def load_monitor_runtime_module():
    fake_common = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common}):
        if "scripts.monitor_runtime" in sys.modules:
            return importlib.reload(sys.modules["scripts.monitor_runtime"])
        return importlib.import_module("scripts.monitor_runtime")


def test_monitor_runtime_script_writes_reports_and_returns_zero(monkeypatch, tmp_path: Path) -> None:
    module = load_monitor_runtime_module()
    samples = [
        RuntimeSample(
            observed_at="2026-01-01T00:00:00+00:00",
            status="ok",
            input_mode="rtsp",
            camera_connected=True,
            cpu_percent=22.0,
            ram_percent=44.0,
            fps=2.0,
            total_frames=10,
            total_alerts=0,
        ),
        RuntimeSample(
            observed_at="2026-01-01T00:01:00+00:00",
            status="ok",
            input_mode="rtsp",
            camera_connected=True,
            cpu_percent=24.0,
            ram_percent=46.0,
            fps=2.2,
            total_frames=20,
            total_alerts=1,
        ),
    ]
    monkeypatch.setattr(module, "collect_runtime_samples", lambda **_kwargs: (samples, 120.0))

    output_dir = tmp_path / "reports"
    exit_code = module.main(
        [
            "--status-path",
            str(tmp_path / "status.json"),
            "--output-dir",
            str(output_dir),
            "--duration-seconds",
            "120",
            "--max-average-cpu-percent",
            "30",
            "--max-peak-cpu-percent",
            "50",
            "--max-ram-percent",
            "60",
            "--min-average-fps",
            "1.5",
            "--max-disconnects",
            "0",
            "--max-stale-samples",
            "0",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "runtime_monitor_samples.csv").exists()
    assert (output_dir / "runtime_monitor_summary.md").exists()
    summary_payload = json.loads((output_dir / "runtime_monitor_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["summary"]["sample_count"] == 2
    assert summary_payload["summary"]["average_cpu_percent"] == 23.0
    assert summary_payload["evaluation"]["passed"] is True


def test_monitor_runtime_script_returns_nonzero_when_thresholds_fail(monkeypatch, tmp_path: Path) -> None:
    module = load_monitor_runtime_module()
    samples = [
        RuntimeSample(
            observed_at="2026-01-01T00:00:00+00:00",
            status="ok",
            input_mode="webcam",
            camera_connected=True,
            cpu_percent=92.0,
            ram_percent=70.0,
            fps=0.3,
            total_frames=10,
            total_alerts=0,
        ),
        RuntimeSample(
            observed_at="2026-01-01T00:01:00+00:00",
            status="ok",
            input_mode="webcam",
            camera_connected=False,
            cpu_percent=96.0,
            ram_percent=74.0,
            fps=0.2,
            total_frames=10,
            total_alerts=0,
        ),
    ]
    monkeypatch.setattr(module, "collect_runtime_samples", lambda **_kwargs: (samples, 60.0))

    output_dir = tmp_path / "reports"
    exit_code = module.main(
        [
            "--status-path",
            str(tmp_path / "status.json"),
            "--output-dir",
            str(output_dir),
            "--duration-seconds",
            "60",
            "--max-average-cpu-percent",
            "80",
            "--max-peak-cpu-percent",
            "95",
            "--max-ram-percent",
            "80",
            "--min-average-fps",
            "1.0",
            "--max-disconnects",
            "0",
            "--max-stale-samples",
            "0",
        ]
    )

    assert exit_code == 1
    summary_payload = json.loads((output_dir / "runtime_monitor_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["evaluation"]["passed"] is False
    assert "FAIL" in (output_dir / "runtime_monitor_summary.md").read_text(encoding="utf-8")
