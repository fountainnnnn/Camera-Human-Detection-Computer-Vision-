from src.runtime_validation import (
    RuntimeSample,
    RuntimeThresholds,
    evaluate_runtime_summary,
    summarize_runtime_samples,
)


def test_summarize_runtime_samples_reports_cpu_ram_fps_disconnects_and_stalls() -> None:
    samples = [
        RuntimeSample(
            observed_at="2026-01-01T00:00:00+00:00",
            status="ok",
            input_mode="rtsp",
            camera_connected=True,
            cpu_percent=20.0,
            ram_percent=40.0,
            fps=2.0,
            total_frames=10,
            total_alerts=0,
        ),
        RuntimeSample(
            observed_at="2026-01-01T00:01:00+00:00",
            status="ok",
            input_mode="rtsp",
            camera_connected=False,
            cpu_percent=30.0,
            ram_percent=50.0,
            fps=1.0,
            total_frames=12,
            total_alerts=1,
        ),
        RuntimeSample(
            observed_at="2026-01-01T00:02:00+00:00",
            status="ok",
            input_mode="rtsp",
            camera_connected=False,
            cpu_percent=10.0,
            ram_percent=45.0,
            fps=0.5,
            total_frames=12,
            total_alerts=1,
        ),
        RuntimeSample(
            observed_at="2026-01-01T00:03:00+00:00",
            status="ok",
            input_mode="rtsp",
            camera_connected=True,
            cpu_percent=40.0,
            ram_percent=55.0,
            fps=2.5,
            total_frames=18,
            total_alerts=2,
        ),
    ]

    summary = summarize_runtime_samples(samples, elapsed_seconds=180.0, exit_code=0)

    assert summary.sample_count == 4
    assert summary.elapsed_seconds == 180.0
    assert summary.average_cpu_percent == 25.0
    assert summary.peak_cpu_percent == 40.0
    assert summary.average_ram_percent == 47.5
    assert summary.peak_ram_percent == 55.0
    assert summary.average_fps == 1.5
    assert summary.peak_fps == 2.5
    assert summary.disconnect_count == 1
    assert summary.stale_sample_count == 1
    assert summary.final_total_frames == 18
    assert summary.final_total_alerts == 2
    assert summary.final_camera_connected is True
    assert summary.exit_code == 0
    assert summary.zero_exit is True


def test_evaluate_runtime_summary_applies_thresholds() -> None:
    samples = [
        RuntimeSample(
            observed_at="2026-01-01T00:00:00+00:00",
            status="ok",
            input_mode="webcam",
            camera_connected=True,
            cpu_percent=70.0,
            ram_percent=45.0,
            fps=1.5,
            total_frames=20,
            total_alerts=0,
        ),
        RuntimeSample(
            observed_at="2026-01-01T00:01:00+00:00",
            status="ok",
            input_mode="webcam",
            camera_connected=False,
            cpu_percent=90.0,
            ram_percent=60.0,
            fps=0.5,
            total_frames=20,
            total_alerts=0,
        ),
    ]
    summary = summarize_runtime_samples(samples, elapsed_seconds=60.0, exit_code=3)
    thresholds = RuntimeThresholds(
        max_average_cpu_percent=75.0,
        max_peak_cpu_percent=85.0,
        max_ram_percent=70.0,
        min_average_fps=1.2,
        max_disconnects=0,
        max_stale_samples=0,
        min_runtime_seconds=120.0,
        require_zero_exit=True,
    )

    evaluation = evaluate_runtime_summary(summary, thresholds)
    checks = {check.name: check for check in evaluation.checks}

    assert evaluation.passed is False
    assert checks["max_average_cpu_percent"].passed is False
    assert checks["max_peak_cpu_percent"].passed is False
    assert checks["max_ram_percent"].passed is True
    assert checks["min_average_fps"].passed is False
    assert checks["max_disconnects"].passed is False
    assert checks["max_stale_samples"].passed is False
    assert checks["min_runtime_seconds"].passed is False
    assert checks["require_zero_exit"].passed is False
