from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class RuntimeSample:
    observed_at: str
    status: str
    input_mode: str
    camera_connected: bool
    cpu_percent: float
    ram_percent: float
    fps: float
    total_frames: int
    total_alerts: int


@dataclass
class RuntimeSummary:
    sample_count: int
    elapsed_seconds: float
    average_cpu_percent: float
    peak_cpu_percent: float
    average_ram_percent: float
    peak_ram_percent: float
    average_fps: float
    peak_fps: float
    disconnect_count: int
    stale_sample_count: int
    final_total_frames: int
    final_total_alerts: int
    final_camera_connected: bool
    input_mode: str
    exit_code: int | None
    zero_exit: bool


@dataclass
class RuntimeThresholds:
    max_average_cpu_percent: float | None = None
    max_peak_cpu_percent: float | None = None
    max_ram_percent: float | None = None
    min_average_fps: float | None = None
    max_disconnects: int | None = None
    max_stale_samples: int | None = None
    min_runtime_seconds: float | None = None
    require_zero_exit: bool = False


@dataclass
class RuntimeCheck:
    name: str
    passed: bool
    actual: Any
    expected: Any


@dataclass
class RuntimeEvaluation:
    passed: bool
    checks: list[RuntimeCheck]


def _round(value: float) -> float:
    return round(value, 2)


def sample_from_payload(payload: dict[str, Any], observed_at: str) -> RuntimeSample:
    return RuntimeSample(
        observed_at=observed_at,
        status=str(payload.get("status", "unknown")),
        input_mode=str(payload.get("input_mode", "unknown")),
        camera_connected=bool(payload.get("camera_connected", False)),
        cpu_percent=float(payload.get("cpu_percent", 0.0)),
        ram_percent=float(payload.get("ram_percent", 0.0)),
        fps=float(payload.get("fps", 0.0)),
        total_frames=int(payload.get("total_frames", 0)),
        total_alerts=int(payload.get("total_alerts", 0)),
    )


def read_status_sample(path: str | Path, observed_at: str) -> RuntimeSample:
    status_path = Path(path)
    with status_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return sample_from_payload(payload, observed_at=observed_at)


def summarize_runtime_samples(
    samples: list[RuntimeSample],
    elapsed_seconds: float,
    exit_code: int | None,
) -> RuntimeSummary:
    if not samples:
        raise ValueError("At least one runtime sample is required")

    disconnect_count = 0
    stale_sample_count = 0
    for previous, current in zip(samples, samples[1:]):
        if previous.camera_connected and not current.camera_connected:
            disconnect_count += 1
        if current.total_frames <= previous.total_frames:
            stale_sample_count += 1

    average_cpu = _round(sum(sample.cpu_percent for sample in samples) / len(samples))
    peak_cpu = _round(max(sample.cpu_percent for sample in samples))
    average_ram = _round(sum(sample.ram_percent for sample in samples) / len(samples))
    peak_ram = _round(max(sample.ram_percent for sample in samples))
    average_fps = _round(sum(sample.fps for sample in samples) / len(samples))
    peak_fps = _round(max(sample.fps for sample in samples))
    last_sample = samples[-1]

    return RuntimeSummary(
        sample_count=len(samples),
        elapsed_seconds=_round(elapsed_seconds),
        average_cpu_percent=average_cpu,
        peak_cpu_percent=peak_cpu,
        average_ram_percent=average_ram,
        peak_ram_percent=peak_ram,
        average_fps=average_fps,
        peak_fps=peak_fps,
        disconnect_count=disconnect_count,
        stale_sample_count=stale_sample_count,
        final_total_frames=last_sample.total_frames,
        final_total_alerts=last_sample.total_alerts,
        final_camera_connected=last_sample.camera_connected,
        input_mode=last_sample.input_mode,
        exit_code=exit_code,
        zero_exit=exit_code == 0,
    )


def evaluate_runtime_summary(summary: RuntimeSummary, thresholds: RuntimeThresholds) -> RuntimeEvaluation:
    checks: list[RuntimeCheck] = []

    if thresholds.max_average_cpu_percent is not None:
        checks.append(
            RuntimeCheck(
                name="max_average_cpu_percent",
                passed=summary.average_cpu_percent <= thresholds.max_average_cpu_percent,
                actual=summary.average_cpu_percent,
                expected=thresholds.max_average_cpu_percent,
            )
        )
    if thresholds.max_peak_cpu_percent is not None:
        checks.append(
            RuntimeCheck(
                name="max_peak_cpu_percent",
                passed=summary.peak_cpu_percent <= thresholds.max_peak_cpu_percent,
                actual=summary.peak_cpu_percent,
                expected=thresholds.max_peak_cpu_percent,
            )
        )
    if thresholds.max_ram_percent is not None:
        checks.append(
            RuntimeCheck(
                name="max_ram_percent",
                passed=summary.peak_ram_percent <= thresholds.max_ram_percent,
                actual=summary.peak_ram_percent,
                expected=thresholds.max_ram_percent,
            )
        )
    if thresholds.min_average_fps is not None:
        checks.append(
            RuntimeCheck(
                name="min_average_fps",
                passed=summary.average_fps >= thresholds.min_average_fps,
                actual=summary.average_fps,
                expected=thresholds.min_average_fps,
            )
        )
    if thresholds.max_disconnects is not None:
        checks.append(
            RuntimeCheck(
                name="max_disconnects",
                passed=summary.disconnect_count <= thresholds.max_disconnects,
                actual=summary.disconnect_count,
                expected=thresholds.max_disconnects,
            )
        )
    if thresholds.max_stale_samples is not None:
        checks.append(
            RuntimeCheck(
                name="max_stale_samples",
                passed=summary.stale_sample_count <= thresholds.max_stale_samples,
                actual=summary.stale_sample_count,
                expected=thresholds.max_stale_samples,
            )
        )
    if thresholds.min_runtime_seconds is not None:
        checks.append(
            RuntimeCheck(
                name="min_runtime_seconds",
                passed=summary.elapsed_seconds >= thresholds.min_runtime_seconds,
                actual=summary.elapsed_seconds,
                expected=thresholds.min_runtime_seconds,
            )
        )
    if thresholds.require_zero_exit:
        checks.append(
            RuntimeCheck(
                name="require_zero_exit",
                passed=summary.zero_exit,
                actual=summary.exit_code,
                expected=0,
            )
        )

    return RuntimeEvaluation(
        passed=all(check.passed for check in checks) if checks else True,
        checks=checks,
    )


def write_runtime_samples_csv(path: str | Path, samples: list[RuntimeSample]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(RuntimeSample)])
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return output_path


def write_runtime_summary_json(
    path: str | Path,
    summary: RuntimeSummary,
    evaluation: RuntimeEvaluation,
    thresholds: RuntimeThresholds,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "evaluation": {
            "passed": evaluation.passed,
            "checks": [asdict(check) for check in evaluation.checks],
        },
        "thresholds": asdict(thresholds),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def render_runtime_summary_markdown(
    summary: RuntimeSummary,
    evaluation: RuntimeEvaluation,
    thresholds: RuntimeThresholds,
) -> str:
    lines = [
        "# Runtime Monitor Summary",
        "",
        f"Overall: {'PASS' if evaluation.passed else 'FAIL'}",
        "",
        "## Summary",
        "",
        f"- input mode: `{summary.input_mode}`",
        f"- elapsed seconds: `{summary.elapsed_seconds}`",
        f"- sample count: `{summary.sample_count}`",
        f"- average CPU percent: `{summary.average_cpu_percent}`",
        f"- peak CPU percent: `{summary.peak_cpu_percent}`",
        f"- average RAM percent: `{summary.average_ram_percent}`",
        f"- peak RAM percent: `{summary.peak_ram_percent}`",
        f"- average FPS: `{summary.average_fps}`",
        f"- peak FPS: `{summary.peak_fps}`",
        f"- disconnect count: `{summary.disconnect_count}`",
        f"- stale sample count: `{summary.stale_sample_count}`",
        f"- final total frames: `{summary.final_total_frames}`",
        f"- final total alerts: `{summary.final_total_alerts}`",
        f"- final camera connected: `{summary.final_camera_connected}`",
        f"- exit code: `{summary.exit_code}`",
        "",
        "## Threshold checks",
        "",
    ]

    if evaluation.checks:
        for check in evaluation.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"- {status} `{check.name}` actual=`{check.actual}` expected=`{check.expected}`")
    else:
        lines.append("- No thresholds configured.")

    lines.extend(
        [
            "",
            "## Configured thresholds",
            "",
            f"- max average CPU percent: `{thresholds.max_average_cpu_percent}`",
            f"- max peak CPU percent: `{thresholds.max_peak_cpu_percent}`",
            f"- max RAM percent: `{thresholds.max_ram_percent}`",
            f"- min average FPS: `{thresholds.min_average_fps}`",
            f"- max disconnects: `{thresholds.max_disconnects}`",
            f"- max stale samples: `{thresholds.max_stale_samples}`",
            f"- min runtime seconds: `{thresholds.min_runtime_seconds}`",
            f"- require zero exit: `{thresholds.require_zero_exit}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_runtime_summary_markdown(
    path: str | Path,
    summary: RuntimeSummary,
    evaluation: RuntimeEvaluation,
    thresholds: RuntimeThresholds,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_runtime_summary_markdown(summary, evaluation, thresholds),
        encoding="utf-8",
    )
    return output_path
