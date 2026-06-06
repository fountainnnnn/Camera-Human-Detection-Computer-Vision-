from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AcceptanceThresholds:
    min_alert_precision: float = 0.9
    min_clip_recall: float = 0.8
    max_false_positives_per_hour: float = 0.5
    max_average_cpu_percent: float = 80.0
    max_peak_cpu_percent: float = 95.0
    min_average_fps: float = 1.0
    max_disconnects: int = 0
    max_stale_samples: int = 2
    min_runtime_seconds: float = 1800.0
    require_runtime_monitor: bool = True


@dataclass
class AcceptanceCheck:
    name: str
    passed: bool
    actual: Any
    expected: Any
    detail: str = ""


@dataclass
class Recommendation:
    category: str
    summary: str


@dataclass
class AcceptanceReport:
    passed: bool
    metrics: dict[str, Any]
    checks: list[AcceptanceCheck]
    recommendations: list[Recommendation]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_acceptance_report(
    *,
    clip_rows: list[dict[str, Any]],
    runtime_monitor_payload: dict[str, Any] | None,
    thresholds: AcceptanceThresholds,
) -> AcceptanceReport:
    checks: list[AcceptanceCheck] = []
    recommendations: list[Recommendation] = []

    validation_present = bool(clip_rows)
    checks.append(
        AcceptanceCheck(
            name="validation_data_present",
            passed=validation_present,
            actual=len(clip_rows),
            expected="> 0 clip rows",
            detail="Alert-level validation requires real clip results.",
        )
    )
    if not validation_present:
        recommendations.append(
            Recommendation(
                category="data",
                summary="Collect labeled validation clips under data/raw and data/intervals before claiming precision or recall.",
            )
        )

    total_alerts = sum(int(row.get("alert_count", 0)) for row in clip_rows)
    true_alerts = sum(int(row.get("true_alert_count", 0)) for row in clip_rows)
    false_alerts = sum(int(row.get("false_alert_count", 0)) for row in clip_rows)
    positive_clips = sum(1 for row in clip_rows if _to_bool(row.get("had_human_gt", False)))
    alerted_positive_clips = sum(1 for row in clip_rows if _to_bool(row.get("alerted_on_positive_clip", False)))
    missed_positive_clips = sum(1 for row in clip_rows if _to_bool(row.get("missed_positive_clip", False)))
    negative_duration_hours = sum(
        max(float(row.get("duration_seconds", 0.0)), 0.0) / 3600.0
        for row in clip_rows
        if not _to_bool(row.get("had_human_gt", False))
    )
    alert_precision = round(_safe_div(true_alerts, total_alerts), 4)
    clip_recall = round(_safe_div(alerted_positive_clips, positive_clips), 4)
    false_positives_per_hour = round(_safe_div(false_alerts, negative_duration_hours), 4) if negative_duration_hours > 0 else 0.0

    metrics: dict[str, Any] = {
        "clip_count": len(clip_rows),
        "total_alerts": total_alerts,
        "true_alerts": true_alerts,
        "false_alerts": false_alerts,
        "positive_clips": positive_clips,
        "alerted_positive_clips": alerted_positive_clips,
        "missed_positive_clips": missed_positive_clips,
        "negative_duration_hours": round(negative_duration_hours, 4),
        "alert_precision": alert_precision,
        "clip_recall": clip_recall,
        "false_positives_per_hour": false_positives_per_hour,
    }

    checks.extend(
        [
            AcceptanceCheck("alert_precision", alert_precision >= thresholds.min_alert_precision, alert_precision, thresholds.min_alert_precision),
            AcceptanceCheck("clip_recall", clip_recall >= thresholds.min_clip_recall, clip_recall, thresholds.min_clip_recall),
            AcceptanceCheck(
                "false_positives_per_hour",
                false_positives_per_hour <= thresholds.max_false_positives_per_hour,
                false_positives_per_hour,
                thresholds.max_false_positives_per_hour,
            ),
        ]
    )

    runtime_present = runtime_monitor_payload is not None
    checks.append(
        AcceptanceCheck(
            name="runtime_monitor_present",
            passed=(runtime_present or not thresholds.require_runtime_monitor),
            actual=runtime_present,
            expected=not thresholds.require_runtime_monitor or True,
            detail="CPU and stability stories require runtime monitor evidence.",
        )
    )
    if not runtime_present:
        recommendations.append(
            Recommendation(
                category="runtime",
                summary="Run scripts/monitor_runtime.py against a live instance to collect CPU, FPS, disconnect, and stale-sample evidence.",
            )
        )

    runtime_summary = (runtime_monitor_payload or {}).get("summary", {})
    runtime_metrics = {
        "average_cpu_percent": float(runtime_summary.get("average_cpu_percent", 0.0)),
        "peak_cpu_percent": float(runtime_summary.get("peak_cpu_percent", 0.0)),
        "average_fps": float(runtime_summary.get("average_fps", 0.0)),
        "disconnect_count": int(runtime_summary.get("disconnect_count", 0)),
        "stale_sample_count": int(runtime_summary.get("stale_sample_count", 0)),
        "elapsed_seconds": float(runtime_summary.get("elapsed_seconds", 0.0)),
    }
    metrics.update(runtime_metrics)

    if runtime_present:
        checks.extend(
            [
                AcceptanceCheck("average_cpu_percent", runtime_metrics["average_cpu_percent"] <= thresholds.max_average_cpu_percent, runtime_metrics["average_cpu_percent"], thresholds.max_average_cpu_percent),
                AcceptanceCheck("peak_cpu_percent", runtime_metrics["peak_cpu_percent"] <= thresholds.max_peak_cpu_percent, runtime_metrics["peak_cpu_percent"], thresholds.max_peak_cpu_percent),
                AcceptanceCheck("average_fps", runtime_metrics["average_fps"] >= thresholds.min_average_fps, runtime_metrics["average_fps"], thresholds.min_average_fps),
                AcceptanceCheck("disconnect_count", runtime_metrics["disconnect_count"] <= thresholds.max_disconnects, runtime_metrics["disconnect_count"], thresholds.max_disconnects),
                AcceptanceCheck("stale_sample_count", runtime_metrics["stale_sample_count"] <= thresholds.max_stale_samples, runtime_metrics["stale_sample_count"], thresholds.max_stale_samples),
                AcceptanceCheck("runtime_seconds", runtime_metrics["elapsed_seconds"] >= thresholds.min_runtime_seconds, runtime_metrics["elapsed_seconds"], thresholds.min_runtime_seconds),
            ]
        )

    if alert_precision < thresholds.min_alert_precision or false_positives_per_hour > thresholds.max_false_positives_per_hour:
        recommendations.append(
            Recommendation(
                category="false_positives",
                summary="Increase model.confidence_threshold, raise detection.min_positive_frames, extend detection.min_detection_duration_seconds, and tighten zones to reduce false positives.",
            )
        )
    if clip_recall < thresholds.min_clip_recall:
        recommendations.append(
            Recommendation(
                category="recall",
                summary="Lower model.confidence_threshold carefully or relax detection.min_positive_frames / detection.min_detection_duration_seconds after reviewing missed positive clips.",
            )
        )

    subset_rows = {}
    for row in clip_rows:
        subset = str(row.get("subset", "unspecified"))
        entry = subset_rows.setdefault(subset, {"positive": 0, "alerted_positive": 0, "false_alerts": 0})
        if _to_bool(row.get("had_human_gt", False)):
            entry["positive"] += 1
            if _to_bool(row.get("alerted_on_positive_clip", False)):
                entry["alerted_positive"] += 1
        entry["false_alerts"] += int(row.get("false_alert_count", 0))

    weak_night_subsets = []
    for subset, entry in subset_rows.items():
        recall = _safe_div(entry["alerted_positive"], entry["positive"]) if entry["positive"] else None
        if subset in {"ir", "night_vision", "night"} and ((recall is not None and recall < thresholds.min_clip_recall) or entry["false_alerts"] > 0):
            weak_night_subsets.append(subset)
    if weak_night_subsets:
        recommendations.append(
            Recommendation(
                category="fine_tune",
                summary=f"Fine-tune the model on IR/night footage and expand labeled subsets for {', '.join(sorted(weak_night_subsets))}.",
            )
        )

    if runtime_present and (
        runtime_metrics["average_cpu_percent"] > thresholds.max_average_cpu_percent
        or runtime_metrics["peak_cpu_percent"] > thresholds.max_peak_cpu_percent
    ):
        recommendations.append(
            Recommendation(
                category="cpu",
                summary="Lower detection.sample_fps or model.inference_size, or run on a supported accelerator if available, to reduce CPU pressure.",
            )
        )
    if runtime_present and (
        runtime_metrics["disconnect_count"] > thresholds.max_disconnects
        or runtime_metrics["stale_sample_count"] > thresholds.max_stale_samples
    ):
        recommendations.append(
            Recommendation(
                category="stability",
                summary="Inspect camera/network stability, reconnect behavior, and long-run logging because runtime monitoring saw disconnects or stalled frame counts.",
            )
        )

    passed = all(check.passed for check in checks)
    return AcceptanceReport(passed=passed, metrics=metrics, checks=checks, recommendations=recommendations)


def render_acceptance_report_markdown(report: AcceptanceReport, thresholds: AcceptanceThresholds) -> str:
    lines = [
        "# Acceptance Report",
        "",
        f"Overall: {'PASS' if report.passed else 'FAIL'}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in report.metrics.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        detail = f" ({check.detail})" if check.detail else ""
        lines.append(f"- {status} `{check.name}` actual=`{check.actual}` expected=`{check.expected}`{detail}")
    lines.extend(["", "## Recommendations", ""])
    if report.recommendations:
        for item in report.recommendations:
            lines.append(f"- {item.summary}")
    else:
        lines.append("- No changes recommended.")
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"- min_alert_precision: `{thresholds.min_alert_precision}`",
            f"- min_clip_recall: `{thresholds.min_clip_recall}`",
            f"- max_false_positives_per_hour: `{thresholds.max_false_positives_per_hour}`",
            f"- max_average_cpu_percent: `{thresholds.max_average_cpu_percent}`",
            f"- max_peak_cpu_percent: `{thresholds.max_peak_cpu_percent}`",
            f"- min_average_fps: `{thresholds.min_average_fps}`",
            f"- max_disconnects: `{thresholds.max_disconnects}`",
            f"- max_stale_samples: `{thresholds.max_stale_samples}`",
            f"- min_runtime_seconds: `{thresholds.min_runtime_seconds}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_acceptance_report_json(path: str | Path, report: AcceptanceReport, thresholds: AcceptanceThresholds) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": report.passed,
        "metrics": report.metrics,
        "checks": [asdict(check) for check in report.checks],
        "recommendations": [asdict(item) for item in report.recommendations],
        "thresholds": asdict(thresholds),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_acceptance_report_markdown(path: str | Path, report: AcceptanceReport, thresholds: AcceptanceThresholds) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_acceptance_report_markdown(report, thresholds), encoding="utf-8")
    return output_path
