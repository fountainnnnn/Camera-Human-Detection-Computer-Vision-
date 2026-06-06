from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlertFailureAnalysis:
    false_positive_clip_count: int
    missed_positive_clip_count: int
    false_positive_subsets: list[dict[str, Any]]
    missed_positive_subsets: list[dict[str, Any]]
    top_false_positive_frames: list[dict[str, Any]]
    low_confidence_positive_frames: list[dict[str, Any]]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def analyze_alert_failures(
    *,
    clip_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> AlertFailureAnalysis:
    false_positive_clips = [row for row in clip_rows if _to_bool(row.get("false_positive_clip", False))]
    missed_positive_clips = [row for row in clip_rows if _to_bool(row.get("missed_positive_clip", False))]

    false_positive_subsets: dict[str, dict[str, Any]] = {}
    for row in false_positive_clips:
        subset = str(row.get("subset", "unspecified"))
        entry = false_positive_subsets.setdefault(subset, {"subset": subset, "clip_count": 0, "false_alert_count": 0})
        entry["clip_count"] += 1
        entry["false_alert_count"] += int(row.get("false_alert_count", 0))

    missed_positive_subsets: dict[str, dict[str, Any]] = {}
    for row in missed_positive_clips:
        subset = str(row.get("subset", "unspecified"))
        entry = missed_positive_subsets.setdefault(subset, {"subset": subset, "clip_count": 0})
        entry["clip_count"] += 1

    top_false_positive_frames = sorted(
        [
            row
            for row in frame_rows
            if not _to_bool(row.get("has_human_gt", False)) and int(row.get("detection_count", 0)) > 0
        ],
        key=lambda row: (-float(row.get("detected_confidence", 0.0)), str(row.get("image_name", ""))),
    )

    low_confidence_positive_frames = sorted(
        [
            row
            for row in frame_rows
            if _to_bool(row.get("has_human_gt", False)) and int(row.get("detection_count", 0)) > 0
        ],
        key=lambda row: (float(row.get("detected_confidence", 0.0)), str(row.get("image_name", ""))),
    )

    return AlertFailureAnalysis(
        false_positive_clip_count=len(false_positive_clips),
        missed_positive_clip_count=len(missed_positive_clips),
        false_positive_subsets=sorted(false_positive_subsets.values(), key=lambda row: (-row["clip_count"], row["subset"])),
        missed_positive_subsets=sorted(missed_positive_subsets.values(), key=lambda row: (-row["clip_count"], row["subset"])),
        top_false_positive_frames=top_false_positive_frames,
        low_confidence_positive_frames=low_confidence_positive_frames,
    )
