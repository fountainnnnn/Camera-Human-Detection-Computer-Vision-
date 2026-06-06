from __future__ import annotations

import pandas as pd


REQUIRED_INTERVAL_COLUMNS = {"clip_id", "start_seconds", "end_seconds"}


def apply_intervals_to_manifest(
    manifest: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    required_manifest = {"clip_id", "timestamp_seconds"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_manifest)}")
    missing_intervals = REQUIRED_INTERVAL_COLUMNS - set(intervals.columns)
    if missing_intervals:
        raise ValueError(f"Intervals missing required columns: {sorted(missing_intervals)}")

    labeled = manifest.copy()
    labeled["has_human_gt"] = False
    if "subset" not in labeled.columns:
        labeled["subset"] = "unspecified"
    if "has_human_gt" not in intervals.columns:
        intervals = intervals.copy()
        intervals["has_human_gt"] = True

    for row in intervals.to_dict(orient="records"):
        clip_id = str(row["clip_id"])
        start_seconds = float(row["start_seconds"])
        end_seconds = float(row["end_seconds"])
        has_human_gt = bool(row.get("has_human_gt", True))
        mask = (
            (labeled["clip_id"].astype(str) == clip_id)
            & (labeled["timestamp_seconds"].astype(float) >= start_seconds)
            & (labeled["timestamp_seconds"].astype(float) <= end_seconds)
        )
        labeled.loc[mask, "has_human_gt"] = has_human_gt
        if "subset" in row and pd.notna(row["subset"]):
            labeled.loc[labeled["clip_id"].astype(str) == clip_id, "subset"] = str(row["subset"])
    return labeled
