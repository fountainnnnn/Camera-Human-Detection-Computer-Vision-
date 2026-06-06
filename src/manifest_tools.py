from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {"image", "clip_id", "timestamp_seconds", "has_human_gt"}


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if "subset" not in manifest.columns:
        manifest["subset"] = "unspecified"
    return manifest


def merge_manifests(paths: list[str | Path]) -> pd.DataFrame:
    manifests = [read_manifest(path) for path in paths]
    if not manifests:
        return pd.DataFrame(columns=sorted(REQUIRED_MANIFEST_COLUMNS | {"subset"}))
    merged = pd.concat(manifests, ignore_index=True)
    merged = merged.drop_duplicates(subset=["image", "clip_id", "timestamp_seconds"]).reset_index(drop=True)
    return merged
