from __future__ import annotations

from pathlib import Path

import pandas as pd

VALIDATION_SUBSETS = ("day", "low_light", "ir", "night_vision")


def interval_template_dataframe(clip_id: str, subset: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "clip_id": clip_id,
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "subset": subset,
                "has_human_gt": True,
                "notes": "Replace placeholder rows with one row per human-present interval.",
            }
        ]
    )


def ensure_interval_template(path: str | Path, clip_id: str, subset: str) -> Path:
    template_path = Path(path)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    if not template_path.exists():
        interval_template_dataframe(clip_id, subset).to_csv(template_path, index=False)
    return template_path


def validation_workspace_directories(raw_root: str | Path, intervals_dir: str | Path) -> dict[str, Path]:
    raw_root_path = Path(raw_root)
    intervals_path = Path(intervals_dir)
    directories = {"raw_root": raw_root_path, "intervals_dir": intervals_path}
    for subset in VALIDATION_SUBSETS:
        directories[f"raw_{subset}"] = raw_root_path / subset
    return directories


def create_validation_workspace(raw_root: str | Path, intervals_dir: str | Path) -> dict[str, Path]:
    directories = validation_workspace_directories(raw_root, intervals_dir)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def collection_checklist_dataframe(raw_root: str | Path, intervals_dir: str | Path) -> pd.DataFrame:
    raw_root_path = Path(raw_root)
    intervals_path = Path(intervals_dir)
    rows = []
    for subset in VALIDATION_SUBSETS:
        rows.append(
            {
                "subset": subset,
                "raw_directory": str(raw_root_path / subset),
                "interval_csv_pattern": str(intervals_path / "<clip_id>.csv"),
                "minimum_clips": 1,
                "capture_notes": f"Record at least one {subset} clip with real scene motion and at least one human-present interval.",
                "label_notes": "Create one CSV per clip_id and replace placeholder rows with exact human-present time intervals.",
            }
        )
    return pd.DataFrame(rows)


def write_collection_checklist(path: str | Path, raw_root: str | Path, intervals_dir: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection_checklist_dataframe(raw_root, intervals_dir).to_csv(output_path, index=False)
    return output_path


def write_validation_collection_guide(path: str | Path, raw_root: str | Path, intervals_dir: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    directories = validation_workspace_directories(raw_root, intervals_dir)
    lines = [
        "# Validation Collection Guide",
        "",
        "1. Record clips into the subset folders under the raw root.",
        "2. Keep filenames stable; the inferred `clip_id` becomes the required interval CSV name.",
        "3. Create one interval CSV per clip in the intervals directory.",
        "4. Replace template rows with exact human-present time ranges in seconds.",
        "5. Run `python .\\scripts\\run_validation_pipeline.py --config .\\config.yaml --raw-root .\\data\\raw --intervals-dir .\\data\\intervals --output-root .\\data\\validation_runs\\latest --create-missing-interval-templates`.",
        "",
        "## Expected subset folders",
        "",
    ]
    for subset in VALIDATION_SUBSETS:
        lines.append(f"- `{subset}`: `{directories[f'raw_{subset}']}`")
    lines.extend(
        [
            "",
            "## Interval files",
            "",
            f"- store CSVs in `{directories['intervals_dir']}`",
            "- file name must match the inferred `clip_id`",
            "- required columns: `clip_id`, `start_seconds`, `end_seconds`",
            "- optional columns: `subset`, `has_human_gt`, `notes`",
            "",
            "## Evidence target",
            "",
            "- cover `day`, `low_light`, `ir`, and `night_vision`",
            "- include positive human-present intervals and ordinary negative footage",
            "- keep enough clips to measure false positives, missed alerts, and latency",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def coverage_row(
    *,
    clip_id: str,
    subset: str,
    source_video: str,
    manifest: str,
    intervals_csv: str,
    has_intervals: bool,
    included_in_eval: bool,
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "subset": subset,
        "source_video": source_video,
        "manifest": manifest,
        "intervals_csv": intervals_csv,
        "has_intervals": has_intervals,
        "included_in_eval": included_in_eval,
    }


def coverage_summary(coverage: pd.DataFrame) -> dict[str, int]:
    if coverage.empty:
        return {
            "total_clips": 0,
            "clips_with_intervals": 0,
            "clips_included_in_eval": 0,
            "clips_missing_intervals": 0,
        }
    return {
        "total_clips": int(len(coverage)),
        "clips_with_intervals": int(coverage["has_intervals"].sum()),
        "clips_included_in_eval": int(coverage["included_in_eval"].sum()),
        "clips_missing_intervals": int((~coverage["has_intervals"]).sum()),
    }


def write_coverage_summary(path: str | Path, coverage: pd.DataFrame) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = coverage_summary(coverage)
    lines = [
        "# Validation Coverage Summary",
        "",
        f"- total clips: {summary['total_clips']}",
        f"- clips with interval labels: {summary['clips_with_intervals']}",
        f"- clips included in evaluation: {summary['clips_included_in_eval']}",
        f"- clips missing interval labels: {summary['clips_missing_intervals']}",
    ]
    if not coverage.empty:
        missing = coverage.loc[~coverage["has_intervals"], ["clip_id", "subset", "intervals_csv"]]
        if not missing.empty:
            lines.extend(["", "## Missing Interval Files", ""])
            for row in missing.to_dict(orient="records"):
                lines.append(f"- `{row['clip_id']}` (`{row['subset']}`): `{row['intervals_csv']}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
