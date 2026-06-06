from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd
import common  # noqa: F401

from src.interval_tools import apply_intervals_to_manifest
from src.manifest_tools import merge_manifests, read_manifest
from src.validation_tools import coverage_row, coverage_summary, ensure_interval_template, write_coverage_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch extraction, labeling, merging, and alert evaluation for validation clips.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--intervals-dir", required=True)
    parser.add_argument("--output-root", default="data/validation_runs/latest")
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--include-unlabeled", action="store_true", help="Include manifests without matching interval CSVs.")
    parser.add_argument(
        "--create-missing-interval-templates",
        action="store_true",
        help="Create starter interval CSVs for clips that do not yet have labels.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    images_dir = output_root / "images"
    manifests_dir = output_root / "manifests_raw"
    labeled_manifests_dir = output_root / "manifests_labeled"
    merged_manifest_path = output_root / "merged_manifest.csv"
    merged_labeled_manifest_path = output_root / "merged_labeled_manifest.csv"
    clip_index_path = output_root / "clip_index.csv"
    coverage_path = output_root / "label_coverage.csv"
    coverage_summary_path = output_root / "label_coverage_summary.md"
    reports_dir = output_root / "reports"
    intervals_dir = Path(args.intervals_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    labeled_manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/extract_clip_batch.py",
            "--raw-root",
            args.raw_root,
            "--output-images",
            str(images_dir),
            "--output-manifests-dir",
            str(manifests_dir),
            "--merged-manifest",
            str(merged_manifest_path),
            "--clip-index",
            str(clip_index_path),
            "--sample-seconds",
            str(args.sample_seconds),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    clip_index = pd.read_csv(clip_index_path)
    labeled_manifest_paths: list[Path] = []
    coverage_rows: list[dict[str, object]] = []

    for row in clip_index.to_dict(orient="records"):
        clip_id = str(row["clip_id"])
        raw_manifest_path = Path(row["manifest"])
        interval_path = intervals_dir / f"{clip_id}.csv"
        raw_manifest = read_manifest(raw_manifest_path)
        has_intervals = interval_path.exists()

        if has_intervals:
            intervals = pd.read_csv(interval_path)
            labeled_manifest = apply_intervals_to_manifest(raw_manifest, intervals)
            labeled_path = labeled_manifests_dir / raw_manifest_path.name
            labeled_manifest.to_csv(labeled_path, index=False)
            labeled_manifest_paths.append(labeled_path)
        elif args.include_unlabeled:
            labeled_path = labeled_manifests_dir / raw_manifest_path.name
            raw_manifest.to_csv(labeled_path, index=False)
            labeled_manifest_paths.append(labeled_path)
        else:
            if args.create_missing_interval_templates:
                ensure_interval_template(interval_path, clip_id, str(row.get("subset", "unspecified")))
        coverage_rows.append(
            coverage_row(
                clip_id=clip_id,
                subset=str(row.get("subset", "unspecified")),
                source_video=str(row.get("source_video", "")),
                manifest=str(raw_manifest_path),
                intervals_csv=str(interval_path),
                has_intervals=has_intervals,
                included_in_eval=has_intervals or args.include_unlabeled,
            )
        )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(coverage_path, index=False)
    write_coverage_summary(coverage_summary_path, coverage_df)
    merged_labeled = merge_manifests(labeled_manifest_paths)
    merged_labeled.to_csv(merged_labeled_manifest_path, index=False)

    if not args.skip_eval and labeled_manifest_paths:
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_alerts.py",
                "--config",
                args.config,
                "--images",
                str(images_dir),
                "--manifest",
                str(merged_labeled_manifest_path),
                "--output-dir",
                str(reports_dir),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )

    summary = coverage_summary(coverage_df)
    if not args.skip_eval and summary["clips_included_in_eval"] == 0:
        print(f"output_root={output_root}")
        print(f"coverage={coverage_path}")
        print(f"coverage_summary={coverage_summary_path}")
        print("error=no_labeled_clips_for_evaluation")
        return 2

    print(f"output_root={output_root}")
    print(f"coverage={coverage_path}")
    print(f"coverage_summary={coverage_summary_path}")
    print(f"merged_manifest={merged_labeled_manifest_path}")
    print(f"included_clips={summary['clips_included_in_eval']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
