from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd
import common  # noqa: F401

from src.manifest_tools import merge_manifests
from src.video_tools import infer_clip_metadata, list_video_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract sampled frames and manifests for every raw validation clip.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-images", required=True)
    parser.add_argument("--output-manifests-dir", required=True)
    parser.add_argument("--merged-manifest", required=True)
    parser.add_argument("--clip-index", default="")
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--default-subset", default="unspecified")
    parser.add_argument(
        "--has-human-gt",
        action="store_true",
        help="Mark every extracted frame as human-present by default. Edit manifests later for mixed clips.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_images = Path(args.output_images)
    manifests_dir = Path(args.output_manifests_dir)
    merged_manifest = Path(args.merged_manifest)
    clip_index_path = Path(args.clip_index) if args.clip_index else merged_manifest.parent / "clip_index.csv"

    output_images.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    merged_manifest.parent.mkdir(parents=True, exist_ok=True)
    clip_index_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    clip_rows: list[dict[str, object]] = []

    for video_path in list_video_files(raw_root):
        metadata = infer_clip_metadata(video_path, raw_root, default_subset=args.default_subset)
        manifest_path = manifests_dir / f"{metadata.clip_id}.csv"
        command = [
            sys.executable,
            "scripts/extract_video_frames.py",
            "--video",
            str(video_path),
            "--output-images",
            str(output_images),
            "--output-manifest",
            str(manifest_path),
            "--clip-id",
            metadata.clip_id,
            "--subset",
            metadata.subset,
            "--sample-seconds",
            str(args.sample_seconds),
        ]
        if args.has_human_gt:
            command.append("--has-human-gt")
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
        manifest_paths.append(manifest_path)

        manifest_df = pd.read_csv(manifest_path)
        duration_seconds = (
            float(manifest_df["timestamp_seconds"].max()) - float(manifest_df["timestamp_seconds"].min())
            if not manifest_df.empty
            else 0.0
        )
        clip_rows.append(
            {
                "clip_id": metadata.clip_id,
                "subset": metadata.subset,
                "source_video": str(video_path),
                "frames": len(manifest_df),
                "duration_seconds": round(duration_seconds, 3),
                "manifest": str(manifest_path),
            }
        )

    merged = merge_manifests(manifest_paths)
    merged.to_csv(merged_manifest, index=False)
    pd.DataFrame(clip_rows).to_csv(clip_index_path, index=False)
    print(f"merged_manifest={merged_manifest}")
    print(f"clip_index={clip_index_path}")
    print(f"clips={len(clip_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
