from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

import common  # noqa: F401
from src.video_tools import build_frame_image_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract sampled frames from a video clip and write a manifest.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-images", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--subset", default="unspecified")
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument(
        "--has-human-gt",
        action="store_true",
        help="Mark every extracted frame as human-present by default. Edit the manifest later for mixed clips.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_images = Path(args.output_images)
    output_manifest = Path(args.output_manifest)
    output_images.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        raise RuntimeError("Video FPS could not be determined")
    frame_interval = max(1, int(round(args.sample_seconds * fps)))

    rows = []
    frame_index = 0
    saved_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % frame_interval == 0:
                timestamp_seconds = frame_index / fps
                image_name = build_frame_image_name(args.clip_id, saved_index)
                image_path = output_images / image_name
                cv2.imwrite(str(image_path), frame)
                rows.append(
                    {
                        "image": image_name,
                        "clip_id": args.clip_id,
                        "subset": args.subset,
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "has_human_gt": bool(args.has_human_gt),
                    }
                )
                saved_index += 1
            frame_index += 1
    finally:
        capture.release()

    pd.DataFrame(rows).to_csv(output_manifest, index=False)
    print(f"frames={saved_index}")
    print(f"manifest={output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
