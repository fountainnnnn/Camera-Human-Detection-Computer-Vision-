from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a starter frame manifest for evaluation footage.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", default="data/test/manifest.csv")
    parser.add_argument("--subset", default="unspecified")
    parser.add_argument("--clip-id", default="clip-001")
    parser.add_argument("--frame-interval-seconds", type=float, default=0.5)
    args = parser.parse_args()

    image_dir = Path(args.images)
    rows = []
    for index, image_path in enumerate(sorted(image_dir.glob("*"))):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rows.append(
            {
                "image": image_path.name,
                "clip_id": args.clip_id,
                "subset": args.subset,
                "timestamp_seconds": round(index * args.frame_interval_seconds, 3),
                "has_human_gt": False,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"manifest={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
