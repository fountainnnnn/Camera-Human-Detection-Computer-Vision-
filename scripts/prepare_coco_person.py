from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.coco_tools import prepare_coco_person_subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a COCO person-only subset with YOLO labels.")
    parser.add_argument("--annotations-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-root", default="data/coco_person")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--include-crowd", action="store_true")
    args = parser.parse_args()

    counts = prepare_coco_person_subset(
        annotations_json=Path(args.annotations_json),
        images_dir=Path(args.images_dir),
        output_root=Path(args.output_root),
        split=args.split,
        max_images=args.max_images,
        include_crowd=args.include_crowd,
    )
    print(
        "images={images} boxes={boxes} missing_images={missing_images} output={output}".format(
            **counts,
            output=args.output_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
