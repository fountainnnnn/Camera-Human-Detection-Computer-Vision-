from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.openimages_tools import DEFAULT_PERSON_LABELS, prepare_openimages_person_subset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Open Images person bounding-box CSVs into YOLO labels and image-id manifests."
    )
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--class-descriptions-csv", required=True)
    parser.add_argument("--output-root", default="data/openimages_person")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--label-name", action="append", dest="label_names")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--include-group-of", action="store_true")
    args = parser.parse_args()

    counts = prepare_openimages_person_subset(
        annotations_csv=Path(args.annotations_csv),
        class_descriptions_csv=Path(args.class_descriptions_csv),
        output_root=Path(args.output_root),
        split=args.split,
        label_names=tuple(args.label_names or DEFAULT_PERSON_LABELS),
        max_images=args.max_images,
        include_group_of=args.include_group_of,
    )
    print(f"images={counts['images']} boxes={counts['boxes']} output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
