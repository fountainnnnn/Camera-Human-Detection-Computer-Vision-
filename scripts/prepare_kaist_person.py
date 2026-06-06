from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.kaist_tools import DEFAULT_HUMAN_LABELS, prepare_kaist_person_subset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a KAIST visible or LWIR person-only YOLO subset.")
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-root", default="data/kaist_person")
    parser.add_argument("--split", default="train")
    parser.add_argument("--modality", choices=("visible", "lwir"), default="visible")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--human-label", action="append", dest="human_labels")
    args = parser.parse_args(argv)

    counts = prepare_kaist_person_subset(
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
        output_root=Path(args.output_root),
        split=args.split,
        modality=args.modality,
        max_images=args.max_images,
        human_labels=tuple(args.human_labels or DEFAULT_HUMAN_LABELS),
    )
    print(
        "images={images} boxes={boxes} missing_images={missing_images} empty_labels={empty_labels} output={output}".format(
            **counts,
            output=args.output_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
