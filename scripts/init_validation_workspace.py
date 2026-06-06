from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.validation_tools import (
    create_validation_workspace,
    write_collection_checklist,
    write_validation_collection_guide,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the raw clip folders, intervals directory, and operator checklist for real-footage validation."
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--intervals-dir", default="data/intervals")
    parser.add_argument("--output-root", default="data/validation_setup")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    intervals_dir = Path(args.intervals_dir)
    output_root = Path(args.output_root)
    checklist_path = output_root / "validation_collection_checklist.csv"
    guide_path = output_root / "validation_collection_guide.md"

    create_validation_workspace(raw_root, intervals_dir)
    write_collection_checklist(checklist_path, raw_root, intervals_dir)
    write_validation_collection_guide(guide_path, raw_root, intervals_dir)

    print(f"raw_root={raw_root}")
    print(f"intervals_dir={intervals_dir}")
    print(f"checklist={checklist_path}")
    print(f"guide={guide_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
