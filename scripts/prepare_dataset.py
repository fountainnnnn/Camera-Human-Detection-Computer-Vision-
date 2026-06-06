from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import common  # noqa: F401
import pandas as pd

from src.dataset_tools import DatasetExample, collect_examples, split_examples


def copy_split(items: list[DatasetExample], split: str, output_root: Path) -> None:
    image_dir = output_root / split / "images"
    label_dir = output_root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for item in items:
        target_image = image_dir / item.image_path.name
        target_label = label_dir / f"{item.image_path.stem}.txt"
        shutil.copy2(item.image_path, target_image)
        if item.label_path is not None:
            shutil.copy2(item.label_path, target_label)
        else:
            target_label.write_text("", encoding="utf-8")
        manifest_rows.append(
            {
                "image": target_image.name,
                "label": target_label.name,
                "split": split,
                "subset": item.subset,
                "has_human_gt": item.has_human_gt,
                "source_image": str(item.image_path),
                "source_label": str(item.label_path) if item.label_path is not None else "",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output_root / split / "manifest.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO train/val/test dataset splits.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--negatives", help="Optional directory of negative images with no humans.")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = collect_examples(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        negatives_dir=Path(args.negatives) if args.negatives else None,
    )
    output_root = Path(args.output_root)
    splits = split_examples(
        examples=items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, split_items in splits.items():
        copy_split(split_items, split_name, output_root)
    print(f"prepared={len(items)} output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
