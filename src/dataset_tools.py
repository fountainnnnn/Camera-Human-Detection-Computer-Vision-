from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetExample:
    image_path: Path
    label_path: Path | None
    subset: str
    has_human_gt: bool


def _subset_for_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "unspecified"
    if len(relative.parts) <= 1:
        return "unspecified"
    return relative.parts[0]


def collect_examples(
    images_dir: Path,
    labels_dir: Path,
    negatives_dir: Path | None = None,
) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            examples.append(
                DatasetExample(
                    image_path=image_path,
                    label_path=label_path,
                    subset=_subset_for_path(image_path, images_dir),
                    has_human_gt=True,
                )
            )
    if negatives_dir is not None and negatives_dir.exists():
        for image_path in sorted(negatives_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            examples.append(
                DatasetExample(
                    image_path=image_path,
                    label_path=None,
                    subset=_subset_for_path(image_path, negatives_dir),
                    has_human_gt=False,
                )
            )
    return examples


def split_examples(
    examples: list[DatasetExample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[DatasetExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
