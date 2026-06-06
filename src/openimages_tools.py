from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PERSON_LABELS = ("Person", "Man", "Woman", "Boy", "Girl")


@dataclass(frozen=True)
class OpenImagesBox:
    image_id: str
    label_name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    is_group_of: bool

    @property
    def yolo_line(self) -> str:
        x_center = (self.xmin + self.xmax) / 2
        y_center = (self.ymin + self.ymax) / 2
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def load_class_map(class_descriptions_csv: Path) -> dict[str, str]:
    with class_descriptions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return {row[0]: row[1] for row in reader if len(row) >= 2}


def selected_label_mids(class_map: dict[str, str], label_names: tuple[str, ...]) -> set[str]:
    wanted = {name.lower() for name in label_names}
    mids = {mid for mid, display_name in class_map.items() if display_name.lower() in wanted}
    missing = sorted(wanted - {class_map[mid].lower() for mid in mids})
    if missing:
        raise ValueError(f"missing Open Images class descriptions: {', '.join(missing)}")
    return mids


def iter_person_boxes(
    annotations_csv: Path,
    selected_mids: set[str],
    include_group_of: bool = False,
) -> list[OpenImagesBox]:
    boxes: list[OpenImagesBox] = []
    with annotations_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["LabelName"] not in selected_mids:
                continue
            is_group_of = row.get("IsGroupOf", "0") == "1"
            if is_group_of and not include_group_of:
                continue
            boxes.append(
                OpenImagesBox(
                    image_id=row["ImageID"],
                    label_name=row["LabelName"],
                    xmin=float(row["XMin"]),
                    xmax=float(row["XMax"]),
                    ymin=float(row["YMin"]),
                    ymax=float(row["YMax"]),
                    is_group_of=is_group_of,
                )
            )
    return boxes


def prepare_openimages_person_subset(
    annotations_csv: Path,
    class_descriptions_csv: Path,
    output_root: Path,
    split: str,
    label_names: tuple[str, ...] = DEFAULT_PERSON_LABELS,
    max_images: int | None = None,
    include_group_of: bool = False,
) -> dict[str, int]:
    class_map = load_class_map(class_descriptions_csv)
    selected_mids = selected_label_mids(class_map, label_names)
    boxes = iter_person_boxes(annotations_csv, selected_mids, include_group_of=include_group_of)

    grouped: dict[str, list[OpenImagesBox]] = {}
    for box in boxes:
        if max_images is not None and box.image_id not in grouped and len(grouped) >= max_images:
            continue
        grouped.setdefault(box.image_id, []).append(box)

    labels_dir = output_root / split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_ids_path = output_root / split / "image_ids.txt"
    manifest_path = output_root / split / "manifest.csv"
    image_ids_path.parent.mkdir(parents=True, exist_ok=True)

    with image_ids_path.open("w", encoding="utf-8", newline="\n") as image_ids:
        for image_id in sorted(grouped):
            image_ids.write(f"{split}/{image_id}\n")
            lines = [box.yolo_line for box in grouped[image_id]]
            (labels_dir / f"{image_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "label", "split", "subset", "has_human_gt", "source_dataset", "image_id"],
        )
        writer.writeheader()
        for image_id in sorted(grouped):
            writer.writerow(
                {
                    "image": f"{image_id}.jpg",
                    "label": f"{image_id}.txt",
                    "split": split,
                    "subset": "openimages_person",
                    "has_human_gt": "true",
                    "source_dataset": "openimages",
                    "image_id": image_id,
                }
            )

    return {"images": len(grouped), "boxes": sum(len(items) for items in grouped.values())}
