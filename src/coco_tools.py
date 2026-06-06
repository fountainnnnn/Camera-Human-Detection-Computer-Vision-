from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


PERSON_CATEGORY = "person"


@dataclass(frozen=True)
class CocoBox:
    image_id: int
    file_name: str
    width: int
    height: int
    bbox: tuple[float, float, float, float]

    @property
    def yolo_line(self) -> str:
        x, y, width, height = self.bbox
        image_width = float(self.width)
        image_height = float(self.height)
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        return f"0 {x_center:.6f} {y_center:.6f} {width / image_width:.6f} {height / image_height:.6f}"


def _load_payload(annotations_json: Path) -> dict[str, object]:
    return json.loads(annotations_json.read_text(encoding="utf-8"))


def _person_category_id(payload: dict[str, object]) -> int:
    for category in payload.get("categories", []):
        if isinstance(category, dict) and str(category.get("name", "")).lower() == PERSON_CATEGORY:
            return int(category["id"])
    raise ValueError("COCO annotations do not contain a person category")


def iter_coco_person_boxes(
    annotations_json: Path,
    max_images: int | None = None,
    include_crowd: bool = False,
) -> list[CocoBox]:
    payload = _load_payload(annotations_json)
    person_id = _person_category_id(payload)
    images = {
        int(image["id"]): image
        for image in payload.get("images", [])
        if isinstance(image, dict) and "id" in image
    }

    boxes_by_image: dict[int, list[CocoBox]] = {}
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        if int(annotation.get("category_id", -1)) != person_id:
            continue
        if annotation.get("iscrowd", 0) and not include_crowd:
            continue
        image_id = int(annotation["image_id"])
        image = images.get(image_id)
        if image is None:
            continue
        if max_images is not None and image_id not in boxes_by_image and len(boxes_by_image) >= max_images:
            continue
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            continue
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue
        boxes_by_image.setdefault(image_id, []).append(
            CocoBox(
                image_id=image_id,
                file_name=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
                bbox=tuple(float(value) for value in bbox),
            )
        )

    return [box for image_id in sorted(boxes_by_image) for box in boxes_by_image[image_id]]


def prepare_coco_person_subset(
    annotations_json: Path,
    images_dir: Path,
    output_root: Path,
    split: str,
    max_images: int | None = None,
    include_crowd: bool = False,
) -> dict[str, int]:
    boxes = iter_coco_person_boxes(
        annotations_json=annotations_json,
        max_images=max_images,
        include_crowd=include_crowd,
    )

    grouped: dict[str, list[CocoBox]] = {}
    for box in boxes:
        grouped.setdefault(box.file_name, []).append(box)

    output_images = output_root / split / "images"
    output_labels = output_root / split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / split / "manifest.csv"

    copied_images = 0
    missing_images = 0
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write("image,label,split,subset,has_human_gt,source_dataset,image_id,source_image\n")
        for file_name, image_boxes in sorted(grouped.items()):
            source_image = images_dir / file_name
            if not source_image.exists():
                missing_images += 1
                continue
            target_image = output_images / file_name
            target_label = output_labels / f"{Path(file_name).stem}.txt"
            shutil.copy2(source_image, target_image)
            target_label.write_text(
                "\n".join(box.yolo_line for box in image_boxes),
                encoding="utf-8",
            )
            copied_images += 1
            manifest.write(
                f"{file_name},{target_label.name},{split},coco_person,true,coco,"
                f"{image_boxes[0].image_id},{source_image}\n"
            )

    return {"images": copied_images, "boxes": sum(len(items) for items in grouped.values()), "missing_images": missing_images}
