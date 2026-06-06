from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import common  # noqa: F401


PERSON_CLASS_ID = 0


def convert_bbox_to_yolo(width: float, height: float, xmin: float, ymin: float, xmax: float, ymax: float) -> str:
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"{PERSON_CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_voc(xml_dir: Path, output_dir: Path) -> int:
    converted = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for xml_path in xml_dir.glob("*.xml"):
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            continue
        width = float(size.findtext("width", "1"))
        height = float(size.findtext("height", "1"))
        lines: list[str] = []
        for obj in root.findall("object"):
            if obj.findtext("name", "").lower() != "person":
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            line = convert_bbox_to_yolo(
                width,
                height,
                float(bbox.findtext("xmin", "0")),
                float(bbox.findtext("ymin", "0")),
                float(bbox.findtext("xmax", "0")),
                float(bbox.findtext("ymax", "0")),
            )
            lines.append(line)
        (output_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        converted += 1
    return converted


def convert_coco(annotation_file: Path, output_dir: Path) -> int:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload.get("images", [])}
    categories = {item["id"]: item["name"].lower() for item in payload.get("categories", [])}
    grouped: dict[int, list[str]] = {}

    for annotation in payload.get("annotations", []):
        if categories.get(annotation["category_id"]) != "person":
            continue
        image = images.get(annotation["image_id"])
        if image is None:
            continue
        x, y, w, h = annotation["bbox"]
        line = convert_bbox_to_yolo(
            float(image["width"]),
            float(image["height"]),
            float(x),
            float(y),
            float(x + w),
            float(y + h),
        )
        grouped.setdefault(annotation["image_id"], []).append(line)

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, lines in grouped.items():
        stem = Path(images[image_id]["file_name"]).stem
        (output_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(grouped)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert VOC or COCO person annotations to YOLO format.")
    parser.add_argument("--format", choices=["voc", "coco"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.format == "voc":
        converted = convert_voc(Path(args.input), Path(args.output))
    else:
        converted = convert_coco(Path(args.input), Path(args.output))
    print(f"converted={converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
