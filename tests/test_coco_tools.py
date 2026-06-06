from pathlib import Path

from src.coco_tools import prepare_coco_person_subset


def test_prepare_coco_person_subset_writes_yolo_labels_and_manifest(tmp_path: Path) -> None:
    annotations = tmp_path / "instances.json"
    images_dir = tmp_path / "images"
    output_root = tmp_path / "out"
    images_dir.mkdir()
    (images_dir / "000000000001.jpg").write_bytes(b"fake image")
    (images_dir / "000000000002.jpg").write_bytes(b"fake image")

    annotations.write_text(
        """
{
  "images": [
    {"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 200},
    {"id": 2, "file_name": "000000000002.jpg", "width": 50, "height": 50}
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 18, "name": "dog"}
  ],
  "annotations": [
    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
    {"id": 11, "image_id": 1, "category_id": 18, "bbox": [1, 2, 3, 4], "iscrowd": 0},
    {"id": 12, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 1}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    counts = prepare_coco_person_subset(
        annotations_json=annotations,
        images_dir=images_dir,
        output_root=output_root,
        split="val2017",
    )

    assert counts == {"images": 1, "boxes": 1, "missing_images": 0}
    assert (output_root / "val2017" / "images" / "000000000001.jpg").exists()
    assert (output_root / "val2017" / "labels" / "000000000001.txt").read_text(encoding="utf-8") == (
        "0 0.250000 0.200000 0.300000 0.200000"
    )
    manifest = (output_root / "val2017" / "manifest.csv").read_text(encoding="utf-8")
    assert "coco_person" in manifest
    assert "000000000001.jpg" in manifest
