import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.prepare_coco_person" in sys.modules:
            return importlib.reload(sys.modules["scripts.prepare_coco_person"])
        return importlib.import_module("scripts.prepare_coco_person")


def test_prepare_coco_person_script_runs(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    annotations = tmp_path / "instances.json"
    images_dir = tmp_path / "images"
    output_root = tmp_path / "out"
    images_dir.mkdir()
    (images_dir / "000000000001.jpg").write_bytes(b"fake image")
    annotations.write_text(
        """
{
  "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 100}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0}]
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_coco_person.py",
            "--annotations-json",
            str(annotations),
            "--images-dir",
            str(images_dir),
            "--output-root",
            str(output_root),
            "--split",
            "val2017",
        ],
    )

    assert module.main() == 0
    assert (output_root / "val2017" / "labels" / "000000000001.txt").exists()
