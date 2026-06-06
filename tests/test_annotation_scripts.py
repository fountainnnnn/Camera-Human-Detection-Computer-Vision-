import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_convert_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.convert_annotations" in sys.modules:
            return importlib.reload(sys.modules["scripts.convert_annotations"])
        return importlib.import_module("scripts.convert_annotations")


def test_convert_voc_keeps_only_person_labels(tmp_path: Path) -> None:
    module = load_convert_module()
    xml_dir = tmp_path / "xml"
    output_dir = tmp_path / "labels"
    xml_dir.mkdir()

    (xml_dir / "frame1.xml").write_text(
        """
<annotation>
  <size><width>200</width><height>100</height></size>
  <object><name>person</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax></bndbox></object>
  <object><name>cat</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
</annotation>
""".strip(),
        encoding="utf-8",
    )

    converted = module.convert_voc(xml_dir, output_dir)

    assert converted == 1
    lines = (output_dir / "frame1.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["0 0.200000 0.300000 0.200000 0.400000"]


def test_convert_coco_keeps_only_person_annotations(tmp_path: Path) -> None:
    module = load_convert_module()
    annotation_file = tmp_path / "annotations.json"
    output_dir = tmp_path / "labels"
    annotation_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "frame1.jpg", "width": 200, "height": 100}],
                "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [20, 10, 40, 40]},
                    {"image_id": 1, "category_id": 2, "bbox": [1, 1, 2, 2]},
                ],
            }
        ),
        encoding="utf-8",
    )

    converted = module.convert_coco(annotation_file, output_dir)

    assert converted == 1
    lines = (output_dir / "frame1.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["0 0.200000 0.300000 0.200000 0.400000"]


def load_catalog_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.download_datasets" in sys.modules:
            return importlib.reload(sys.modules["scripts.download_datasets"])
        return importlib.import_module("scripts.download_datasets")


def test_dataset_catalog_records_access_mode_for_all_sources() -> None:
    module = load_catalog_module()

    assert module.DATASET_CATALOG
    assert all(entry["access_mode"] in {"direct", "form_gated", "manual_review"} for entry in module.DATASET_CATALOG.values())
    assert all(entry["source_url"].startswith("https://") for entry in module.DATASET_CATALOG.values())
    assert module.DATASET_CATALOG["openimages"]["manual_download"] is False
