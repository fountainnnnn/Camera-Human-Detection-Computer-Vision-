import importlib
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.fetch_coco_assets" in sys.modules:
            return importlib.reload(sys.modules["scripts.fetch_coco_assets"])
        return importlib.import_module("scripts.fetch_coco_assets")


def test_fetch_coco_assets_reuses_downloaded_archives_and_extracts(tmp_path: Path) -> None:
    module = load_module()
    archive = tmp_path / "annotations_trainval2017.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("annotations/instances_val2017.json", "{}")

    module.DEFAULT_URLS["annotations"] = "http://example.test/annotations_trainval2017.zip"

    def fake_urlretrieve(url: str, target: Path) -> None:
        target.write_bytes(archive.read_bytes())

    module.urlretrieve = fake_urlretrieve

    downloaded = module.fetch_coco_assets(tmp_path / "out", include_val_images=False)

    assert downloaded["annotations"].exists()
    assert (tmp_path / "out" / "annotations" / "instances_val2017.json").exists()
