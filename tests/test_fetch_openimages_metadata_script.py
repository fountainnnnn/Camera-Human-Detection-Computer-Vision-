import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.fetch_openimages_metadata" in sys.modules:
            return importlib.reload(sys.modules["scripts.fetch_openimages_metadata"])
        return importlib.import_module("scripts.fetch_openimages_metadata")


def test_fetch_metadata_downloads_defaults_without_train(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    requested: list[str] = []

    def fake_urlretrieve(url: str, target: Path):
        requested.append(url)
        target.write_text("csv", encoding="utf-8")
        return str(target), None

    monkeypatch.setattr(module, "urlretrieve", fake_urlretrieve)
    downloaded = module.fetch_metadata(tmp_path)

    assert set(downloaded) == {"class_descriptions", "validation_boxes", "test_boxes"}
    assert all(path.exists() for path in downloaded.values())
    assert not any("train-annotations" in url for url in requested)


def test_fetch_metadata_can_include_large_train_boxes(tmp_path: Path, monkeypatch) -> None:
    module = load_module()

    def fake_urlretrieve(url: str, target: Path):
        target.write_text(url, encoding="utf-8")
        return str(target), None

    monkeypatch.setattr(module, "urlretrieve", fake_urlretrieve)
    downloaded = module.fetch_metadata(tmp_path, include_train_boxes=True)

    assert "train_boxes" in downloaded
    assert "oidv6-train-annotations-bbox.csv" in downloaded["train_boxes"].name
