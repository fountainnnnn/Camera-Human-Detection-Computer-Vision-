import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.download_openimages_images" in sys.modules:
            return importlib.reload(sys.modules["scripts.download_openimages_images"])
        return importlib.import_module("scripts.download_openimages_images")


def test_download_images_uses_manifest_rows(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    image_ids = tmp_path / "image_ids.txt"
    output_root = tmp_path / "out"
    image_ids.write_text("validation/img1\nvalidation/img2\n", encoding="utf-8")
    requested: list[str] = []

    def fake_urlretrieve(url: str, target: Path):
        requested.append(url)
        target.write_bytes(b"jpg")
        return str(target), None

    monkeypatch.setattr(module, "urlretrieve", fake_urlretrieve)
    counts = module.download_images(image_ids, output_root, max_images=1, base_url="https://example.test")

    assert counts == {"downloaded": 1, "skipped": 0, "failed": 0}
    assert requested == ["https://example.test/validation/img1.jpg"]
    assert (output_root / "validation" / "images" / "img1.jpg").read_bytes() == b"jpg"
