import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.prepare_kaist_person" in sys.modules:
            return importlib.reload(sys.modules["scripts.prepare_kaist_person"])
        return importlib.import_module("scripts.prepare_kaist_person")


def test_prepare_kaist_person_script_runs(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    calls = []

    def fake_prepare_kaist_person_subset(**kwargs):
        calls.append(kwargs)
        return {"images": 1, "boxes": 2, "missing_images": 0, "empty_labels": 0}

    monkeypatch.setattr(module, "prepare_kaist_person_subset", fake_prepare_kaist_person_subset)

    assert module.main(
        [
            "--annotations-dir",
            str(tmp_path / "annotations"),
            "--images-dir",
            str(tmp_path / "images"),
            "--output-root",
            str(tmp_path / "out"),
            "--modality",
            "lwir",
            "--human-label",
            "person",
            "--human-label",
            "people",
        ]
    ) == 0
    assert calls[0]["modality"] == "lwir"
    assert calls[0]["human_labels"] == ("person", "people")
