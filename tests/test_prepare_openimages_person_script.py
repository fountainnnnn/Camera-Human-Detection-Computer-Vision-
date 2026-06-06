import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.prepare_openimages_person" in sys.modules:
            return importlib.reload(sys.modules["scripts.prepare_openimages_person"])
        return importlib.import_module("scripts.prepare_openimages_person")


def test_prepare_openimages_person_script_runs(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    classes = tmp_path / "classes.csv"
    annotations = tmp_path / "boxes.csv"
    output_root = tmp_path / "out"
    classes.write_text(
        "/m/person,Person\n/m/man,Man\n/m/woman,Woman\n/m/boy,Boy\n/m/girl,Girl\n",
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside",
                "img1,xclick,/m/person,1,0.10,0.50,0.20,0.80,0,0,0,0,0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_openimages_person.py",
            "--annotations-csv",
            str(annotations),
            "--class-descriptions-csv",
            str(classes),
            "--output-root",
            str(output_root),
            "--split",
            "train",
        ],
    )

    assert module.main() == 0
    assert (output_root / "train" / "labels" / "img1.txt").exists()
