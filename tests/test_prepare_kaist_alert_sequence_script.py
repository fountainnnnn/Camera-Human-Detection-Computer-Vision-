import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.prepare_kaist_alert_sequence" in sys.modules:
            return importlib.reload(sys.modules["scripts.prepare_kaist_alert_sequence"])
        return importlib.import_module("scripts.prepare_kaist_alert_sequence")


def test_prepare_kaist_alert_sequence_script_runs(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    calls = []

    def fake_prepare_kaist_alert_sequence(**kwargs):
        calls.append(kwargs)
        return {
            "frames": 3,
            "positive_frames": 1,
            "negative_frames": 2,
            "clips": 2,
            "missing_images": 0,
        }

    monkeypatch.setattr(module, "prepare_kaist_alert_sequence", fake_prepare_kaist_alert_sequence)

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
            "--source-fps",
            "20",
            "--sample-fps",
            "2",
            "--window-seconds",
            "30",
        ]
    ) == 0
    assert calls[0]["modality"] == "lwir"
    assert calls[0]["source_fps"] == 20
    assert calls[0]["sample_fps"] == 2
    assert calls[0]["window_seconds"] == 30
