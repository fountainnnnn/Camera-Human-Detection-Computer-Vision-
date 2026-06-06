import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.evaluate_alerts" in sys.modules:
            return importlib.reload(sys.modules["scripts.evaluate_alerts"])
        return importlib.import_module("scripts.evaluate_alerts")


def test_evaluate_alerts_applies_model_and_threshold_overrides(monkeypatch, tmp_path) -> None:
    module = load_module()
    captured = {}
    config = SimpleNamespace(
        model=SimpleNamespace(path="models/yolo11n.pt", confidence_threshold=0.6),
        detection=SimpleNamespace(
            rolling_window_size=3,
            min_positive_frames=2,
            min_detection_duration_seconds=1.0,
            alert_cooldown_seconds=60.0,
        ),
    )

    class FakeDetector:
        def __init__(self, received_config):
            captured["model_path"] = received_config.model.path
            captured["threshold"] = received_config.model.confidence_threshold

    class FakeManifest:
        columns = ["image", "clip_id", "timestamp_seconds", "has_human_gt"]

        def __setitem__(self, key, value):
            self.columns.append(key)

        def to_dict(self, orient):
            assert orient == "records"
            return []

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "YOLOPersonDetector", FakeDetector)
    monkeypatch.setattr(module.pd, "read_csv", lambda _path: FakeManifest())

    assert module.main(
        [
            "--images",
            str(tmp_path / "images"),
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--model-path",
            "runs/detect/train-5/weights/best.pt",
            "--confidence-threshold",
            "0.25",
        ]
    ) == 0

    assert captured["model_path"] == "runs/detect/train-5/weights/best.pt"
    assert captured["threshold"] == 0.25
