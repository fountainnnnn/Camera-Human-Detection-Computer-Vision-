import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeYOLO:
    last_instance = None

    def __init__(self, path: str) -> None:
        self.path = path
        self.train_kwargs = None
        FakeYOLO.last_instance = self

    def train(self, **kwargs) -> None:
        self.train_kwargs = kwargs


def load_train_module():
    fake_common = SimpleNamespace()
    fake_ultralytics = SimpleNamespace(YOLO=FakeYOLO)
    fake_yaml = SimpleNamespace(safe_dump=lambda payload, sort_keys=False: str(payload))
    with patch.dict(sys.modules, {"common": fake_common, "ultralytics": fake_ultralytics, "yaml": fake_yaml}):
        if "scripts.train" in sys.modules:
            return importlib.reload(sys.modules["scripts.train"])
        return importlib.import_module("scripts.train")


def test_train_script_writes_dataset_yaml_and_calls_yolo_train(monkeypatch, tmp_path: Path) -> None:
    module = load_train_module()
    config = SimpleNamespace(model=SimpleNamespace(path="models/yolo11n.pt", device="auto"))

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "resolve_model_device", lambda value: "cpu" if value == "auto" else value)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--dataset-root", str(tmp_path), "--epochs", "5", "--imgsz", "512"],
    )

    assert module.main() == 0
    dataset_yaml = tmp_path / "dataset.yaml"
    assert dataset_yaml.exists()
    text = dataset_yaml.read_text(encoding="utf-8")
    assert "train/images" in text
    assert "val/images" in text
    assert "test/images" in text
    assert "person" in text
    assert FakeYOLO.last_instance is not None
    assert FakeYOLO.last_instance.path == "models/yolo11n.pt"
    assert FakeYOLO.last_instance.train_kwargs["epochs"] == 5
    assert FakeYOLO.last_instance.train_kwargs["imgsz"] == 512
    assert FakeYOLO.last_instance.train_kwargs["device"] == "cpu"
    assert FakeYOLO.last_instance.train_kwargs["mosaic"] == 0.2
    assert FakeYOLO.last_instance.train_kwargs["erasing"] == 0.2
