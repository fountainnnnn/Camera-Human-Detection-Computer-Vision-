import importlib
import sys
from types import SimpleNamespace

import scripts.common as script_common

sys.modules.setdefault("common", script_common)


def load_module():
    if "scripts.evaluate" in sys.modules:
        return sys.modules["scripts.evaluate"]
    return importlib.import_module("scripts.evaluate")


def test_load_evaluation_config_accepts_model_path_override(monkeypatch) -> None:
    module = load_module()
    config = SimpleNamespace(model=SimpleNamespace(path="models/yolo11n.pt"))

    monkeypatch.setattr(module, "load_config", lambda _path: config)

    result = module.load_evaluation_config("config.yaml", "runs/detect/train/weights/best.pt")

    assert result.model.path == "runs/detect/train/weights/best.pt"


def test_parse_thresholds_accepts_custom_csv() -> None:
    module = load_module()

    assert module.parse_thresholds("0.25, 0.5,0.95") == [0.25, 0.5, 0.95]


def test_parse_thresholds_rejects_out_of_range_values() -> None:
    module = load_module()

    try:
        module.parse_thresholds("0.5,1.2")
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")
