from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.config import load_config, resolve_model_device


def test_load_config_uses_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.yaml")
    assert config.input.mode == "webcam"
    assert config.input.loop_image_folder is False
    assert config.model.device == "auto"
    assert config.runtime.max_runtime_seconds is None


def test_resolve_model_device_falls_back_to_cpu_when_torch_missing() -> None:
    with patch("src.config.importlib.import_module", side_effect=ModuleNotFoundError):
        assert resolve_model_device("auto") == "cpu"


def test_resolve_model_device_prefers_cuda_when_available() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    with patch("src.config.importlib.import_module", return_value=fake_torch):
        assert resolve_model_device("auto") == "cuda:0"


def test_resolve_model_device_keeps_explicit_request() -> None:
    assert resolve_model_device("cpu") == "cpu"
