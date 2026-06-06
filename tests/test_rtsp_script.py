import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch


def load_test_rtsp_module():
    fake_common = SimpleNamespace()
    fake_cv2 = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common, "cv2": fake_cv2}):
        if "scripts.test_rtsp" in sys.modules:
            return importlib.reload(sys.modules["scripts.test_rtsp"])
        return importlib.import_module("scripts.test_rtsp")


def test_run_ffprobe_returns_stdout_when_command_succeeds(monkeypatch) -> None:
    module = load_test_rtsp_module()

    monkeypatch.setattr(module.shutil, "which", lambda name: "C:/ffprobe.exe" if name == "ffprobe" else None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{\"streams\":[]}', stderr=""),
    )

    assert module.run_ffprobe("rtsp://camera/stream2", 10) == '{"streams":[]}'


def test_run_ffprobe_raises_when_binary_missing(monkeypatch) -> None:
    module = load_test_rtsp_module()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    try:
        module.run_ffprobe("rtsp://camera/stream2", 10)
    except RuntimeError as exc:
        assert "ffprobe" in str(exc)
    else:
        raise AssertionError("run_ffprobe should have raised when ffprobe is unavailable")
