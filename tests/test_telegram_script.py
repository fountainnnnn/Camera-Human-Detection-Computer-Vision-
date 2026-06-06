import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_test_telegram_module(fake_cv2):
    fake_common = SimpleNamespace()
    fake_dotenv = SimpleNamespace(load_dotenv=lambda: None)
    with patch.dict(sys.modules, {"common": fake_common, "cv2": fake_cv2, "dotenv": fake_dotenv}):
        if "scripts.test_telegram" in sys.modules:
            return importlib.reload(sys.modules["scripts.test_telegram"])
        return importlib.import_module("scripts.test_telegram")


def test_test_telegram_script_creates_image_and_sends(monkeypatch, tmp_path: Path, capsys) -> None:
    writes = []
    text_calls = []
    fake_cv2 = SimpleNamespace(
        putText=lambda *args: text_calls.append(args),
        imwrite=lambda path, image: writes.append((path, image)) or True,
        FONT_HERSHEY_SIMPLEX=0,
    )
    module = load_test_telegram_module(fake_cv2)
    sent = []
    config = SimpleNamespace(telegram=SimpleNamespace(enabled=False))

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(enabled=False, send_photo=lambda path, caption: sent.append((Path(path), caption)) or True)),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["test_telegram.py"])

    assert module.main() == 0
    assert text_calls
    assert writes
    assert sent
    assert sent[0][0].name == "telegram_test.jpg"
    assert "Security AI Telegram test alert" == sent[0][1]
    assert "sent=reports" in capsys.readouterr().out
