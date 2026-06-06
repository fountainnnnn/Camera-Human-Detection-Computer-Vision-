import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.fetch_kaist_assets" in sys.modules:
            return importlib.reload(sys.modules["scripts.fetch_kaist_assets"])
        return importlib.import_module("scripts.fetch_kaist_assets")


def test_fetch_kaist_assets_print_only_outputs_gdown_command(tmp_path: Path, capsys) -> None:
    module = load_module()

    assert module.main(["--output-dir", str(tmp_path), "--variant", "preview", "--print-only"]) == 0

    output = capsys.readouterr().out
    assert "gdown" in output
    assert "11nhHpmuh2FUjrLNfGs51R2Mqqy1GTjY8" in output


def test_fetch_kaist_assets_invokes_gdown(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    target = module.fetch_kaist_assets(tmp_path, "preview")

    assert target == tmp_path / "kaist_preview"
    assert calls[0][1] is True
    assert "gdown" in calls[0][0]
