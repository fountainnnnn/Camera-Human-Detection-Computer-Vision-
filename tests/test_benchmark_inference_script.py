import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace()}):
        if "scripts.benchmark_inference" in sys.modules:
            return importlib.reload(sys.modules["scripts.benchmark_inference"])
        return importlib.import_module("scripts.benchmark_inference")


def test_benchmark_inference_script_writes_reports(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "reports"
    image_dir.mkdir()
    (image_dir / "frame.jpg").write_bytes(b"fake")

    fake_report = SimpleNamespace(summary=SimpleNamespace(), frames=[])

    monkeypatch.setattr(module, "YOLOPersonDetector", lambda _config: object())
    monkeypatch.setattr(module, "run_inference_benchmark", lambda **_kwargs: fake_report)
    monkeypatch.setattr(module, "write_benchmark_json", lambda path, _report: Path(path).write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(module, "write_benchmark_markdown", lambda path, _report: Path(path).write_text("# ok", encoding="utf-8"))

    exit_code = module.main(
        [
            "--images",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--inference-size",
            "512",
            "--confidence-threshold",
            "0.6",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "inference_benchmark.json").exists()
    assert (output_dir / "inference_benchmark.md").exists()
