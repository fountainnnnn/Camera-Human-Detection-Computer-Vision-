import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_smoke_validation_review_module():
    fake_common = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common}):
        if "scripts.smoke_validation_review" in sys.modules:
            return importlib.reload(sys.modules["scripts.smoke_validation_review"])
        return importlib.import_module("scripts.smoke_validation_review")


def test_smoke_validation_review_creates_demo_reports_and_runs_bundle(tmp_path: Path) -> None:
    module = load_smoke_validation_review_module()
    calls = []

    monkeypatch_target = module
    monkeypatch_target.run_review_bundle = lambda reports_dir, runtime_summary: calls.append((Path(reports_dir), Path(runtime_summary))) or 0

    output_dir = tmp_path / "demo_reports"
    exit_code = module.main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "alert_clip_results.csv").exists()
    assert (output_dir / "alert_frame_predictions.csv").exists()
    assert (output_dir / "runtime_monitor" / "runtime_monitor_summary.json").exists()
    assert calls == [(output_dir, output_dir / "runtime_monitor" / "runtime_monitor_summary.json")]
