import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_review_validation_run_module():
    fake_common = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common}):
        if "scripts.review_validation_run" in sys.modules:
            return importlib.reload(sys.modules["scripts.review_validation_run"])
        return importlib.import_module("scripts.review_validation_run")


def test_review_validation_run_script_orchestrates_reports(monkeypatch, tmp_path: Path) -> None:
    module = load_review_validation_run_module()

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    runtime_dir = reports_dir / "runtime_monitor"
    runtime_dir.mkdir()

    for path in [
        reports_dir / "alert_clip_results.csv",
        reports_dir / "alert_frame_predictions.csv",
        runtime_dir / "runtime_monitor_summary.json",
    ]:
        path.write_text("placeholder", encoding="utf-8")

    calls = []

    monkeypatch.setattr(
        module,
        "run_analysis",
        lambda reports_root: calls.append(("analysis", Path(reports_root))) or 0,
    )
    monkeypatch.setattr(
        module,
        "run_tuning",
        lambda reports_root: calls.append(("tuning", Path(reports_root))) or 0,
    )
    monkeypatch.setattr(
        module,
        "run_acceptance",
        lambda reports_root, runtime_summary: calls.append(("acceptance", Path(reports_root), Path(runtime_summary))) or 0,
    )

    exit_code = module.main(
        [
            "--reports-dir",
            str(reports_dir),
            "--runtime-summary",
            str(runtime_dir / "runtime_monitor_summary.json"),
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("analysis", reports_dir),
        ("tuning", reports_dir),
        ("acceptance", reports_dir, runtime_dir / "runtime_monitor_summary.json"),
    ]
    index_path = reports_dir / "validation_review_index.md"
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")
    assert "alert_failure_analysis.md" in content
    assert "alert_threshold_recommendations.md" in content
    assert "acceptance_report.md" in content
