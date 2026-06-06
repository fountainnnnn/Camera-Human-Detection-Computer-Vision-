import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_build_acceptance_report_module():
    fake_common = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common}):
        if "scripts.build_acceptance_report" in sys.modules:
            return importlib.reload(sys.modules["scripts.build_acceptance_report"])
        return importlib.import_module("scripts.build_acceptance_report")


def test_build_acceptance_report_script_writes_outputs_and_fails_when_targets_miss(tmp_path: Path) -> None:
    module = load_build_acceptance_report_module()

    clip_results = tmp_path / "alert_clip_results.csv"
    clip_results.write_text(
        "\n".join(
            [
                "clip_id,subset,alert_count,true_alert_count,false_alert_count,had_human_gt,alerted_on_positive_clip,missed_positive_clip,false_positive_clip,duration_seconds",
                "clip-a,day,1,1,0,True,True,False,False,30",
                "clip-b,ir,0,0,0,True,False,True,False,20",
                "clip-c,night_vision,1,0,1,False,False,False,True,3600",
            ]
        ),
        encoding="utf-8",
    )
    runtime_summary = tmp_path / "runtime_monitor_summary.json"
    runtime_summary.write_text(
        json.dumps(
            {
                "summary": {
                    "average_cpu_percent": 88.0,
                    "peak_cpu_percent": 96.0,
                    "average_fps": 0.8,
                    "disconnect_count": 1,
                    "stale_sample_count": 3,
                    "elapsed_seconds": 900.0,
                },
                "evaluation": {"passed": False, "checks": []},
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    exit_code = module.main(
        [
            "--clip-results",
            str(clip_results),
            "--runtime-summary",
            str(runtime_summary),
            "--output-dir",
            str(output_dir),
            "--min-alert-precision",
            "0.9",
            "--min-clip-recall",
            "0.8",
            "--max-false-positives-per-hour",
            "0.5",
            "--max-average-cpu-percent",
            "80",
            "--max-peak-cpu-percent",
            "95",
            "--min-average-fps",
            "1.0",
            "--max-disconnects",
            "0",
            "--max-stale-samples",
            "2",
            "--min-runtime-seconds",
            "1800",
        ]
    )

    assert exit_code == 1
    assert (output_dir / "acceptance_report.md").exists()
    payload = json.loads((output_dir / "acceptance_report.json").read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["metrics"]["alert_precision"] == 0.5
    assert "Fine-tune the model on IR/night footage" in (output_dir / "acceptance_report.md").read_text(encoding="utf-8")
