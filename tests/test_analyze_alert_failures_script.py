import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_analyze_alert_failures_module():
    fake_common = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common}):
        if "scripts.analyze_alert_failures" in sys.modules:
            return importlib.reload(sys.modules["scripts.analyze_alert_failures"])
        return importlib.import_module("scripts.analyze_alert_failures")


def test_analyze_alert_failures_script_writes_reports(tmp_path: Path) -> None:
    module = load_analyze_alert_failures_module()

    clip_results = tmp_path / "alert_clip_results.csv"
    clip_results.write_text(
        "\n".join(
            [
                "clip_id,subset,alert_count,true_alert_count,false_alert_count,had_human_gt,alerted_on_positive_clip,missed_positive_clip,false_positive_clip,duration_seconds",
                "ir-missed,ir,0,0,0,True,False,True,False,22",
                "night-fp,night_vision,2,0,2,False,False,False,True,60",
            ]
        ),
        encoding="utf-8",
    )
    frame_predictions = tmp_path / "alert_frame_predictions.csv"
    frame_predictions.write_text(
        "\n".join(
            [
                "clip_id,subset,timestamp_seconds,has_human_gt,detected_confidence,detection_count,image_name",
                "night-fp,night_vision,11.0,False,0.91,1,night-fp-001.jpg",
                "ir-missed,ir,5.0,True,0.34,1,ir-missed-001.jpg",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    exit_code = module.main(
        [
            "--clip-results",
            str(clip_results),
            "--frame-predictions",
            str(frame_predictions),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "alert_failure_analysis.md").exists()
    assert (output_dir / "false_positive_frames.csv").exists()
    assert (output_dir / "missed_positive_clips.csv").exists()
    content = (output_dir / "alert_failure_analysis.md").read_text(encoding="utf-8")
    assert "night_vision" in content
    assert "ir-missed" in content
