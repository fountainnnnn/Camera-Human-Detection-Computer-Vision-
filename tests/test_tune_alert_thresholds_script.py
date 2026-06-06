import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_tune_alert_thresholds_module():
    fake_common = SimpleNamespace()
    with patch.dict(sys.modules, {"common": fake_common}):
        if "scripts.tune_alert_thresholds" in sys.modules:
            return importlib.reload(sys.modules["scripts.tune_alert_thresholds"])
        return importlib.import_module("scripts.tune_alert_thresholds")


def test_tune_alert_thresholds_script_writes_sweep_outputs(tmp_path: Path) -> None:
    module = load_tune_alert_thresholds_module()

    frame_predictions = tmp_path / "alert_frame_predictions.csv"
    frame_predictions.write_text(
        "\n".join(
            [
                "clip_id,subset,timestamp_seconds,has_human_gt,detected_confidence,detection_count,image_name",
                "clip-positive,day,0.0,True,0.55,1,frame1.jpg",
                "clip-positive,day,1.0,True,0.65,1,frame2.jpg",
                "clip-positive,day,2.0,True,0.75,1,frame3.jpg",
                "clip-negative,day,0.0,False,0.0,0,frame4.jpg",
                "clip-negative,day,1.0,False,0.70,1,frame5.jpg",
                "clip-negative,day,2.0,False,0.72,1,frame6.jpg",
                "clip-negative,day,3.0,False,0.74,1,frame7.jpg",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    exit_code = module.main(
        [
            "--frame-predictions",
            str(frame_predictions),
            "--output-dir",
            str(output_dir),
            "--confidence-thresholds",
            "0.5,0.7",
            "--min-positive-frames",
            "2,3",
            "--rolling-window-sizes",
            "3",
            "--min-detection-duration-seconds",
            "1.0",
            "--cooldown-seconds",
            "60",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "alert_threshold_sweep.csv").exists()
    assert (output_dir / "alert_threshold_recommendations.md").exists()
    content = (output_dir / "alert_threshold_recommendations.md").read_text(encoding="utf-8")
    assert "Recommended setting" in content
    assert "alert precision" in content
