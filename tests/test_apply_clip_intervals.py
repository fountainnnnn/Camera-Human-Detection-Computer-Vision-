import pandas as pd
from pathlib import Path


def test_apply_clip_intervals_updates_manifest(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {"image": "a.jpg", "clip_id": "clip-1", "timestamp_seconds": 0.0},
            {"image": "b.jpg", "clip_id": "clip-1", "timestamp_seconds": 1.0},
            {"image": "c.jpg", "clip_id": "clip-1", "timestamp_seconds": 2.0},
        ]
    )
    intervals = pd.DataFrame(
        [
            {"clip_id": "clip-1", "start_seconds": 0.5, "end_seconds": 1.5, "subset": "night"},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    intervals_path = tmp_path / "intervals.csv"
    manifest.to_csv(manifest_path, index=False)
    intervals.to_csv(intervals_path, index=False)

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/apply_clip_intervals.py", "--manifest", str(manifest_path), "--intervals", str(intervals_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    updated = pd.read_csv(manifest_path)
    assert updated["has_human_gt"].tolist() == [False, True, False]
    assert updated["subset"].tolist() == ["night", "night", "night"]
