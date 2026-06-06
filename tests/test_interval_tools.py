import pandas as pd

from src.interval_tools import apply_intervals_to_manifest


def test_apply_intervals_to_manifest_marks_ranges() -> None:
    manifest = pd.DataFrame(
        [
            {"image": "a.jpg", "clip_id": "clip-1", "timestamp_seconds": 0.0, "subset": "unspecified"},
            {"image": "b.jpg", "clip_id": "clip-1", "timestamp_seconds": 1.0, "subset": "unspecified"},
            {"image": "c.jpg", "clip_id": "clip-1", "timestamp_seconds": 2.0, "subset": "unspecified"},
        ]
    )
    intervals = pd.DataFrame(
        [
            {"clip_id": "clip-1", "start_seconds": 0.5, "end_seconds": 1.5, "subset": "night"},
        ]
    )
    labeled = apply_intervals_to_manifest(manifest, intervals)
    assert labeled["has_human_gt"].tolist() == [False, True, False]
    assert labeled["subset"].tolist() == ["night", "night", "night"]
