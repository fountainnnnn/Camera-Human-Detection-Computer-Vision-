from pathlib import Path

from src.manifest_tools import merge_manifests, read_manifest


def test_read_manifest_adds_default_subset(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image,clip_id,timestamp_seconds,has_human_gt\nframe1.jpg,clip-1,0.0,False\n",
        encoding="utf-8",
    )
    manifest = read_manifest(manifest_path)
    assert "subset" in manifest.columns
    assert manifest.iloc[0]["subset"] == "unspecified"


def test_merge_manifests_deduplicates_rows(tmp_path: Path) -> None:
    manifest_a = tmp_path / "a.csv"
    manifest_b = tmp_path / "b.csv"
    csv_text = "image,clip_id,timestamp_seconds,has_human_gt,subset\nframe1.jpg,clip-1,0.0,False,night\n"
    manifest_a.write_text(csv_text, encoding="utf-8")
    manifest_b.write_text(csv_text, encoding="utf-8")
    merged = merge_manifests([manifest_a, manifest_b])
    assert len(merged) == 1
