from pathlib import Path

from src.video_tools import build_frame_image_name, infer_clip_metadata, list_video_files, sanitize_clip_id


def test_sanitize_clip_id_normalizes_names() -> None:
    assert sanitize_clip_id("Night Hallway (IR)!") == "night-hallway-ir"


def test_build_frame_image_name_uses_zero_padding() -> None:
    assert build_frame_image_name("clip-1", 12) == "clip-1_000012.jpg"


def test_infer_clip_metadata_uses_subset_folder(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    subset_dir = raw_root / "night"
    subset_dir.mkdir(parents=True)
    video_path = subset_dir / "Night Hallway.mp4"
    video_path.write_bytes(b"video")
    metadata = infer_clip_metadata(video_path, raw_root)
    assert metadata.subset == "night"
    assert metadata.clip_id == "night-hallway"


def test_list_video_files_filters_extensions(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "a.mp4").write_bytes(b"video")
    (raw_root / "b.txt").write_text("x", encoding="utf-8")
    files = list_video_files(raw_root)
    assert files == [raw_root / "a.mp4"]
