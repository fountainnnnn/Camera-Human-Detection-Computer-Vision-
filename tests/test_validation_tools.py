from pathlib import Path

import pandas as pd

from src.validation_tools import (
    VALIDATION_SUBSETS,
    collection_checklist_dataframe,
    coverage_row,
    coverage_summary,
    create_validation_workspace,
    ensure_interval_template,
    interval_template_dataframe,
    validation_workspace_directories,
    write_collection_checklist,
    write_coverage_summary,
    write_validation_collection_guide,
)


def test_interval_template_dataframe_contains_placeholder() -> None:
    df = interval_template_dataframe("clip-1", "night")
    assert list(df.columns) == ["clip_id", "start_seconds", "end_seconds", "subset", "has_human_gt", "notes"]
    assert df.iloc[0]["clip_id"] == "clip-1"
    assert df.iloc[0]["subset"] == "night"


def test_ensure_interval_template_creates_file_once(tmp_path: Path) -> None:
    template_path = tmp_path / "clip-1.csv"
    ensure_interval_template(template_path, "clip-1", "night")
    original = pd.read_csv(template_path)
    ensure_interval_template(template_path, "clip-1", "day")
    second = pd.read_csv(template_path)
    assert original.equals(second)


def test_validation_workspace_directories_and_creation(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    intervals_dir = tmp_path / "intervals"
    directories = validation_workspace_directories(raw_root, intervals_dir)
    assert directories["raw_root"] == raw_root
    assert directories["intervals_dir"] == intervals_dir
    created = create_validation_workspace(raw_root, intervals_dir)
    for path in created.values():
        assert path.exists()
        assert path.is_dir()


def test_collection_checklist_and_guide(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    intervals_dir = tmp_path / "intervals"
    checklist = collection_checklist_dataframe(raw_root, intervals_dir)
    assert checklist["subset"].tolist() == list(VALIDATION_SUBSETS)
    checklist_path = tmp_path / "checklist.csv"
    guide_path = tmp_path / "guide.md"
    write_collection_checklist(checklist_path, raw_root, intervals_dir)
    write_validation_collection_guide(guide_path, raw_root, intervals_dir)
    written = pd.read_csv(checklist_path)
    assert written["subset"].tolist() == list(VALIDATION_SUBSETS)
    guide_text = guide_path.read_text(encoding="utf-8")
    assert "Validation Collection Guide" in guide_text
    assert "night_vision" in guide_text


def test_coverage_row_fields() -> None:
    row = coverage_row(
        clip_id="clip-1",
        subset="night",
        source_video="video.mp4",
        manifest="manifest.csv",
        intervals_csv="clip-1.csv",
        has_intervals=False,
        included_in_eval=False,
    )
    assert row["clip_id"] == "clip-1"
    assert row["included_in_eval"] is False


def test_coverage_summary_counts_and_markdown(tmp_path: Path) -> None:
    coverage = pd.DataFrame(
        [
            coverage_row(
                clip_id="clip-1",
                subset="night",
                source_video="a.mp4",
                manifest="a.csv",
                intervals_csv="clip-1.csv",
                has_intervals=True,
                included_in_eval=True,
            ),
            coverage_row(
                clip_id="clip-2",
                subset="ir",
                source_video="b.mp4",
                manifest="b.csv",
                intervals_csv="clip-2.csv",
                has_intervals=False,
                included_in_eval=False,
            ),
        ]
    )
    summary = coverage_summary(coverage)
    assert summary == {
        "total_clips": 2,
        "clips_with_intervals": 1,
        "clips_included_in_eval": 1,
        "clips_missing_intervals": 1,
    }
    output_path = tmp_path / "coverage.md"
    write_coverage_summary(output_path, coverage)
    text = output_path.read_text(encoding="utf-8")
    assert "clips missing interval labels: 1" in text
    assert "`clip-2` (`ir`): `clip-2.csv`" in text
