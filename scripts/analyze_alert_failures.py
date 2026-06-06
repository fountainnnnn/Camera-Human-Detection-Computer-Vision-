from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import common  # noqa: F401

from src.error_analysis import analyze_alert_failures


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze alert false positives and missed positive clips.")
    parser.add_argument("--clip-results", default="reports/alert_clip_results.csv")
    parser.add_argument("--frame-predictions", default="reports/alert_frame_predictions.csv")
    parser.add_argument("--output-dir", default="reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    clip_rows = _read_csv_rows(Path(args.clip_results))
    frame_rows = _read_csv_rows(Path(args.frame_predictions))
    report = analyze_alert_failures(clip_rows=clip_rows, frame_rows=frame_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv_rows(
        output_dir / "false_positive_frames.csv",
        report.top_false_positive_frames,
        ["clip_id", "subset", "timestamp_seconds", "has_human_gt", "detected_confidence", "detection_count", "image_name"],
    )
    _write_csv_rows(
        output_dir / "missed_positive_clips.csv",
        [row for row in clip_rows if str(row.get("missed_positive_clip", "")).strip().lower() in {"1", "true", "yes"}],
        list(clip_rows[0].keys()) if clip_rows else ["clip_id"],
    )

    summary_path = output_dir / "alert_failure_analysis.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Alert Failure Analysis",
                "",
                f"- false positive clips: {report.false_positive_clip_count}",
                f"- missed positive clips: {report.missed_positive_clip_count}",
                "",
                "## False positive subsets",
                "",
                *[
                    f"- {row['subset']}: clips={row['clip_count']} false_alerts={row['false_alert_count']}"
                    for row in report.false_positive_subsets
                ],
                "",
                "## Missed positive subsets",
                "",
                *[
                    f"- {row['subset']}: clips={row['clip_count']}"
                    for row in report.missed_positive_subsets
                ],
                "",
                "## Top false positive frames",
                "",
                *[
                    f"- {row.get('image_name', '')} subset={row.get('subset', '')} confidence={row.get('detected_confidence', '')}"
                    for row in report.top_false_positive_frames[:10]
                ],
                "",
                "## Lowest-confidence positive frames",
                "",
                *[
                    f"- {row.get('image_name', '')} subset={row.get('subset', '')} confidence={row.get('detected_confidence', '')}"
                    for row in report.low_confidence_positive_frames[:10]
                ],
            ]
        ),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
