from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.kaist_tools import prepare_kaist_alert_sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a sampled KAIST sequence manifest for alert-level evaluation.")
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modality", choices=["visible", "lwir"], default="lwir")
    parser.add_argument("--subset", default="kaist_lwir_sequence")
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)

    counts = prepare_kaist_alert_sequence(
        annotations_dir=Path(args.annotations_dir),
        images_dir=Path(args.images_dir),
        output_root=Path(args.output_root),
        modality=args.modality,
        subset=args.subset,
        source_fps=args.source_fps,
        sample_fps=args.sample_fps,
        window_seconds=args.window_seconds,
        max_frames=args.max_frames,
    )
    print(
        " ".join(
            [
                f"frames={counts['frames']}",
                f"positive_frames={counts['positive_frames']}",
                f"negative_frames={counts['negative_frames']}",
                f"clips={counts['clips']}",
                f"missing_images={counts['missing_images']}",
                f"output={args.output_root}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
