from __future__ import annotations

import argparse

import common  # noqa: F401

from src.camera import create_source
from src.config import load_config
from src.detector import YOLOPersonDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Run person detection on a video clip.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    config.input.mode = "video_file"
    config.input.video_path = args.video

    source = create_source(config)
    detector = YOLOPersonDetector(config)
    processed = 0
    positive_frames = 0

    try:
        while processed < args.frames:
            packet = source.read()
            if packet is None:
                break
            detections = detector.detect(packet.frame)
            if detections:
                positive_frames += 1
            processed += 1
            print(f"frame={processed} detections={len(detections)}")
    finally:
        source.release()

    print(f"video_frames={processed} positive_frames={positive_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
