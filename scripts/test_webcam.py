from __future__ import annotations

import argparse
from pathlib import Path

import common  # noqa: F401

from src.camera import create_source
from src.config import load_config
from src.detector import YOLOPersonDetector
from src.utils import annotate_frame, ensure_dir, save_image, timestamp_slug


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate webcam input and person detection.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = load_config(args.config)
    config.input.mode = "webcam"

    source = create_source(config)
    detector = YOLOPersonDetector(config)
    processed = 0
    try:
        while processed < args.frames:
            packet = source.read()
            if packet is None:
                print("No webcam frame received.")
                return 1
            detections = detector.detect(packet.frame)
            if args.save_snapshots:
                annotated = annotate_frame(
                    packet.snapshot_frame if packet.snapshot_frame is not None else packet.frame,
                    detections,
                    config.camera.name,
                    packet.timestamp.isoformat(timespec="seconds"),
                )
                output_dir = ensure_dir(args.output_dir)
                output_path = Path(output_dir) / f"webcam_test_{timestamp_slug(packet.timestamp)}_{processed + 1:03d}.jpg"
                save_image(output_path, annotated)
                print(f"snapshot={output_path}")
            print(f"frame={processed + 1} detections={len(detections)}")
            processed += 1
    finally:
        source.release()

    print(f"Webcam test completed: {processed} frames processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
