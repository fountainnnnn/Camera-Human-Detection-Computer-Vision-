from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import common  # noqa: F401

from src.config import load_config
from src.detector import YOLOPersonDetector
from src.utils import annotate_frame, ensure_dir, save_image, timestamp_slug


def main() -> int:
    parser = argparse.ArgumentParser(description="Run person detection on a single image.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Failed to read image: {args.image}")
        return 1

    config = load_config(args.config)
    detector = YOLOPersonDetector(config)
    detections = detector.detect(frame)
    annotated = annotate_frame(frame, detections, config.camera.name, timestamp_slug())
    output_dir = ensure_dir(args.output_dir)
    output_path = output_dir / f"test_image_{timestamp_slug()}.jpg"
    save_image(output_path, annotated)
    print(f"detections={len(detections)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
