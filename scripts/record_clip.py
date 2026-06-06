from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import common  # noqa: F401

from src.camera import create_source
from src.config import load_config
from src.utils import ensure_dir, timestamp_slug


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a short validation clip from webcam or RTSP input.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["webcam", "rtsp"], help="Override input mode for recording.")
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--prefix", default="clip")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config.input.mode = args.mode
    if config.input.mode not in {"webcam", "rtsp"}:
        raise ValueError("record_clip.py supports webcam or rtsp modes only")

    output_dir = ensure_dir(args.output_dir)
    output_path = output_dir / f"{args.prefix}_{config.input.mode}_{timestamp_slug()}.mp4"

    source = create_source(config)
    writer = None
    recorded_frames = 0
    frame_interval = 1.0 / max(args.max_fps, 0.1)
    started_at = time.monotonic()

    try:
        while time.monotonic() - started_at < args.duration_seconds:
            packet = source.read()
            if packet is None:
                print("No frame received while recording.")
                time.sleep(0.25)
                continue
            frame = packet.frame
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, args.max_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open output video writer: {output_path}")
            writer.write(frame)
            recorded_frames += 1
            time.sleep(frame_interval)
    finally:
        source.release()
        if writer is not None:
            writer.release()

    print(f"video={output_path}")
    print(f"frames={recorded_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
