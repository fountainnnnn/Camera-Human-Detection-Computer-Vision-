from __future__ import annotations

import argparse
import shutil
import subprocess

import common  # noqa: F401

from src.camera import create_source
from src.config import load_config
from src.detector import YOLOPersonDetector


def run_ffprobe(rtsp_url: str, timeout_seconds: int) -> str:
    if not rtsp_url:
        raise RuntimeError("RTSP URL is required for ffprobe validation.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe was not found on PATH.")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-show_entries",
        "stream=index,codec_name,width,height",
        "-of",
        "json",
        rtsp_url,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed.")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RTSP input and person detection.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--ffprobe", action="store_true")
    parser.add_argument("--ffprobe-timeout", type=int, default=15)
    args = parser.parse_args()

    config = load_config(args.config)
    config.input.mode = "rtsp"

    if args.ffprobe:
        try:
            probe_output = run_ffprobe(config.input.rtsp_detection_url, args.ffprobe_timeout)
        except RuntimeError as exc:
            print(f"ffprobe_error={exc}")
            return 1
        print(f"ffprobe_ok={probe_output}")

    source = create_source(config)
    detector = YOLOPersonDetector(config)
    processed = 0

    try:
        while processed < args.frames:
            packet = source.read()
            if packet is None:
                print("No RTSP frame received.")
                return 1
            detections = detector.detect(packet.frame)
            print(f"frame={processed + 1} detections={len(detections)}")
            processed += 1
    finally:
        source.release()

    print(f"RTSP test completed: {processed} frames processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
