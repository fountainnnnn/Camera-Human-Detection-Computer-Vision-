from __future__ import annotations

import argparse
import time

import common  # noqa: F401

from src.camera import create_source
from src.config import load_config
from src.utils import ensure_dir, save_image, timestamp_slug


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture webcam sample frames.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output-dir", default="data/local_test/webcam_captures")
    args = parser.parse_args()

    config = load_config(args.config)
    config.input.mode = "webcam"
    source = create_source(config)
    output_dir = ensure_dir(args.output_dir)

    try:
        for _ in range(args.count):
            packet = source.read()
            if packet is None:
                print("No webcam frame received.")
                return 1
            path = output_dir / f"webcam_{timestamp_slug()}.jpg"
            save_image(path, packet.frame)
            print(f"saved={path}")
            time.sleep(args.interval)
    finally:
        source.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
