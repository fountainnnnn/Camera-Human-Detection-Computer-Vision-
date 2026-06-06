from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import common  # noqa: F401
from dotenv import load_dotenv

from src.config import load_config
from src.telegram_bot import TelegramAlertClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a Telegram test image.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    config.telegram.enabled = True

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, "Security AI Telegram Test", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    output = Path("reports") / "telegram_test.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)

    client = TelegramAlertClient.from_config(config)
    client.enabled = True
    client.send_photo(output, "Security AI Telegram test alert")
    print(f"sent={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
