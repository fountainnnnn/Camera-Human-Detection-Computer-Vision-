from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import AppConfig


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


@dataclass
class TelegramAlertClient:
    enabled: bool
    bot_token: str | None
    chat_id: str | None
    timeout_seconds: int = 15

    @classmethod
    def from_config(cls, config: AppConfig) -> "TelegramAlertClient":
        return cls(
            enabled=config.telegram.enabled,
            bot_token=os.getenv(config.telegram.bot_token_env),
            chat_id=os.getenv(config.telegram.chat_id_env),
        )

    def send_photo(self, image_path: str | Path, caption: str) -> bool:
        if not self.enabled:
            return False
        if not self.bot_token or not self.chat_id:
            raise RuntimeError("Telegram is enabled but credentials are missing from .env")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        with Path(image_path).open("rb") as image_file:
            response = requests.post(
                url,
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": image_file},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return True


def build_caption(
    config: AppConfig,
    timestamp_text: str,
    confidence: float,
    detection_duration_seconds: float,
    cooldown_seconds: float,
) -> str:
    return (
        f"{config.camera.name}\n"
        f"mode: {config.input.mode}\n"
        f"time: {timestamp_text}\n"
        f"confidence: {confidence:.2f}\n"
        f"detection_duration: {_format_seconds(detection_duration_seconds)}\n"
        f"cooldown: {_format_seconds(cooldown_seconds)}"
    )
