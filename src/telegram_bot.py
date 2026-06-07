from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

import requests
from requests import HTTPError

from .config import AppConfig


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _parse_chat_ids(chat_id: str | None) -> list[str]:
    if not chat_id:
        return []
    normalized = chat_id.replace(";", ",").replace("\n", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def build_alert_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "联系儿子", "url": "https://wa.me/6591863614"},
                {"text": "联系丈夫", "url": "https://wa.me/6593838469"},
            ],
        ]
    }


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
        chat_ids = _parse_chat_ids(self.chat_id)
        if not self.bot_token or not chat_ids:
            raise RuntimeError("Telegram is enabled but credentials are missing from .env")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        with Path(image_path).open("rb") as image_file:
            for chat_id in chat_ids:
                image_file.seek(0)
                response = requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                        "reply_markup": json.dumps(build_alert_keyboard(), ensure_ascii=False),
                    },
                    files={"photo": image_file},
                    timeout=self.timeout_seconds,
                )
                try:
                    response.raise_for_status()
                except HTTPError:
                    raise RuntimeError(
                        f"Telegram sendPhoto failed for chat_id={chat_id} with HTTP {response.status_code}: {response.text}"
                    ) from None
        return True


def build_caption(
    config: AppConfig,
    timestamp_text: str,
    person_count: int,
    confidence: float,
    detection_duration_seconds: float,
    cooldown_seconds: float,
) -> str:
    people_text = "1 个人" if person_count == 1 else f"{person_count} 个人"
    confidence_percent = confidence * 100
    return (
        f"{config.camera.name} 检测到 {people_text}，最高置信度为 {confidence_percent:.0f}%。"
        f"检测模式为 {config.input.mode}，检测持续时间为 {_format_seconds(detection_duration_seconds)}。"
        f"告警时间为 {timestamp_text}，下一次告警冷却时间为 {_format_seconds(cooldown_seconds)}。"
    )
