from pathlib import Path
import json
from unittest.mock import Mock, patch

import pytest

from src.config import AppConfig
from src.telegram_bot import TelegramAlertClient, build_caption


def test_send_photo_returns_false_when_disabled(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    client = TelegramAlertClient(enabled=False, bot_token=None, chat_id=None)
    assert client.send_photo(image_path, "caption") is False


def test_send_photo_raises_when_credentials_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    client = TelegramAlertClient(enabled=True, bot_token=None, chat_id="123")
    with pytest.raises(RuntimeError, match="credentials are missing"):
        client.send_photo(image_path, "caption")


def test_send_photo_posts_image_to_telegram(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image-bytes")
    response = Mock()
    client = TelegramAlertClient(enabled=True, bot_token="token", chat_id="123")

    with patch("src.telegram_bot.requests.post", return_value=response) as post:
        assert client.send_photo(image_path, "caption") is True

    post.assert_called_once()
    _, kwargs = post.call_args
    assert kwargs["data"]["chat_id"] == "123"
    assert kwargs["data"]["caption"] == "caption"
    reply_markup = json.loads(kwargs["data"]["reply_markup"])
    assert reply_markup["inline_keyboard"][0][0]["text"] == "联系儿子"
    assert reply_markup["inline_keyboard"][0][0]["url"] == "https://wa.me/6591863614"
    assert reply_markup["inline_keyboard"][0][1]["text"] == "联系丈夫"
    assert reply_markup["inline_keyboard"][0][1]["url"] == "https://wa.me/6593838469"
    assert len(reply_markup["inline_keyboard"]) == 1
    assert kwargs["timeout"] == 15
    assert "photo" in kwargs["files"]
    assert kwargs["files"]["photo"].name.endswith("image.jpg")
    response.raise_for_status.assert_called_once()


def test_build_caption_includes_human_alert_context() -> None:
    config = AppConfig()
    config.camera.name = "Front Door"
    config.input.mode = "rtsp"
    caption = build_caption(
        config=config,
        timestamp_text="2026-06-05T12:00:00+08:00",
        person_count=2,
        confidence=0.91,
        detection_duration_seconds=2.5,
        cooldown_seconds=60.0,
    )
    assert caption.startswith("Front Door 检测到 2 个人")
    assert "最高置信度为 91%" in caption
    assert "检测模式为 rtsp" in caption
    assert "检测持续时间为 2.5s" in caption
    assert "下一次告警冷却时间为 60.0s" in caption
