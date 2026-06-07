from __future__ import annotations

import argparse
import json
import os
import time

import common  # noqa: F401
import requests
from requests import RequestException
from dotenv import load_dotenv

from src.config import load_config


def _send_message(token: str, chat_id: int, text: str, reply_markup: dict | None = None) -> None:
    data = {"chat_id": chat_id, "text": text}
    if reply_markup is not None:
        data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        timeout=15,
    )
    response.raise_for_status()


def _answer_callback(token: str, callback_query_id: str, text: str = "") -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{token}/answerCallbackQuery",
        data={"callback_query_id": callback_query_id, "text": text},
        timeout=15,
    )
    response.raise_for_status()


def _police_confirmation_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "确认报警", "callback_data": "final_confirm_police_call"},
                {"text": "取消", "callback_data": "cancel_police_call"},
            ]
        ]
    }


def _reply_text(message_text: str, chat_id: int) -> str:
    text = message_text.strip().lower()
    if text in {"/start", "start"}:
        return "Security AI bot is online. Send /chatid to get this chat ID."
    if text in {"/chatid", "chatid"}:
        return f"TELEGRAM_CHAT_ID={chat_id}"
    if text in {"/help", "help"}:
        return "Commands: /start, /chatid, /help"
    return "Security AI bot received your message. Send /chatid if you are setting up alerts."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a simple Telegram reply bot for setup/debugging.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Process available updates once and exit.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    token = os.getenv(config.telegram.bot_token_env)
    if not token:
        raise RuntimeError(f"{config.telegram.bot_token_env} is missing from .env")

    offset: int | None = None
    print("Telegram reply bot running. Press Ctrl+C to stop.")
    while True:
        try:
            response = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": offset, "timeout": 10},
                timeout=15,
            )
        except RequestException:
            raise RuntimeError("Telegram API request failed. Check network access and bot token.")
        if response.status_code == 401:
            raise RuntimeError("Telegram API rejected the bot token. Copy a fresh token from BotFather into .env.")
        if response.status_code >= 400:
            raise RuntimeError(f"Telegram API request failed with HTTP {response.status_code}.")
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API returned an error: {payload}")

        for update in payload.get("result", []):
            offset = int(update["update_id"]) + 1
            callback_query = update.get("callback_query")
            if callback_query:
                callback_id = callback_query.get("id")
                data = callback_query.get("data")
                message = callback_query.get("message") or {}
                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if callback_id and data == "confirm_police_call" and chat_id is not None:
                    _answer_callback(token, callback_id, "请再次确认。")
                    _send_message(
                        token,
                        int(chat_id),
                        "请确认是否拨打马来西亚警方 999。只有确认后才会打开拨号。",
                        _police_confirmation_keyboard(),
                    )
                    print(f"sent police confirmation chat_id={chat_id}")
                elif callback_id and data == "cancel_police_call":
                    _answer_callback(token, callback_id, "已取消。")
                elif callback_id and data == "final_confirm_police_call" and chat_id is not None:
                    _answer_callback(token, callback_id, "请立即拨打 999。")
                    _send_message(token, int(chat_id), "请立即拨打马来西亚警方紧急号码 999。")
                continue

            message = update.get("message") or update.get("edited_message")
            if not message:
                continue
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue
            text = message.get("text") or ""
            _send_message(token, int(chat_id), _reply_text(text, int(chat_id)))
            print(f"replied chat_id={chat_id}")

        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 0.1))


if __name__ == "__main__":
    raise SystemExit(main())
