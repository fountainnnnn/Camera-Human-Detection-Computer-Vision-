from __future__ import annotations

import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from .utils import ensure_dir


def _build_handler(path: Path, level: int) -> TimedRotatingFileHandler:
    handler = TimedRotatingFileHandler(path, when="midnight", backupCount=14, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    return handler


def setup_logging(log_dir: str | Path) -> tuple[logging.Logger, logging.Logger]:
    directory = ensure_dir(log_dir)
    app_log = directory / "app.log"
    error_log = directory / "errors.log"
    detection_log = directory / "detections.log"

    logger = logging.getLogger("security_ai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)
    logger.addHandler(_build_handler(app_log, logging.INFO))
    logger.addHandler(_build_handler(error_log, logging.ERROR))

    detection_logger = logging.getLogger("security_ai.detections")
    detection_logger.setLevel(logging.INFO)
    detection_logger.handlers.clear()
    detection_logger.propagate = False
    detection_logger.addHandler(_build_handler(detection_log, logging.INFO))

    return logger, detection_logger


def write_status(path: str | Path, payload: dict) -> None:
    status_path = Path(path)
    ensure_dir(status_path.parent)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
