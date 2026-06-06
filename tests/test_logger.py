import json
import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

def _flush_and_close(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()


def test_setup_logging_writes_app_error_and_detection_logs(tmp_path: Path) -> None:
    with patch.dict("sys.modules", {"cv2": SimpleNamespace()}):
        logger_module = importlib.import_module("src.logger")
        logger, detection_logger = logger_module.setup_logging(tmp_path)

        logger.info("application started")
        logger.error("camera failure")
        detection_logger.info("person detected")

        _flush_and_close(logger)
        _flush_and_close(detection_logger)

    app_log = (tmp_path / "app.log").read_text(encoding="utf-8")
    error_log = (tmp_path / "errors.log").read_text(encoding="utf-8")
    detection_log = (tmp_path / "detections.log").read_text(encoding="utf-8")

    assert "application started" in app_log
    assert "camera failure" in app_log
    assert "camera failure" in error_log
    assert "person detected" in detection_log


def test_write_status_creates_parent_directory_and_json_file(tmp_path: Path) -> None:
    status_path = tmp_path / "runtime" / "status.json"
    payload = {"status": "ok", "total_alerts": 3}

    with patch.dict("sys.modules", {"cv2": SimpleNamespace()}):
        logger_module = importlib.import_module("src.logger")
        logger_module.write_status(status_path, payload)

    assert json.loads(status_path.read_text(encoding="utf-8")) == payload
