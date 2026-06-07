from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.alert_logic import TemporalAlertFilter
    from src.camera import create_source
    from src.config import AppConfig, load_config
    from src.detector import YOLOPersonDetector
    from src.health import HealthServer
    from src.logger import setup_logging, write_status
    from src.metrics import RuntimeMetrics
    from src.telegram_bot import TelegramAlertClient, build_caption
    from src.utils import annotate_frame, ensure_dir, now_local, save_image, timestamp_slug
else:
    from .alert_logic import TemporalAlertFilter
    from .camera import create_source
    from .config import AppConfig, load_config
    from .detector import YOLOPersonDetector
    from .health import HealthServer
    from .logger import setup_logging, write_status
    from .metrics import RuntimeMetrics
    from .telegram_bot import TelegramAlertClient, build_caption
    from .utils import annotate_frame, ensure_dir, now_local, save_image, timestamp_slug


def _status_payload(config: AppConfig, metrics: RuntimeMetrics, camera_connected: bool) -> dict:
    snapshot = metrics.snapshot(config.input.mode, camera_connected)
    return snapshot.__dict__


def _debug_snapshot_path(alert_dir: Path, input_mode: str, frame_index: int, timestamp_text: str) -> Path:
    safe_stamp = timestamp_text.replace(":", "").replace("-", "").replace("T", "_").replace("+", "_")
    return alert_dir / "test_snapshots" / f"{safe_stamp}_{frame_index:06d}_{input_mode}.jpg"


def run(config_path: str = "config.yaml") -> int:
    load_dotenv()
    config = load_config(config_path)

    alert_dir = ensure_dir(config.storage.alert_dir)
    log_dir = ensure_dir(config.storage.log_dir)
    logger, detection_logger = setup_logging(log_dir)

    logger.info("Starting Security AI")
    logger.info("Input mode: %s", config.input.mode)

    metrics = RuntimeMetrics()
    source = create_source(config)
    detector = YOLOPersonDetector(config)
    logger.info("Model device: %s", detector.device)
    alert_filter = TemporalAlertFilter(
        rolling_window_size=config.detection.rolling_window_size,
        min_positive_frames=config.detection.min_positive_frames,
        min_detection_duration_seconds=config.detection.min_detection_duration_seconds,
        cooldown_seconds=config.detection.alert_cooldown_seconds,
    )
    telegram = TelegramAlertClient.from_config(config)

    camera_connected = False
    should_stop = False
    health_server: HealthServer | None = None
    status_path = Path(log_dir) / "status.json"

    def _stop(*_args) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if config.runtime.health_endpoint_enabled:
        health_server = HealthServer(
            host="127.0.0.1",
            port=config.runtime.health_port,
            provider=lambda: _status_payload(config, metrics, camera_connected),
        )
        health_server.start()
        logger.info("Health endpoint started on 127.0.0.1:%s", config.runtime.health_port)

    frame_interval = 1.0 / max(config.detection.sample_fps, 0.1)
    next_frame_at = time.monotonic()
    last_status_write = 0.0
    started_at = time.monotonic()

    try:
        while not should_stop:
            if config.runtime.max_runtime_seconds is not None and time.monotonic() - started_at >= config.runtime.max_runtime_seconds:
                logger.info("Configured max runtime reached; shutting down")
                break
            wait_time = next_frame_at - time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)
            next_frame_at = time.monotonic() + frame_interval

            packet = source.read()
            if packet is None:
                if camera_connected:
                    logger.warning("Camera stream unavailable from %s", config.input.mode)
                else:
                    logger.warning("No frame available from %s", config.input.mode)
                camera_connected = False
                if getattr(source, "finite", False):
                    break
                continue

            if not camera_connected:
                logger.info("Camera stream connected for %s", config.input.mode)
            camera_connected = True
            metrics.note_frame(packet.timestamp)
            detections = detector.detect(packet.frame)
            has_detection = bool(detections)
            decision_time = now_local()
            if has_detection:
                metrics.note_detection(decision_time)

            decision = alert_filter.evaluate(decision_time, has_detection)
            best_confidence = detections[0].confidence if detections else 0.0

            logger.info(
                "frame processed mode=%s detections=%s confidence=%.2f positives=%s cooldown=%.1f reason=%s",
                config.input.mode,
                len(detections),
                best_confidence,
                decision.positive_frames,
                decision.cooldown_remaining_seconds,
                decision.reason,
            )

            if decision.triggered and detections:
                alert_time = now_local()
                annotated = annotate_frame(
                    packet.frame,
                    detections,
                    config.camera.name,
                    alert_time.isoformat(timespec="seconds"),
                )
                alert_path = alert_dir / f"{timestamp_slug(alert_time)}_{config.input.mode}.jpg"
                save_image(alert_path, annotated)
                caption = build_caption(
                    config=config,
                    timestamp_text=alert_time.isoformat(timespec="seconds"),
                    person_count=len(detections),
                    confidence=best_confidence,
                    detection_duration_seconds=decision.detection_duration_seconds,
                    cooldown_seconds=config.detection.alert_cooldown_seconds,
                )
                try:
                    telegram.send_photo(alert_path, caption)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Telegram send failed: %s", exc)
                metrics.note_alert(alert_time)
                detection_logger.info(
                    "alert camera=%s mode=%s confidence=%.2f duration=%.2f image=%s",
                    config.camera.name,
                    config.input.mode,
                    best_confidence,
                    decision.detection_duration_seconds,
                    alert_path,
                )

            debug_frame = None
            if config.runtime.debug_view or config.runtime.save_debug_frames:
                debug_frame = annotate_frame(
                    packet.frame,
                    detections,
                    config.camera.name,
                    packet.timestamp.isoformat(timespec="seconds"),
                )
            if config.runtime.save_debug_frames and debug_frame is not None:
                save_image(
                    _debug_snapshot_path(
                        alert_dir=alert_dir,
                        input_mode=config.input.mode,
                        frame_index=metrics.total_frames,
                        timestamp_text=packet.timestamp.isoformat(timespec="microseconds"),
                    ),
                    debug_frame,
                )
            if config.runtime.debug_view and debug_frame is not None:
                cv2.imshow("Security AI Debug", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now_monotonic = time.monotonic()
            if now_monotonic - last_status_write >= config.runtime.heartbeat_interval_seconds:
                write_status(status_path, _status_payload(config, metrics, camera_connected))
                last_status_write = now_monotonic

    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal runtime error: %s", exc)
        return 1
    finally:
        source.release()
        if health_server is not None:
            health_server.stop()
        cv2.destroyAllWindows()
        write_status(status_path, _status_payload(config, metrics, camera_connected))
        logger.info("Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
