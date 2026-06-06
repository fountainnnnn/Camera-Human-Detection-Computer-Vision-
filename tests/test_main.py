from datetime import datetime, timezone
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.config import AppConfig


class FakeFrame:
    def __init__(self, name: str = "frame") -> None:
        self.name = name

    def copy(self):
        return self


class FakeSource:
    finite = True

    def __init__(self, packets) -> None:
        self._packets = list(packets)
        self.released = False

    def read(self):
        if self._packets:
            return self._packets.pop(0)
        return None

    def release(self) -> None:
        self.released = True


class FakeStreamingSource(FakeSource):
    finite = False


class FakeDetector:
    def __init__(self, detections) -> None:
        self.device = "cpu"
        self._detections = detections

    def detect(self, _frame):
        return list(self._detections)


class FakeLogger:
    def __init__(self) -> None:
        self.infos = []
        self.warnings = []
        self.errors = []
        self.exceptions = []

    def info(self, *_args, **_kwargs) -> None:
        self.infos.append(_args)

    def warning(self, *_args, **_kwargs) -> None:
        self.warnings.append(_args)

    def error(self, *_args, **_kwargs) -> None:
        self.errors.append(_args)

    def exception(self, *_args, **_kwargs) -> None:
        self.exceptions.append(_args)


def load_main_module(fake_cv2):
    fake_dotenv = SimpleNamespace(load_dotenv=lambda: None)
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 0.0,
        virtual_memory=lambda: SimpleNamespace(percent=0.0),
    )
    with patch.dict(sys.modules, {"cv2": fake_cv2, "dotenv": fake_dotenv, "psutil": fake_psutil}):
        if "src.main" in sys.modules:
            return importlib.reload(sys.modules["src.main"])
        return importlib.import_module("src.main")


def test_run_shows_debug_preview_when_enabled(monkeypatch) -> None:
    imshow_calls = []
    waitkey_calls = []
    fake_cv2 = SimpleNamespace(
        imshow=lambda name, frame: imshow_calls.append((name, frame)),
        waitKey=lambda delay: waitkey_calls.append(delay) or ord("q"),
        destroyAllWindows=lambda: None,
    )
    main_module = load_main_module(fake_cv2)

    packet = SimpleNamespace(
        frame=FakeFrame(),
        snapshot_frame=None,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    source = FakeSource([packet])
    detector = FakeDetector([])
    config = AppConfig()
    config.runtime.debug_view = True
    config.runtime.save_debug_frames = False
    config.runtime.health_endpoint_enabled = False
    config.telegram.enabled = False

    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "create_source", lambda _config: source)
    monkeypatch.setattr(main_module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(
        main_module,
        "TemporalAlertFilter",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda *_args: SimpleNamespace(triggered=False, reason="ok", detection_duration_seconds=0.0)
        ),
    )
    monkeypatch.setattr(
        main_module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(send_photo=lambda *_args, **_kwargs: False)),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _log_dir: (FakeLogger(), FakeLogger()))
    monkeypatch.setattr(main_module, "annotate_frame", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(main_module, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(main_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module.signal, "signal", lambda *_args, **_kwargs: None)

    assert main_module.run("config.yaml") == 0
    assert source.released is True
    assert len(imshow_calls) == 1
    assert waitkey_calls == [1]


def test_run_saves_debug_frames_when_enabled(monkeypatch, tmp_path: Path) -> None:
    fake_cv2 = SimpleNamespace(
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda *_args, **_kwargs: -1,
        destroyAllWindows=lambda: None,
    )
    main_module = load_main_module(fake_cv2)

    packet = SimpleNamespace(
        frame=FakeFrame(),
        snapshot_frame=None,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc),
    )
    source = FakeSource([packet])
    detector = FakeDetector([SimpleNamespace(confidence=0.91, box=(1, 2, 3, 4))])
    config = AppConfig()
    config.runtime.debug_view = False
    config.runtime.save_debug_frames = True
    config.runtime.health_endpoint_enabled = False
    config.telegram.enabled = False
    config.storage.alert_dir = str(tmp_path / "alerts")

    saved_paths = []

    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "create_source", lambda _config: source)
    monkeypatch.setattr(main_module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(
        main_module,
        "TemporalAlertFilter",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda *_args: SimpleNamespace(triggered=False, reason="ok", detection_duration_seconds=0.0)
        ),
    )
    monkeypatch.setattr(
        main_module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(send_photo=lambda *_args, **_kwargs: False)),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _log_dir: (FakeLogger(), FakeLogger()))
    monkeypatch.setattr(main_module, "annotate_frame", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(main_module, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "save_image", lambda path, _frame: saved_paths.append(Path(path)) or Path(path))
    monkeypatch.setattr(main_module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(main_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module.signal, "signal", lambda *_args, **_kwargs: None)

    assert main_module.run("config.yaml") == 0
    assert len(saved_paths) == 1
    assert saved_paths[0].parent.name == "test_snapshots"
    assert saved_paths[0].name.endswith("_000001_webcam.jpg")


def test_run_recovers_after_temporary_source_drop(monkeypatch) -> None:
    waitkey_calls = []
    fake_cv2 = SimpleNamespace(
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda delay: waitkey_calls.append(delay) or ord("q"),
        destroyAllWindows=lambda: None,
    )
    main_module = load_main_module(fake_cv2)

    packet = SimpleNamespace(
        frame=FakeFrame(),
        snapshot_frame=None,
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    source = FakeStreamingSource([None, packet])
    detector = FakeDetector([])
    config = AppConfig()
    config.input.mode = "rtsp"
    config.runtime.debug_view = True
    config.runtime.save_debug_frames = False
    config.runtime.health_endpoint_enabled = False
    config.telegram.enabled = False
    logger = FakeLogger()

    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "create_source", lambda _config: source)
    monkeypatch.setattr(main_module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(
        main_module,
        "TemporalAlertFilter",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda *_args: SimpleNamespace(triggered=False, reason="ok", detection_duration_seconds=0.0)
        ),
    )
    monkeypatch.setattr(
        main_module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(send_photo=lambda *_args, **_kwargs: False)),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _log_dir: (logger, FakeLogger()))
    monkeypatch.setattr(main_module, "annotate_frame", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(main_module, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(main_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module.signal, "signal", lambda *_args, **_kwargs: None)

    assert main_module.run("config.yaml") == 0
    assert any(args[0] == "No frame available from %s" for args in logger.warnings)
    assert any(args[0] == "Camera stream connected for %s" for args in logger.infos)
    assert waitkey_calls == [1]


def test_run_saves_alert_snapshot_locally_when_detection_is_confirmed(monkeypatch, tmp_path: Path) -> None:
    fake_cv2 = SimpleNamespace(
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda *_args, **_kwargs: -1,
        destroyAllWindows=lambda: None,
    )
    main_module = load_main_module(fake_cv2)

    packet = SimpleNamespace(
        frame=FakeFrame("frame"),
        snapshot_frame=FakeFrame("snapshot"),
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    detection = SimpleNamespace(confidence=0.93, box=(1, 2, 3, 4))
    source = FakeSource([packet])
    detector = FakeDetector([detection])
    config = AppConfig()
    config.input.mode = "webcam"
    config.runtime.debug_view = False
    config.runtime.save_debug_frames = False
    config.runtime.health_endpoint_enabled = False
    config.telegram.enabled = True
    config.storage.alert_dir = str(tmp_path / "alerts")
    sent = []
    saved_paths = []
    detection_logger = FakeLogger()
    annotate_calls = []

    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "create_source", lambda _config: source)
    monkeypatch.setattr(main_module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(
        main_module,
        "TemporalAlertFilter",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda *_args: SimpleNamespace(triggered=True, reason="confirmed", detection_duration_seconds=2.5)
        ),
    )
    monkeypatch.setattr(
        main_module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(send_photo=lambda path, caption: sent.append((Path(path), caption)) or True)),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _log_dir: (FakeLogger(), detection_logger))
    monkeypatch.setattr(main_module, "annotate_frame", lambda frame, *_args, **_kwargs: annotate_calls.append(frame) or frame)
    monkeypatch.setattr(main_module, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "save_image", lambda path, _frame: saved_paths.append(Path(path)) or Path(path))
    monkeypatch.setattr(main_module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(main_module, "now_local", lambda: datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc))
    monkeypatch.setattr(main_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module.signal, "signal", lambda *_args, **_kwargs: None)

    assert main_module.run("config.yaml") == 0
    assert len(saved_paths) == 1
    assert saved_paths[0].parent == tmp_path / "alerts"
    assert saved_paths[0].name.endswith("_webcam.jpg")
    assert len(sent) == 1
    assert sent[0][0] == saved_paths[0]
    assert annotate_calls == [packet.snapshot_frame]
    assert any(args and args[0].startswith("alert camera=") for args in detection_logger.infos)


def test_run_continues_loop_after_confirmed_alert(monkeypatch) -> None:
    fake_cv2 = SimpleNamespace(
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda *_args, **_kwargs: -1,
        destroyAllWindows=lambda: None,
    )
    main_module = load_main_module(fake_cv2)

    first_packet = SimpleNamespace(
        frame=FakeFrame("frame-1"),
        snapshot_frame=FakeFrame("snapshot-1"),
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    second_packet = SimpleNamespace(
        frame=FakeFrame("frame-2"),
        snapshot_frame=FakeFrame("snapshot-2"),
        timestamp=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
    )
    source = FakeSource([first_packet, second_packet])
    detector = FakeDetector([SimpleNamespace(confidence=0.9, box=(1, 2, 3, 4))])
    config = AppConfig()
    config.runtime.debug_view = False
    config.runtime.save_debug_frames = False
    config.runtime.health_endpoint_enabled = False
    config.telegram.enabled = False
    logger = FakeLogger()
    alert_results = iter(
        [
            SimpleNamespace(triggered=True, reason="confirmed", detection_duration_seconds=2.0),
            SimpleNamespace(triggered=False, reason="cooldown_active", detection_duration_seconds=2.0),
        ]
    )

    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "create_source", lambda _config: source)
    monkeypatch.setattr(main_module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(
        main_module,
        "TemporalAlertFilter",
        lambda **_kwargs: SimpleNamespace(evaluate=lambda *_args: next(alert_results)),
    )
    monkeypatch.setattr(
        main_module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(send_photo=lambda *_args, **_kwargs: False)),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _log_dir: (logger, FakeLogger()))
    monkeypatch.setattr(main_module, "annotate_frame", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(main_module, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "save_image", lambda path, _frame: Path(path))
    monkeypatch.setattr(main_module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(main_module, "now_local", lambda: datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc))
    monkeypatch.setattr(main_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module.signal, "signal", lambda *_args, **_kwargs: None)

    assert main_module.run("config.yaml") == 0
    processed_logs = [args for args in logger.infos if args and args[0] == "frame processed mode=%s detections=%s reason=%s"]
    assert len(processed_logs) == 2


def test_run_stops_when_configured_max_runtime_is_reached(monkeypatch) -> None:
    fake_cv2 = SimpleNamespace(
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda *_args, **_kwargs: -1,
        destroyAllWindows=lambda: None,
    )
    main_module = load_main_module(fake_cv2)

    packet = SimpleNamespace(
        frame=FakeFrame("frame"),
        snapshot_frame=None,
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    source = FakeStreamingSource([packet, packet, packet])
    detector = FakeDetector([])
    config = AppConfig()
    config.runtime.debug_view = False
    config.runtime.save_debug_frames = False
    config.runtime.health_endpoint_enabled = False
    config.runtime.max_runtime_seconds = 1.0
    logger = FakeLogger()
    monotonic_value = {"value": -0.3}

    def fake_monotonic():
        monotonic_value["value"] += 0.3
        return monotonic_value["value"]

    monkeypatch.setattr(main_module, "load_config", lambda _path: config)
    monkeypatch.setattr(main_module, "create_source", lambda _config: source)
    monkeypatch.setattr(main_module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(
        main_module,
        "TemporalAlertFilter",
        lambda **_kwargs: SimpleNamespace(
            evaluate=lambda *_args: SimpleNamespace(triggered=False, reason="ok", detection_duration_seconds=0.0)
        ),
    )
    monkeypatch.setattr(
        main_module.TelegramAlertClient,
        "from_config",
        classmethod(lambda cls, _config: SimpleNamespace(send_photo=lambda *_args, **_kwargs: False)),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _log_dir: (logger, FakeLogger()))
    monkeypatch.setattr(main_module, "annotate_frame", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(main_module, "write_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(main_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(main_module.signal, "signal", lambda *_args, **_kwargs: None)

    assert main_module.run("config.yaml") == 0
    assert any(args[0] == "Configured max runtime reached; shutting down" for args in logger.infos)
