from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InputConfig:
    mode: str = "webcam"
    webcam_index: int = 0
    video_path: str = ""
    image_folder: str = ""
    rtsp_detection_url: str = ""
    rtsp_snapshot_url: str = ""
    reconnect_delay_seconds: int = 5
    max_reconnect_attempts: int = 999999
    loop_image_folder: bool = False


@dataclass
class CameraConfig:
    name: str = "Test Camera"


@dataclass
class ModelConfig:
    path: str = "models/yolo11n.pt"
    person_class_id: int = 0
    confidence_threshold: float = 0.65
    inference_size: int = 640
    device: str = "auto"


@dataclass
class DetectionConfig:
    sample_fps: float = 2.0
    rolling_window_size: int = 6
    min_positive_frames: int = 3
    min_detection_duration_seconds: float = 2.0
    alert_cooldown_seconds: float = 60.0
    min_box_area_ratio: float = 0.01
    max_box_area_ratio: float = 0.80


@dataclass
class ZonesConfig:
    enabled: bool = False
    monitored_zones: list[list[list[float]]] = field(default_factory=list)
    ignore_zones: list[list[list[float]]] = field(default_factory=list)


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"


@dataclass
class StorageConfig:
    alert_dir: str = "alerts"
    log_dir: str = "logs"
    max_alert_age_days: int = 14
    max_log_age_days: int = 14


@dataclass
class RuntimeConfig:
    debug_view: bool = True
    save_debug_frames: bool = False
    health_endpoint_enabled: bool = True
    health_port: int = 8765
    heartbeat_interval_seconds: int = 30
    max_runtime_seconds: float | None = None


@dataclass
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(payload: dict[str, Any]) -> AppConfig:
    return AppConfig(
        input=InputConfig(**payload.get("input", {})),
        camera=CameraConfig(**payload.get("camera", {})),
        model=ModelConfig(**payload.get("model", {})),
        detection=DetectionConfig(**payload.get("detection", {})),
        zones=ZonesConfig(**payload.get("zones", {})),
        telegram=TelegramConfig(**payload.get("telegram", {})),
        storage=StorageConfig(**payload.get("storage", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
    )


def default_config() -> AppConfig:
    return AppConfig()


def resolve_model_device(requested_device: str) -> str:
    normalized = requested_device.strip().lower() if requested_device else "auto"
    if normalized not in {"", "auto"}:
        return requested_device

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return "cpu"

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda:0"

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"

    return "cpu"


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return default_config()

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    base = default_config().to_dict()
    merged = _merge_dicts(base, payload)
    return _build_config(merged)


def save_example_config(path: str | Path = "config.example.yaml") -> None:
    import yaml

    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(default_config().to_dict(), handle, sort_keys=False)
