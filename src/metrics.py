from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

import psutil


@dataclass
class RuntimeSnapshot:
    status: str
    input_mode: str
    camera_connected: bool
    last_frame_time: str | None
    last_detection_time: str | None
    last_alert_time: str | None
    cpu_percent: float
    ram_percent: float
    fps: float
    total_frames: int
    total_alerts: int


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.frame_times: deque[float] = deque(maxlen=120)
        self.last_frame_time: datetime | None = None
        self.last_detection_time: datetime | None = None
        self.last_alert_time: datetime | None = None
        self.total_frames = 0
        self.total_alerts = 0

    def note_frame(self, timestamp: datetime) -> None:
        with self._lock:
            self.last_frame_time = timestamp
            self.total_frames += 1
            self.frame_times.append(timestamp.timestamp())

    def note_detection(self, timestamp: datetime) -> None:
        with self._lock:
            self.last_detection_time = timestamp

    def note_alert(self, timestamp: datetime) -> None:
        with self._lock:
            self.last_alert_time = timestamp
            self.total_alerts += 1

    def fps(self) -> float:
        with self._lock:
            if len(self.frame_times) < 2:
                return 0.0
            duration = self.frame_times[-1] - self.frame_times[0]
            if duration <= 0:
                return 0.0
            return round((len(self.frame_times) - 1) / duration, 2)

    def snapshot(self, input_mode: str, camera_connected: bool) -> RuntimeSnapshot:
        with self._lock:
            if len(self.frame_times) < 2:
                fps = 0.0
            else:
                duration = self.frame_times[-1] - self.frame_times[0]
                fps = round((len(self.frame_times) - 1) / duration, 2) if duration > 0 else 0.0
            return RuntimeSnapshot(
                status="ok",
                input_mode=input_mode,
                camera_connected=camera_connected,
                last_frame_time=self.last_frame_time.isoformat() if self.last_frame_time else None,
                last_detection_time=self.last_detection_time.isoformat() if self.last_detection_time else None,
                last_alert_time=self.last_alert_time.isoformat() if self.last_alert_time else None,
                cpu_percent=psutil.cpu_percent(interval=None),
                ram_percent=psutil.virtual_memory().percent,
                fps=fps,
                total_frames=self.total_frames,
                total_alerts=self.total_alerts,
            )
