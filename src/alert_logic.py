from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AlertDecision:
    triggered: bool
    reason: str
    positive_frames: int
    detection_duration_seconds: float
    cooldown_remaining_seconds: float


class TemporalAlertFilter:
    def __init__(
        self,
        rolling_window_size: int,
        min_positive_frames: int,
        min_detection_duration_seconds: float,
        cooldown_seconds: float,
    ) -> None:
        self.history: deque[tuple[datetime, bool]] = deque(maxlen=rolling_window_size)
        self.min_positive_frames = min_positive_frames
        self.min_detection_duration_seconds = min_detection_duration_seconds
        self.cooldown_seconds = cooldown_seconds
        self.current_positive_start: datetime | None = None
        self.last_alert_time: datetime | None = None

    def evaluate(self, timestamp: datetime, has_detection: bool) -> AlertDecision:
        self.history.append((timestamp, has_detection))

        if has_detection and self.current_positive_start is None:
            self.current_positive_start = timestamp
        if not has_detection:
            recent_positive = any(flag for _, flag in self.history)
            if not recent_positive:
                self.current_positive_start = None

        positive_frames = sum(1 for _, flag in self.history if flag)
        duration = 0.0
        if self.current_positive_start is not None:
            duration = max(0.0, (timestamp - self.current_positive_start).total_seconds())

        cooldown_remaining = 0.0
        if self.last_alert_time is not None:
            cooldown_remaining = max(
                0.0,
                self.cooldown_seconds - (timestamp - self.last_alert_time).total_seconds(),
            )

        if positive_frames < self.min_positive_frames:
            return AlertDecision(False, "insufficient_positive_frames", positive_frames, duration, cooldown_remaining)
        if duration < self.min_detection_duration_seconds:
            return AlertDecision(False, "insufficient_duration", positive_frames, duration, cooldown_remaining)
        if not has_detection:
            return AlertDecision(False, "current_frame_no_detection", positive_frames, duration, cooldown_remaining)
        if cooldown_remaining > 0:
            return AlertDecision(False, "cooldown_active", positive_frames, duration, cooldown_remaining)

        self.last_alert_time = timestamp
        return AlertDecision(True, "confirmed", positive_frames, duration, 0.0)
