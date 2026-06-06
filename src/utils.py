from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_local() -> datetime:
    return datetime.now().astimezone()


def timestamp_slug(moment: datetime | None = None) -> str:
    stamp = moment or now_local()
    return stamp.strftime("%Y%m%d_%H%M%S")


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def normalized_center(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2) / max(width, 1)
    center_y = ((y1 + y2) / 2) / max(height, 1)
    return center_x, center_y


def box_allowed(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    monitored_zones: list[list[list[float]]],
    ignore_zones: list[list[list[float]]],
) -> bool:
    height, width = frame_shape[:2]
    center = normalized_center(box, width, height)
    if ignore_zones and any(point_in_polygon(center, zone) for zone in ignore_zones):
        return False
    if monitored_zones:
        return any(point_in_polygon(center, zone) for zone in monitored_zones)
    return True


def draw_zones(frame: np.ndarray, zones: Iterable[list[list[float]]], color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    for zone in zones:
        points = np.array(
            [[int(px * width), int(py * height)] for px, py in zone],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], True, color, 2)


def annotate_frame(
    frame: np.ndarray,
    detections: Iterable[object],
    camera_name: str,
    timestamp_text: str,
) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"person {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"{camera_name} {timestamp_text}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def save_image(path: str | Path, frame: np.ndarray) -> Path:
    image_path = Path(path)
    ensure_dir(image_path.parent)
    cv2.imwrite(str(image_path), frame)
    return image_path


def format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"
