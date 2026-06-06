from __future__ import annotations

from dataclasses import dataclass
from .config import AppConfig, resolve_model_device
from .utils import box_allowed


@dataclass
class PersonDetection:
    box: tuple[int, int, int, int]
    confidence: float
    area_ratio: float


class YOLOPersonDetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_model_device(config.model.device)
        from ultralytics import YOLO

        self.model = YOLO(config.model.path)

    def detect(self, frame) -> list[PersonDetection]:
        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            imgsz=self.config.model.inference_size,
            conf=self.config.model.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        detections: list[PersonDetection] = []
        monitored = self.config.zones.monitored_zones if self.config.zones.enabled else []
        ignore = self.config.zones.ignore_zones if self.config.zones.enabled else []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id != self.config.model.person_class_id:
                    continue
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                area_ratio = ((x2 - x1) * (y2 - y1)) / max(width * height, 1)
                if area_ratio < self.config.detection.min_box_area_ratio:
                    continue
                if area_ratio > self.config.detection.max_box_area_ratio:
                    continue
                if not box_allowed((x1, y1, x2, y2), frame.shape, monitored, ignore):
                    continue
                detections.append(
                    PersonDetection(
                        box=(x1, y1, x2, y2),
                        confidence=confidence,
                        area_ratio=area_ratio,
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections
