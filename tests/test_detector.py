from types import SimpleNamespace
from unittest.mock import patch
import importlib
import sys

import numpy  # noqa: F401

from src.config import AppConfig


class FakeYOLO:
    last_instance = None

    def __init__(self, path: str) -> None:
        self.path = path
        self.last_predict_kwargs = None
        FakeYOLO.last_instance = self

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs
        return []


class FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = [FakeScalar(class_id)]
        self.conf = [FakeScalar(confidence)]
        self.xyxy = [FakeVector(xyxy)]


def test_detector_uses_resolved_device_for_prediction() -> None:
    fake_module = SimpleNamespace(YOLO=FakeYOLO)
    fake_cv2 = SimpleNamespace()
    frame = SimpleNamespace(shape=(20, 30, 3))
    with patch.dict(sys.modules, {"ultralytics": fake_module, "cv2": fake_cv2}):
        detector_module = importlib.import_module("src.detector")
        with patch.object(detector_module, "resolve_model_device", return_value="cuda:0"):
            YOLOPersonDetector = detector_module.YOLOPersonDetector
            config = AppConfig()
            config.model.confidence_threshold = 0.72
            config.model.inference_size = 512
            detector = YOLOPersonDetector(config)
            detector.detect(frame)
    assert detector.device == "cuda:0"
    assert FakeYOLO.last_instance is not None
    assert FakeYOLO.last_instance.path == "models/yolo11n.pt"
    assert FakeYOLO.last_instance.last_predict_kwargs["conf"] == 0.72
    assert FakeYOLO.last_instance.last_predict_kwargs["imgsz"] == 512
    assert FakeYOLO.last_instance.last_predict_kwargs["device"] == "cuda:0"


def test_detector_filters_non_person_small_and_ignored_boxes() -> None:
    fake_cv2 = SimpleNamespace()

    class FakeFilteringYOLO(FakeYOLO):
        def predict(self, **kwargs):
            self.last_predict_kwargs = kwargs
            return [
                SimpleNamespace(
                    boxes=[
                        FakeBox(0, 0.92, [20, 20, 60, 80]),   # valid
                        FakeBox(1, 0.99, [20, 20, 60, 80]),   # wrong class
                        FakeBox(0, 0.95, [0, 0, 5, 5]),       # too small
                        FakeBox(0, 0.88, [150, 10, 190, 70]), # in ignore zone
                    ]
                )
            ]

    fake_module = SimpleNamespace(YOLO=FakeFilteringYOLO)
    with patch.dict(sys.modules, {"ultralytics": fake_module, "cv2": fake_cv2}):
        detector_module = importlib.import_module("src.detector")
        with patch.object(detector_module, "resolve_model_device", return_value="cpu"):
            config = AppConfig()
            config.detection.min_box_area_ratio = 0.02
            config.zones.enabled = True
            config.zones.ignore_zones = [[[0.7, 0.0], [1.0, 0.0], [1.0, 1.0], [0.7, 1.0]]]
            detector = detector_module.YOLOPersonDetector(config)
            detections = detector.detect(SimpleNamespace(shape=(100, 200, 3)))

    assert len(detections) == 1
    assert detections[0].box == (20, 20, 60, 80)


def test_detector_requires_monitored_zone_when_enabled() -> None:
    fake_cv2 = SimpleNamespace()

    class FakeZoneYOLO(FakeYOLO):
        def predict(self, **kwargs):
            self.last_predict_kwargs = kwargs
            return [
                SimpleNamespace(
                    boxes=[
                        FakeBox(0, 0.91, [10, 10, 50, 90]),    # outside monitored zone
                        FakeBox(0, 0.87, [120, 10, 170, 90]),  # inside monitored zone
                    ]
                )
            ]

    fake_module = SimpleNamespace(YOLO=FakeZoneYOLO)
    with patch.dict(sys.modules, {"ultralytics": fake_module, "cv2": fake_cv2}):
        detector_module = importlib.import_module("src.detector")
        with patch.object(detector_module, "resolve_model_device", return_value="cpu"):
            config = AppConfig()
            config.zones.enabled = True
            config.zones.monitored_zones = [[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]]
            detector = detector_module.YOLOPersonDetector(config)
            detections = detector.detect(SimpleNamespace(shape=(100, 200, 3)))

    assert len(detections) == 1
    assert detections[0].box == (120, 10, 170, 90)
