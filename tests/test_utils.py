import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeFrame:
    def __init__(self) -> None:
        self.shape = (100, 200, 3)

    def copy(self):
        return self


def load_utils_module(fake_cv2):
    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        if "src.utils" in sys.modules:
            return importlib.reload(sys.modules["src.utils"])
        return importlib.import_module("src.utils")


def test_annotate_frame_draws_detection_box_and_labels() -> None:
    rectangle_calls = []
    text_calls = []
    fake_cv2 = SimpleNamespace(
        rectangle=lambda *args: rectangle_calls.append(args),
        putText=lambda *args: text_calls.append(args),
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=1,
    )
    utils = load_utils_module(fake_cv2)
    frame = FakeFrame()
    detections = [SimpleNamespace(box=(10, 20, 30, 40), confidence=0.91)]

    annotated = utils.annotate_frame(frame, detections, "Front Door", "2026-06-05T12:00:00")

    assert annotated is frame
    assert rectangle_calls
    assert rectangle_calls[0][1:3] == ((10, 20), (30, 40))
    assert any("person 0.91" in call[1] for call in text_calls)
    assert any("Front Door 2026-06-05T12:00:00" in call[1] for call in text_calls)


def test_save_image_creates_parent_directory(monkeypatch, tmp_path: Path) -> None:
    writes = []
    fake_cv2 = SimpleNamespace(imwrite=lambda path, frame: writes.append((path, frame)) or True)
    utils = load_utils_module(fake_cv2)
    image_path = tmp_path / "alerts" / "frame.jpg"
    frame = FakeFrame()

    result = utils.save_image(image_path, frame)

    assert result == image_path
    assert image_path.parent.exists()
    assert writes == [(str(image_path), frame)]
