from datetime import datetime, timezone
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeFrame:
    def copy(self):
        return self


class FakeSource:
    def __init__(self, packets) -> None:
        self._packets = list(packets)
        self.released = False

    def read(self):
        if self._packets:
            return self._packets.pop(0)
        return None

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def __init__(self, detections_per_frame) -> None:
        self._detections_per_frame = list(detections_per_frame)

    def detect(self, _frame):
        if self._detections_per_frame:
            return self._detections_per_frame.pop(0)
        return []


def load_test_webcam_module():
    with patch.dict(sys.modules, {"common": SimpleNamespace(), "cv2": SimpleNamespace()}):
        if "scripts.test_webcam" in sys.modules:
            return importlib.reload(sys.modules["scripts.test_webcam"])
        return importlib.import_module("scripts.test_webcam")


def test_webcam_script_prints_detection_status_and_releases_source(monkeypatch, capsys) -> None:
    module = load_test_webcam_module()
    packet = SimpleNamespace(
        frame=FakeFrame(),
        snapshot_frame=FakeFrame(),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    source = FakeSource([packet, packet])
    detector = FakeDetector([[SimpleNamespace(confidence=0.8)], []])
    config = SimpleNamespace(input=SimpleNamespace(mode="initial"), camera=SimpleNamespace(name="Test Camera"))

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "create_source", lambda _config: source)
    monkeypatch.setattr(module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(sys, "argv", ["test_webcam.py", "--frames", "2"])

    assert module.main() == 0
    assert source.released is True
    assert config.input.mode == "webcam"

    output = capsys.readouterr().out
    assert "frame=1 detections=1" in output
    assert "frame=2 detections=0" in output
    assert "Webcam test completed: 2 frames processed." in output


def test_webcam_script_can_save_snapshots(monkeypatch, capsys, tmp_path: Path) -> None:
    module = load_test_webcam_module()
    packet = SimpleNamespace(
        frame=FakeFrame(),
        snapshot_frame=FakeFrame(),
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    source = FakeSource([packet])
    detector = FakeDetector([[SimpleNamespace(confidence=0.9)]])
    config = SimpleNamespace(input=SimpleNamespace(mode="initial"), camera=SimpleNamespace(name="Test Camera"))
    saved_paths = []

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "create_source", lambda _config: source)
    monkeypatch.setattr(module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(module, "annotate_frame", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(module, "ensure_dir", lambda path: Path(path))
    monkeypatch.setattr(module, "save_image", lambda path, _frame: saved_paths.append(Path(path)) or Path(path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_webcam.py", "--frames", "1", "--save-snapshots", "--output-dir", str(tmp_path)],
    )

    assert module.main() == 0
    assert len(saved_paths) == 1
    assert saved_paths[0].parent == tmp_path
    assert saved_paths[0].name.startswith("webcam_test_")
    assert "snapshot=" in capsys.readouterr().out


def test_webcam_script_returns_error_when_no_frame_arrives(monkeypatch, capsys) -> None:
    module = load_test_webcam_module()
    source = FakeSource([None])
    detector = FakeDetector([])
    config = SimpleNamespace(input=SimpleNamespace(mode="initial"), camera=SimpleNamespace(name="Test Camera"))

    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "create_source", lambda _config: source)
    monkeypatch.setattr(module, "YOLOPersonDetector", lambda _config: detector)
    monkeypatch.setattr(sys, "argv", ["test_webcam.py", "--frames", "1"])

    assert module.main() == 1
    assert source.released is True
    assert "No webcam frame received." in capsys.readouterr().out
