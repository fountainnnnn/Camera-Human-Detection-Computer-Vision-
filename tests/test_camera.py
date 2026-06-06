import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.config import AppConfig


class FakeVideoCapture:
    def __init__(self, *args) -> None:
        self.args = args

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return

    def read(self):
        return False, None


def load_camera_module(fake_cv2=None):
    fake_cv2 = fake_cv2 or SimpleNamespace(
        CAP_DSHOW=700,
        CAP_FFMPEG=1900,
        VideoCapture=FakeVideoCapture,
        imread=lambda path: None,
    )
    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        return importlib.import_module("src.camera")


def test_create_source_supports_all_configured_input_modes(tmp_path: Path) -> None:
    camera_module = load_camera_module()

    webcam_config = AppConfig()
    webcam_config.input.mode = "webcam"
    webcam_config.input.webcam_index = 2
    webcam_source = camera_module.create_source(webcam_config)
    assert isinstance(webcam_source, camera_module.WebcamSource)
    assert webcam_source.webcam_index == 2

    rtsp_config = AppConfig()
    rtsp_config.input.mode = "rtsp"
    rtsp_config.input.rtsp_detection_url = "rtsp://detect"
    rtsp_config.input.rtsp_snapshot_url = "rtsp://snapshot"
    rtsp_config.input.reconnect_delay_seconds = 9
    rtsp_source = camera_module.create_source(rtsp_config)
    assert isinstance(rtsp_source, camera_module.RTSPSource)
    assert rtsp_source.detection_url == "rtsp://detect"
    assert rtsp_source.snapshot_url == "rtsp://snapshot"
    assert rtsp_source.reconnect_delay_seconds == 9

    video_config = AppConfig()
    video_config.input.mode = "video_file"
    video_config.input.video_path = str(tmp_path / "clip.mp4")
    video_source = camera_module.create_source(video_config)
    assert isinstance(video_source, camera_module.VideoFileSource)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame1.jpg").write_text("x", encoding="utf-8")
    (image_dir / "frame2.png").write_text("x", encoding="utf-8")
    (image_dir / "notes.txt").write_text("x", encoding="utf-8")

    image_config = AppConfig()
    image_config.input.mode = "image_folder"
    image_config.input.image_folder = str(image_dir)
    image_source = camera_module.create_source(image_config)
    assert isinstance(image_source, camera_module.ImageFolderSource)
    assert [path.name for path in image_source.paths] == ["frame1.jpg", "frame2.png"]
    assert image_source.finite is True

    image_config.input.loop_image_folder = True
    looping_source = camera_module.create_source(image_config)
    assert isinstance(looping_source, camera_module.ImageFolderSource)
    assert looping_source.finite is False


def test_create_source_rejects_unknown_mode() -> None:
    camera_module = load_camera_module()
    config = AppConfig()
    config.input.mode = "unknown_mode"

    with pytest.raises(ValueError, match="Unsupported input mode"):
        camera_module.create_source(config)


def test_image_folder_source_loops_when_enabled(tmp_path: Path) -> None:
    frames = {"frame1.jpg": SimpleNamespace(copy=lambda: "snapshot")}

    def fake_imread(path):
        return frames[Path(path).name]

    fake_cv2 = SimpleNamespace(
        CAP_DSHOW=700,
        CAP_FFMPEG=1900,
        VideoCapture=FakeVideoCapture,
        imread=fake_imread,
    )
    camera_module = load_camera_module(fake_cv2)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame1.jpg").write_text("x", encoding="utf-8")

    source = camera_module.ImageFolderSource(str(image_dir), loop=True)

    assert source.read() is not None
    assert source.read() is not None


def test_webcam_source_reopens_capture_when_closed(monkeypatch) -> None:
    frame = SimpleNamespace(copy=lambda: "snapshot-frame")

    class ClosedCapture:
        def __init__(self, *args) -> None:
            self.args = args
            self.released = False

        def isOpened(self) -> bool:
            return False

        def read(self):
            return False, None

        def release(self) -> None:
            self.released = True

    class OpenCapture(ClosedCapture):
        def isOpened(self) -> bool:
            return True

        def read(self):
            return True, frame

    created = []

    def make_capture(*args):
        created.append(args)
        if len(created) == 1:
            return ClosedCapture(*args)
        return OpenCapture(*args)

    fake_cv2 = SimpleNamespace(CAP_DSHOW=700, CAP_FFMPEG=1900, VideoCapture=make_capture, imread=lambda path: None)
    camera_module = load_camera_module(fake_cv2)
    monkeypatch.setattr(camera_module.time, "sleep", lambda *_args, **_kwargs: None)

    source = camera_module.WebcamSource(3)
    packet = source.read()

    assert packet is not None
    assert packet.frame is frame
    assert packet.snapshot_frame == "snapshot-frame"
    assert created == [(3, 700), (3, 700)]


def test_webcam_source_returns_none_when_frame_read_fails(monkeypatch) -> None:
    class OpenCapture:
        def __init__(self, *args) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self):
            return False, None

        def release(self) -> None:
            self.released = True

    capture = OpenCapture()
    fake_cv2 = SimpleNamespace(
        CAP_DSHOW=700,
        CAP_FFMPEG=1900,
        VideoCapture=lambda *args: capture,
        imread=lambda path: None,
    )
    camera_module = load_camera_module(fake_cv2)
    monkeypatch.setattr(camera_module.time, "sleep", lambda *_args, **_kwargs: None)

    source = camera_module.WebcamSource(0)

    assert source.read() is None
    assert capture.released is True
