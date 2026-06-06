import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch


class FakeFrame:
    def __init__(self, name: str, copy_result=None) -> None:
        self.name = name
        self._copy_result = copy_result if copy_result is not None else self

    def copy(self):
        return self._copy_result


class FakeCapture:
    def __init__(self, opened: bool, reads: list[tuple[bool, object | None]]) -> None:
        self.opened = opened
        self.reads = list(reads)
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True
        self.opened = False


def load_camera_module(fake_cv2):
    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        if "src.camera" in sys.modules:
            return importlib.reload(sys.modules["src.camera"])
        return importlib.import_module("src.camera")


def test_rtsp_source_uses_snapshot_stream_for_alert_frame(monkeypatch) -> None:
    captures = []
    detection_frame = FakeFrame("detect", copy_result=FakeFrame("detect-copy"))
    snapshot_frame = FakeFrame("snapshot")

    def make_capture(url, backend):
        captures.append((url, backend))
        if url == "rtsp://detect":
            return FakeCapture(True, [(True, detection_frame)])
        if url == "rtsp://snapshot":
            return FakeCapture(True, [(True, snapshot_frame)])
        raise AssertionError(f"unexpected capture url: {url}")

    fake_cv2 = SimpleNamespace(CAP_FFMPEG=1900, VideoCapture=make_capture)
    camera_module = load_camera_module(fake_cv2)
    monkeypatch.setattr(camera_module.time, "sleep", lambda *_args, **_kwargs: None)

    source = camera_module.RTSPSource("rtsp://detect", "rtsp://snapshot", reconnect_delay_seconds=3)
    packet = source.read()

    assert packet is not None
    assert packet.frame is detection_frame
    assert packet.snapshot_frame is snapshot_frame
    assert captures == [
        ("rtsp://detect", 1900),
        ("rtsp://snapshot", 1900),
    ]


def test_rtsp_source_falls_back_to_detection_frame_when_snapshot_fails(monkeypatch) -> None:
    detection_copy = FakeFrame("detect-copy")
    detection_frame = FakeFrame("detect", copy_result=detection_copy)
    snapshot_capture = FakeCapture(True, [(False, None)])

    def make_capture(url, backend):
        if url == "rtsp://detect":
            return FakeCapture(True, [(True, detection_frame)])
        if url == "rtsp://snapshot":
            return snapshot_capture
        raise AssertionError(f"unexpected capture url: {url}")

    fake_cv2 = SimpleNamespace(CAP_FFMPEG=1900, VideoCapture=make_capture)
    camera_module = load_camera_module(fake_cv2)
    monkeypatch.setattr(camera_module.time, "sleep", lambda *_args, **_kwargs: None)

    source = camera_module.RTSPSource("rtsp://detect", "rtsp://snapshot", reconnect_delay_seconds=3)
    packet = source.read()

    assert packet is not None
    assert packet.snapshot_frame is detection_copy
    assert snapshot_capture.released is True


def test_rtsp_source_returns_none_and_releases_capture_when_stream_drops(monkeypatch) -> None:
    detection_capture = FakeCapture(True, [(False, None)])

    def make_capture(url, backend):
        if url == "rtsp://detect":
            return detection_capture
        raise AssertionError(f"unexpected capture url: {url}")

    fake_cv2 = SimpleNamespace(CAP_FFMPEG=1900, VideoCapture=make_capture)
    camera_module = load_camera_module(fake_cv2)
    sleep_calls = []
    monkeypatch.setattr(camera_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    source = camera_module.RTSPSource("rtsp://detect", "", reconnect_delay_seconds=7)

    assert source.read() is None
    assert detection_capture.released is True
    assert sleep_calls == [7]
