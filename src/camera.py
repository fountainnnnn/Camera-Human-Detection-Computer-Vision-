from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from .config import AppConfig
from .utils import now_local


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: datetime
    snapshot_frame: np.ndarray | None = None


class VideoSource:
    finite = False

    def read(self) -> FramePacket | None:
        raise NotImplementedError

    def read_snapshot(self) -> np.ndarray | None:
        return None

    def release(self) -> None:
        return


class OpenCVSource(VideoSource):
    def __init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class WebcamSource(OpenCVSource):
    def __init__(self, webcam_index: int) -> None:
        super().__init__()
        self.webcam_index = webcam_index
        self._open()

    def _open(self) -> None:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        self.capture = cv2.VideoCapture(self.webcam_index, backend)

    def read(self) -> FramePacket | None:
        if self.capture is None or not self.capture.isOpened():
            self._open()
            time.sleep(0.5)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.release()
            return None
        return FramePacket(frame=frame, timestamp=now_local(), snapshot_frame=frame.copy())


class RTSPSource(OpenCVSource):
    def __init__(
        self,
        detection_url: str,
        snapshot_url: str,
        reconnect_delay_seconds: int,
        threaded: bool = True,
    ) -> None:
        super().__init__()
        self.detection_url = detection_url
        self.snapshot_url = snapshot_url
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.threaded = threaded
        self._snapshot_capture: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: datetime | None = None
        self._lock = threading.Lock()
        self._stop_reader = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._open()
        if self.threaded:
            self._reader_thread = threading.Thread(target=self._reader_loop, name="rtsp-frame-reader", daemon=True)
            self._reader_thread.start()

    def _open(self) -> None:
        self.capture = cv2.VideoCapture(self.detection_url, cv2.CAP_FFMPEG)
        buffer_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_prop is not None and callable(getattr(self.capture, "set", None)):
            self.capture.set(buffer_prop, 1)

    def _open_snapshot(self) -> None:
        if self.snapshot_url:
            self._snapshot_capture = cv2.VideoCapture(self.snapshot_url, cv2.CAP_FFMPEG)

    def _read_snapshot(self) -> np.ndarray | None:
        if not self.snapshot_url:
            return None
        if self._snapshot_capture is None or not self._snapshot_capture.isOpened():
            self._open_snapshot()
        if self._snapshot_capture is None:
            return None
        ok, frame = self._snapshot_capture.read()
        if not ok or frame is None:
            self._snapshot_capture.release()
            self._snapshot_capture = None
            return None
        return frame

    def _read_from_capture(self) -> FramePacket | None:
        if self.capture is None or not self.capture.isOpened():
            self._open()
            time.sleep(self.reconnect_delay_seconds)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            super().release()
            time.sleep(self.reconnect_delay_seconds)
            return None
        return FramePacket(frame=frame, timestamp=now_local(), snapshot_frame=None)

    def _reader_loop(self) -> None:
        while not self._stop_reader.is_set():
            packet = self._read_from_capture()
            if packet is None:
                continue
            with self._lock:
                self._latest_frame = packet.frame
                self._latest_timestamp = packet.timestamp

    def read(self) -> FramePacket | None:
        if not self.threaded:
            return self._read_from_capture()

        deadline = time.monotonic() + max(0.5, min(float(self.reconnect_delay_seconds), 2.0))
        while not self._stop_reader.is_set():
            with self._lock:
                frame = self._latest_frame
                timestamp = self._latest_timestamp
            if frame is not None and timestamp is not None:
                return FramePacket(frame=frame, timestamp=timestamp, snapshot_frame=None)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
        return None

    def read_snapshot(self) -> np.ndarray | None:
        return self._read_snapshot()

    def release(self) -> None:
        self._stop_reader.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        super().release()
        if self._snapshot_capture is not None:
            self._snapshot_capture.release()
            self._snapshot_capture = None


class VideoFileSource(OpenCVSource):
    finite = True

    def __init__(self, path: str) -> None:
        super().__init__()
        self.capture = cv2.VideoCapture(path)

    def read(self) -> FramePacket | None:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        return FramePacket(frame=frame, timestamp=now_local(), snapshot_frame=frame.copy())


class ImageFolderSource(VideoSource):

    def __init__(self, folder: str, loop: bool = False) -> None:
        self.paths = [
            path
            for path in sorted(Path(folder).glob("*"))
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        self.index = 0
        self.loop = loop
        self.finite = not loop

    def read(self) -> FramePacket | None:
        if self.index >= len(self.paths):
            if self.loop and self.paths:
                self.index = 0
            else:
                return None
        if not self.paths:
            return None
        path = self.paths[self.index]
        self.index += 1
        frame = cv2.imread(str(path))
        if frame is None:
            return None
        return FramePacket(frame=frame, timestamp=now_local(), snapshot_frame=frame.copy())


def create_source(config: AppConfig) -> VideoSource:
    mode = config.input.mode.lower()
    if mode == "webcam":
        return WebcamSource(config.input.webcam_index)
    if mode == "rtsp":
        return RTSPSource(
            detection_url=config.input.rtsp_detection_url,
            snapshot_url=config.input.rtsp_snapshot_url,
            reconnect_delay_seconds=config.input.reconnect_delay_seconds,
        )
    if mode == "video_file":
        return VideoFileSource(config.input.video_path)
    if mode == "image_folder":
        return ImageFolderSource(config.input.image_folder, loop=config.input.loop_image_folder)
    raise ValueError(f"Unsupported input mode: {config.input.mode}")
