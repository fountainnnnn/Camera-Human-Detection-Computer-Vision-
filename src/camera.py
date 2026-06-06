from __future__ import annotations

import time
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
    def __init__(self, detection_url: str, snapshot_url: str, reconnect_delay_seconds: int) -> None:
        super().__init__()
        self.detection_url = detection_url
        self.snapshot_url = snapshot_url
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._snapshot_capture: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        self.capture = cv2.VideoCapture(self.detection_url, cv2.CAP_FFMPEG)

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

    def read(self) -> FramePacket | None:
        if self.capture is None or not self.capture.isOpened():
            self._open()
            time.sleep(self.reconnect_delay_seconds)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.release()
            time.sleep(self.reconnect_delay_seconds)
            return None
        snapshot = self._read_snapshot()
        if snapshot is None:
            snapshot = frame.copy()
        return FramePacket(frame=frame, timestamp=now_local(), snapshot_frame=snapshot)

    def release(self) -> None:
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
