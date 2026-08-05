"""Reusable OpenCV camera for built-in and USB camera devices."""

from __future__ import annotations

from typing import Any

import cv2


class LocalCamera:
    """Read frames from one local camera device."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._capture: Any = None

    @property
    def is_opened(self) -> bool:
        return bool(self._capture is not None and self._capture.isOpened())

    def open(self) -> None:
        self.release()
        capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise ConnectionError(f"Could not open local camera {self.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture

    def read(self) -> tuple[bool, Any]:
        if not self.is_opened:
            try:
                self.open()
            except ConnectionError:
                return False, None
        return self._capture.read()

    def read_latest(self) -> tuple[bool, Any]:
        """Match the RTSP camera interface used by camera consumers."""
        return self.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "LocalCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
