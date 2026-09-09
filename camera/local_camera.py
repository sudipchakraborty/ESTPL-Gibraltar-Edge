"""Low-latency reusable OpenCV camera for local USB devices."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2


class LocalCamera:
    """Continuously drain a USB camera and expose only its newest frame."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        reconnect_delay: float = 0.5,
    ) -> None:
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self._capture: Any = None
        self._capture_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_frame: Any = None
        self._frame_sequence = 0
        self._last_read_sequence = 0
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    @property
    def is_opened(self) -> bool:
        with self._capture_lock:
            return bool(self._capture is not None and self._capture.isOpened())

    def _create_capture(self) -> Any:
        capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise ConnectionError(f"Could not open local camera {self.index}")

        # MJPEG avoids the very low frame rates many USB cameras negotiate for
        # uncompressed 720p. Apply it before dimensions and FPS.
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def open(self) -> None:
        self.release()
        capture = self._create_capture()
        with self._capture_lock:
            self._capture = capture
        with self._frame_lock:
            self._latest_frame = None
            self._frame_sequence = 0
            self._last_read_sequence = 0
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"USB-{self.index}-latest-frame-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def read(self) -> tuple[bool, Any]:
        """Return the newest available frame without building a queue."""
        if self._reader_thread is None or not self._reader_thread.is_alive():
            try:
                self.open()
            except ConnectionError:
                return False, None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not self._stop_event.is_set():
            success, frame = self.read_latest()
            if success:
                return True, frame
            self._stop_event.wait(0.005)
        return False, None

    def read_latest(self) -> tuple[bool, Any]:
        """Return a fresh latest frame immediately; never block the GUI."""
        with self._frame_lock:
            if (
                self._latest_frame is None
                or self._frame_sequence == self._last_read_sequence
            ):
                return False, None
            self._last_read_sequence = self._frame_sequence
            return True, self._latest_frame

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._capture_lock:
                capture = self._capture
            if capture is None:
                if self._stop_event.wait(self.reconnect_delay):
                    break
                try:
                    capture = self._create_capture()
                except ConnectionError:
                    continue
                with self._capture_lock:
                    if self._stop_event.is_set():
                        capture.release()
                        break
                    self._capture = capture
            try:
                success, frame = capture.read()
            except cv2.error:
                success, frame = False, None
            if success and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_sequence += 1
                continue
            with self._capture_lock:
                if self._capture is capture:
                    self._capture = None
            capture.release()

    def release(self) -> None:
        self._stop_event.set()
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()
        if (
            self._reader_thread is not None
            and self._reader_thread is not threading.current_thread()
        ):
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None
        with self._frame_lock:
            self._latest_frame = None

    def __enter__(self) -> "LocalCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
