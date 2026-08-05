"""RTSP camera reader built on OpenCV."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2


class RTSPCamera:
    """Read frames from an RTSP stream and reconnect after read failures."""

    def __init__(
        self,
        url: str,
        *,
        transport: str = "tcp",
        buffer_size: int = 1,
        reconnect_delay: float = 2.0,
        read_timeout: float = 2.0,
        hardware_acceleration: bool = True,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("A non-empty RTSP URL is required.")
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("Camera URL must start with rtsp:// or rtsps://.")
        if transport not in {"tcp", "udp"}:
            raise ValueError("RTSP transport must be 'tcp' or 'udp'.")

        self.url = url.strip()
        self.transport = transport
        self.buffer_size = max(1, int(buffer_size))
        self.reconnect_delay = max(0.0, float(reconnect_delay))
        self.read_timeout = max(0.1, float(read_timeout))
        self.hardware_acceleration = bool(hardware_acceleration)
        self._capture: Any = None
        self._capture_lock = threading.Lock()
        self._frame_ready = threading.Condition()
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
        """Create a low-latency FFmpeg capture."""
        # These FFmpeg options prevent old packets from accumulating. The
        # background reader below still provides the primary latency guarantee
        # by continuously draining the decoder.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}"
            "|fflags;nobuffer"
            "|flags;low_delay"
            "|max_delay;0"
            "|reorder_queue_size;0"
            "|probesize;32"
            "|analyzeduration;0"
        )
        capture_parameters = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            2000,
            cv2.CAP_PROP_N_THREADS,
            1,
        ]
        if self.hardware_acceleration:
            capture_parameters.extend(
                [
                    cv2.CAP_PROP_HW_ACCELERATION,
                    cv2.VIDEO_ACCELERATION_ANY,
                ]
            )
        capture = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
            capture_parameters,
        )
        if not capture.isOpened():
            capture.release()
            raise ConnectionError("Unable to open the configured RTSP stream.")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return capture

    def open(self) -> None:
        """Open the stream and continuously retain only its newest frame."""
        self.release()
        capture = self._create_capture()
        with self._capture_lock:
            self._capture = capture
        with self._frame_ready:
            self._latest_frame = None
            self._frame_sequence = 0
            self._last_read_sequence = 0
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="RTSP-latest-frame-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def read(self) -> tuple[bool, Any]:
        """Return the newest frame, never an old frame from a growing queue."""
        if self._reader_thread is None or not self._reader_thread.is_alive():
            try:
                self.open()
            except ConnectionError:
                return False, None

        deadline = time.monotonic() + self.read_timeout
        with self._frame_ready:
            while (
                self._frame_sequence == self._last_read_sequence
                and not self._stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._frame_ready.wait(timeout=remaining)

            if self._latest_frame is None:
                return False, None

            self._last_read_sequence = self._frame_sequence
            # The reader replaces this ndarray rather than modifying it, so
            # the single Jutemill consumer can safely use it without copying
            # an entire 1920x1080 frame.
            return True, self._latest_frame

    def read_latest(self) -> tuple[bool, Any]:
        """Return a new latest frame immediately, without blocking the GUI."""
        with self._frame_ready:
            if (
                self._latest_frame is None
                or self._frame_sequence == self._last_read_sequence
            ):
                return False, None
            self._last_read_sequence = self._frame_sequence
            return True, self._latest_frame

    def _reader_loop(self) -> None:
        """Drain RTSP continuously so slow analysis cannot create video lag."""
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
                # Releasing VideoCapture during shutdown can interrupt a
                # blocking FFmpeg read and raise from OpenCV's C++ layer.
                success, frame = False, None
            if success and frame is not None:
                with self._frame_ready:
                    self._latest_frame = frame
                    self._frame_sequence += 1
                    self._frame_ready.notify_all()
                continue

            with self._capture_lock:
                if self._capture is capture:
                    self._capture = None
            capture.release()
            if self._stop_event.is_set():
                break

        with self._frame_ready:
            self._frame_ready.notify_all()

    def release(self) -> None:
        """Stop the reader and release the OpenCV capture handle."""
        self._stop_event.set()
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()
        with self._frame_ready:
            self._frame_ready.notify_all()
        if (
            self._reader_thread is not None
            and self._reader_thread is not threading.current_thread()
        ):
            self._reader_thread.join(timeout=3.0)
        self._reader_thread = None

    def __enter__(self) -> "RTSPCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
