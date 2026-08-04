"""Non-blocking outbound video publishing for MediaMTX."""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from typing import Any

import av
import cv2
import numpy as np


class MediaMTXPublisher:
    """Publish only the latest OpenCV frame to a MediaMTX RTSP path.

    Encoding and network I/O run in a background thread, so a slow or
    disconnected network never blocks the camera-processing loop.
    """

    def __init__(
        self,
        url: str,
        *,
        fps: float = 10,
        bitrate: int = 1_500_000,
        codec: str = "libx264",
        transport: str = "tcp",
        reconnect_delay_seconds: float = 3,
        copy_frames: bool = True,
    ) -> None:
        if not url.strip():
            raise ValueError("A MediaMTX publish URL is required.")

        self.url = url.strip()
        self.fps = max(1.0, float(fps))
        self.bitrate = max(100_000, int(bitrate))
        self.codec = codec
        self.transport = transport
        self.reconnect_delay = max(0.1, float(reconnect_delay_seconds))
        self.copy_frames = bool(copy_frames)

        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the outbound publisher thread."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="MediaMTXPublisher",
                daemon=True,
            )
            self._thread.start()
        print(f"[MEDIA PUBLISHER] Target: {self.url}", flush=True)

    def publish(self, frame: np.ndarray) -> int:
        """Replace the pending frame without waiting for the network."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise TypeError("publish() requires a BGR OpenCV image.")

        with self._condition:
            self._latest_frame = frame.copy() if self.copy_frames else frame
            self._sequence += 1
            sequence = self._sequence
            self._condition.notify()
        return sequence

    def _wait_for_frame(
        self,
        previous_sequence: int,
    ) -> tuple[np.ndarray | None, int]:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._running or self._sequence != previous_sequence,
                timeout=1,
            )
            if not self._running:
                return None, previous_sequence
            return self._latest_frame, self._sequence

    def _open_output(
        self,
        frame: np.ndarray,
    ) -> tuple[av.container.OutputContainer, Any]:
        height, width = frame.shape[:2]
        if width % 2:
            width -= 1
        if height % 2:
            height -= 1

        container = av.open(
            self.url,
            mode="w",
            format="rtsp",
            options={"rtsp_transport": self.transport},
        )
        stream = container.add_stream(self.codec, rate=round(self.fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = self.bitrate
        stream.time_base = Fraction(1, round(self.fps))
        stream.codec_context.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "bf": "0",
            "g": str(max(1, round(self.fps * 2))),
        }
        return container, stream

    @staticmethod
    def _prepare_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _publish_until_disconnected(self) -> None:
        with self._condition:
            starting_sequence = self._sequence
        frame, sequence = self._wait_for_frame(starting_sequence)
        if frame is None:
            return

        container = None
        try:
            container, stream = self._open_output(frame)
            pts = 0
            frame_interval = 1 / self.fps
            next_frame_at = time.monotonic()
            print("[MEDIA PUBLISHER] Connected", flush=True)

            while self._running:
                frame, sequence = self._wait_for_frame(sequence)
                if frame is None:
                    break
                remaining = next_frame_at - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                    with self._condition:
                        if self._latest_frame is not None:
                            frame = self._latest_frame
                            sequence = self._sequence

                prepared = self._prepare_frame(frame, stream.width, stream.height)
                video_frame = av.VideoFrame.from_ndarray(
                    prepared,
                    format="bgr24",
                )
                video_frame.pts = pts
                video_frame.time_base = stream.time_base
                pts += 1
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                next_frame_at = max(
                    next_frame_at + frame_interval,
                    time.monotonic(),
                )

            for packet in stream.encode():
                container.mux(packet)
        finally:
            if container is not None:
                container.close()

    def _run(self) -> None:
        while self._running:
            try:
                self._publish_until_disconnected()
            except Exception as error:
                if self._running:
                    print(
                        "[MEDIA PUBLISHER] Connection failed; retrying: "
                        f"{error}",
                        flush=True,
                    )
                    time.sleep(self.reconnect_delay)

    def close(self, timeout: float = 5) -> None:
        """Stop publishing and close the background thread."""
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(0, float(timeout)))
        self._thread = None
        print("[MEDIA PUBLISHER] Stopped", flush=True)

    def __enter__(self) -> "MediaMTXPublisher":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
