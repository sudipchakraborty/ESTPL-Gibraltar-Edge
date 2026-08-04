"""Publish OpenCV frames to browsers through an embedded WebRTC server."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


VIDEO_CLOCK_RATE = 90_000
VIEWER_FILE = Path(__file__).resolve().parent / "viewer.html"


class OfferRequest(BaseModel):
    sdp: str
    type: str


class _LatestFrameBuffer:
    def __init__(
        self,
        *,
        target_width: int | None,
        target_height: int | None,
        copy_frames: bool,
    ) -> None:
        if (target_width is None) != (target_height is None):
            raise ValueError(
                "target_width and target_height must be provided together."
            )
        self.target_width = (
            max(2, int(target_width)) if target_width is not None else None
        )
        self.target_height = (
            max(2, int(target_height)) if target_height is not None else None
        )
        self.copy_frames = bool(copy_frames)
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._closed = False

    def publish(self, frame: np.ndarray) -> int:
        if not isinstance(frame, np.ndarray):
            raise TypeError("VideoSender.publish expects an OpenCV ndarray.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frame must have BGR shape (height, width, 3).")
        if self.target_width is not None and self.target_height is not None:
            if (
                frame.shape[1] != self.target_width
                or frame.shape[0] != self.target_height
            ):
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )
        elif self.copy_frames:
            frame = frame.copy()

        with self._condition:
            if self._closed:
                return self._sequence
            self._frame = frame
            self._sequence += 1
            self._condition.notify_all()
            return self._sequence

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float,
    ) -> tuple[int, np.ndarray | None]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._sequence <= after_sequence
                and not self._closed
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._sequence, self._frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _LatestFrameTrack(VideoStreamTrack):
    def __init__(
        self,
        frame_buffer: _LatestFrameBuffer,
        *,
        fps: float,
        placeholder_width: int,
        placeholder_height: int,
    ) -> None:
        super().__init__()
        self.frame_buffer = frame_buffer
        self.fps = max(1.0, float(fps))
        self.placeholder_width = placeholder_width
        self.placeholder_height = placeholder_height
        self._sequence = -1
        self._last_frame: np.ndarray | None = None
        self._pts = 0
        self._pts_step = max(1, round(VIDEO_CLOCK_RATE / self.fps))

    async def recv(self) -> VideoFrame:
        sequence, frame = await asyncio.to_thread(
            self.frame_buffer.wait_for_frame,
            self._sequence,
            1.0,
        )
        if frame is not None:
            self._sequence = sequence
            self._last_frame = frame
        elif self._last_frame is None:
            self._last_frame = np.zeros(
                (self.placeholder_height, self.placeholder_width, 3),
                dtype=np.uint8,
            )

        video_frame = VideoFrame.from_ndarray(
            self._last_frame,
            format="bgr24",
        )
        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        self._pts += self._pts_step
        return video_frame


class VideoSender:
    """Host WebRTC signaling and publish only the newest OpenCV frame."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        fps: float = 10.0,
        target_width: int | None = None,
        target_height: int | None = None,
        copy_frames: bool = True,
        ice_servers: list[dict[str, Any]] | None = None,
        preferred_codec: str = "video/H264",
        allowed_origins: list[str] | None = None,
        log_level: str = "warning",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.fps = max(1.0, float(fps))
        self.preferred_codec = preferred_codec
        self.log_level = log_level
        self.frame_buffer = _LatestFrameBuffer(
            target_width=target_width,
            target_height=target_height,
            copy_frames=copy_frames,
        )
        self.placeholder_width = target_width or 640
        self.placeholder_height = target_height or 360
        self.peer_connections: set[RTCPeerConnection] = set()
        self.total_frames_published = 0
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._rtc_configuration = RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=entry["urls"],
                    username=entry.get("username"),
                    credential=entry.get("credential"),
                )
                for entry in (ice_servers or [])
            ]
        )

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            yield
            await self._close_peer_connections()

        self.app = FastAPI(
            title="VisualAI VideoSender",
            version="1.0.0",
            lifespan=lifespan,
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins or ["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        self._register_routes()

    @property
    def viewer_url(self) -> str:
        display_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{display_host}:{self.port}"

    def publish(self, frame: np.ndarray) -> int:
        """Replace the buffered frame and return its sequence number."""
        sequence = self.frame_buffer.publish(frame)
        self.total_frames_published = sequence
        return sequence

    def _register_routes(self) -> None:
        @self.app.get("/")
        async def viewer() -> FileResponse:
            return FileResponse(VIEWER_FILE)

        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "running",
                "frames_published": self.total_frames_published,
                "active_viewers": len(self.peer_connections),
            }

        @self.app.post("/offer")
        async def offer(request: OfferRequest) -> dict[str, str]:
            peer = RTCPeerConnection(configuration=self._rtc_configuration)
            self.peer_connections.add(peer)

            @peer.on("connectionstatechange")
            async def connection_state_change() -> None:
                if peer.connectionState in {
                    "failed",
                    "closed",
                    "disconnected",
                }:
                    await peer.close()
                    self.peer_connections.discard(peer)

            try:
                await peer.setRemoteDescription(
                    RTCSessionDescription(
                        sdp=request.sdp,
                        type=request.type,
                    )
                )
                track = _LatestFrameTrack(
                    self.frame_buffer,
                    fps=self.fps,
                    placeholder_width=self.placeholder_width,
                    placeholder_height=self.placeholder_height,
                )
                sender = peer.addTrack(track)
                capabilities = RTCRtpSender.getCapabilities("video")
                preferred = [
                    codec
                    for codec in capabilities.codecs
                    if codec.mimeType.lower() == self.preferred_codec.lower()
                ]
                if preferred:
                    transceiver = next(
                        item
                        for item in peer.getTransceivers()
                        if item.sender == sender
                    )
                    transceiver.setCodecPreferences(preferred)

                answer = await peer.createAnswer()
                await peer.setLocalDescription(answer)
                return {
                    "sdp": peer.localDescription.sdp,
                    "type": peer.localDescription.type,
                }
            except Exception as error:
                await peer.close()
                self.peer_connections.discard(peer)
                raise HTTPException(status_code=400, detail=str(error)) from error

    async def _close_peer_connections(self) -> None:
        peers = list(self.peer_connections)
        self.peer_connections.clear()
        if peers:
            await asyncio.gather(
                *(peer.close() for peer in peers),
                return_exceptions=True,
            )

    def _run(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
        )
        self._server = uvicorn.Server(config)
        print(
            f"[VIDEO SENDER] Viewer available at {self.viewer_url}",
            flush=True,
        )
        self._server.run()

    def start(self, startup_timeout: float = 5.0) -> None:
        """Start the WebRTC server in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="VideoSenderServer",
            daemon=True,
        )
        self._thread.start()

        connect_host = (
            "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        )
        deadline = time.monotonic() + max(0.1, float(startup_timeout))
        while time.monotonic() < deadline:
            if self._thread is None or not self._thread.is_alive():
                raise RuntimeError("VideoSender stopped during startup.")
            try:
                with socket.create_connection(
                    (connect_host, self.port),
                    timeout=0.2,
                ):
                    print("[VIDEO SENDER] Ready", flush=True)
                    return
            except OSError:
                time.sleep(0.05)
        self.close()
        raise TimeoutError(
            f"VideoSender did not start on {connect_host}:{self.port}."
        )

    def close(self, timeout: float = 5.0) -> None:
        """Stop publishing and shut down the WebRTC server."""
        self.frame_buffer.close()
        if self._server is not None:
            self._server.should_exit = True
        if (
            self._thread is not None
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=max(0.0, float(timeout)))
        self._thread = None
        self._server = None

    def __enter__(self) -> "VideoSender":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
