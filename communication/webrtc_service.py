"""Reusable threaded multi-camera WebRTC service."""

import threading
import uvicorn
from communication.webrtc_server import WebRTCServer


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def publish(self, frame):
        with self._lock:
            self._frame = frame

    def get_latest_frame(self):
        with self._lock:
            return self._frame


class MultiCameraWebRTCService:
    def __init__(self, camera_ids, host="0.0.0.0", port=8000, fps=10, target_width=960):
        self.providers = {camera_id: LatestFrame() for camera_id in camera_ids}
        self.web_server = WebRTCServer(self.providers, fps=fps, target_width=target_width)
        config = uvicorn.Config(self.web_server.app, host=host, port=int(port), log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.server.run, name="GibraltarWebRTC", daemon=True)
        self.thread.start()
        print(f"[WEBRTC] Multi-camera server starting on port {self.server.config.port}", flush=True)

    def publish(self, camera_id, frame):
        provider = self.providers.get(camera_id)
        if provider is not None:
            provider.publish(frame)

    def close(self, timeout=5.0):
        self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        self.thread = None
