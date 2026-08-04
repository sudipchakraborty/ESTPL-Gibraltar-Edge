"""Queue JSON alerts and deliver them to a remote Socket.IO server."""

from __future__ import annotations

import os
import json
import queue
import threading
import time
from collections.abc import Mapping
from typing import Any

import socketio
from dotenv import load_dotenv


class AlertSender:
    """Send JSON-compatible objects without blocking the camera loop."""

    def __init__(
        self,
        server_url: str,
        *,
        event_name: str = "inspection_status",
        auth_token: str | None = None,
        reconnect_delay: float = 2.0,
        queue_size: int = 100,
        auto_start: bool = True,
    ) -> None:
        if not isinstance(server_url, str) or not server_url.strip():
            raise ValueError("A non-empty remote server URL is required.")
        if not isinstance(event_name, str) or not event_name.strip():
            raise ValueError("A non-empty Socket.IO event name is required.")

        self.server_url = server_url.strip()
        self.event_name = event_name.strip()
        self.auth_token = auth_token.strip() if auth_token else None
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._socket = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=self.reconnect_delay,
        )
        self._register_events()

        if auto_start:
            self.start()

    @classmethod
    def from_env(
        cls,
        variable_name: str = "SOCKET_SERVER_URL",
        **kwargs: Any,
    ) -> "AlertSender":
        """Create a sender using a URL stored in the project .env file."""
        load_dotenv()
        server_url = os.getenv(variable_name, "").strip()
        if not server_url:
            raise ValueError(f"{variable_name} is missing from the .env file.")
        auth_token = os.getenv("SOCKET_AUTH_TOKEN", "").strip() or None
        return cls(server_url, auth_token=auth_token, **kwargs)

    @property
    def connected(self) -> bool:
        return bool(self._socket.connected)

    def _register_events(self) -> None:
        @self._socket.event
        def connect() -> None:
            print("[ALERT SENDER] Connected to remote server")

        @self._socket.event
        def disconnect() -> None:
            print("[ALERT SENDER] Disconnected from remote server")

        @self._socket.event
        def connect_error(error: Any) -> None:
            print("[ALERT SENDER] Connection error:", error)

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="RemoteAlertSender",
            daemon=True,
        )
        self._worker.start()

    def send(self, payload: Mapping[str, Any]) -> bool:
        """Queue one JSON-compatible mapping and return immediately."""
        if not isinstance(payload, Mapping):
            raise TypeError("Alert payload must be a mapping/JSON object.")

        alert = dict(payload)
        try:
            json.dumps(alert)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Alert payload contains values that are not JSON-compatible."
            ) from error
        try:
            self._queue.put_nowait(alert)
            return True
        except queue.Full:
            # Preserve recent industrial events instead of allowing an old
            # backlog to grow while the remote server is unavailable.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(alert)
                print("[ALERT SENDER] Queue full; oldest alert discarded")
                return True
            except queue.Full:
                return False

    def _connect(self) -> bool:
        if self._socket.connected:
            return True
        try:
            self._socket.connect(
                self.server_url,
                auth=(
                    {"token": self.auth_token}
                    if self.auth_token
                    else None
                ),
                transports=["websocket", "polling"],
                wait_timeout=5,
            )
        except Exception as error:
            print("[ALERT SENDER] Unable to connect:", error)
        return bool(self._socket.connected)

    def _worker_loop(self) -> None:
        pending: dict[str, Any] | None = None
        while not self._stop_event.is_set():
            if not self._connect():
                self._stop_event.wait(self.reconnect_delay)
                continue

            if pending is None:
                try:
                    pending = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue

            try:
                self._socket.emit(self.event_name, pending)
                print("[ALERT SENDER] JSON alert sent")
                self._queue.task_done()
                pending = None
            except Exception as error:
                print("[ALERT SENDER] Send failed; will retry:", error)
                if self._socket.connected:
                    self._socket.disconnect()
                self._stop_event.wait(self.reconnect_delay)

    def close(self, timeout: float = 3.0) -> None:
        """Stop the background worker and close the remote connection."""
        self._stop_event.set()
        if (
            self._worker is not None
            and self._worker is not threading.current_thread()
        ):
            self._worker.join(timeout=max(0.0, float(timeout)))
        self._worker = None
        if self._socket.connected:
            self._socket.disconnect()

    def __enter__(self) -> "AlertSender":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
