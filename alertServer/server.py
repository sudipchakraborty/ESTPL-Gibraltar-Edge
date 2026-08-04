"""Reusable Socket.IO server for receiving and broadcasting alert JSON."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import socketio
import uvicorn


async def _health_app(
    scope: dict[str, Any],
    receive: Any,
    send: Any,
) -> None:
    del receive
    if scope["type"] == "http" and scope.get("path") == "/health":
        body = b'{"status":"ok"}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})
        return
    await send(
        {
            "type": "http.response.start",
            "status": 404,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"detail":"Not Found"}'})


class AlertServer:
    """Receive ``inspection_status`` and broadcast ``alert_received``."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 5000,
        incoming_event: str = "inspection_status",
        outgoing_event: str = "alert_received",
        cors_allowed_origins: str | list[str] = "*",
        recent_alert_limit: int = 100,
        log_level: str = "warning",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.incoming_event = incoming_event
        self.outgoing_event = outgoing_event
        self.log_level = log_level
        self.recent_alerts: deque[dict[str, Any]] = deque(
            maxlen=max(1, int(recent_alert_limit))
        )
        self.socket_server = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors_allowed_origins,
        )
        self.app = socketio.ASGIApp(
            self.socket_server,
            other_asgi_app=_health_app,
        )
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._register_events()

    def _register_events(self) -> None:
        @self.socket_server.event
        async def connect(
            session_id: str,
            environ: dict[str, Any],
            auth: Any = None,
        ) -> None:
            del environ, auth
            print(f"[ALERT SERVER] Client connected: {session_id}", flush=True)

        @self.socket_server.event
        async def disconnect(session_id: str) -> None:
            print(f"[ALERT SERVER] Client disconnected: {session_id}", flush=True)

        async def receive_alert(
            session_id: str,
            data: Any,
        ) -> dict[str, Any]:
            received_at = datetime.now(timezone.utc).isoformat()
            if not isinstance(data, Mapping):
                return {
                    "received": False,
                    "error": "Alert must be a JSON object.",
                    "received_at": received_at,
                }

            alert = dict(data)
            try:
                json.dumps(alert)
            except (TypeError, ValueError):
                return {
                    "received": False,
                    "error": "Alert contains non-JSON-compatible values.",
                    "received_at": received_at,
                }

            envelope = {
                "received_at": received_at,
                "alert": alert,
            }
            self.recent_alerts.append(envelope)
            print(
                "[ALERT SERVER] Alert received:\n"
                + json.dumps(envelope, indent=2, ensure_ascii=False),
                flush=True,
            )
            await self.socket_server.emit(self.outgoing_event, envelope)
            return {
                "received": True,
                "received_at": received_at,
            }

        self.socket_server.on(self.incoming_event)(receive_alert)

    def _build_server(self) -> uvicorn.Server:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
        )
        return uvicorn.Server(config)

    def run(self) -> None:
        """Run the server in the current thread until stopped."""
        self._server = self._build_server()
        print(
            f"[ALERT SERVER] Listening on http://{self.host}:{self.port}",
            flush=True,
        )
        self._server.run()

    def start_background(self, startup_timeout: float = 5.0) -> None:
        """Start an embedded server without blocking the calling application."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.run,
            name="EmbeddedAlertServer",
            daemon=True,
        )
        self._thread.start()

        connect_host = (
            "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        )
        deadline = time.monotonic() + max(0.1, float(startup_timeout))
        while time.monotonic() < deadline:
            if self._thread is None or not self._thread.is_alive():
                raise RuntimeError("Alert server stopped during startup.")
            try:
                with socket.create_connection(
                    (connect_host, self.port),
                    timeout=0.2,
                ):
                    print("[ALERT SERVER] Ready", flush=True)
                    return
            except OSError:
                time.sleep(0.05)
        self.stop()
        raise TimeoutError(
            f"Alert server did not start on {connect_host}:{self.port}."
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Request shutdown and wait for a background server to exit."""
        if self._server is not None:
            self._server.should_exit = True
        if (
            self._thread is not None
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=max(0.0, float(timeout)))
        self._thread = None
        self._server = None

    def __enter__(self) -> "AlertServer":
        self.start_background()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
