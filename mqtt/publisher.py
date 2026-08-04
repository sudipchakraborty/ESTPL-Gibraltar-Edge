"""Reusable MQTT JSON alert publisher."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Mapping
from typing import Any

import paho.mqtt.client as paho


class MqttAlertPublisher:
    """Publish JSON alerts without blocking the camera-processing loop."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        topic: str,
        username: str | None = None,
        password: str | None = None,
        qos: int = 1,
        keepalive: int = 60,
        use_tls: bool = False,
        client_id: str | None = None,
    ) -> None:
        self.host = self._required("host", host)
        self.port = int(port)
        self.topic = self._required("topic", topic)
        self.qos = max(0, min(2, int(qos)))
        self.keepalive = max(10, int(keepalive))
        self.connected = threading.Event()

        generated_id = client_id or f"visualai-edge-{uuid.uuid4().hex[:10]}"
        self.client = paho.Client(
            callback_api_version=paho.CallbackAPIVersion.VERSION2,
            client_id=generated_id,
            protocol=paho.MQTTv311,
        )
        if username:
            self.client.username_pw_set(username, password)
        if use_tls:
            self.client.tls_set()

        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.max_queued_messages_set(100)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    @staticmethod
    def _required(name: str, value: Any) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"MQTT {name} is required.")
        return normalized

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MqttAlertPublisher":
        """Build a publisher while resolving credentials from environment."""
        username_env = str(config.get("username_env", "MQTT_USERNAME"))
        password_env = str(config.get("password_env", "MQTT_PASSWORD"))
        username = os.getenv(username_env)
        password = os.getenv(password_env)
        if not username:
            raise ValueError(f"MQTT username is missing: set {username_env}.")
        if not password:
            raise ValueError(f"MQTT password is missing: set {password_env}.")

        return cls(
            host=str(config["host"]),
            port=int(config.get("port", 1883)),
            topic=str(config["topic"]),
            username=username,
            password=password,
            qos=int(config.get("qos", 1)),
            keepalive=int(config.get("keepalive", 60)),
            use_tls=bool(config.get("use_tls", False)),
            client_id=config.get("client_id"),
        )

    def _on_connect(
        self,
        _client: paho.Client,
        _userdata: Any,
        _flags: paho.ConnectFlags,
        reason_code: paho.ReasonCode,
        _properties: paho.Properties | None,
    ) -> None:
        if reason_code == 0:
            self.connected.set()
            print(
                f"[MQTT] Connected to {self.host}:{self.port}; "
                f"topic={self.topic}",
                flush=True,
            )
        else:
            self.connected.clear()
            print(f"[MQTT] Connection rejected: {reason_code}", flush=True)

    def _on_disconnect(
        self,
        _client: paho.Client,
        _userdata: Any,
        _disconnect_flags: paho.DisconnectFlags,
        reason_code: paho.ReasonCode,
        _properties: paho.Properties | None,
    ) -> None:
        self.connected.clear()
        if reason_code != 0:
            print(f"[MQTT] Disconnected unexpectedly: {reason_code}", flush=True)

    def start(self) -> None:
        """Connect asynchronously and start automatic reconnect handling."""
        self.client.connect_async(self.host, self.port, self.keepalive)
        self.client.loop_start()

    def send(self, payload: Mapping[str, Any]) -> bool:
        """Queue a JSON alert for MQTT delivery."""
        message = json.dumps(dict(payload), ensure_ascii=False)
        result = self.client.publish(
            self.topic,
            payload=message,
            qos=self.qos,
            retain=False,
        )
        accepted = result.rc == paho.MQTT_ERR_SUCCESS
        if not accepted:
            print(f"[MQTT] Alert queue failed: rc={result.rc}", flush=True)
        return accepted

    def close(self) -> None:
        """Disconnect and stop the MQTT network thread."""
        self.client.disconnect()
        self.client.loop_stop()
        self.connected.clear()

    def __enter__(self) -> "MqttAlertPublisher":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
