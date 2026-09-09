"""Persistence and validation for per-camera audio-alert settings."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CAMERA_NAMES = ("camera-1", "camera-2", "camera-3")
EVENT_NAMES = ("matched", "mismatch")


def default_audio_alert_config() -> dict[str, Any]:
    settings: dict[str, Any] = {
        camera_name: {
            "enabled": False,
            "matched": {
                "enabled": True, "kind": "tone", "use_default": True, "path": ""
            },
            "mismatch": {
                "enabled": False, "kind": "tone", "use_default": True, "path": ""
            },
        }
        for camera_name in CAMERA_NAMES
    }
    settings["system_volume"] = 80
    return settings


def normalize_audio_alert_config(value: object) -> dict[str, Any]:
    normalized = default_audio_alert_config()
    if not isinstance(value, dict):
        return normalized
    try:
        normalized["system_volume"] = max(
            0,
            min(100, int(value.get("system_volume", 80))),
        )
    except (TypeError, ValueError):
        normalized["system_volume"] = 80
    for camera_name in CAMERA_NAMES:
        candidate = value.get(camera_name)
        if not isinstance(candidate, dict):
            continue
        camera = normalized[camera_name]
        camera["enabled"] = bool(candidate.get("enabled", camera["enabled"]))
        for event_name in EVENT_NAMES:
            event_candidate = candidate.get(event_name)
            if not isinstance(event_candidate, dict):
                continue
            event = camera[event_name]
            event["enabled"] = bool(event_candidate.get("enabled", event["enabled"]))
            kind = str(event_candidate.get("kind", event["kind"])).lower()
            event["kind"] = kind if kind in {"voice", "tone"} else "tone"
            event["use_default"] = bool(
                event_candidate.get("use_default", event["use_default"])
            )
            event["path"] = str(event_candidate.get("path", "")).strip()
    return normalized


class AudioAlertConfigStore:
    """Read and atomically update audio settings inside a project JSON file."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def load(self) -> dict[str, Any]:
        try:
            with self._config_path.open("r", encoding="utf-8") as config_file:
                project_config = json.load(config_file)
        except (OSError, json.JSONDecodeError, TypeError):
            return default_audio_alert_config()
        return normalize_audio_alert_config(project_config.get("audio_alerts"))

    def save(self, settings: object) -> dict[str, Any]:
        normalized = normalize_audio_alert_config(settings)
        try:
            with self._config_path.open("r", encoding="utf-8") as config_file:
                project_config = json.load(config_file)
        except (OSError, json.JSONDecodeError, TypeError):
            project_config = {}
        if not isinstance(project_config, dict):
            project_config = {}
        project_config["audio_alerts"] = deepcopy(normalized)
        temporary_path = self._config_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as config_file:
            json.dump(project_config, config_file, indent=4)
            config_file.write("\n")
        temporary_path.replace(self._config_path)
        return normalized
