"""YOLO smoke detection and alert generation."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


class SmokeDetectionMonitor:
    """Detect smoke and queue a debounced alert."""

    def __init__(self, config: dict, repository_dir: Path) -> None:
        model_path = Path(config["model_path"])
        if not model_path.is_absolute():
            model_path = repository_dir / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Smoke detection model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.device = config.get("device", "cpu")
        self.confidence = float(config.get("confidence_threshold", 0.3))
        self.smoke_classes = {
            str(name).strip().casefold()
            for name in config.get("smoke_class_names", ["smoke"])
        }
        self.minimum_frames = max(
            1, int(config.get("minimum_consecutive_frames", 3))
        )
        self.alert_cooldown = float(config.get("alert_cooldown_seconds", 10))
        self.absence_frames_to_reset = max(
            1, int(config.get("absence_frames_to_reset", 10))
        )

        available_names = {
            str(name).casefold() for name in self.model.names.values()
        }
        if not available_names.intersection(self.smoke_classes):
            raise ValueError(
                "The smoke model has none of the configured smoke classes. "
                f"Model classes: {sorted(available_names)}; "
                f"configured classes: {sorted(self.smoke_classes)}"
            )

        self._detection_frames = 0
        self._absence_frames = 0
        self._alert_active = False
        self._last_alert_time = 0.0
        self._pending_alert: dict[str, Any] | None = None

    def process(self, frame):
        """Annotate smoke and queue an alert after consecutive detections."""
        result = self.model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[dict[str, Any]] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(self.model.names[class_id])
            if class_name.casefold() not in self.smoke_classes:
                continue
            detection = {
                "bbox": tuple(map(int, box.xyxy[0].tolist())),
                "confidence": float(box.conf.item()),
                "class_name": class_name,
            }
            detections.append(detection)
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(
                frame,
                f"Smoke {detection['confidence']:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

        now = time.monotonic()
        if detections:
            self._absence_frames = 0
            self._detection_frames += 1
            if (
                self._detection_frames >= self.minimum_frames
                and not self._alert_active
                and now - self._last_alert_time >= self.alert_cooldown
            ):
                self._last_alert_time = now
                self._alert_active = True
                self._pending_alert = {
                    "smoke_count": len(detections),
                    "maximum_confidence": max(
                        detection["confidence"] for detection in detections
                    ),
                    "detected_at": datetime.now().astimezone().isoformat(),
                    "detections": detections,
                }
                print(
                    f"[SMOKE ALERT] Smoke detected: count={len(detections)}",
                    flush=True,
                )
        else:
            self._detection_frames = 0
            self._absence_frames += 1
            if self._absence_frames >= self.absence_frames_to_reset:
                self._alert_active = False

        return frame

    def pop_alert(self) -> dict[str, Any] | None:
        """Return and clear the latest queued smoke alert."""
        alert = self._pending_alert
        self._pending_alert = None
        return alert
