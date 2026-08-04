"""Person-presence monitoring for a gate or perimeter camera."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


class PeoplePresenceMonitor:
    """Detect people, annotate the frame, and report night-time presence."""

    def __init__(self, config: dict, repository_dir: Path) -> None:
        model_path = Path(config["model_path"])
        if not model_path.is_absolute():
            model_path = repository_dir / model_path

        self.model = YOLO(str(model_path))
        self.device = config.get("device", "cpu")
        self.confidence = float(config.get("confidence_threshold", 0.3))
        self.night_start_hour = int(config.get("night_start_hour", 18))
        self.night_end_hour = int(config.get("night_end_hour", 6))
        self.alert_cooldown = float(config.get("alert_cooldown_seconds", 10))
        self.absence_frames_to_reset = int(
            config.get("absence_frames_to_reset", 10)
        )

        if not 0 <= self.night_start_hour <= 23:
            raise ValueError("night_start_hour must be between 0 and 23.")
        if not 0 <= self.night_end_hour <= 23:
            raise ValueError("night_end_hour must be between 0 and 23.")

        self._presence_active = False
        self._absence_frames = 0
        self._last_alert_time = 0.0
        self._pending_alert: dict[str, Any] | None = None

    def _is_night(self, current_time: datetime) -> bool:
        hour = current_time.hour
        if self.night_start_hour == self.night_end_hour:
            return True
        if self.night_start_hour < self.night_end_hour:
            return self.night_start_hour <= hour < self.night_end_hour
        return hour >= self.night_start_hour or hour < self.night_end_hour

    def _detect_people(self, frame) -> list[dict[str, Any]]:
        result = self.model.predict(
            frame,
            classes=[0],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            detections.append(
                {
                    "bbox": tuple(map(int, box.xyxy[0].tolist())),
                    "confidence": float(box.conf.item()),
                }
            )
        return detections

    def process(self, frame):
        """Annotate people and queue an alert when night presence begins."""
        detections = self._detect_people(frame)
        current_time = datetime.now().astimezone()
        night_active = self._is_night(current_time)
        now = time.monotonic()

        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            color = (0, 0, 255) if night_active else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Person {index} {detection['confidence']:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if detections and night_active:
            self._absence_frames = 0
            if (
                not self._presence_active
                and now - self._last_alert_time >= self.alert_cooldown
            ):
                self._last_alert_time = now
                self._pending_alert = {
                    "people_count": len(detections),
                    "maximum_confidence": max(
                        detection["confidence"] for detection in detections
                    ),
                    "detected_at": current_time.isoformat(),
                }
                print(
                    "[GATE ALERT] Person present during night hours: "
                    f"count={len(detections)}",
                    flush=True,
                )
            self._presence_active = True
        elif detections:
            self._absence_frames = 0
            self._presence_active = False
        else:
            self._absence_frames += 1
            if self._absence_frames >= self.absence_frames_to_reset:
                self._presence_active = False

        status_text = (
            f"NIGHT WATCH ACTIVE  |  PEOPLE: {len(detections)}"
            if night_active
            else f"DAY MONITORING  |  PEOPLE: {len(detections)}"
        )
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1] - 1, 45),
            (22, 22, 22),
            -1,
        )
        cv2.putText(
            frame,
            status_text,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 80, 255) if night_active and detections else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def pop_alert(self) -> dict[str, Any] | None:
        """Return and clear the latest queued presence alert."""
        alert = self._pending_alert
        self._pending_alert = None
        return alert
