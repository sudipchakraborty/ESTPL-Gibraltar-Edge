"""Bathroom entrance monitoring using person detection and tracking."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO


class BathroomEntryMonitor:
    """Detect and track people crossing into a bathroom entrance."""

    def __init__(self, config: dict, repository_dir: Path) -> None:
        model_path = Path(config["model_path"])
        if not model_path.is_absolute():
            model_path = repository_dir / model_path

        self.model = YOLO(str(model_path))
        self.device = config.get("device", "cpu")
        self.confidence = float(config.get("confidence_threshold", 0.3))
        self.reference_width = int(config["reference_frame_width"])
        self.reference_height = int(config["reference_frame_height"])
        self.reference_zone = tuple(map(int, config["entrance_zone"]))
        self.reference_line_y = int(config["crossing_line_y"])
        self.entry_direction = config.get("entry_direction", "up")
        self.maximum_distance = float(config.get("maximum_tracking_distance", 80))
        self.maximum_missed = int(config.get("maximum_missed_frames", 12))
        self.alarm_duration = float(config.get("alarm_duration_seconds", 5))
        self.cooldown = float(config.get("alarm_cooldown_seconds", 10))

        self.tracks: dict[int, dict] = {}
        self.next_track_id = 1
        self.last_alarm_time = 0.0
        self.alarm_until = 0.0

    def _scaled_geometry(self, frame) -> tuple[tuple[int, int, int, int], int]:
        height, width = frame.shape[:2]
        scale_x = width / self.reference_width
        scale_y = height / self.reference_height
        x1, y1, x2, y2 = self.reference_zone
        zone = (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        return zone, round(self.reference_line_y * scale_y)

    def _detect_people(self, frame) -> list[dict]:
        result = self.model.predict(
            frame,
            classes=[0],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "point": ((x1 + x2) // 2, y2),
                    "confidence": float(box.conf.item()),
                }
            )
        return detections

    def _update_tracks(self, detections: list[dict]) -> list[dict]:
        unmatched_tracks = set(self.tracks)
        assignments = []

        for detection in detections:
            point_x, point_y = detection["point"]
            best_id = None
            best_distance = self.maximum_distance

            for track_id in unmatched_tracks:
                old_x, old_y = self.tracks[track_id]["point"]
                distance = ((point_x - old_x) ** 2 + (point_y - old_y) ** 2) ** 0.5
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1
                previous_point = detection["point"]
                self.tracks[best_id] = {
                    "point": detection["point"],
                    "missed": 0,
                    "alarm_triggered": False,
                }
            else:
                previous_point = self.tracks[best_id]["point"]
                unmatched_tracks.remove(best_id)

            track = self.tracks[best_id]
            track["previous_point"] = previous_point
            track["point"] = detection["point"]
            track["bbox"] = detection["bbox"]
            track["confidence"] = detection["confidence"]
            track["missed"] = 0
            track["id"] = best_id
            assignments.append(track)

        for track_id in unmatched_tracks:
            self.tracks[track_id]["missed"] += 1
            if self.tracks[track_id]["missed"] > self.maximum_missed:
                del self.tracks[track_id]

        return assignments

    def _entered(
        self,
        track: dict,
        zone: tuple[int, int, int, int],
        line_y: int,
    ) -> bool:
        x1, _y1, x2, _y2 = zone
        previous_x, previous_y = track["previous_point"]
        current_x, current_y = track["point"]
        within_entrance = x1 <= current_x <= x2 or x1 <= previous_x <= x2

        if self.entry_direction == "up":
            crossed = previous_y > line_y >= current_y
        else:
            crossed = previous_y < line_y <= current_y

        return within_entrance and crossed and not track["alarm_triggered"]

    def process(self, frame):
        """Process one frame and return it with monitoring overlays."""
        zone, line_y = self._scaled_geometry(frame)
        tracks = self._update_tracks(self._detect_people(frame))
        now = time.monotonic()

        for track in tracks:
            if self._entered(track, zone, line_y):
                track["alarm_triggered"] = True
                if now - self.last_alarm_time >= self.cooldown:
                    self.last_alarm_time = now
                    self.alarm_until = now + self.alarm_duration
                    self.on_bathroom_entry(track["id"])

            x1, y1, x2, y2 = track["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {track['id']} {track['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        zone_x1, zone_y1, zone_x2, zone_y2 = zone
        cv2.rectangle(
            frame,
            (zone_x1, zone_y1),
            (zone_x2, zone_y2),
            (0, 255, 255),
            2,
        )
        cv2.line(
            frame,
            (zone_x1, line_y),
            (zone_x2, line_y),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Bathroom entrance",
            (zone_x1, max(20, zone_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if now < self.alarm_until:
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1] - 1, 55),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                "ALARM: PERSON ENTERED BATHROOM",
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    @staticmethod
    def on_bathroom_entry(track_id: int) -> None:
        """Alarm hook: MQTT publishing can be added here later."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[BATHROOM ENTRY ALARM] time={timestamp} track_id={track_id}")
