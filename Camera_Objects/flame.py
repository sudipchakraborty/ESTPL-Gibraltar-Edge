"""Flame monitoring for fixed-camera video streams."""

from __future__ import annotations

import time

import cv2
import numpy as np


class FlameMonitor:
    """Detect small flickering orange or blue flames and raise an alarm."""

    def __init__(self, config: dict) -> None:
        self.minimum_area = float(config.get("minimum_area", 20))
        self.maximum_area = float(config.get("maximum_area", 5000))
        self.motion_threshold = int(config.get("motion_threshold", 12))
        self.minimum_frames = int(config.get("minimum_consecutive_frames", 3))
        self.cooldown = float(config.get("alarm_cooldown_seconds", 10))
        self.alarm_duration = float(config.get("alarm_duration_seconds", 5))

        self.previous_gray = None
        self.consecutive_detections = 0
        self.last_alarm_time = 0.0
        self.alarm_until = 0.0

    @staticmethod
    def _flame_colour_mask(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow/orange flame.
        warm_mask = cv2.inRange(
            hsv,
            np.array([0, 120, 180], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )

        # Blue flame commonly produced by gas lighters.
        blue_mask = cv2.inRange(
            hsv,
            np.array([90, 100, 140], dtype=np.uint8),
            np.array([135, 255, 255], dtype=np.uint8),
        )

        colour_mask = cv2.bitwise_or(warm_mask, blue_mask)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN, kernel)

    def _candidate_boxes(self, frame) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return []

        difference = cv2.absdiff(self.previous_gray, gray)
        self.previous_gray = gray
        _, motion_mask = cv2.threshold(
            difference,
            self.motion_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        motion_mask = cv2.dilate(
            motion_mask,
            np.ones((7, 7), dtype=np.uint8),
            iterations=2,
        )

        candidates = cv2.bitwise_and(
            self._flame_colour_mask(frame),
            motion_mask,
        )
        contours, _ = cv2.findContours(
            candidates,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.minimum_area <= area <= self.maximum_area:
                boxes.append(cv2.boundingRect(contour))
        return boxes

    def process(self, frame):
        """Process one frame and return it with flame alarm overlays."""
        boxes = self._candidate_boxes(frame)
        now = time.monotonic()

        if boxes:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        if (
            self.consecutive_detections >= self.minimum_frames
            and now - self.last_alarm_time >= self.cooldown
        ):
            self.last_alarm_time = now
            self.alarm_until = now + self.alarm_duration
            self.on_flame_detected()

        for x, y, width, height in boxes:
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Possible flame",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if now < self.alarm_until:
            cv2.rectangle(
                frame,
                (0, 60),
                (frame.shape[1] - 1, 115),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                "ALARM: FLAME DETECTED",
                (15, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    @staticmethod
    def on_flame_detected() -> None:
        """Terminal alarm hook; MQTT publishing can be added here later."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n!!! [FLAME ALARM] Flame detected at {timestamp} !!!\n")
