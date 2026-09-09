"""Non-UI USB camera discovery reusable by desktop applications."""

from __future__ import annotations

from typing import Any

import cv2


def discover_usb_cameras(max_devices: int = 10) -> list[dict[str, Any]]:
    """Return OpenCV device indexes that can be opened on this computer."""
    devices: list[dict[str, Any]] = []
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    for device_index in range(max(0, max_devices)):
        capture = cv2.VideoCapture(device_index, backend)
        try:
            if capture.isOpened():
                devices.append(
                    {
                        "index": device_index,
                        "label": f"USB Camera {device_index}",
                    }
                )
        finally:
            capture.release()
    return devices
