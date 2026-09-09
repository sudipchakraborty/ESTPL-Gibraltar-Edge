"""Helpers for attaching camera evidence to inspection events."""

import base64


def jpeg_evidence(image_data: bytes | None) -> dict | None:
    """Return a transport-safe JPEG payload, or None when no frame exists."""
    if not image_data:
        return None
    return {
        "content_type": "image/jpeg",
        "data": base64.b64encode(image_data).decode("ascii"),
    }
