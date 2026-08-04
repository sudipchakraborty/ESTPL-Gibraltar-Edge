from __future__ import annotations

import numpy as np

from .result import OCRResult


def draw_result(frame: np.ndarray, result: OCRResult, *, copy: bool = True) -> np.ndarray:
    """Draw OCR boxes and labels on an OpenCV frame."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Drawing requires opencv-python") from exc

    output = frame.copy() if copy else frame
    for word in result.words:
        box = word.box
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            x1, y1, x2, y2 = (int(value) for value in box)
        else:
            points = [(int(point[0]), int(point[1])) for point in box]
            if not points:
                continue
            xs, ys = zip(*points)
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{word.text} ({word.confidence:.2f})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output
