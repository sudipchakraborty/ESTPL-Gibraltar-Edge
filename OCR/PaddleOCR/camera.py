from __future__ import annotations

from .config import OCRConfig
from .drawing import draw_result
from .engine import OCR


def run_camera(camera_index: int = 0, config: OCRConfig | None = None) -> None:
    """Run this module as a complete camera OCR application."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Camera mode requires opencv-python") from exc

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera {camera_index}")

    try:
        with OCR(config) as reader:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                result = reader.read(frame)
                output = draw_result(frame, result, copy=False)
                cv2.putText(
                    output,
                    f"OCR: {result.inference_time_ms:.0f} ms | Q/Esc: quit",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("PaddleOCR Camera", output)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
