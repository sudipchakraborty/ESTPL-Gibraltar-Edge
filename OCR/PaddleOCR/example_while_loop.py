"""Minimal integration example for another project."""

import cv2

from ocr import OCR
from ocr.drawing import draw_result


capture = cv2.VideoCapture(0)

try:
    # Model initialization happens once, before the loop.
    with OCR() as reader:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            result = reader.read(frame)
            print(result.text)
            cv2.imshow("OCR", draw_result(frame, result))

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
finally:
    capture.release()
    cv2.destroyAllWindows()
