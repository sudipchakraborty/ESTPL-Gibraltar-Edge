from __future__ import annotations

import argparse
import json

from .camera import run_camera
from .config import OCRConfig
from .engine import OCR


def main() -> int:
    parser = argparse.ArgumentParser(description="Reusable PaddleOCR camera module")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", help="read one image instead of opening a camera")
    parser.add_argument("--device", default="gpu:0", help="gpu:0 or cpu")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    config = OCRConfig(language=args.language, device=args.device)

    if not args.image:
        run_camera(args.camera, config)
        return 0

    import cv2

    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"could not read image: {args.image}")
    with OCR(config) as reader:
        print(json.dumps(reader.read(image).as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
