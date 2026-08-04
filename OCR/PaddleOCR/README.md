# Reusable camera OCR module

This folder is self-contained and may be copied into another Python project.
It keeps the fast settings from the working application: one reusable
PaddleOCR pipeline and no document-orientation, unwarping, or text-line
orientation stages. It explicitly selects `PP-OCRv5_mobile_det` and
`en_PP-OCRv5_mobile_rec`; this is important because PaddleOCR 3.7 otherwise
defaults to the larger PP-OCRv6 medium models.

## Install

Use Python 3.11. For the same CUDA setup as this project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r ocr\requirements-gpu.txt
```

For a machine without a compatible NVIDIA GPU, install
`ocr/requirements-cpu.txt` and create the reader with
`OCR(OCRConfig(device="cpu"))`.

The default recognition model is English. For another language, provide the
matching PaddleOCR mobile recognition model name in `OCRConfig`. The
`language` setting is used when both model-name fields are set to `None` and
PaddleOCR is allowed to choose the models.

PaddleOCR downloads its mobile OCR models on first use. Later runs reuse the
downloaded cache. To deploy without downloads, set `detection_model_dir` and
`recognition_model_dir` in `OCRConfig` to copied local model directories.

## Import from another project

Keep the `ocr` folder at the project root:

```python
import cv2
from ocr import OCR

camera = cv2.VideoCapture(0)
with OCR() as reader:                 # load once, outside the loop
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        result = reader.read(frame)   # OCRResult
        print(result.text)
```

`result.words` contains each text line, confidence, and bounding box.
`reader.infer(frame)` is an alias for `reader.read(frame)`. The engine is
locked during inference, so one instance can safely be shared by threads
(calls are serialized).

## Standalone use

```powershell
.\.venv\Scripts\python -m ocr --camera 0
.\.venv\Scripts\python -m ocr --image samples\test.jpg
.\.venv\Scripts\python -m ocr --camera 0 --device cpu
```

Press Q or Escape to close camera mode.
