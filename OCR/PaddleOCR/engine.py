from __future__ import annotations

import threading
import time
import sys
import types
from typing import Any

import numpy as np

from .config import OCRConfig
from .result import OCRResult, OCRWord


class OCR:
    """Initialize PaddleOCR once and reuse it for every camera frame."""

    def __init__(self, config: OCRConfig | None = None, *, auto_initialize: bool = True):
        self.config = config or OCRConfig()
        self._pipeline: Any = None
        self._lock = threading.RLock()
        if auto_initialize:
            self.initialize()

    @property
    def initialized(self) -> bool:
        return self._pipeline is not None

    def initialize(self) -> "OCR":
        with self._lock:
            if self._pipeline is not None:
                return self

            # PaddleX imports ModelScope even when its model host is not used.
            # On Windows, ModelScope eagerly imports CUDA PyTorch; loading that
            # beside a different CUDA build of Paddle causes a cuDNN DLL
            # collision. Give PaddleX the one API it references while it is
            # imported, then restore normal import behavior for other projects.
            modelscope_module = sys.modules.get("modelscope")
            modelscope_stubbed = sys.platform == "win32" and modelscope_module is None
            if modelscope_stubbed:
                modelscope_stub = types.ModuleType("modelscope")

                def unavailable_modelscope_download(*_args: Any, **_kwargs: Any) -> None:
                    raise RuntimeError(
                        "ModelScope downloads are disabled for PaddleOCR on Windows; "
                        "use cached models or another PaddleX model source."
                    )

                modelscope_stub.snapshot_download = unavailable_modelscope_download
                sys.modules["modelscope"] = modelscope_stub
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed. Install ocr/requirements-gpu.txt "
                    "or ocr/requirements-cpu.txt."
                ) from exc
            finally:
                if modelscope_stubbed:
                    sys.modules.pop("modelscope", None)

            options: dict[str, Any] = {
                "device": self.config.device,
                # Pin the intended small models; PaddleOCR 3.7 otherwise
                # silently defaults to the larger PP-OCRv6 medium pair.
                "text_detection_model_name": self.config.detection_model_name,
                "text_recognition_model_name": self.config.recognition_model_name,
                # These disabled document stages are essential for camera speed.
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "text_det_thresh": self.config.detection_threshold,
                "text_rec_score_thresh": self.config.recognition_threshold,
            }
            if (
                self.config.detection_model_name is None
                and self.config.recognition_model_name is None
                and self.config.detection_model_dir is None
                and self.config.recognition_model_dir is None
            ):
                options["lang"] = self.config.language
            if self.config.detection_limit_side_len is not None:
                options["text_det_limit_side_len"] = self.config.detection_limit_side_len
            if self.config.recognition_batch_size is not None:
                options["text_recognition_batch_size"] = self.config.recognition_batch_size
            if self.config.detection_model_dir:
                options["text_detection_model_dir"] = self.config.detection_model_dir
            if self.config.recognition_model_dir:
                options["text_recognition_model_dir"] = self.config.recognition_model_dir

            self._pipeline = PaddleOCR(**options)
            return self

    def read(self, frame: np.ndarray) -> OCRResult:
        """Recognize text in one OpenCV BGR/gray frame."""
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise TypeError("frame must be a 2D or 3D NumPy/OpenCV image")
        if frame.size == 0:
            raise ValueError("frame is empty")

        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("OCR is closed; call initialize() before read()")
            started = time.perf_counter()
            pages = self._pipeline.predict(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        words: list[OCRWord] = []
        if pages:
            page = pages[0]
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            boxes = page.get("rec_boxes", [])
            for text, score, box in zip(texts, scores, boxes):
                normalized_box = box.tolist() if hasattr(box, "tolist") else list(box)
                words.append(
                    OCRWord(
                        text=str(text),
                        confidence=float(score),
                        box=normalized_box,
                    )
                )

        return OCRResult(
            words=words,
            inference_time_ms=elapsed_ms,
            image_width=int(frame.shape[1]),
            image_height=int(frame.shape[0]),
        )

    infer = read

    def close(self) -> None:
        with self._lock:
            self._pipeline = None

    shutdown = close

    def __enter__(self) -> "OCR":
        return self.initialize()

    def __exit__(self, *_: object) -> None:
        self.close()
