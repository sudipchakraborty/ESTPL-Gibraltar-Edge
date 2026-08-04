"""Reusable, camera-friendly PaddleOCR module."""

from .config import OCRConfig
from .engine import OCR
from .result import OCRResult, OCRWord

__all__ = ["OCR", "OCRConfig", "OCRResult", "OCRWord"]
__version__ = "1.0.0"
