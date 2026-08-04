from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Settings for the reusable PaddleOCR pipeline."""

    language: str = "en"
    device: str = "gpu:0"
    detection_model_name: str | None = "PP-OCRv5_mobile_det"
    recognition_model_name: str | None = "en_PP-OCRv5_mobile_rec"
    detection_threshold: float = 0.3
    recognition_threshold: float = 0.5
    detection_limit_side_len: int | None = None
    recognition_batch_size: int | None = None
    detection_model_dir: str | None = None
    recognition_model_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must be 'cpu', 'gpu:0', or another Paddle device")
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be between 0 and 1")
        if not 0.0 <= self.recognition_threshold <= 1.0:
            raise ValueError("recognition_threshold must be between 0 and 1")
        if self.detection_limit_side_len is not None and self.detection_limit_side_len <= 0:
            raise ValueError("detection_limit_side_len must be positive")
        if self.recognition_batch_size is not None and self.recognition_batch_size <= 0:
            raise ValueError("recognition_batch_size must be positive")
