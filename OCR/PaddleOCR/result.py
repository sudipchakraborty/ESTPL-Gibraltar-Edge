from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class OCRWord:
    text: str
    confidence: float
    box: list[Any]


@dataclass(frozen=True, slots=True)
class OCRResult:
    words: list[OCRWord] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0

    @property
    def text(self) -> str:
        """All recognized lines joined with a newline."""
        return "\n".join(word.text for word in self.words)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "inference_time_ms": self.inference_time_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "words": [
                {
                    "text": word.text,
                    "confidence": word.confidence,
                    "box": word.box,
                }
                for word in self.words
            ],
        }
