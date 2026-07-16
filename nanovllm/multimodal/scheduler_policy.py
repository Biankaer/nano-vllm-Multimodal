from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MultimodalBatchPolicy:
    """Admission limits for request-level Qwen2.5-VL continuous batching."""

    max_batch_size: int = 8
    max_images_per_batch: int = 8
    max_text_tokens_per_batch: int = 16384
    max_image_tokens_per_batch: int = 8192
    batch_wait_ms: float = 5.0
    enable_image_bucketization: bool = True
    low_detail_image_tokens: int = 256
    auto_detail_image_tokens: int = 1024
    high_detail_image_tokens: int = 2048

    def __post_init__(self) -> None:
        if min(
            self.max_batch_size,
            self.max_images_per_batch,
            self.max_text_tokens_per_batch,
            self.max_image_tokens_per_batch,
        ) < 1:
            raise ValueError("multimodal batch limits must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms cannot be negative")

    def estimate_image_tokens(self, detail: str) -> int:
        return {
            "low": self.low_detail_image_tokens,
            "high": self.high_detail_image_tokens,
        }.get(detail, self.auto_detail_image_tokens)
