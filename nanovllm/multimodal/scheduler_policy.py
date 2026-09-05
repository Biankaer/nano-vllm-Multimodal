from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MultimodalBatchPolicy:
    """Admission limits for request-level Qwen2.5-VL continuous batching."""

    max_batch_size: int = 8
    max_images_per_batch: int = 8
    max_text_tokens_per_batch: int = 16384
    max_image_tokens_per_batch: int = 8192
    # ``batch_wait_ms`` is the legacy fixed-window option. New callers should
    # use a short quiet window plus a bounded maximum wait.
    batch_wait_ms: float | None = None
    batch_quiet_ms: float | None = None
    batch_max_wait_ms: float | None = None
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
        waits = (self.batch_wait_ms, self.batch_quiet_ms, self.batch_max_wait_ms)
        if any(value is not None and value < 0 for value in waits):
            raise ValueError("multimodal batch waits cannot be negative")
        if self.effective_quiet_ms > self.effective_max_wait_ms:
            raise ValueError("batch_quiet_ms cannot exceed batch_max_wait_ms")

    @property
    def effective_quiet_ms(self) -> float:
        if self.batch_quiet_ms is not None:
            return self.batch_quiet_ms
        if self.batch_wait_ms is not None:
            return self.batch_wait_ms
        return 3.0

    @property
    def effective_max_wait_ms(self) -> float:
        if self.batch_max_wait_ms is not None:
            return self.batch_max_wait_ms
        if self.batch_wait_ms is not None:
            return self.batch_wait_ms
        return 20.0

    def estimate_image_tokens(self, detail: str) -> int:
        return {
            "low": self.low_detail_image_tokens,
            "high": self.high_detail_image_tokens,
        }.get(detail, self.auto_detail_image_tokens)
