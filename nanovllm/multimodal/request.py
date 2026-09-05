from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from nanovllm.multimodal.schemas import MultimodalRequest


@dataclass(slots=True)
class MultimodalGenerationRequest:
    request: MultimodalRequest
    max_new_tokens: int
    temperature: float
    top_p: float = 1.0
    ignore_eos: bool = False
    arrival_time: float = field(default_factory=monotonic)
    internal_id: str = field(default_factory=lambda: f"mm-internal-{uuid4().hex}", init=False)

    def __post_init__(self) -> None:
        if not self.request.request_id:
            self.request.request_id = f"mm-{uuid4().hex}"
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def request_id(self) -> str:
        return self.request.request_id or ""

    @property
    def image_count(self) -> int:
        return len(self.request.images)

    @property
    def estimated_text_tokens(self) -> int:
        # A tokenizer-independent admission estimate. The model processor remains
        # the source of truth when the batch is materialized.
        return max(1, (len(self.request.text) + 3) // 4)

    @property
    def generation_key(self) -> tuple[int, float, float, bool]:
        return (
            self.max_new_tokens,
            round(self.temperature, 6),
            round(self.top_p, 6),
            self.ignore_eos,
        )
