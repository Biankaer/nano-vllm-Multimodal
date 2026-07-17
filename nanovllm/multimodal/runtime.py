from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisualTokenSpan:
    """A contiguous run of prompt tokens backed by one vision feature tensor."""

    start: int
    length: int
    feature_key: str

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True, slots=True)
class MultimodalPromptMetadata:
    """Small, serializable metadata for a materialized multimodal prompt."""

    position_ids: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    rope_delta: int
    visual_spans: tuple[VisualTokenSpan, ...] = ()

    def validate(self, prompt_length: int) -> None:
        if prompt_length < 1:
            raise ValueError("prompt length must be positive")
        if len(self.position_ids) != 3:
            raise ValueError("Qwen2.5-VL position_ids must contain exactly three axes")
        if any(len(axis) != prompt_length for axis in self.position_ids):
            raise ValueError("position_ids axis length must match prompt length")

        previous_end = 0
        for index, span in enumerate(self.visual_spans):
            if span.start < 0:
                raise ValueError("visual span start cannot be negative")
            if span.length < 1:
                raise ValueError("visual span length must be positive")
            if not span.feature_key:
                raise ValueError("visual span feature_key cannot be empty")
            if span.end > prompt_length:
                raise ValueError("visual span exceeds prompt length")
            if index > 0 and span.start < previous_end:
                raise ValueError("visual spans cannot overlap")
            previous_end = span.end

    def visual_identity_at(self, token_index: int) -> str | None:
        if token_index < 0 or token_index >= len(self.position_ids[0]):
            raise IndexError("token index is outside the prompt")
        for span in self.visual_spans:
            if span.start <= token_index < span.end:
                return f"{span.feature_key}:{token_index - span.start}"
            if span.start > token_index:
                break
        return None
