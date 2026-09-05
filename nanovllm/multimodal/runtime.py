from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class VisionInput:
    """Processor output needed to encode one visual asset."""

    feature_key: str
    pixel_values: torch.Tensor
    grid_thw: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class VisionFeatureBundle:
    """Final and optional intermediate embeddings for one visual asset."""

    final_embedding: torch.Tensor
    deepstack_embeddings: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.final_embedding, torch.Tensor):
            raise TypeError("final vision embedding must be a tensor")
        if self.final_embedding.ndim != 2:
            raise ValueError("final vision embedding must be rank-2")
        if not isinstance(self.deepstack_embeddings, tuple):
            raise TypeError("deepstack vision embeddings must be a tuple")
        expected_shape = self.final_embedding.shape
        for embedding in self.deepstack_embeddings:
            if not isinstance(embedding, torch.Tensor):
                raise TypeError("deepstack vision embeddings must be tensors")
            if embedding.ndim != 2:
                raise ValueError("deepstack vision embeddings must be rank-2")
            if embedding.shape != expected_shape:
                raise ValueError(
                    "all vision embeddings must have the same token and hidden dimensions"
                )

    @property
    def token_count(self) -> int:
        return self.final_embedding.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.final_embedding.shape[1]

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.final_embedding, *self.deepstack_embeddings)
        )

    def detach(self) -> "VisionFeatureBundle":
        return VisionFeatureBundle(
            self.final_embedding.detach(),
            tuple(tensor.detach() for tensor in self.deepstack_embeddings),
        )

    def contiguous(self) -> "VisionFeatureBundle":
        return VisionFeatureBundle(
            self.final_embedding.contiguous(),
            tuple(tensor.contiguous() for tensor in self.deepstack_embeddings),
        )

    def to(self, *args, **kwargs) -> "VisionFeatureBundle":
        return VisionFeatureBundle(
            self.final_embedding.to(*args, **kwargs),
            tuple(tensor.to(*args, **kwargs) for tensor in self.deepstack_embeddings),
        )


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


@dataclass(frozen=True, slots=True)
class MaterializedMultimodalPrompt:
    """Tokenized prompt plus model-independent visual encoder inputs."""

    token_ids: tuple[int, ...]
    metadata: MultimodalPromptMetadata
    vision_inputs: tuple[VisionInput, ...]
