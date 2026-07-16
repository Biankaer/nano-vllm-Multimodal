from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from nanovllm.multimodal.request import MultimodalGenerationRequest
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy


@dataclass(slots=True)
class MultimodalBatch:
    requests: list[MultimodalGenerationRequest]
    text_tokens: int
    image_tokens: int
    image_count: int


class MultimodalBatcher:
    """FIFO scheduler with compatibility and text/image token budgets."""

    def __init__(self, policy: MultimodalBatchPolicy):
        self.policy = policy
        self._waiting: deque[MultimodalGenerationRequest] = deque()

    def add(self, request: MultimodalGenerationRequest) -> None:
        image_tokens = self._estimate_image_tokens(request)
        if request.image_count > self.policy.max_images_per_batch:
            raise ValueError("request image count exceeds max_images_per_batch")
        if request.estimated_text_tokens > self.policy.max_text_tokens_per_batch:
            raise ValueError("request text estimate exceeds max_text_tokens_per_batch")
        if image_tokens > self.policy.max_image_tokens_per_batch:
            raise ValueError("request image token estimate exceeds max_image_tokens_per_batch")
        self._waiting.append(request)

    def pop_batch(self) -> MultimodalBatch:
        if not self._waiting:
            raise IndexError("cannot schedule an empty multimodal queue")

        first = self._waiting.popleft()
        selected = [first]
        text_tokens = first.estimated_text_tokens
        image_tokens = self._estimate_image_tokens(first)
        image_count = first.image_count
        deferred: deque[MultimodalGenerationRequest] = deque()

        while self._waiting and len(selected) < self.policy.max_batch_size:
            candidate = self._waiting.popleft()
            candidate_text_tokens = candidate.estimated_text_tokens
            candidate_image_tokens = self._estimate_image_tokens(candidate)
            if self._is_compatible(first, candidate) and self._fits(
                text_tokens + candidate_text_tokens,
                image_tokens + candidate_image_tokens,
                image_count + candidate.image_count,
            ):
                selected.append(candidate)
                text_tokens += candidate_text_tokens
                image_tokens += candidate_image_tokens
                image_count += candidate.image_count
            else:
                deferred.append(candidate)

        deferred.extend(self._waiting)
        self._waiting = deferred
        return MultimodalBatch(selected, text_tokens, image_tokens, image_count)

    def _is_compatible(
        self,
        first: MultimodalGenerationRequest,
        candidate: MultimodalGenerationRequest,
    ) -> bool:
        if first.generation_key != candidate.generation_key:
            return False
        if not self.policy.enable_image_bucketization:
            return True
        return self._image_bucket(first) == self._image_bucket(candidate)

    def _fits(self, text_tokens: int, image_tokens: int, image_count: int) -> bool:
        return (
            text_tokens <= self.policy.max_text_tokens_per_batch
            and image_tokens <= self.policy.max_image_tokens_per_batch
            and image_count <= self.policy.max_images_per_batch
        )

    def _estimate_image_tokens(self, request: MultimodalGenerationRequest) -> int:
        return sum(self.policy.estimate_image_tokens(image.detail) for image in request.request.images)

    @staticmethod
    def _image_bucket(request: MultimodalGenerationRequest) -> tuple[int, tuple[str, ...]]:
        return request.image_count, tuple(image.detail for image in request.request.images)

    def __len__(self) -> int:
        return len(self._waiting)
