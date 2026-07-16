from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nanovllm.multimodal.cache import ImageFeatureCache
from nanovllm.multimodal.image_loader import load_image
from nanovllm.multimodal.schemas import MultimodalRequest


class VisionEncoder(Protocol):
    def encode_images(self, images: list[object]) -> list[object]:
        ...


@dataclass(slots=True)
class ProcessedMultimodalRequest:
    text: str
    image_features: list[object]
    image_cache_hits: int
    image_cache_misses: int


class MultimodalProcessor:
    """Preprocess images and optionally reuse cached image features."""

    def __init__(self, vision_encoder: VisionEncoder, cache: ImageFeatureCache | None = None):
        self.vision_encoder = vision_encoder
        self.cache = cache or ImageFeatureCache()

    def process(self, request: MultimodalRequest) -> ProcessedMultimodalRequest:
        features: list[object] = []
        uncached_images: list[object] = []
        uncached_keys: list[str] = []
        hits = 0
        misses = 0

        for image_input in request.images:
            loaded = load_image(image_input)
            cached = self.cache.get(loaded.cache_key)
            if cached is not None:
                features.append(cached)
                hits += 1
                continue
            uncached_images.append(loaded.image)
            uncached_keys.append(loaded.cache_key)
            misses += 1

        if uncached_images:
            encoded = self.vision_encoder.encode_images(uncached_images)
            for key, feature in zip(uncached_keys, encoded):
                self.cache.put(key, feature)
                features.append(feature)

        return ProcessedMultimodalRequest(
            text=request.text,
            image_features=features,
            image_cache_hits=hits,
            image_cache_misses=misses,
        )
