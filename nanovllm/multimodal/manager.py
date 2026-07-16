from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path

from nanovllm.multimodal.batching import MultimodalBatch
from nanovllm.multimodal.cache import ImageAssetCache, ImageFeatureCache
from nanovllm.multimodal.image_loader import load_image
from nanovllm.multimodal.schemas import ImageInput


@dataclass(slots=True)
class MultimodalManagerStats:
    prepared_requests: int
    prepared_images: int
    image_asset_cache_entries: int
    image_asset_cache_hits: int
    image_asset_cache_misses: int
    image_asset_cache_evictions: int
    image_feature_cache_entries: int
    image_feature_cache_hits: int
    image_feature_cache_misses: int
    image_feature_cache_evictions: int


class MultimodalManager:
    """Own image materialization caches below the serving API layer.

    The asset cache is active in the HuggingFace bridge today. The feature cache
    is the engine-owned hook for the later split vision-encoder execution path.
    """

    def __init__(self, *, asset_cache_capacity: int = 256, feature_cache_capacity: int = 256):
        self.asset_cache = ImageAssetCache(asset_cache_capacity)
        self.feature_cache = ImageFeatureCache(feature_cache_capacity)
        self.prepared_requests = 0
        self.prepared_images = 0

    def prepare_batch(self, batch: MultimodalBatch) -> list[list[object]]:
        prepared: list[list[object]] = []
        for generation_request in batch.requests:
            request_images = [self._load_cached(image) for image in generation_request.request.images]
            prepared.append(request_images)
            self.prepared_requests += 1
            self.prepared_images += len(request_images)
        return prepared

    def get_image_features(self, cache_key: str) -> object | None:
        return self.feature_cache.get(cache_key)

    def put_image_features(self, cache_key: str, features: object) -> None:
        self.feature_cache.put(cache_key, features)

    def stats(self) -> MultimodalManagerStats:
        asset_stats = self.asset_cache.stats
        feature_stats = self.feature_cache.stats
        return MultimodalManagerStats(
            prepared_requests=self.prepared_requests,
            prepared_images=self.prepared_images,
            image_asset_cache_entries=len(self.asset_cache),
            image_asset_cache_hits=asset_stats.hits,
            image_asset_cache_misses=asset_stats.misses,
            image_asset_cache_evictions=asset_stats.evictions,
            image_feature_cache_entries=len(self.feature_cache),
            image_feature_cache_hits=feature_stats.hits,
            image_feature_cache_misses=feature_stats.misses,
            image_feature_cache_evictions=feature_stats.evictions,
        )

    def _load_cached(self, image_input: ImageInput) -> object:
        key = image_source_cache_key(image_input)
        cached = self.asset_cache.get(key)
        if cached is not None:
            return cached
        loaded = load_image(image_input)
        self.asset_cache.put(key, loaded.image)
        return loaded.image


def image_source_cache_key(image_input: ImageInput) -> str:
    if image_input.source_type == "base64":
        encoded = image_input.source.split(",", 1)[-1]
        digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
        return f"base64:{digest}"
    if image_input.source_type == "path":
        path = Path(image_input.source).expanduser().resolve()
        stat = path.stat()
        return f"path:{path}:{stat.st_mtime_ns}:{stat.st_size}"
    return f"url:{hashlib.sha256(image_input.source.encode('utf-8')).hexdigest()}"
