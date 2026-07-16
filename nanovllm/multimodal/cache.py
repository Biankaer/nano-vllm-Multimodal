from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ImageFeatureCache:
    """Small LRU cache for precomputed image features.

    This is intentionally model-agnostic. Later VLM integration can store tensors
    or CPU-side feature blobs without changing the serving API.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._items: OrderedDict[str, Any] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        value = self._items.get(key)
        if value is None:
            self.stats.misses += 1
            return None
        self._items.move_to_end(key)
        self.stats.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        if key in self._items:
            self._items[key] = value
            self._items.move_to_end(key)
            return
        self._items[key] = value
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
            self.stats.evictions += 1
