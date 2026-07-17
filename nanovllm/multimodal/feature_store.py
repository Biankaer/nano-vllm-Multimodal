from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import torch


@dataclass(frozen=True, slots=True)
class VisionFeatureStoreStats:
    entries: int
    resident_bytes: int
    byte_budget: int
    hits: int
    misses: int
    evictions: int


@dataclass(slots=True)
class _FeatureEntry:
    tensor: torch.Tensor
    nbytes: int
    pin_count: int = 0


class VisionFeatureStore:
    def __init__(self, byte_budget: int) -> None:
        if byte_budget < 1:
            raise ValueError("feature cache byte budget must be positive")
        self.byte_budget = byte_budget
        self.resident_bytes = 0
        self._entries: OrderedDict[str, _FeatureEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: str) -> torch.Tensor | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.tensor

    def peek(self, key: str) -> torch.Tensor | None:
        with self._lock:
            entry = self._entries.get(key)
            return entry.tensor if entry is not None else None

    def put(self, key: str, tensor: torch.Tensor) -> None:
        if not key:
            raise ValueError("feature cache key cannot be empty")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("feature cache values must be tensors")
        stored = tensor.detach().contiguous()
        nbytes = stored.numel() * stored.element_size()

        with self._lock:
            previous = self._entries.get(key)
            if previous is not None and previous.pin_count:
                raise RuntimeError(f"cannot replace pinned feature entry: {key}")
            if previous is not None:
                self.resident_bytes -= previous.nbytes
                del self._entries[key]

            required = self.resident_bytes + nbytes
            self._entries[key] = _FeatureEntry(stored, nbytes)
            self.resident_bytes = required
            self._evict_to_budget(protected_key=key)
            if self.resident_bytes > self.byte_budget:
                del self._entries[key]
                self.resident_bytes -= nbytes
                if previous is not None:
                    self._entries[key] = previous
                    self.resident_bytes += previous.nbytes
                raise MemoryError(
                    "vision feature cache budget exceeded: "
                    f"required={required}, budget={self.byte_budget}"
                )

    def put_many(self, tensors: dict[str, torch.Tensor]) -> None:
        prepared: OrderedDict[str, _FeatureEntry] = OrderedDict()
        for key, tensor in tensors.items():
            if not key:
                raise ValueError("feature cache key cannot be empty")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("feature cache values must be tensors")
            stored = tensor.detach().contiguous()
            prepared[key] = _FeatureEntry(
                tensor=stored,
                nbytes=stored.numel() * stored.element_size(),
            )

        working_set_bytes = sum(entry.nbytes for entry in prepared.values())
        if working_set_bytes > self.byte_budget:
            raise MemoryError(
                "vision feature cache budget exceeded: "
                f"required={working_set_bytes}, budget={self.byte_budget}"
            )

        with self._lock:
            for key in prepared:
                previous = self._entries.get(key)
                if previous is not None and previous.pin_count:
                    raise RuntimeError(f"cannot replace pinned feature entry: {key}")

            retained = OrderedDict(
                (key, entry)
                for key, entry in self._entries.items()
                if key not in prepared
            )
            resident_bytes = sum(entry.nbytes for entry in retained.values()) + working_set_bytes
            victims: list[str] = []
            while resident_bytes > self.byte_budget:
                victim_key = next(
                    (
                        key
                        for key, entry in retained.items()
                        if entry.pin_count == 0
                    ),
                    None,
                )
                if victim_key is None:
                    raise MemoryError(
                        "vision feature cache budget exceeded: "
                        f"required={resident_bytes}, budget={self.byte_budget}"
                    )
                resident_bytes -= retained[victim_key].nbytes
                del retained[victim_key]
                victims.append(victim_key)

            retained.update(prepared)
            self._entries = retained
            self.resident_bytes = resident_bytes
            self._evictions += len(victims)

    @contextmanager
    def pin(self, keys: tuple[str, ...]) -> Iterator[dict[str, torch.Tensor]]:
        unique_keys = tuple(dict.fromkeys(keys))
        with self._lock:
            missing = [key for key in unique_keys if key not in self._entries]
            if missing:
                raise KeyError(f"cannot pin missing feature keys: {missing}")
            for key in unique_keys:
                self._entries[key].pin_count += 1
                self._entries.move_to_end(key)
            tensors = {key: self._entries[key].tensor for key in unique_keys}
        try:
            yield tensors
        finally:
            with self._lock:
                for key in unique_keys:
                    entry = self._entries.get(key)
                    if entry is None or entry.pin_count < 1:
                        raise RuntimeError(f"invalid feature pin state for {key}")
                    entry.pin_count -= 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.resident_bytes = 0

    def stats(self) -> VisionFeatureStoreStats:
        with self._lock:
            return VisionFeatureStoreStats(
                entries=len(self._entries),
                resident_bytes=self.resident_bytes,
                byte_budget=self.byte_budget,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    def _evict_to_budget(self, *, protected_key: str) -> None:
        while self.resident_bytes > self.byte_budget:
            victim_key = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if key != protected_key and entry.pin_count == 0
                ),
                None,
            )
            if victim_key is None:
                return
            victim = self._entries.pop(victim_key)
            self.resident_bytes -= victim.nbytes
            self._evictions += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
