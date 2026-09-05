from __future__ import annotations

from threading import Lock
from typing import Sequence

import torch

from nanovllm.layers.paged_kv.base import (
    PagedKVAllocation,
    validate_gather_inputs,
    validate_store_inputs,
)
from nanovllm.layers.paged_kv.loader import load_paged_kv_extension


_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_EXTENSION_LOAD_LOCK = Lock()
_EXTENSION_READY = False


def reset_cuda_backend_extension_state_for_tests() -> None:
    global _EXTENSION_READY
    with _EXTENSION_LOAD_LOCK:
        _EXTENSION_READY = False


class CUDAPagedKVBackend:
    name = "cuda"

    @staticmethod
    def _load() -> None:
        global _EXTENSION_READY
        if _EXTENSION_READY:
            return
        with _EXTENSION_LOAD_LOCK:
            if not _EXTENSION_READY:
                load_paged_kv_extension()
                _EXTENSION_READY = True

    def allocate(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> PagedKVAllocation:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("CUDA Paged KV allocation requires a CUDA device")
        if dtype not in _DTYPE_NAMES:
            raise ValueError(f"CUDA Paged KV allocation does not support dtype {dtype}")
        self._load()
        device_index = (
            torch.cuda.current_device()
            if resolved_device.index is None
            else resolved_device.index
        )
        owner = torch.classes.nanovllm_paged_kv.PagedKVCache(
            list(shape), _DTYPE_NAMES[dtype], device_index
        )
        return PagedKVAllocation(owner.tensor(), owner)

    def copy_slots_async(
        self,
        allocation: PagedKVAllocation,
        host_slots: torch.Tensor,
    ) -> torch.Tensor:
        if allocation.owner is None:
            raise ValueError("CUDA Paged KV allocation has no native owner")
        return allocation.owner.copy_slots_async(host_slots)

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        validate_store_inputs(key, value, key_cache, value_cache, slot_mapping)
        if not key.is_cuda:
            raise ValueError("CUDA Paged KV Store requires CUDA tensors")
        self._load()
        torch.ops.nanovllm_paged_kv.store(
            key, value, key_cache, value_cache, slot_mapping
        )

    def gather(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        validate_gather_inputs(cache, block_table, sequence_length)
        if not cache.is_cuda:
            raise ValueError("CUDA Paged KV Gather requires CUDA tensors")
        self._load()
        return torch.ops.nanovllm_paged_kv.gather(
            cache, block_table, sequence_length
        )
