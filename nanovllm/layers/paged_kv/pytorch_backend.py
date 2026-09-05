from __future__ import annotations

from typing import Sequence

import torch

from nanovllm.layers.paged_kv.base import (
    PagedKVAllocation,
    validate_gather_inputs,
    validate_store_inputs,
)


class PyTorchPagedKVBackend:
    name = "pytorch"

    def allocate(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> PagedKVAllocation:
        return PagedKVAllocation(torch.empty(tuple(shape), dtype=dtype, device=device))

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        validate_store_inputs(key, value, key_cache, value_cache, slot_mapping)
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].to(dtype=torch.long)
        key_cache.flatten(0, 1).index_copy_(0, slots, key[valid])
        value_cache.flatten(0, 1).index_copy_(0, slots, value[valid])

    def gather(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        dimensions = validate_gather_inputs(cache, block_table, sequence_length)
        if dimensions.required_blocks == 0:
            return cache.new_empty((0, dimensions.num_kv_heads, dimensions.head_dim))
        block_ids = block_table[:dimensions.required_blocks].to(dtype=torch.long)
        return cache.index_select(0, block_ids).flatten(0, 1)[:sequence_length]
