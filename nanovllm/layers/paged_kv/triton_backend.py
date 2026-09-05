from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from nanovllm.layers.paged_kv.base import (
    PagedKVAllocation,
    validate_gather_inputs,
    validate_store_inputs,
)


@triton.jit
def _store_paged_kv_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_index = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_index)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    if slot != -1:
        key = tl.load(key_ptr + token_index * key_stride + offsets, mask=mask)
        value = tl.load(value_ptr + token_index * value_stride + offsets, mask=mask)
        cache_offsets = slot * D + offsets
        tl.store(key_cache_ptr + cache_offsets, key, mask=mask)
        tl.store(value_cache_ptr + cache_offsets, value, mask=mask)


@triton.jit
def _gather_paged_kv_kernel(
    cache_ptr,
    block_table_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    output_token = tl.program_id(0)
    logical_block = output_token // BLOCK_SIZE
    block_offset = output_token % BLOCK_SIZE
    physical_block = tl.load(block_table_ptr + logical_block)
    source_slot = physical_block * BLOCK_SIZE + block_offset
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    values = tl.load(cache_ptr + source_slot * D + offsets, mask=mask)
    tl.store(output_ptr + output_token * D + offsets, values, mask=mask)


class TritonPagedKVBackend:
    name = "triton"

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
        dimensions = validate_store_inputs(
            key, value, key_cache, value_cache, slot_mapping
        )
        if not key.is_cuda:
            raise ValueError("Triton Paged KV Store requires CUDA tensors")
        d = dimensions.num_kv_heads * dimensions.head_dim
        if key_cache.stride(1) != d or value_cache.stride(1) != d:
            raise ValueError("cache slots must be contiguous")
        if dimensions.num_tokens == 0:
            return
        block_d = triton.next_power_of_2(d)
        _store_paged_kv_kernel[(dimensions.num_tokens,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            key_cache,
            value_cache,
            slot_mapping,
            D=d,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 2048 else 4,
        )

    def gather(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        dimensions = validate_gather_inputs(cache, block_table, sequence_length)
        if not cache.is_cuda:
            raise ValueError("Triton Paged KV Gather requires CUDA tensors")
        output = cache.new_empty(
            (sequence_length, dimensions.num_kv_heads, dimensions.head_dim)
        )
        if sequence_length == 0:
            return output
        d = dimensions.num_kv_heads * dimensions.head_dim
        if cache.stride(1) != d:
            raise ValueError("cache slots must be contiguous")
        block_d = triton.next_power_of_2(d)
        _gather_paged_kv_kernel[(sequence_length,)](
            cache,
            block_table,
            output,
            BLOCK_SIZE=dimensions.block_size,
            D=d,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 2048 else 4,
        )
        return output
