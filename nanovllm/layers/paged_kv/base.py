from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


@dataclass(frozen=True, slots=True)
class StoreDimensions:
    num_tokens: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size


@dataclass(frozen=True, slots=True)
class GatherDimensions:
    sequence_length: int
    required_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int


@dataclass(slots=True)
class PagedKVAllocation:
    tensor: torch.Tensor
    owner: object | None = None


class PagedKVBackend(Protocol):
    name: str

    def allocate(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> PagedKVAllocation:
        ...
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        ...

    def gather(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        ...


def index_value_validation_enabled(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return True
    value = os.environ.get("NANOVLLM_PAGED_KV_VALIDATE_INDICES", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_store_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> StoreDimensions:
    if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
        raise ValueError("key and value must have matching shape [tokens, kv_heads, head_dim]")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("key and value inner dimensions must be contiguous")
    if key_cache.ndim != 4 or value_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("key and value cache shape must match [blocks, block, kv_heads, head_dim]")
    num_tokens, num_kv_heads, head_dim = key.shape
    num_blocks, block_size, cache_heads, cache_head_dim = key_cache.shape
    if (cache_heads, cache_head_dim) != (num_kv_heads, head_dim):
        raise ValueError("cache shape does not match key/value kv_heads and head_dim")
    tensors = (key, value, key_cache, value_cache, slot_mapping)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all paged KV Store tensors must be on the same device")
    if not (key.dtype == value.dtype == key_cache.dtype == value_cache.dtype):
        raise ValueError("key, value, and caches must have matching dtype")
    if slot_mapping.dtype != torch.int32 or slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be a one-dimensional int32 tensor")
    if slot_mapping.numel() != num_tokens:
        raise ValueError("slot_mapping length must equal the number of input tokens")
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    num_slots = num_blocks * block_size
    if slot_mapping.numel() and index_value_validation_enabled(slot_mapping):
        invalid = (slot_mapping < -1) | (slot_mapping >= num_slots)
        if bool(torch.any(invalid)):
            raise ValueError(f"slot_mapping contains a slot outside [-1, {num_slots})")
    return StoreDimensions(
        num_tokens=num_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def validate_gather_inputs(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_length: int,
) -> GatherDimensions:
    if cache.ndim != 4:
        raise ValueError("cache must have shape [blocks, block, kv_heads, head_dim]")
    if cache.stride(-1) != 1:
        raise ValueError("cache inner dimensions must be contiguous")
    if block_table.dtype != torch.int32 or block_table.ndim != 1:
        raise ValueError("block_table must be a one-dimensional int32 tensor")
    if block_table.device != cache.device:
        raise ValueError("cache and block_table must be on the same device")
    if not block_table.is_contiguous():
        raise ValueError("block_table must be contiguous")
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    num_blocks, block_size, num_kv_heads, head_dim = cache.shape
    required_blocks = (sequence_length + block_size - 1) // block_size
    if block_table.numel() < required_blocks:
        raise ValueError("block_table does not cover the requested sequence length")
    selected = block_table[:required_blocks]
    if (
        selected.numel()
        and index_value_validation_enabled(selected)
        and bool(torch.any((selected < 0) | (selected >= num_blocks)))
    ):
        raise ValueError("block_table contains an invalid physical block")
    return GatherDimensions(
        sequence_length=sequence_length,
        required_blocks=required_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def resolve_paged_kv_backend_name(
    requested: str,
    *,
    cuda_available: bool,
    cuda_extension_available: bool,
) -> str:
    if requested not in {"auto", "pytorch", "triton", "cuda"}:
        raise ValueError(
            "paged_kv_backend must be one of: auto, pytorch, triton, cuda"
        )
    if requested != "auto":
        return requested
    if cuda_available and cuda_extension_available:
        return "cuda"
    if cuda_available:
        return "triton"
    return "pytorch"
