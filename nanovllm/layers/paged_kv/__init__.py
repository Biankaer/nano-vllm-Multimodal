from __future__ import annotations

from nanovllm.layers.paged_kv.base import (
    GatherDimensions,
    index_value_validation_enabled,
    PagedKVAllocation,
    PagedKVBackend,
    StoreDimensions,
    resolve_paged_kv_backend_name,
    validate_gather_inputs,
    validate_store_inputs,
)


def create_paged_kv_backend(name: str) -> PagedKVBackend:
    if name == "pytorch":
        from nanovllm.layers.paged_kv.pytorch_backend import PyTorchPagedKVBackend

        return PyTorchPagedKVBackend()
    if name == "triton":
        from nanovllm.layers.paged_kv.triton_backend import TritonPagedKVBackend

        return TritonPagedKVBackend()
    if name == "cuda":
        from nanovllm.layers.paged_kv.cuda_backend import CUDAPagedKVBackend

        return CUDAPagedKVBackend()
    raise ValueError(f"unknown paged KV backend: {name}")


__all__ = [
    "GatherDimensions",
    "index_value_validation_enabled",
    "PagedKVAllocation",
    "PagedKVBackend",
    "StoreDimensions",
    "create_paged_kv_backend",
    "resolve_paged_kv_backend_name",
    "validate_gather_inputs",
    "validate_store_inputs",
]
