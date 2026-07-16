from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name: str) -> Any:
    # Keep lightweight submodules importable without initializing CUDA/model
    # dependencies. `from nanovllm import LLM` preserves the original API.
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module 'nanovllm' has no attribute {name!r}")
