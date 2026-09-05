"""Multimodal serving and native-runtime primitives."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanovllm.multimodal.cache import ImageAssetCache, ImageFeatureCache
    from nanovllm.multimodal.manager import MultimodalManager
    from nanovllm.multimodal.qwen25_vl import Qwen25VLBackend, Qwen25VLConfig
    from nanovllm.multimodal.qwen25_vl_runtime import (
        build_qwen25_vl_prompt_metadata,
        build_vision_feature_key,
        merge_visual_embeddings,
    )
    from nanovllm.multimodal.qwen3_vl_processor import Qwen3VLPromptProcessor
    from nanovllm.multimodal.qwen3_vl_runtime import build_qwen3_vl_prompt_metadata
    from nanovllm.multimodal.runtime import (
        MaterializedMultimodalPrompt,
        MultimodalPromptMetadata,
        VisionFeatureBundle,
        VisionInput,
        VisualTokenSpan,
    )
    from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
    from nanovllm.multimodal.schemas import ImageInput, MultimodalRequest


__all__ = [
    "ImageFeatureCache",
    "ImageAssetCache",
    "ImageInput",
    "MultimodalBatchPolicy",
    "MultimodalManager",
    "MaterializedMultimodalPrompt",
    "MultimodalPromptMetadata",
    "MultimodalRequest",
    "Qwen25VLBackend",
    "Qwen25VLConfig",
    "Qwen3VLPromptProcessor",
    "VisionFeatureBundle",
    "VisionInput",
    "VisualTokenSpan",
    "build_qwen25_vl_prompt_metadata",
    "build_qwen3_vl_prompt_metadata",
    "build_vision_feature_key",
    "merge_visual_embeddings",
]


_LAZY_IMPORTS = {
    "ImageAssetCache": ("nanovllm.multimodal.cache", "ImageAssetCache"),
    "ImageFeatureCache": ("nanovllm.multimodal.cache", "ImageFeatureCache"),
    "ImageInput": ("nanovllm.multimodal.schemas", "ImageInput"),
    "MultimodalBatchPolicy": ("nanovllm.multimodal.scheduler_policy", "MultimodalBatchPolicy"),
    "MultimodalManager": ("nanovllm.multimodal.manager", "MultimodalManager"),
    "MaterializedMultimodalPrompt": (
        "nanovllm.multimodal.runtime", "MaterializedMultimodalPrompt"
    ),
    "MultimodalPromptMetadata": ("nanovllm.multimodal.runtime", "MultimodalPromptMetadata"),
    "MultimodalRequest": ("nanovllm.multimodal.schemas", "MultimodalRequest"),
    "Qwen25VLBackend": ("nanovllm.multimodal.qwen25_vl", "Qwen25VLBackend"),
    "Qwen25VLConfig": ("nanovllm.multimodal.qwen25_vl", "Qwen25VLConfig"),
    "Qwen3VLPromptProcessor": (
        "nanovllm.multimodal.qwen3_vl_processor",
        "Qwen3VLPromptProcessor",
    ),
    "VisionFeatureBundle": ("nanovllm.multimodal.runtime", "VisionFeatureBundle"),
    "VisionInput": ("nanovllm.multimodal.runtime", "VisionInput"),
    "VisualTokenSpan": ("nanovllm.multimodal.runtime", "VisualTokenSpan"),
    "build_qwen25_vl_prompt_metadata": (
        "nanovllm.multimodal.qwen25_vl_runtime",
        "build_qwen25_vl_prompt_metadata",
    ),
    "build_qwen3_vl_prompt_metadata": (
        "nanovllm.multimodal.qwen3_vl_runtime",
        "build_qwen3_vl_prompt_metadata",
    ),
    "build_vision_feature_key": (
        "nanovllm.multimodal.qwen25_vl_runtime",
        "build_vision_feature_key",
    ),
    "merge_visual_embeddings": (
        "nanovllm.multimodal.qwen25_vl_runtime",
        "merge_visual_embeddings",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'nanovllm.multimodal' has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
