from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from glob import glob
import os
from typing import Any
from safetensors import safe_open

from nanovllm.utils.loader import load_model


@dataclass(frozen=True, slots=True)
class Qwen3VLCheckpointAudit:
    language_weights: int
    visual_weights: int
    lm_head_weights: int


def audit_qwen3_vl_checkpoint(model_path: str) -> Qwen3VLCheckpointAudit:
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise ValueError(f"Qwen3-VL checkpoint has no safetensors files: {model_path}")
    seen: dict[str, str] = {}
    counts = {"language": 0, "visual": 0, "lm_head": 0}
    prefixes = (
        ("model.language_model.", "language"),
        ("model.visual.", "visual"),
        ("lm_head.", "lm_head"),
    )
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                previous = seen.get(key)
                if previous is not None:
                    raise ValueError(
                        f"Qwen3-VL checkpoint weight appears in multiple files: "
                        f"{key} ({previous}, {filename})"
                    )
                seen[key] = filename
                for prefix, partition in prefixes:
                    if key.startswith(prefix):
                        counts[partition] += 1
                        break
                else:
                    raise ValueError(f"unknown Qwen3-VL checkpoint weight: {key}")
    for partition in ("language", "visual"):
        if counts[partition] == 0:
            raise ValueError(f"Qwen3-VL checkpoint is missing {partition} weights")
    return Qwen3VLCheckpointAudit(
        language_weights=counts["language"],
        visual_weights=counts["visual"],
        lm_head_weights=counts["lm_head"],
    )


class QwenVLAdapter(ABC):
    """Small model-family boundary for native Qwen vision-language runtime setup."""

    model_type: str
    vision_weight_prefix: str
    deepstack_layer_count: int
    default_vision_attention_backend: str

    def __init__(self, hf_config: Any) -> None:
        self.hf_config = hf_config

    @property
    def text_config(self):
        config = getattr(self.hf_config, "text_config", None)
        if config is None:
            raise ValueError(f"{self.model_type} config is missing text_config")
        return config

    @abstractmethod
    def create_language_model(self, model_class=None, **kwargs): ...

    @abstractmethod
    def create_prompt_processor(self, model_path: str): ...

    def create_vision_provider(self, model_path: str, **kwargs):
        from nanovllm.multimodal.vision_provider import VisionFeatureProvider

        return VisionFeatureProvider.from_pretrained(model_path, **kwargs)

    def audit_checkpoint(self, model_path: str):
        return None

    @abstractmethod
    def load_language_weights(self, model, model_path: str): ...


class Qwen25VLAdapter(QwenVLAdapter):
    model_type = "qwen2_5_vl"
    vision_weight_prefix = "visual."
    deepstack_layer_count = 0
    default_vision_attention_backend = "sdpa"

    def create_language_model(self, model_class=None, **kwargs):
        if model_class is None:
            from nanovllm.models.qwen25_vl import Qwen25VLForCausalLM
            model_class = Qwen25VLForCausalLM
        return model_class(self.text_config)

    def create_prompt_processor(self, model_path: str):
        from nanovllm.multimodal.qwen25_vl_processor import Qwen25VLPromptProcessor
        return Qwen25VLPromptProcessor.from_pretrained(model_path)

    def load_language_weights(self, model, model_path: str):
        return load_model(model, model_path, include_prefixes=("model.",))


class Qwen3VLAdapter(QwenVLAdapter):
    model_type = "qwen3_vl"
    vision_weight_prefix = "model.visual."
    deepstack_layer_count = 3
    default_vision_attention_backend = "flash_attention_2"

    def create_language_model(self, model_class=None, **kwargs):
        if model_class is None:
            from nanovllm.models.qwen3_vl import Qwen3VLForCausalLM
            model_class = Qwen3VLForCausalLM
        return model_class(self.text_config, **kwargs)

    def audit_checkpoint(self, model_path: str):
        return audit_qwen3_vl_checkpoint(model_path)

    def create_prompt_processor(self, model_path: str):
        from nanovllm.multimodal.qwen3_vl_processor import Qwen3VLPromptProcessor
        return Qwen3VLPromptProcessor.from_pretrained(model_path)

    def load_language_weights(self, model, model_path: str):
        return load_model(
            model,
            model_path,
            include_prefixes=("model.language_model.", "lm_head."),
        )


_ADAPTERS = {
    Qwen25VLAdapter.model_type: Qwen25VLAdapter,
    Qwen3VLAdapter.model_type: Qwen3VLAdapter,
}


def resolve_qwen_vl_adapter(hf_config) -> QwenVLAdapter:
    model_type = getattr(hf_config, "model_type", None)
    adapter_class = _ADAPTERS.get(model_type)
    if adapter_class is None:
        raise ValueError(f"unsupported Qwen-VL model type: {model_type!r}")
    return adapter_class(hf_config)


__all__ = [
    "QwenVLAdapter", "Qwen25VLAdapter", "Qwen3VLAdapter",
    "resolve_qwen_vl_adapter", "audit_qwen3_vl_checkpoint",
    "Qwen3VLCheckpointAudit",
]
