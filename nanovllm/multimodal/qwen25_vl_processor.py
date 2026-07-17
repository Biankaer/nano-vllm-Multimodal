from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from nanovllm.multimodal.qwen25_vl_runtime import (
    build_qwen25_vl_prompt_metadata,
    build_vision_feature_key,
)
from nanovllm.multimodal.runtime import MultimodalPromptMetadata


@dataclass(frozen=True, slots=True)
class VisionInput:
    feature_key: str
    pixel_values: torch.Tensor
    grid_thw: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MaterializedQwen25VLPrompt:
    token_ids: tuple[int, ...]
    metadata: MultimodalPromptMetadata
    vision_inputs: tuple[VisionInput, ...]


class Qwen25VLPromptProcessor:
    def __init__(
        self,
        processor,
        *,
        model_fingerprint: str,
        processor_fingerprint: str,
        image_token_id: int,
        video_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        if not model_fingerprint or not processor_fingerprint:
            raise ValueError("model and processor fingerprints cannot be empty")
        self.processor = processor
        self.model_fingerprint = model_fingerprint
        self.processor_fingerprint = processor_fingerprint
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.spatial_merge_size = spatial_merge_size

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen25VLPromptProcessor":
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(model_path)
        if getattr(config, "model_type", None) != "qwen2_5_vl":
            raise ValueError("native multimodal processing requires a Qwen2.5-VL checkpoint")
        processor = AutoProcessor.from_pretrained(model_path)
        image_processor = getattr(processor, "image_processor", None)
        processor_state = (
            image_processor.to_dict()
            if image_processor is not None and hasattr(image_processor, "to_dict")
            else {"class": type(image_processor).__qualname__}
        )
        processor_fingerprint = _hash_json(
            {
                "processor_class": type(processor).__qualname__,
                "image_processor": processor_state,
            }
        )
        vision_config = config.vision_config
        return cls(
            processor,
            model_fingerprint=_fingerprint_model_path(model_path),
            processor_fingerprint=processor_fingerprint,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            spatial_merge_size=vision_config.spatial_merge_size,
        )

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def materialize(
        self,
        messages: list[dict[str, Any]],
        images: Sequence[object],
        details: Sequence[str],
    ) -> MaterializedQwen25VLPrompt:
        if len(images) != len(details):
            raise ValueError("every image must have one detail setting")
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=list(images) if images else None,
            videos=None,
            padding=False,
            return_tensors="pt",
        )
        input_ids = _field(inputs, "input_ids")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("processor input_ids must have shape [1, prompt_length]")
        token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
        if self.video_token_id in token_ids:
            raise ValueError("native Qwen2.5-VL mode does not support video inputs")

        if not images:
            metadata = build_qwen25_vl_prompt_metadata(
                token_ids=token_ids,
                image_grid_thw=(),
                image_feature_keys=(),
                image_token_id=self.image_token_id,
                spatial_merge_size=self.spatial_merge_size,
            )
            return MaterializedQwen25VLPrompt(token_ids, metadata, ())

        pixel_values = _field(inputs, "pixel_values")
        image_grid_thw = _field(inputs, "image_grid_thw")
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise ValueError("image_grid_thw must have shape [num_images, 3]")
        if image_grid_thw.shape[0] != len(images):
            raise ValueError("processor image grids do not match the image count")

        grids = tuple(
            tuple(int(value) for value in row.tolist())
            for row in image_grid_thw
        )
        patch_counts = [grid_t * grid_h * grid_w for grid_t, grid_h, grid_w in grids]
        if pixel_values.ndim != 2 or pixel_values.shape[0] != sum(patch_counts):
            raise ValueError(
                "processor patch count does not match image grids: "
                f"patches={pixel_values.shape[0]}, expected={sum(patch_counts)}"
            )

        vision_inputs = []
        offset = 0
        for image, detail, grid, patch_count in zip(images, details, grids, patch_counts):
            image_fingerprint = _fingerprint_image(image, detail)
            feature_key = build_vision_feature_key(
                self.model_fingerprint,
                self.processor_fingerprint,
                image_fingerprint,
                grid,
            )
            patches = pixel_values[offset:offset + patch_count].detach().cpu().contiguous()
            vision_inputs.append(VisionInput(feature_key, patches, grid))
            offset += patch_count

        feature_keys = tuple(item.feature_key for item in vision_inputs)
        metadata = build_qwen25_vl_prompt_metadata(
            token_ids=token_ids,
            image_grid_thw=grids,
            image_feature_keys=feature_keys,
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        return MaterializedQwen25VLPrompt(
            token_ids,
            metadata,
            tuple(vision_inputs),
        )


def _field(inputs, name: str) -> torch.Tensor:
    value = getattr(inputs, name, None)
    if value is None and isinstance(inputs, dict):
        value = inputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"processor output is missing tensor {name}")
    return value


def _fingerprint_image(image: object, detail: str) -> str:
    mode = getattr(image, "mode", None)
    size = getattr(image, "size", None)
    tobytes = getattr(image, "tobytes", None)
    if not isinstance(mode, str) or not isinstance(size, tuple) or not callable(tobytes):
        raise TypeError("images must provide PIL-compatible mode, size, and tobytes")
    digest = hashlib.sha256()
    digest.update(mode.encode("utf-8"))
    digest.update(str(size).encode("ascii"))
    digest.update(detail.encode("utf-8"))
    digest.update(tobytes())
    return digest.hexdigest()


def _fingerprint_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser().resolve()
    digest = hashlib.sha256()
    for filename in ("config.json", "model.safetensors.index.json"):
        candidate = path / filename
        if candidate.is_file():
            digest.update(filename.encode("utf-8"))
            digest.update(candidate.read_bytes())
    for candidate in sorted(path.glob("*.safetensors")):
        stat = candidate.stat()
        digest.update(candidate.name.encode("utf-8"))
        digest.update(stat.st_size.to_bytes(8, "little"))
        digest.update(stat.st_mtime_ns.to_bytes(8, "little"))
    value = digest.hexdigest()
    if not any(path.glob("*.safetensors")):
        raise ValueError(f"model path contains no safetensors checkpoints: {path}")
    return value


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
