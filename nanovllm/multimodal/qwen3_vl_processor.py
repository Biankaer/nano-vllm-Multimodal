from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any, Sequence
from nanovllm.multimodal.dependencies import (
    require_transformers_version as _require_transformers_version,
)

from nanovllm.multimodal.qwen25_vl_processor import (
    _field,
    _fingerprint_image,
    _fingerprint_model_path,
    _hash_json,
)
from nanovllm.multimodal.qwen3_vl_runtime import (
    build_qwen3_vl_prompt_metadata,
    build_vision_feature_key,
)
from nanovllm.multimodal.runtime import MaterializedMultimodalPrompt, VisionInput


class Qwen3VLPromptProcessor:
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
        if spatial_merge_size < 1:
            raise ValueError("spatial_merge_size must be positive")
        self.processor = processor
        self.model_fingerprint = model_fingerprint
        self.processor_fingerprint = processor_fingerprint
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.spatial_merge_size = spatial_merge_size

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3VLPromptProcessor":
        require_transformers_version()
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(model_path)
        if getattr(config, "model_type", None) != "qwen3_vl":
            raise ValueError("native multimodal processing requires a Qwen3-VL checkpoint")
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
        return cls(
            processor,
            model_fingerprint=_fingerprint_model_path(model_path),
            processor_fingerprint=processor_fingerprint,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        )

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def materialize(
        self,
        messages: list[dict[str, Any]],
        images: Sequence[object],
        details: Sequence[str],
    ) -> MaterializedMultimodalPrompt:
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
            raise ValueError("native Qwen3-VL mode does not support video inputs")

        if not images:
            metadata = build_qwen3_vl_prompt_metadata(
                token_ids=token_ids,
                image_grid_thw=(),
                image_feature_keys=(),
                image_token_id=self.image_token_id,
                spatial_merge_size=self.spatial_merge_size,
            )
            return MaterializedMultimodalPrompt(token_ids, metadata, ())

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
            actual_patches = pixel_values.shape[0] if pixel_values.ndim else 0
            raise ValueError(
                "processor patch count does not match image grids: "
                f"patches={actual_patches}, expected={sum(patch_counts)}"
            )

        image_fingerprints = tuple(
            _fingerprint_image(image, detail)
            for image, detail in zip(images, details)
        )
        group_fingerprint = _hash_json(
            {
                "images": image_fingerprints,
                "grids": grids,
            }
        )
        vision_inputs = []
        offset = 0
        for occurrence, (image_fingerprint, grid, patch_count) in enumerate(
            zip(image_fingerprints, grids, patch_counts)
        ):
            base_key = build_vision_feature_key(
                self.model_fingerprint,
                self.processor_fingerprint,
                _hash_json(
                    {
                        "group": group_fingerprint,
                        "image": image_fingerprint,
                    }
                ),
                grid,
            )
            feature_key = f"{base_key}:{occurrence}"
            patches = pixel_values[offset:offset + patch_count].detach().cpu().contiguous()
            vision_inputs.append(VisionInput(feature_key, patches, grid))
            offset += patch_count

        metadata = build_qwen3_vl_prompt_metadata(
            token_ids=token_ids,
            image_grid_thw=grids,
            image_feature_keys=tuple(item.feature_key for item in vision_inputs),
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        return MaterializedMultimodalPrompt(token_ids, metadata, tuple(vision_inputs))


def require_transformers_version(current_version: str | None = None) -> None:
    """Compatibility wrapper around the shared lightweight preflight helper."""
    if current_version is None:
        try:
            current_version = importlib_metadata.version("transformers")
        except importlib_metadata.PackageNotFoundError:
            return _require_transformers_version()
    _require_transformers_version(current_version)


__all__ = ["Qwen3VLPromptProcessor", "require_transformers_version"]
