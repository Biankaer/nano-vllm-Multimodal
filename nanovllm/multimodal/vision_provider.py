from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import nn

from nanovllm.multimodal.feature_store import VisionFeatureStore
from nanovllm.multimodal.qwen25_vl_processor import VisionInput
from nanovllm.utils.loader import load_model


def _move_module_preserving_float32_buffers(
    module: nn.Module,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    float32_buffers = {
        name: buffer.detach().clone()
        for name, buffer in module.named_buffers()
        if buffer.dtype == torch.float32
    }
    module.to(device=device, dtype=dtype)
    for qualified_name, buffer in float32_buffers.items():
        parent_name, _, buffer_name = qualified_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        parent._buffers[buffer_name] = buffer.to(device=device)


def _restore_qwen25_vision_rotary_buffer(
    vision_tower: nn.Module,
    vision_config,
    *,
    device: torch.device | str,
) -> None:
    head_dim = vision_config.hidden_size // vision_config.num_heads
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
            / rotary_dim
        )
    )
    vision_tower.rotary_pos_emb._buffers["inv_freq"] = inv_freq.to(device=device)


@dataclass(frozen=True, slots=True)
class VisionProviderStats:
    cache_entries: int
    cache_resident_bytes: int
    cache_byte_budget: int
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    encoder_batches: int
    encoded_images: int
    registered_inputs: int
    prefix_skipped_features: int


class VisionFeatureProvider:
    def __init__(
        self,
        vision_tower: nn.Module,
        *,
        spatial_merge_size: int,
        expected_hidden_size: int,
        feature_store: VisionFeatureStore,
    ) -> None:
        if spatial_merge_size < 1:
            raise ValueError("spatial merge size must be positive")
        if expected_hidden_size < 1:
            raise ValueError("expected hidden size must be positive")
        parameter = next(vision_tower.parameters(), None)
        if parameter is None:
            raise ValueError("vision tower must contain at least one parameter")
        self.vision_tower = vision_tower
        self.spatial_merge_size = spatial_merge_size
        self.expected_hidden_size = expected_hidden_size
        self.feature_store = feature_store
        self.device = parameter.device
        self.dtype = parameter.dtype
        self._registered_inputs: dict[str, VisionInput] = {}
        self._input_refcounts: dict[str, int] = {}
        self._encoder_batches = 0
        self._encoded_images = 0
        self._prefix_skipped_features = 0

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        byte_budget: int,
        dtype: torch.dtype,
        device: torch.device | str,
        attn_implementation: str = "sdpa",
    ) -> "VisionFeatureProvider":
        from transformers import AutoConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        config = AutoConfig.from_pretrained(model_path)
        if getattr(config, "model_type", None) != "qwen2_5_vl":
            raise ValueError("vision provider requires a Qwen2.5-VL checkpoint")
        if (
            attn_implementation == "flash_attention_2"
            and not hasattr(torch.library, "wrap_triton")
        ):
            raise RuntimeError(
                "vision FlashAttention requires torch.library.wrap_triton; "
                "use attn_implementation='sdpa' with this PyTorch version"
            )
        vision_config = config.vision_config
        vision_config._attn_implementation = attn_implementation
        vision_config.torch_dtype = dtype
        vision_tower = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            vision_config
        )
        _move_module_preserving_float32_buffers(
            vision_tower,
            device=device,
            dtype=dtype,
        )
        _restore_qwen25_vision_rotary_buffer(
            vision_tower,
            vision_config,
            device=device,
        )
        vision_tower.eval()
        load_model(
            vision_tower,
            model_path,
            include_prefixes=("visual.",),
            strip_prefix="visual.",
        )
        return cls(
            vision_tower,
            spatial_merge_size=vision_config.spatial_merge_size,
            expected_hidden_size=vision_config.out_hidden_size,
            feature_store=VisionFeatureStore(byte_budget),
        )

    def register_inputs(self, inputs: Sequence[VisionInput]) -> None:
        unique_inputs = {item.feature_key: item for item in inputs}
        for item in unique_inputs.values():
            existing = self._registered_inputs.get(item.feature_key)
            if existing is not None:
                if (
                    existing.grid_thw != item.grid_thw
                    or existing.pixel_values.shape != item.pixel_values.shape
                    or existing.pixel_values.dtype != item.pixel_values.dtype
                    or not torch.equal(existing.pixel_values, item.pixel_values)
                ):
                    raise ValueError(
                        f"conflicting vision input registration for {item.feature_key}"
                    )
                self._input_refcounts[item.feature_key] += 1
                continue
            self._registered_inputs[item.feature_key] = VisionInput(
                feature_key=item.feature_key,
                pixel_values=item.pixel_values.detach().cpu().contiguous(),
                grid_thw=item.grid_thw,
            )
            self._input_refcounts[item.feature_key] = 1

    def release_inputs(self, keys: Sequence[str]) -> None:
        for key in dict.fromkeys(keys):
            refcount = self._input_refcounts.get(key)
            if refcount is None:
                continue
            if refcount > 1:
                self._input_refcounts[key] = refcount - 1
                continue
            self._input_refcounts.pop(key, None)
            self._registered_inputs.pop(key, None)

    def resolve(self, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        unique_keys = tuple(dict.fromkeys(keys))
        cached: dict[str, torch.Tensor] = {}
        missing_keys = []
        for key in unique_keys:
            value = self.feature_store.get(key)
            if value is None:
                missing_keys.append(key)
            else:
                cached[key] = value

        if missing_keys:
            missing_inputs = []
            for key in missing_keys:
                item = self._registered_inputs.get(key)
                if item is None:
                    raise KeyError(f"missing registered vision input for {key}")
                self._validate_grid(item.grid_thw)
                missing_inputs.append(item)

            with self.feature_store.pin(tuple(cached)):
                encoded = self._encode(missing_inputs)
                self.feature_store.put_many(encoded)

        resolved = {}
        for key in unique_keys:
            value = self.feature_store.peek(key)
            if value is None:
                raise RuntimeError(f"resolved vision feature was not retained: {key}")
            resolved[key] = value
        return resolved

    @contextmanager
    def resolve_pinned(
        self,
        keys: Sequence[str],
    ) -> Iterator[dict[str, torch.Tensor]]:
        unique_keys = tuple(dict.fromkeys(keys))
        self.resolve(unique_keys)
        with self.feature_store.pin(unique_keys) as pinned:
            yield pinned

    def record_prefix_skipped_features(self, count: int) -> None:
        if count < 0:
            raise ValueError("prefix skipped feature count cannot be negative")
        self._prefix_skipped_features += count

    def stats(self) -> VisionProviderStats:
        cache_stats = self.feature_store.stats()
        return VisionProviderStats(
            cache_entries=cache_stats.entries,
            cache_resident_bytes=cache_stats.resident_bytes,
            cache_byte_budget=cache_stats.byte_budget,
            cache_hits=cache_stats.hits,
            cache_misses=cache_stats.misses,
            cache_evictions=cache_stats.evictions,
            encoder_batches=self._encoder_batches,
            encoded_images=self._encoded_images,
            registered_inputs=len(self._registered_inputs),
            prefix_skipped_features=self._prefix_skipped_features,
        )

    @torch.inference_mode()
    def _encode(self, inputs: Sequence[VisionInput]) -> dict[str, torch.Tensor]:
        patch_widths = {item.pixel_values.shape[1] for item in inputs}
        if len(patch_widths) != 1:
            raise ValueError("vision patch tensors must have the same feature width")
        pixel_values = torch.cat([item.pixel_values for item in inputs]).to(
            device=self.device,
            dtype=self.dtype,
        )
        grid_thw = torch.tensor(
            [item.grid_thw for item in inputs],
            dtype=torch.int64,
            device=self.device,
        )
        output = self.vision_tower(pixel_values, grid_thw=grid_thw)
        if output.ndim != 2 or output.shape[1] != self.expected_hidden_size:
            raise ValueError(
                "vision output hidden size mismatch: "
                f"shape={tuple(output.shape)}, expected={self.expected_hidden_size}"
            )

        token_counts = [self._merged_token_count(item.grid_thw) for item in inputs]
        if output.shape[0] != sum(token_counts):
            raise ValueError(
                "vision output token count mismatch: "
                f"tokens={output.shape[0]}, expected={sum(token_counts)}"
            )
        chunks = output.split(token_counts, dim=0)
        self._encoder_batches += 1
        self._encoded_images += len(inputs)
        return {
            item.feature_key: chunk.detach().contiguous()
            for item, chunk in zip(inputs, chunks)
        }

    def _merged_token_count(self, grid: tuple[int, int, int]) -> int:
        grid_t, grid_h, grid_w = grid
        return (
            grid_t
            * (grid_h // self.spatial_merge_size)
            * (grid_w // self.spatial_merge_size)
        )

    def _validate_grid(self, grid: tuple[int, int, int]) -> None:
        grid_t, grid_h, grid_w = grid
        if grid_t < 1 or grid_h < 1 or grid_w < 1:
            raise ValueError("vision grid dimensions must be positive")
        if grid_h % self.spatial_merge_size or grid_w % self.spatial_merge_size:
            raise ValueError("vision grid must be divisible by spatial merge size")
