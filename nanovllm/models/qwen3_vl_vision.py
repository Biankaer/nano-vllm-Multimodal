from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.multimodal.runtime import VisionFeatureBundle


def _gelu_pytorch_tanh(value: torch.Tensor) -> torch.Tensor:
    return F.gelu(value, approximate="tanh")


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if getattr(config, "hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("Qwen3-VL vision hidden_act must be gelu_pytorch_tanh")
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(_gelu_pytorch_tanh(self.linear_fc1(hidden_states)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.temporal_patch_size = int(config.temporal_patch_size)
        self.in_channels = int(config.in_channels)
        self.embed_dim = int(config.hidden_size)
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    @property
    def flattened_patch_width(self) -> int:
        return (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("pixel_values must be a rank-2 patch matrix")
        if hidden_states.shape[1] != self.flattened_patch_width:
            raise ValueError(
                "pixel patch feature width does not match Conv3d input: "
                f"expected {self.flattened_patch_width}, got {hidden_states.shape[1]}"
            )
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.to(dtype=self.proj.weight.dtype)
        return self.proj(hidden_states).reshape(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("vision rotary dimension must be a positive even number")
        self.dim = dim
        self.theta = float(theta)
        inv_freq = self._make_inv_freq(torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _make_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._make_inv_freq(self.inv_freq.device)
        return self

    def forward(self, seqlen: int) -> torch.Tensor:
        positions = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(positions, self.inv_freq)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        merge_unit = int(config.spatial_merge_size) ** 2
        self.hidden_size = int(config.hidden_size) * merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else int(config.hidden_size)
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(
            self.hidden_size, int(config.out_hidden_size), bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.reshape(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).reshape(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb_vision(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    query_float = query.float()
    key_float = key.float()
    cos_float = cos.unsqueeze(-2).float()
    sin_float = sin.unsqueeze(-2).float()
    rotated_query = query_float * cos_float + _rotate_half(query_float) * sin_float
    rotated_key = key_float * cos_float + _rotate_half(key_float) * sin_float
    return rotated_query.to(query_dtype), rotated_key.to(key_dtype)


@dataclass(frozen=True, slots=True)
class VisionAttentionMetadata:
    cu_seqlens: torch.Tensor
    segment_bounds: tuple[tuple[int, int], ...]
    max_seqlen: int


def _resolve_flash_attn_varlen_func():
    try:
        from flash_attn import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FlashAttention 2 is unavailable; install flash-attn or use attention_backend='sdpa'"
        ) from error
    return flash_attn_varlen_func


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config, attention_backend: str = "sdpa") -> None:
        super().__init__()
        if attention_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError(
                "vision attention backend must be 'sdpa' or 'flash_attention_2'"
            )
        self.dim = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        if self.dim % self.num_heads:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_backend = attention_backend
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    @staticmethod
    def _normalize_metadata(
        metadata: VisionAttentionMetadata | torch.Tensor, token_count: int
    ) -> VisionAttentionMetadata:
        if isinstance(metadata, VisionAttentionMetadata):
            if metadata.segment_bounds[-1][1] != token_count:
                raise ValueError("attention metadata must end at token count")
            return metadata
        # Compatibility for direct primitive calls. The model itself precomputes the
        # host bounds once and passes VisionAttentionMetadata through all blocks.
        if metadata.ndim != 1 or metadata.numel() < 2:
            raise ValueError("cu_seqlens must be a rank-1 tensor with segment bounds")
        if metadata.dtype != torch.int32:
            raise TypeError("cu_seqlens must use int32")
        bounds = metadata.detach().cpu().tolist()
        if bounds[0] != 0 or bounds[-1] != token_count:
            raise ValueError("cu_seqlens must start at zero and end at token count")
        segments = tuple(zip(bounds, bounds[1:]))
        if any(end <= start for start, end in segments):
            raise ValueError("cu_seqlens segments must be non-empty and increasing")
        return VisionAttentionMetadata(
            cu_seqlens=metadata,
            segment_bounds=segments,
            max_seqlen=max(end - start for start, end in segments),
        )

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        outputs = []
        for start, end in metadata.segment_bounds:
            segment_query = query[start:end].transpose(0, 1).unsqueeze(0)
            segment_key = key[start:end].transpose(0, 1).unsqueeze(0)
            segment_value = value[start:end].transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                segment_query,
                segment_key,
                segment_value,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(output.transpose(1, 2).squeeze(0))
        return torch.cat(outputs, dim=0)

    def _flash(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        if not query.is_cuda:
            raise RuntimeError(
                "FlashAttention vision forward requires CUDA; use attention_backend='sdpa' on CPU"
            )
        flash_attn_varlen_func = _resolve_flash_attn_varlen_func()
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=metadata.cu_seqlens,
            cu_seqlens_k=metadata.cu_seqlens,
            max_seqlen_q=metadata.max_seqlen,
            max_seqlen_k=metadata.max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_metadata: VisionAttentionMetadata | torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        metadata = self._normalize_metadata(attention_metadata, token_count)
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(token_count, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        if cos.shape != (token_count, self.head_dim) or sin.shape != cos.shape:
            raise ValueError("vision rotary cos/sin shape must be [tokens, head_dim]")
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        if self.attention_backend == "sdpa":
            output = self._sdpa(query, key, value, metadata)
        else:
            output = self._flash(query, key, value, metadata)
        return self.proj(output.reshape(token_count, self.dim).contiguous())


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attention_backend: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config, attention_backend)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_metadata: VisionAttentionMetadata | torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attention_metadata, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config, attention_backend: str = "sdpa") -> None:
        super().__init__()
        if attention_backend not in {"sdpa", "flash_attention_2"}:
            raise ValueError(
                "vision attention backend must be 'sdpa' or 'flash_attention_2'"
            )
        self.config = config
        self.spatial_merge_size = int(config.spatial_merge_size)
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        self.spatial_merge_unit = self.spatial_merge_size**2
        position_count = int(config.num_position_embeddings)
        self.num_grid_per_side = math.isqrt(position_count)
        if self.num_grid_per_side**2 != position_count:
            raise ValueError("num_position_embeddings must be a perfect square")
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(position_count, config.hidden_size)
        head_dim = int(config.hidden_size) // int(config.num_heads)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen3VLVisionBlock(config, attention_backend)
            for _ in range(int(config.depth))
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config, use_postshuffle_norm=False
        )
        self.deepstack_visual_indexes = tuple(
            int(index) for index in config.deepstack_visual_indexes
        )
        if len(set(self.deepstack_visual_indexes)) != len(
            self.deepstack_visual_indexes
        ):
            raise ValueError("deepstack_visual_indexes must be unique")
        if any(
            index < 0 or index >= len(self.blocks)
            for index in self.deepstack_visual_indexes
        ):
            raise ValueError("deepstack visual index is outside the vision block range")
        self.deepstack_merger_list = nn.ModuleList(
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        )

    def _validate_inputs(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> list[tuple[int, int, int]]:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("grid_thw shape must be [num_images, 3]")
        if grid_thw.shape[0] == 0:
            raise ValueError("grid_thw must describe at least one image")
        if grid_thw.dtype == torch.bool or grid_thw.is_floating_point():
            raise TypeError("grid_thw must contain integer values")
        grids = [tuple(int(value) for value in row) for row in grid_thw.detach().cpu().tolist()]
        if any(value <= 0 for grid in grids for value in grid):
            raise ValueError("grid_thw values must be positive")
        if any(
            height % self.spatial_merge_size
            or width % self.spatial_merge_size
            for _, height, width in grids
        ):
            raise ValueError("grid height and width must be divisible by spatial_merge_size")
        expected_rows = sum(temporal * height * width for temporal, height, width in grids)
        if pixel_values.ndim != 2:
            raise ValueError("pixel_values must be a rank-2 patch matrix")
        if pixel_values.shape[0] != expected_rows:
            raise ValueError(
                f"pixel patch rows must equal sum(t*h*w): expected {expected_rows}, "
                f"got {pixel_values.shape[0]}"
            )
        if pixel_values.shape[1] != self.patch_embed.flattened_patch_width:
            raise ValueError(
                "pixel patch feature width does not match Conv3d input: "
                f"expected {self.patch_embed.flattened_patch_width}, got {pixel_values.shape[1]}"
            )
        return grids

    def fast_pos_embed_interpolate(
        self,
        grid_thw: torch.Tensor,
        grids: Sequence[tuple[int, int, int]] | None = None,
    ) -> torch.Tensor:
        if grids is None:
            grids = [
                tuple(int(value) for value in row)
                for row in grid_thw.detach().cpu().tolist()
            ]
        index_lists: list[list[int]] = [[] for _ in range(4)]
        weight_lists: list[list[float]] = [[] for _ in range(4)]
        for _, height, width in grids:
            height_positions = torch.linspace(
                0, self.num_grid_per_side - 1, height, dtype=torch.float32
            )
            width_positions = torch.linspace(
                0, self.num_grid_per_side - 1, width, dtype=torch.float32
            )
            height_floor = height_positions.to(torch.int64)
            width_floor = width_positions.to(torch.int64)
            height_ceil = (height_floor + 1).clamp(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clamp(max=self.num_grid_per_side - 1)
            height_fraction = height_positions - height_floor
            width_fraction = width_positions - width_floor
            base_floor = height_floor * self.num_grid_per_side
            base_ceil = height_ceil * self.num_grid_per_side
            indices = (
                (base_floor[:, None] + width_floor[None, :]).flatten(),
                (base_floor[:, None] + width_ceil[None, :]).flatten(),
                (base_ceil[:, None] + width_floor[None, :]).flatten(),
                (base_ceil[:, None] + width_ceil[None, :]).flatten(),
            )
            weights = (
                ((1 - height_fraction)[:, None] * (1 - width_fraction)[None, :]).flatten(),
                ((1 - height_fraction)[:, None] * width_fraction[None, :]).flatten(),
                (height_fraction[:, None] * (1 - width_fraction)[None, :]).flatten(),
                (height_fraction[:, None] * width_fraction[None, :]).flatten(),
            )
            for destination, values in zip(index_lists, indices):
                destination.extend(values.tolist())
            for destination, values in zip(weight_lists, weights):
                destination.extend(values.tolist())

        indices = torch.tensor(
            index_lists, dtype=torch.long, device=self.pos_embed.weight.device
        )
        weights = torch.tensor(
            weight_lists,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        interpolation_terms = self.pos_embed(indices) * weights.unsqueeze(-1)
        interpolated = (
            interpolation_terms[0]
            + interpolation_terms[1]
            + interpolation_terms[2]
            + interpolation_terms[3]
        )
        split_sizes = [height * width for _, height, width in grids]
        outputs = []
        for position, (temporal, height, width) in zip(
            interpolated.split(split_sizes), grids
        ):
            outputs.append(
                position.repeat(temporal, 1)
                .reshape(
                    temporal,
                    height // self.spatial_merge_size,
                    self.spatial_merge_size,
                    width // self.spatial_merge_size,
                    self.spatial_merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
        return torch.cat(outputs, dim=0)

    def rot_pos_emb(
        self,
        grid_thw: torch.Tensor,
        grids: Sequence[tuple[int, int, int]] | None = None,
    ) -> torch.Tensor:
        if grids is None:
            grids = [
                tuple(int(value) for value in row)
                for row in grid_thw.detach().cpu().tolist()
            ]
        max_dimension = max(max(height, width) for _, height, width in grids)
        frequency_table = self.rotary_pos_emb(max_dimension)
        coordinates = []
        merge = self.spatial_merge_size
        for temporal, height, width in grids:
            merged_height = height // merge
            merged_width = width // merge
            block_rows = torch.arange(merged_height, device=frequency_table.device)
            block_columns = torch.arange(merged_width, device=frequency_table.device)
            intra_rows = torch.arange(merge, device=frequency_table.device)
            intra_columns = torch.arange(merge, device=frequency_table.device)
            rows = (
                block_rows[:, None, None, None] * merge
                + intra_rows[None, None, :, None]
            ).expand(merged_height, merged_width, merge, merge)
            columns = (
                block_columns[None, :, None, None] * merge
                + intra_columns[None, None, None, :]
            ).expand(merged_height, merged_width, merge, merge)
            image_coordinates = torch.stack(
                (rows.reshape(-1), columns.reshape(-1)), dim=-1
            )
            coordinates.append(image_coordinates.repeat(temporal, 1))
        return frequency_table[torch.cat(coordinates)].flatten(1)

    @staticmethod
    def _build_attention_metadata(
        grids: Sequence[tuple[int, int, int]], device: torch.device
    ) -> VisionAttentionMetadata:
        lengths = tuple(
            height * width
            for temporal, height, width in grids
            for _ in range(temporal)
        )
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
        return VisionAttentionMetadata(
            cu_seqlens=cu_seqlens,
            segment_bounds=tuple(zip(boundaries, boundaries[1:])),
            max_seqlen=max(lengths),
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> VisionFeatureBundle:
        grids = self._validate_inputs(pixel_values, grid_thw)
        hidden_states = self.patch_embed(pixel_values)
        position_embeddings = self.fast_pos_embed_interpolate(grid_thw, grids)
        position_embeddings = position_embeddings.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states + position_embeddings

        rotary = self.rot_pos_emb(grid_thw, grids).to(device=hidden_states.device)
        rotary = rotary.reshape(hidden_states.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos_sin = (rotary.cos(), rotary.sin())
        attention_metadata = self._build_attention_metadata(
            grids, hidden_states.device
        )

        deepstack_slots = {
            index: (output_index, self.deepstack_merger_list[output_index])
            for output_index, index in enumerate(self.deepstack_visual_indexes)
        }
        deepstack: list[torch.Tensor | None] = [
            None for _ in self.deepstack_visual_indexes
        ]
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, attention_metadata, cos_sin)
            slot = deepstack_slots.get(layer_index)
            if slot is not None:
                output_index, merger = slot
                deepstack[output_index] = merger(hidden_states)
        final_embedding = self.merger(hidden_states)

        if any(embedding is None for embedding in deepstack):
            raise RuntimeError("not all configured DeepStack layers produced features")
        deepstack_embeddings = tuple(
            embedding for embedding in deepstack if embedding is not None
        )

        expected_rows = sum(
            temporal
            * (height // self.spatial_merge_size)
            * (width // self.spatial_merge_size)
            for temporal, height, width in grids
        )
        if final_embedding.shape[0] != expected_rows or any(
            embedding.shape != final_embedding.shape
            for embedding in deepstack_embeddings
        ):
            raise RuntimeError("vision final/DeepStack output row count is inconsistent")
        return VisionFeatureBundle(final_embedding, deepstack_embeddings)


# Short aliases are intentionally kept local to this model namespace while checkpoint
# names remain identical to the official model.visual.* module hierarchy.
VisionMLP = Qwen3VLVisionMLP
VisionAttention = Qwen3VLVisionAttention
VisionBlock = Qwen3VLVisionBlock


__all__ = [
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionRotaryEmbedding",
    "VisionAttention",
    "VisionBlock",
    "VisionMLP",
    "apply_rotary_pos_emb_vision",
]
