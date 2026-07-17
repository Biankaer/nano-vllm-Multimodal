from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch

from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan


GridTHW = tuple[int, int, int]


def build_vision_feature_key(
    model_fingerprint: str,
    processor_fingerprint: str,
    image_fingerprint: str,
    grid_thw: GridTHW,
) -> str:
    """Create a stable identity for the exact vision features used by the decoder."""

    fingerprints = (model_fingerprint, processor_fingerprint, image_fingerprint)
    if any(not value for value in fingerprints):
        raise ValueError("model, processor, and image fingerprints cannot be empty")
    grid = tuple(int(value) for value in grid_thw)
    if len(grid) != 3 or any(value < 1 for value in grid):
        raise ValueError("grid_thw must contain three positive dimensions")

    digest = hashlib.sha256()
    for value in fingerprints:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for value in grid:
        digest.update(value.to_bytes(8, "little", signed=False))
    return f"vision:{digest.hexdigest()}"


def build_qwen25_vl_prompt_metadata(
    *,
    token_ids: Sequence[int],
    image_grid_thw: Sequence[GridTHW],
    image_feature_keys: Sequence[str],
    image_token_id: int,
    spatial_merge_size: int,
) -> MultimodalPromptMetadata:
    """Build image/text MRoPE positions for one unpadded Qwen2.5-VL prompt."""

    prompt_tokens = tuple(int(token_id) for token_id in token_ids)
    if not prompt_tokens:
        raise ValueError("token_ids cannot be empty")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    if len(image_grid_thw) != len(image_feature_keys):
        raise ValueError("every image grid must have one image feature key")

    if not image_grid_thw:
        positions = tuple(range(len(prompt_tokens)))
        metadata = MultimodalPromptMetadata(
            position_ids=(positions, positions, positions),
            rope_delta=0,
        )
        metadata.validate(len(prompt_tokens))
        return metadata

    axes: list[list[int]] = [[], [], []]
    visual_spans: list[VisualTokenSpan] = []
    token_cursor = 0
    max_position = -1

    for grid, feature_key in zip(image_grid_thw, image_feature_keys):
        grid_t, grid_h, grid_w = (int(value) for value in grid)
        if grid_t < 1 or grid_h < 1 or grid_w < 1:
            raise ValueError("image grid dimensions must be positive")
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError("image grid height and width must be divisible by spatial_merge_size")
        if not feature_key:
            raise ValueError("image feature key cannot be empty")

        try:
            image_start = prompt_tokens.index(image_token_id, token_cursor)
        except ValueError as exc:
            raise ValueError("image placeholder tokens do not match image grids") from exc

        text_length = image_start - token_cursor
        text_start_position = max_position + 1
        _append_text_positions(axes, text_start_position, text_length)
        if text_length:
            max_position = text_start_position + text_length - 1

        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        visual_token_count = grid_t * llm_grid_h * llm_grid_w
        image_end = image_start + visual_token_count
        if image_end > len(prompt_tokens) or any(
            token_id != image_token_id for token_id in prompt_tokens[image_start:image_end]
        ):
            raise ValueError("image placeholder token count does not match merged image grid")

        visual_start_position = max_position + 1
        for _ in range(grid_t):
            for height_index in range(llm_grid_h):
                for width_index in range(llm_grid_w):
                    # Static images use a zero temporal interval in Qwen2.5-VL.
                    axes[0].append(visual_start_position)
                    axes[1].append(visual_start_position + height_index)
                    axes[2].append(visual_start_position + width_index)

        visual_max = visual_start_position + max(llm_grid_h, llm_grid_w) - 1
        max_position = max(max_position, visual_max)
        visual_spans.append(
            VisualTokenSpan(
                start=image_start,
                length=visual_token_count,
                feature_key=feature_key,
            )
        )
        token_cursor = image_end

    if image_token_id in prompt_tokens[token_cursor:]:
        raise ValueError("image placeholder tokens do not match image grids")

    trailing_text_length = len(prompt_tokens) - token_cursor
    trailing_text_start = max_position + 1
    _append_text_positions(axes, trailing_text_start, trailing_text_length)
    if trailing_text_length:
        max_position = trailing_text_start + trailing_text_length - 1

    metadata = MultimodalPromptMetadata(
        position_ids=tuple(tuple(axis) for axis in axes),
        rope_delta=max_position + 1 - len(prompt_tokens),
        visual_spans=tuple(visual_spans),
    )
    metadata.validate(len(prompt_tokens))
    return metadata


def _append_text_positions(axes: list[list[int]], start: int, length: int) -> None:
    positions = list(range(start, start + length))
    for axis in axes:
        axis.extend(positions)


def merge_visual_embeddings(
    input_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    *,
    image_token_id: int,
) -> torch.Tensor:
    """Replace image placeholder rows while leaving source embeddings unchanged."""

    if input_ids.ndim != 1:
        raise ValueError("input_ids must have shape [num_tokens]")
    if token_embeddings.ndim != 2 or visual_embeddings.ndim != 2:
        raise ValueError("token and visual embeddings must both be rank-2 tensors")
    if token_embeddings.shape[0] != input_ids.shape[0]:
        raise ValueError("input_ids and token embeddings must contain the same number of tokens")
    if token_embeddings.shape[1] != visual_embeddings.shape[1]:
        raise ValueError("token and visual embedding hidden size must match")
    if token_embeddings.device != visual_embeddings.device:
        raise ValueError("token and visual embeddings must be on the same device")
    if token_embeddings.dtype != visual_embeddings.dtype:
        raise ValueError("token and visual embeddings must have the same dtype")

    image_mask = input_ids.to(device=token_embeddings.device) == image_token_id
    num_image_tokens = int(image_mask.sum().item())
    num_image_features = visual_embeddings.shape[0]
    if num_image_tokens != num_image_features:
        raise ValueError(
            "Image tokens and features do not match: "
            f"tokens={num_image_tokens}, features={num_image_features}"
        )

    merged = token_embeddings.clone()
    merged[image_mask] = visual_embeddings
    return merged
