from __future__ import annotations

from collections.abc import Sequence

from nanovllm.multimodal.qwen25_vl_runtime import build_vision_feature_key
from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan


GridTHW = tuple[int, int, int]


def build_qwen3_vl_prompt_metadata(
    *,
    token_ids: Sequence[int],
    image_grid_thw: Sequence[GridTHW],
    image_feature_keys: Sequence[str],
    image_token_id: int,
    spatial_merge_size: int,
) -> MultimodalPromptMetadata:
    """Build Qwen3-VL static-image MRoPE positions for one unpadded prompt."""

    prompt_tokens = tuple(int(token_id) for token_id in token_ids)
    if not prompt_tokens:
        raise ValueError("token_ids cannot be empty")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    if len(image_grid_thw) != len(image_feature_keys):
        raise ValueError("every image grid must have one image feature key")

    if not image_grid_thw:
        if image_token_id in prompt_tokens:
            raise ValueError("image placeholder tokens do not match image grids")
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

    for raw_grid, feature_key in zip(image_grid_thw, image_feature_keys):
        grid_t, grid_h, grid_w = (int(value) for value in raw_grid)
        if grid_t < 1 or grid_h < 1 or grid_w < 1:
            raise ValueError("image grid dimensions must be positive")
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError(
                "image grid height and width must be divisible by spatial_merge_size"
            )
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
            token_id != image_token_id
            for token_id in prompt_tokens[image_start:image_end]
        ):
            raise ValueError(
                "image placeholder token count does not match merged image grid"
            )

        visual_start_position = max_position + 1
        for temporal_index in range(grid_t):
            for height_index in range(llm_grid_h):
                for width_index in range(llm_grid_w):
                    axes[0].append(visual_start_position + temporal_index)
                    axes[1].append(visual_start_position + height_index)
                    axes[2].append(visual_start_position + width_index)

        visual_max = visual_start_position + max(
            grid_t,
            llm_grid_h,
            llm_grid_w,
        ) - 1
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
    positions = range(start, start + length)
    for axis in axes:
        axis.extend(positions)


__all__ = ["build_qwen3_vl_prompt_metadata", "build_vision_feature_key"]
