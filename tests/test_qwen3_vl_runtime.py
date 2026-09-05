import unittest

import torch

from nanovllm.layers.multimodal_rotary_embedding import (
    MultimodalRotaryEmbedding,
    apply_multimodal_rotary_emb,
)
from nanovllm.multimodal.qwen3_vl_runtime import build_qwen3_vl_prompt_metadata
from nanovllm.multimodal.runtime import VisualTokenSpan


def reference_qwen3_static_image_positions(token_ids, grids, image_token_id, merge_size):
    axes = [[], [], []]
    start = 0
    for grid in grids:
        end = token_ids.index(image_token_id, start)
        base = max((max(axis) for axis in axes if axis), default=-1) + 1
        text_length = end - start
        for axis in axes:
            axis.extend(range(base, base + text_length))

        grid_t, grid_h, grid_w = grid
        llm_h, llm_w = grid_h // merge_size, grid_w // merge_size
        visual_base = base + text_length
        for temporal in range(grid_t):
            for height in range(llm_h):
                for width in range(llm_w):
                    axes[0].append(visual_base + temporal)
                    axes[1].append(visual_base + height)
                    axes[2].append(visual_base + width)
        start = end + grid_t * llm_h * llm_w

    base = max((max(axis) for axis in axes if axis), default=-1) + 1
    trailing = len(token_ids) - start
    for axis in axes:
        axis.extend(range(base, base + trailing))
    return tuple(tuple(axis) for axis in axes)


def reference_interleaved_rotation(query, key, positions, *, base, sections):
    head_dim = query.shape[-1]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    frequencies = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    selected = frequencies[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        index = slice(offset, sections[axis] * 3, 3)
        selected[..., index] = frequencies[axis, ..., index]
    embeddings = torch.cat((selected, selected), dim=-1)
    cos = embeddings.cos().unsqueeze(1)
    sin = embeddings.sin().unsqueeze(1)

    def rotate_half(value):
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


class Qwen3VLPositionTests(unittest.TestCase):
    def test_static_image_positions_match_independent_official_reference(self):
        token_ids = (10, 99, 99, 99, 99, 20, 99, 99, 30)
        grids = ((1, 4, 4), (1, 2, 4))

        metadata = build_qwen3_vl_prompt_metadata(
            token_ids=token_ids,
            image_grid_thw=grids,
            image_feature_keys=("image-a", "image-b"),
            image_token_id=99,
            spatial_merge_size=2,
        )

        expected = reference_qwen3_static_image_positions(token_ids, grids, 99, 2)
        self.assertEqual(metadata.position_ids, expected)
        self.assertEqual(metadata.rope_delta, max(max(axis) for axis in expected) + 1 - len(token_ids))
        self.assertEqual(
            metadata.visual_spans,
            (
                VisualTokenSpan(start=1, length=4, feature_key="image-a"),
                VisualTokenSpan(start=6, length=2, feature_key="image-b"),
            ),
        )

    def test_temporal_axis_matches_qwen3_grid_index_semantics(self):
        metadata = build_qwen3_vl_prompt_metadata(
            token_ids=(99, 99),
            image_grid_thw=((2, 2, 2),),
            image_feature_keys=("image-a",),
            image_token_id=99,
            spatial_merge_size=2,
        )

        self.assertEqual(metadata.position_ids, ((0, 1), (0, 0), (0, 0)))
        self.assertEqual(metadata.rope_delta, 0)

    def test_rejects_extra_placeholder_tokens(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            build_qwen3_vl_prompt_metadata(
                token_ids=(99, 99),
                image_grid_thw=((1, 2, 2),),
                image_feature_keys=("image-a",),
                image_token_id=99,
                spatial_merge_size=2,
            )


class Qwen3InterleavedMRoPETests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_interleaved_inv_freq_uses_cpu_reference_when_constructed_on_cuda(self):
        head_dim = 128
        base = 5_000_000.0
        expected = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu")
                / head_dim
            )
        )
        original_device = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            rotary = MultimodalRotaryEmbedding(
                head_dim=head_dim,
                base=base,
                sections=(24, 20, 20),
                strategy="interleaved",
            )
        finally:
            torch.set_default_device(original_device)

        self.assertEqual(rotary.inv_freq.device.type, "cuda")
        self.assertTrue(torch.equal(rotary.inv_freq.cpu(), expected))

    def test_bfloat16_interleaved_rotation_matches_hf_dtype_order_exactly(self):
        torch.manual_seed(29)
        query = torch.randn(5, 2, 128, dtype=torch.bfloat16)
        key = torch.randn(5, 1, 128, dtype=torch.bfloat16)
        cos = torch.randn(3, 5, 128, dtype=torch.bfloat16)
        sin = torch.randn(3, 5, 128, dtype=torch.bfloat16)
        sections = (24, 20, 20)
        half_dimension = cos.shape[-1] // 2
        selected_cos_half = cos[0, ..., :half_dimension].clone()
        selected_sin_half = sin[0, ..., :half_dimension].clone()
        for axis, offset in ((1, 1), (2, 2)):
            frequency_indices = slice(offset, sections[axis] * 3, 3)
            selected_cos_half[..., frequency_indices] = cos[
                axis, ..., :half_dimension
            ][..., frequency_indices]
            selected_sin_half[..., frequency_indices] = sin[
                axis, ..., :half_dimension
            ][..., frequency_indices]
        selected_cos = torch.cat((selected_cos_half, selected_cos_half), dim=-1)
        selected_sin = torch.cat((selected_sin_half, selected_sin_half), dim=-1)
        selected_cos = selected_cos.unsqueeze(1)
        selected_sin = selected_sin.unsqueeze(1)

        def rotate_half(value):
            first, second = value.chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

        expected_query = (
            query * selected_cos + rotate_half(query) * selected_sin
        )
        expected_key = key * selected_cos + rotate_half(key) * selected_sin

        actual_query, actual_key = apply_multimodal_rotary_emb(
            query, key, cos, sin, sections, strategy="interleaved"
        )

        self.assertTrue(torch.equal(actual_query, expected_query))
        self.assertTrue(torch.equal(actual_key, expected_key))

    def test_interleaved_layout_has_fixed_golden_output_and_preserves_inputs(self):
        query = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]])
        key = torch.tensor([[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]])
        cos = torch.tensor(
            [
                [[1.0, 101.0, 102.0, 4.0, 1.0, 101.0, 102.0, 4.0]],
                [[201.0, 2.0, 203.0, 204.0, 201.0, 2.0, 203.0, 204.0]],
                [[301.0, 302.0, 3.0, 304.0, 301.0, 302.0, 3.0, 304.0]],
            ]
        )
        sin = torch.tensor(
            [
                [[10.0, 1001.0, 1002.0, 40.0, 10.0, 1001.0, 1002.0, 40.0]],
                [[2001.0, 20.0, 2003.0, 2004.0, 2001.0, 20.0, 2003.0, 2004.0]],
                [[3001.0, 3002.0, 30.0, 3004.0, 3001.0, 3002.0, 30.0, 3004.0]],
            ]
        )
        inputs_before = tuple(value.clone() for value in (query, key, cos, sin))

        actual_query, actual_key = apply_multimodal_rotary_emb(
            query,
            key,
            cos,
            sin,
            sections=(2, 1, 1),
            strategy="interleaved",
        )

        # Interleaving selects axes [T, H, W, T] for frequency indices 0..3,
        # then repeats that selection for the second half of each head.
        expected_query = torch.tensor(
            [[[-49.0, -116.0, -201.0, -304.0, 15.0, 52.0, 111.0, 192.0]]]
        )
        expected_key = torch.tensor(
            [[[-32.0, -46.0, -42.0, -20.0, 84.0, 146.0, 186.0, 204.0]]]
        )
        torch.testing.assert_close(actual_query, expected_query)
        torch.testing.assert_close(actual_key, expected_key)
        for actual, before in zip((query, key, cos, sin), inputs_before):
            torch.testing.assert_close(actual, before)

    def test_interleaved_frequency_layout_matches_independent_reference(self):
        torch.manual_seed(17)
        sections = (24, 20, 20)
        positions = torch.tensor(
            (
                (0, 1, 4, 7),
                (0, 2, 5, 8),
                (0, 3, 6, 9),
            )
        )
        query = torch.randn(4, 2, 128)
        key = torch.randn(4, 1, 128)
        rotary = MultimodalRotaryEmbedding(
            head_dim=128,
            base=1_000_000.0,
            sections=sections,
            strategy="interleaved",
        )
        inputs_before = tuple(value.clone() for value in (positions, query, key))

        actual_query, actual_key = rotary(positions, query, key)
        expected_query, expected_key = reference_interleaved_rotation(
            query,
            key,
            positions,
            base=1_000_000.0,
            sections=sections,
        )

        torch.testing.assert_close(actual_query, expected_query)
        torch.testing.assert_close(actual_key, expected_key)
        for actual, before in zip((positions, query, key), inputs_before):
            torch.testing.assert_close(actual, before)

    def test_default_strategy_remains_explicitly_chunked(self):
        positions = torch.arange(3).expand(3, -1)
        query = torch.randn(3, 2, 128)
        key = torch.randn(3, 1, 128)
        arguments = dict(
            head_dim=128,
            base=1_000_000.0,
            sections=(16, 24, 24),
        )

        default = MultimodalRotaryEmbedding(**arguments)(positions, query, key)
        explicit = MultimodalRotaryEmbedding(**arguments, strategy="chunked")(
            positions,
            query,
            key,
        )

        torch.testing.assert_close(default[0], explicit[0])
        torch.testing.assert_close(default[1], explicit[1])

    def test_rejects_unknown_frequency_layout_strategy(self):
        with self.assertRaisesRegex(ValueError, "strategy"):
            MultimodalRotaryEmbedding(
                head_dim=128,
                base=1_000_000.0,
                sections=(24, 20, 20),
                strategy="striped",
            )


if __name__ == "__main__":
    unittest.main()
