import builtins
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanovllm.models.qwen3_vl_vision import (
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionRotaryEmbedding,
    _resolve_flash_attn_varlen_func,
    apply_rotary_pos_emb_vision,
)
from nanovllm.multimodal.runtime import VisionFeatureBundle


def tiny_config(**overrides):
    values = dict(
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        depth=3,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=6,
        num_position_embeddings=9,
        deepstack_visual_indexes=[0, 1, 2],
        hidden_act="gelu_pytorch_tanh",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def independent_abs_pos_reference(weight, grid_thw, merge_size):
    side = math.isqrt(weight.shape[0])
    table = weight.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
    outputs = []
    for temporal, height, width in grid_thw.tolist():
        interpolated = F.interpolate(
            table.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).to(weight.dtype)
        interpolated = interpolated.squeeze(0).permute(1, 2, 0)
        permuted = (
            interpolated.view(
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                -1,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(height * width, -1)
        )
        outputs.append(permuted.repeat(temporal, 1))
    return torch.cat(outputs)


def transformers_abs_pos_reference(weight, grid_thw, merge_size):
    side = math.isqrt(weight.shape[0])
    index_lists = [[] for _ in range(4)]
    weight_lists = [[] for _ in range(4)]
    for _, height, width in grid_thw.tolist():
        height_positions = torch.linspace(0, side - 1, height)
        width_positions = torch.linspace(0, side - 1, width)
        height_floor = height_positions.int()
        width_floor = width_positions.int()
        height_ceil = (height_floor + 1).clip(max=side - 1)
        width_ceil = (width_floor + 1).clip(max=side - 1)
        height_fraction = height_positions - height_floor
        width_fraction = width_positions - width_floor
        base_floor = height_floor * side
        base_ceil = height_ceil * side
        indices = (
            (base_floor[:, None] + width_floor[None]).flatten(),
            (base_floor[:, None] + width_ceil[None]).flatten(),
            (base_ceil[:, None] + width_floor[None]).flatten(),
            (base_ceil[:, None] + width_ceil[None]).flatten(),
        )
        weights = (
            ((1 - height_fraction)[:, None] * (1 - width_fraction)[None]).flatten(),
            ((1 - height_fraction)[:, None] * width_fraction[None]).flatten(),
            (height_fraction[:, None] * (1 - width_fraction)[None]).flatten(),
            (height_fraction[:, None] * width_fraction[None]).flatten(),
        )
        for destination, values in zip(index_lists, indices):
            destination.extend(values.tolist())
        for destination, values in zip(weight_lists, weights):
            destination.extend(values.tolist())
    indices = torch.tensor(index_lists, dtype=torch.long)
    weights = torch.tensor(weight_lists, dtype=weight.dtype)
    terms = weight[indices] * weights[:, :, None]
    interpolated = terms[0] + terms[1] + terms[2] + terms[3]
    outputs = []
    offset = 0
    for temporal, height, width in grid_thw.tolist():
        count = height * width
        position = interpolated[offset : offset + count]
        offset += count
        outputs.append(
            position.repeat(temporal, 1)
            .view(
                temporal, height // merge_size, merge_size,
                width // merge_size, merge_size, -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
    return torch.cat(outputs)


class Qwen3VLVisionPrimitiveTests(unittest.TestCase):
    def test_patch_embed_matches_direct_conv3d(self):
        config = tiny_config(hidden_size=4)
        module = Qwen3VLVisionPatchEmbed(config)
        torch.manual_seed(4)
        pixels = torch.randn(5, 24)

        actual = module(pixels)
        expected = F.conv3d(
            pixels.reshape(-1, 3, 2, 2, 2),
            module.proj.weight,
            module.proj.bias,
            stride=(2, 2, 2),
        ).reshape(-1, 4)

        torch.testing.assert_close(actual, expected)

    def test_learned_abs_pos_interpolation_and_merge_permutation(self):
        model = Qwen3VLVisionModel(tiny_config(depth=0, deepstack_visual_indexes=[]))
        with torch.no_grad():
            model.pos_embed.weight.copy_(
                torch.arange(9 * 8, dtype=torch.float32).reshape(9, 8)
            )
        grid = torch.tensor([[1, 2, 4], [2, 4, 2]])

        actual = model.fast_pos_embed_interpolate(grid)
        expected = independent_abs_pos_reference(model.pos_embed.weight, grid, 2)

        torch.testing.assert_close(actual, expected)

    def test_bfloat16_abs_pos_matches_transformers_addition_order_exactly(self):
        config = tiny_config(
            hidden_size=16,
            num_heads=2,
            num_position_embeddings=16,
            depth=0,
            deepstack_visual_indexes=[],
        )
        model = Qwen3VLVisionModel(config).to(dtype=torch.bfloat16)
        torch.manual_seed(41)
        with torch.no_grad():
            model.pos_embed.weight.copy_(torch.randn_like(model.pos_embed.weight))
        grid = torch.tensor([[1, 8, 8]])

        actual = model.fast_pos_embed_interpolate(grid)
        expected = transformers_abs_pos_reference(model.pos_embed.weight, grid, 2)

        self.assertTrue(torch.equal(actual, expected))

    def test_vision_rope_positions_and_rotation_match_independent_reference(self):
        model = Qwen3VLVisionModel(tiny_config(depth=0, deepstack_visual_indexes=[]))
        grid = torch.tensor([[1, 2, 4]])
        frequencies = model.rot_pos_emb(grid)

        # Merge-block order is (0,0),(0,1),(1,0),(1,1), then the next block.
        coords = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
        )
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, 2, 2, dtype=torch.float32) / 2)
        )
        expected_freqs = (coords.float().unsqueeze(-1) * inv_freq).flatten(1)
        torch.testing.assert_close(frequencies, expected_freqs)

        torch.manual_seed(2)
        query = torch.randn(8, 2, 4, dtype=torch.float16)
        key = torch.randn(8, 2, 4, dtype=torch.float16)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = embedding.cos(), embedding.sin()
        actual_q, actual_k = apply_rotary_pos_emb_vision(query, key, cos, sin)

        def rotate_half(value):
            return torch.cat((-value[..., 2:], value[..., :2]), dim=-1)

        expected_q = (
            query.float() * cos[:, None].float()
            + rotate_half(query.float()) * sin[:, None].float()
        ).to(query.dtype)
        expected_k = (
            key.float() * cos[:, None].float()
            + rotate_half(key.float()) * sin[:, None].float()
        ).to(key.dtype)
        torch.testing.assert_close(actual_q, expected_q)
        torch.testing.assert_close(actual_k, expected_k)

    def test_rotary_embedding_buffer_is_nonpersistent(self):
        rotary = Qwen3VLVisionRotaryEmbedding(dim=8)
        self.assertNotIn("inv_freq", rotary.state_dict())

    def test_rotary_frequency_buffer_stays_float32_like_hf_from_pretrained(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                rotary = Qwen3VLVisionRotaryEmbedding(dim=8).to(dtype=dtype)
                self.assertEqual(rotary.inv_freq.dtype, torch.float32)
                self.assertEqual(rotary(4).dtype, torch.float32)


class Qwen3VLVisionAttentionTests(unittest.TestCase):
    def test_sdpa_matches_independent_segmented_noncausal_reference(self):
        torch.manual_seed(3)
        attention = Qwen3VLVisionAttention(tiny_config(), "sdpa").eval()
        hidden = torch.randn(5, 8)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        cos = torch.ones(5, 4)
        sin = torch.zeros(5, 4)

        actual = attention(hidden, cu_seqlens, (cos, sin))
        qkv = attention.qkv(hidden).reshape(5, 3, 2, 4).permute(1, 0, 2, 3)
        outputs = []
        for start, end in ((0, 3), (3, 5)):
            q = qkv[0, start:end].transpose(0, 1).unsqueeze(0)
            k = qkv[1, start:end].transpose(0, 1).unsqueeze(0)
            value = qkv[2, start:end].transpose(0, 1).unsqueeze(0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(4)
            probs = scores.softmax(dim=-1)
            outputs.append(torch.matmul(probs, value).transpose(1, 2).reshape(end - start, 8))
        expected = attention.proj(torch.cat(outputs))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        changed = hidden.clone()
        changed[2].add_(10)
        changed_output = attention(changed, cu_seqlens, (cos, sin))
        self.assertFalse(torch.allclose(actual[0], changed_output[0]))

    def test_flash_backend_is_lazy_and_cpu_forward_has_clear_error(self):
        attention = Qwen3VLVisionAttention(tiny_config(), "flash_attention_2")
        original_import = builtins.__import__

        def reject_flash(name, *args, **kwargs):
            if name.startswith("flash_attn"):
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_flash):
            with self.assertRaisesRegex(RuntimeError, "FlashAttention|CUDA"):
                attention(
                    torch.randn(2, 8),
                    torch.tensor([0, 2], dtype=torch.int32),
                    (torch.ones(2, 4), torch.zeros(2, 4)),
                )

    def test_flash_dependency_resolution_has_clear_import_error(self):
        original_import = builtins.__import__

        def reject_flash(name, *args, **kwargs):
            if name.startswith("flash_attn"):
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_flash):
            with self.assertRaisesRegex(RuntimeError, "install flash-attn"):
                _resolve_flash_attn_varlen_func()

    def test_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "attention backend"):
            Qwen3VLVisionAttention(tiny_config(), "eager")


class Qwen3VLVisionModelTests(unittest.TestCase):
    def test_tiny_forward_returns_final_and_three_deepstack_features(self):
        torch.manual_seed(8)
        model = Qwen3VLVisionModel(tiny_config(), attention_backend="sdpa").eval()
        grid = torch.tensor([[1, 2, 4]])
        pixels = torch.randn(8, 24)

        output = model(pixels, grid)

        self.assertIsInstance(output, VisionFeatureBundle)
        self.assertEqual(output.final_embedding.shape, (2, 6))
        self.assertEqual(len(output.deepstack_embeddings), 3)
        for feature in output.deepstack_embeddings:
            self.assertEqual(feature.shape, (2, 6))

    def test_multiple_images_are_attention_isolated(self):
        torch.manual_seed(11)
        model = Qwen3VLVisionModel(
            tiny_config(depth=2, deepstack_visual_indexes=[0]),
            attention_backend="sdpa",
        ).eval()
        grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
        pixels = torch.randn(12, 24)
        changed = pixels.clone()
        changed[4:].add_(100)

        original_output = model(pixels, grid)
        changed_output = model(changed, grid)

        torch.testing.assert_close(
            original_output.final_embedding[:1], changed_output.final_embedding[:1]
        )
        torch.testing.assert_close(
            original_output.deepstack_embeddings[0][:1],
            changed_output.deepstack_embeddings[0][:1],
        )
        self.assertFalse(
            torch.allclose(
                original_output.final_embedding[1:],
                changed_output.final_embedding[1:],
            )
        )

    def test_multiblock_forward_only_materializes_grid_metadata_once(self):
        model = Qwen3VLVisionModel(tiny_config(), attention_backend="sdpa").eval()
        grid = torch.tensor([[1, 2, 4]])
        pixels = torch.randn(8, 24)

        original_cpu = torch.Tensor.cpu
        cpu_calls = 0

        def count_cpu(tensor, *args, **kwargs):
            nonlocal cpu_calls
            cpu_calls += 1
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", count_cpu):
            model(pixels, grid)

        self.assertEqual(cpu_calls, 1)

    def test_deepstack_features_follow_configured_index_order(self):
        model = Qwen3VLVisionModel(
            tiny_config(deepstack_visual_indexes=[2, 0, 1]),
            attention_backend="sdpa",
        ).eval()
        for value, merger in enumerate(model.deepstack_merger_list):
            merger.forward = (
                lambda hidden_states, value=value: torch.full(
                    (hidden_states.shape[0] // 4, 6),
                    float(value),
                    device=hidden_states.device,
                )
            )

        output = model(torch.randn(8, 24), torch.tensor([[1, 2, 4]]))

        self.assertEqual(
            [feature[0, 0].item() for feature in output.deepstack_embeddings],
            [0.0, 1.0, 2.0],
        )

    def test_state_dict_names_match_checkpoint_after_model_visual_prefix_strip(self):
        model = Qwen3VLVisionModel(tiny_config())
        keys = set(model.state_dict())
        expected = {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight"}
        for block_index in range(3):
            for norm in ("norm1", "norm2"):
                expected.update(
                    {
                        f"blocks.{block_index}.{norm}.weight",
                        f"blocks.{block_index}.{norm}.bias",
                    }
                )
            for module in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
                expected.update(
                    {
                        f"blocks.{block_index}.{module}.weight",
                        f"blocks.{block_index}.{module}.bias",
                    }
                )
        for prefix in (
            "merger",
            "deepstack_merger_list.0",
            "deepstack_merger_list.1",
            "deepstack_merger_list.2",
        ):
            for module in ("norm", "linear_fc1", "linear_fc2"):
                expected.update({f"{prefix}.{module}.weight", f"{prefix}.{module}.bias"})
        self.assertEqual(keys, expected)
        self.assertNotIn("rotary_pos_emb.inv_freq", keys)
        clone = Qwen3VLVisionModel(tiny_config())
        result = clone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_validates_grid_patch_rows_width_and_merge_divisibility(self):
        model = Qwen3VLVisionModel(tiny_config(), attention_backend="sdpa")
        valid_pixels = torch.randn(8, 24)
        cases = (
            (valid_pixels, torch.tensor([1, 2, 4]), "shape"),
            (valid_pixels, torch.tensor([[0, 2, 4]]), "positive"),
            (valid_pixels, torch.tensor([[1, 3, 4]]), "divisible"),
            (valid_pixels[:-1], torch.tensor([[1, 2, 4]]), "patch rows"),
            (torch.randn(8, 23), torch.tensor([[1, 2, 4]]), "width"),
        )
        for pixels, grid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    model(pixels, grid)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for SDPA/Flash parity"
    )
    def test_flash_matches_sdpa_when_flash_attn_is_available(self):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            self.skipTest("flash-attn is unavailable")
        config = tiny_config(depth=1, deepstack_visual_indexes=[0])
        torch.manual_seed(19)
        sdpa = Qwen3VLVisionModel(config, attention_backend="sdpa").cuda().half().eval()
        flash = Qwen3VLVisionModel(config, attention_backend="flash_attention_2").cuda().half().eval()
        flash.load_state_dict(sdpa.state_dict())
        grid = torch.tensor([[1, 2, 4]], device="cuda")
        pixels = torch.randn(8, 24, device="cuda", dtype=torch.float16)

        expected = sdpa(pixels, grid)
        actual = flash(pixels, grid)
        torch.testing.assert_close(actual.final_embedding, expected.final_embedding, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            actual.deepstack_embeddings[0],
            expected.deepstack_embeddings[0],
            rtol=2e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
