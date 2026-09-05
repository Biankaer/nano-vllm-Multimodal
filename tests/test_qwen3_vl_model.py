import unittest
import tempfile
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

import nanovllm.models.qwen3_vl as qwen3_vl_module
from nanovllm.models.qwen3_vl import Qwen3VLForCausalLM
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


def tiny_qwen3_vl_text_config(**overrides):
    values = dict(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=3,
        intermediate_size=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=5_000_000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        vocab_size=32,
        tie_word_embeddings=True,
        attention_bias=False,
        head_dim=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _ResidualAddingLayer(nn.Module):
    """A controllable stand-in that preserves NanoInfer's fused residual form."""

    def __init__(self, layer_delta: float):
        super().__init__()
        self.layer_delta = layer_delta
        self.inputs = []

    def forward(self, positions, hidden_states, residual):
        complete_input = (
            hidden_states if residual is None else hidden_states + residual
        )
        self.inputs.append(complete_input.detach().clone())
        return torch.full_like(complete_input, self.layer_delta), complete_input


class _FusedResidualIdentityNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states + residual, None


class Qwen3VLModelStructureTests(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_hf_aligned_linear_pads_prefix_to_full_context_shape(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weight = torch.eye(2)
        cu_q = torch.tensor([0, 2], dtype=torch.int32)
        cu_k = torch.tensor([0, 5], dtype=torch.int32)
        set_context(
            True,
            cu_q,
            cu_k,
            2,
            5,
            full_context_lens=torch.tensor([8], dtype=torch.int32),
            query_start_lens=torch.tensor([3], dtype=torch.int32),
        )

        observed_shapes = []
        original_linear = torch.nn.functional.linear

        def capture_shape(value, projection, bias=None):
            observed_shapes.append(tuple(value.shape))
            return original_linear(value, projection, bias)

        with patch("nanovllm.models.qwen3_vl.nn.functional.linear", side_effect=capture_shape):
            actual = qwen3_vl_module.hf_aligned_linear(hidden, weight)

        self.assertEqual(observed_shapes, [(1, 8, 2)])
        torch.testing.assert_close(actual, hidden)

    def test_hf_aligned_linear_keeps_packed_requests_numerically_isolated(self):
        hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        weight = torch.eye(2)
        cu = torch.tensor([0, 2, 6], dtype=torch.int32)
        set_context(True, cu, cu, 4, 4)

        observed_shapes = []
        original_linear = torch.nn.functional.linear

        def capture_shape(value, projection, bias=None):
            observed_shapes.append(tuple(value.shape))
            return original_linear(value, projection, bias)

        with patch("nanovllm.models.qwen3_vl.nn.functional.linear", side_effect=capture_shape):
            actual = qwen3_vl_module.hf_aligned_linear(hidden, weight)

        self.assertEqual(observed_shapes, [(1, 2, 2), (1, 4, 2)])
        torch.testing.assert_close(actual, hidden)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_sdpa_language_backend_uses_hf_aligned_projection_and_mlp_paths(self, *_):
        model = Qwen3VLForCausalLM(
            tiny_qwen3_vl_text_config(),
            attention_backend="sdpa",
        )
        layer = model.model.language_model.layers[0]

        self.assertEqual(layer.self_attn.attention_backend, "sdpa")
        self.assertEqual(layer.self_attn.projection_mode, "separate")
        self.assertEqual(layer.self_attn.output_projection_mode, "hf_aligned")
        self.assertEqual(layer.mlp.projection_mode, "separate_3d")
        self.assertEqual(type(layer.self_attn.attn).__name__, "SDPAAttention")

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_flash_language_backend_preserves_fused_production_paths(self, *_):
        model = Qwen3VLForCausalLM(
            tiny_qwen3_vl_text_config(),
            attention_backend="flash_attention_2",
        )
        layer = model.model.language_model.layers[0]

        self.assertEqual(layer.self_attn.attention_backend, "flash_attention_2")
        self.assertEqual(layer.self_attn.projection_mode, "fused")
        self.assertEqual(layer.self_attn.output_projection_mode, "fused")
        self.assertEqual(layer.mlp.projection_mode, "fused")
        self.assertEqual(type(layer.self_attn.attn).__name__, "Attention")

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_rejects_unknown_language_attention_backend(self, *_):
        with self.assertRaisesRegex(ValueError, "attention backend"):
            Qwen3VLForCausalLM(
                tiny_qwen3_vl_text_config(),
                attention_backend="eager",
            )

    def test_text_rmsnorm_matches_hf_eager_formula_exactly_in_bfloat16(self):
        norm = qwen3_vl_module.Qwen3VLTextRMSNorm(2048, eps=1e-6).to(
            dtype=torch.bfloat16
        )
        torch.manual_seed(43)
        hidden_states = torch.randn(8, 2048, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.randn_like(norm.weight))

        actual = norm(hidden_states)
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(-1, keepdim=True) + 1e-6
        )
        expected = norm.weight * normalized.to(hidden_states.dtype)

        self.assertTrue(torch.equal(actual, expected))

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_checkpoint_namespace_is_complete_and_strictly_roundtrips(self, *_):
        config = tiny_qwen3_vl_text_config()
        model = Qwen3VLForCausalLM(config)
        expected = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        }
        for layer_index in range(config.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_index}"
            expected.update(
                {
                    f"{prefix}.self_attn.qkv_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight",
                    f"{prefix}.mlp.gate_up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                }
            )

        self.assertEqual(set(model.state_dict()), expected)
        clone = Qwen3VLForCausalLM(config)
        result = clone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_embedding_and_lm_head_share_the_same_parameter(self, *_):
        model = Qwen3VLForCausalLM(tiny_qwen3_vl_text_config())

        self.assertIs(
            model.lm_head.weight,
            model.model.language_model.embed_tokens.weight,
        )
        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.model.language_model.embed_tokens.weight.data_ptr(),
        )

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_rejects_untied_qwen3_vl_checkpoint_configuration(self, *_):
        with self.assertRaisesRegex(ValueError, "tied"):
            Qwen3VLForCausalLM(
                tiny_qwen3_vl_text_config(tie_word_embeddings=False)
            )

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_tied_checkpoint_embedding_is_reachable_through_strict_loader(self, *_):
        config = tiny_qwen3_vl_text_config(num_hidden_layers=0)
        model = Qwen3VLForCausalLM(config)
        checkpoint = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size,
                config.hidden_size,
            ),
            "model.language_model.norm.weight": torch.randn(config.hidden_size),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(checkpoint, str(Path(directory) / "model.safetensors"))

            report = load_model(model, directory)

        self.assertIn(
            "model.language_model.embed_tokens.weight",
            report.loaded_parameters,
        )
        torch.testing.assert_close(
            model.model.language_model.embed_tokens.weight,
            checkpoint["model.language_model.embed_tokens.weight"],
        )
        self.assertIs(
            model.lm_head.weight,
            model.model.language_model.embed_tokens.weight,
        )

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_tied_checkpoint_accepts_matching_lm_head_alias(self, *_):
        config = tiny_qwen3_vl_text_config(num_hidden_layers=0)
        model = Qwen3VLForCausalLM(config)
        tied_weight = torch.randn(config.vocab_size, config.hidden_size)
        checkpoint = {
            "model.language_model.embed_tokens.weight": tied_weight,
            "model.language_model.norm.weight": torch.randn(config.hidden_size),
            "lm_head.weight": tied_weight.clone(),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(checkpoint, str(Path(directory) / "model.safetensors"))

            report = load_model(model, directory)

        self.assertIn("lm_head.weight", report.loaded_parameters)
        self.assertIs(
            model.lm_head.weight,
            model.model.language_model.embed_tokens.weight,
        )

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_tied_checkpoint_can_load_only_the_lm_head_alias(self, *_):
        config = tiny_qwen3_vl_text_config(num_hidden_layers=0)
        model = Qwen3VLForCausalLM(config)
        tied_weight = torch.randn(config.vocab_size, config.hidden_size)
        checkpoint = {
            "model.language_model.norm.weight": torch.randn(config.hidden_size),
            "lm_head.weight": tied_weight,
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(checkpoint, str(Path(directory) / "model.safetensors"))

            report = load_model(model, directory)

        torch.testing.assert_close(model.lm_head.weight, tied_weight)
        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.model.language_model.embed_tokens.weight.data_ptr(),
        )
        self.assertEqual(
            report.loaded_parameters,
            (
                "lm_head.weight",
                "model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
            ),
        )

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_strict_loader_reaches_all_unpacked_projection_weights(self, *_):
        config = tiny_qwen3_vl_text_config(num_hidden_layers=1)
        model = Qwen3VLForCausalLM(config)
        prefix = "model.language_model.layers.0"
        q_shape = (config.num_attention_heads * config.head_dim, config.hidden_size)
        kv_shape = (config.num_key_value_heads * config.head_dim, config.hidden_size)
        checkpoint = {
            "model.language_model.embed_tokens.weight": torch.randn(
                config.vocab_size,
                config.hidden_size,
            ),
            "model.language_model.norm.weight": torch.randn(config.hidden_size),
            f"{prefix}.self_attn.q_proj.weight": torch.full(q_shape, 1.0),
            f"{prefix}.self_attn.k_proj.weight": torch.full(kv_shape, 2.0),
            f"{prefix}.self_attn.v_proj.weight": torch.full(kv_shape, 3.0),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
            ),
            f"{prefix}.self_attn.q_norm.weight": torch.randn(config.head_dim),
            f"{prefix}.self_attn.k_norm.weight": torch.randn(config.head_dim),
            f"{prefix}.mlp.gate_proj.weight": torch.full(
                (config.intermediate_size, config.hidden_size),
                4.0,
            ),
            f"{prefix}.mlp.up_proj.weight": torch.full(
                (config.intermediate_size, config.hidden_size),
                5.0,
            ),
            f"{prefix}.mlp.down_proj.weight": torch.randn(
                config.hidden_size,
                config.intermediate_size,
            ),
            f"{prefix}.input_layernorm.weight": torch.randn(config.hidden_size),
            f"{prefix}.post_attention_layernorm.weight": torch.randn(
                config.hidden_size
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(checkpoint, str(Path(directory) / "model.safetensors"))

            report = load_model(model, directory)

        qkv = model.model.language_model.layers[0].self_attn.qkv_proj.weight
        expected_qkv = torch.cat(
            (
                checkpoint[f"{prefix}.self_attn.q_proj.weight"],
                checkpoint[f"{prefix}.self_attn.k_proj.weight"],
                checkpoint[f"{prefix}.self_attn.v_proj.weight"],
            )
        )
        torch.testing.assert_close(qkv, expected_qkv)
        gate_up = model.model.language_model.layers[0].mlp.gate_up_proj.weight
        expected_gate_up = torch.cat(
            (
                checkpoint[f"{prefix}.mlp.gate_proj.weight"],
                checkpoint[f"{prefix}.mlp.up_proj.weight"],
            )
        )
        torch.testing.assert_close(gate_up, expected_gate_up)
        self.assertEqual(report.skipped_checkpoint_weights, ())

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_attention_uses_qk_norm_and_interleaved_multimodal_rope(self, *_):
        model = Qwen3VLForCausalLM(tiny_qwen3_vl_text_config())
        attention = model.model.language_model.layers[0].self_attn

        self.assertEqual(model.position_axes, 3)
        self.assertEqual(attention.head_dim, 8)
        self.assertIsNone(attention.qkv_proj.bias)
        self.assertIsNone(attention.o_proj.bias)
        self.assertEqual(attention.rotary_emb.head_dim, 8)
        self.assertEqual(attention.rotary_emb.sections, (2, 1, 1))
        self.assertEqual(attention.rotary_emb.strategy, "interleaved")
        self.assertAlmostEqual(
            float(attention.rotary_emb.inv_freq[1]),
            5_000_000.0 ** (-2 / 8),
        )
        self.assertEqual(tuple(attention.q_norm.weight.shape), (8,))
        self.assertEqual(tuple(attention.k_norm.weight.shape), (8,))

    def test_packed_module_mapping_matches_qwen3_loader_contract(self):
        self.assertEqual(
            Qwen3VLForCausalLM.packed_modules_mapping,
            {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            },
        )


class Qwen3VLForwardTests(unittest.TestCase):
    def _model_with_fake_layers(self):
        with patch("torch.distributed.get_rank", return_value=0), patch(
            "torch.distributed.get_world_size", return_value=1
        ):
            model = Qwen3VLForCausalLM(tiny_qwen3_vl_text_config())
        with torch.no_grad():
            model.model.language_model.embed_tokens.weight.zero_()
        layers = nn.ModuleList(
            [_ResidualAddingLayer(1.0), _ResidualAddingLayer(2.0), _ResidualAddingLayer(3.0)]
        )
        model.model.language_model.layers = layers
        model.model.language_model.norm = _FusedResidualIdentityNorm()
        return model, layers

    def test_plain_text_forward_uses_input_ids_and_no_visual_payload(self):
        model, _ = self._model_with_fake_layers()
        input_ids = torch.tensor([1, 2, 3])
        positions = torch.arange(3).expand(3, -1)
        embedded = model.embed(input_ids).detach()

        actual = model(input_ids, positions)

        torch.testing.assert_close(actual, embedded + 6.0)

    def test_inputs_embeds_path_does_not_read_token_embedding(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(4, 16)
        positions = torch.arange(4).expand(3, -1)

        with patch.object(
            model.model.language_model.embed_tokens,
            "forward",
            side_effect=AssertionError("embedding lookup should not run"),
        ):
            actual = model(None, positions, inputs_embeds=embeddings)

        torch.testing.assert_close(actual, embeddings + 6.0)

    def test_deepstack_is_added_after_layers_zero_one_and_two(self):
        model, layers = self._model_with_fake_layers()
        embeddings = torch.zeros(5, 16)
        rows = torch.tensor([1, 4], dtype=torch.long)
        deepstack = (
            torch.full((2, 16), 10.0),
            torch.full((2, 16), 20.0),
            torch.full((2, 16), 30.0),
        )

        actual = model(
            None,
            torch.arange(5).expand(3, -1),
            inputs_embeds=embeddings,
            visual_token_rows=rows,
            deepstack_visual_embeds=deepstack,
        )

        expected_layer_one_input = torch.full_like(embeddings, 1.0)
        expected_layer_one_input[rows] += 10.0
        expected_layer_two_input = expected_layer_one_input + 2.0
        expected_layer_two_input[rows] += 20.0
        expected_output = expected_layer_two_input + 3.0
        expected_output[rows] += 30.0
        torch.testing.assert_close(layers[0].inputs[0], embeddings)
        torch.testing.assert_close(layers[1].inputs[0], expected_layer_one_input)
        torch.testing.assert_close(layers[2].inputs[0], expected_layer_two_input)
        torch.testing.assert_close(actual, expected_output)
        torch.testing.assert_close(actual[[0, 2, 3]], torch.full((3, 16), 6.0))
        torch.testing.assert_close(embeddings, torch.zeros_like(embeddings))

    def test_final_layer_deepstack_receives_gradient_without_inplace_mutation(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.zeros(3, 16, requires_grad=True)
        deepstack = tuple(
            torch.full((1, 16), value, requires_grad=True)
            for value in (10.0, 20.0, 30.0)
        )

        output = model(
            None,
            torch.arange(3).expand(3, -1),
            inputs_embeds=embeddings,
            visual_token_rows=torch.tensor([1]),
            deepstack_visual_embeds=deepstack,
        )
        output.sum().backward()

        for feature in deepstack:
            torch.testing.assert_close(feature.grad, torch.ones_like(feature))
        torch.testing.assert_close(embeddings, torch.zeros_like(embeddings))

    def test_decode_accepts_empty_visual_payload(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(2, 16)

        actual = model(
            None,
            torch.arange(2).expand(3, -1),
            inputs_embeds=embeddings,
            visual_token_rows=None,
            deepstack_visual_embeds=(),
        )

        torch.testing.assert_close(actual, embeddings + 6.0)

    def test_requires_exactly_one_token_input(self):
        model, _ = self._model_with_fake_layers()
        positions = torch.arange(2).expand(3, -1)
        embeddings = torch.randn(2, 16)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(None, positions)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(torch.tensor([1, 2]), positions, inputs_embeds=embeddings)

    def test_visual_rows_and_deepstack_must_be_present_together(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(3, 16)
        positions = torch.arange(3).expand(3, -1)
        features = tuple(torch.randn(1, 16) for _ in range(3))

        with self.assertRaisesRegex(ValueError, "together"):
            model(
                None,
                positions,
                inputs_embeds=embeddings,
                visual_token_rows=torch.tensor([1]),
                deepstack_visual_embeds=(),
            )
        with self.assertRaisesRegex(ValueError, "together"):
            model(
                None,
                positions,
                inputs_embeds=embeddings,
                deepstack_visual_embeds=features,
            )

    def test_rejects_wrong_deepstack_count_and_shape(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(3, 16)
        positions = torch.arange(3).expand(3, -1)
        rows = torch.tensor([1])

        with self.assertRaisesRegex(ValueError, "exactly three"):
            model(
                None,
                positions,
                inputs_embeds=embeddings,
                visual_token_rows=rows,
                deepstack_visual_embeds=(torch.randn(1, 16),),
            )
        invalid_shapes = (
            (torch.randn(16), torch.randn(1, 16), torch.randn(1, 16)),
            (torch.randn(2, 16), torch.randn(1, 16), torch.randn(1, 16)),
            (torch.randn(1, 15), torch.randn(1, 16), torch.randn(1, 16)),
        )
        for features in invalid_shapes:
            with self.subTest(shapes=tuple(tuple(item.shape) for item in features)):
                with self.assertRaisesRegex(ValueError, "DeepStack"):
                    model(
                        None,
                        positions,
                        inputs_embeds=embeddings,
                        visual_token_rows=rows,
                        deepstack_visual_embeds=features,
                    )

    def test_rejects_invalid_visual_rows(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(3, 16)
        positions = torch.arange(3).expand(3, -1)
        features = tuple(torch.randn(2, 16) for _ in range(3))
        invalid_rows = (
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0, 3], dtype=torch.long),
            torch.tensor([1, 1], dtype=torch.long),
        )

        for rows in invalid_rows:
            with self.subTest(rows=rows):
                with self.assertRaises((ValueError, TypeError, IndexError)):
                    model(
                        None,
                        positions,
                        inputs_embeds=embeddings,
                        visual_token_rows=rows,
                        deepstack_visual_embeds=features,
                    )

    def test_cuda_visual_row_contract_never_reads_values_in_python(self):
        source = inspect.getsource(
            type(self._model_with_fake_layers()[0].model.language_model)._validate_deepstack
        )
        cuda_branch = source.split("if visual_token_rows.is_cuda:", 1)[1].split(
            "else:",
            1,
        )[0]

        self.assertNotIn("torch.any", cuda_branch)
        self.assertNotIn("torch.unique", cuda_branch)
        self.assertNotIn("_assert", cuda_branch)
        self.assertIn("trusted", cuda_branch.lower())

    def test_empty_visual_rows_do_not_clone_hidden_states(self):
        hidden_states = torch.randn(3, 16)

        output = type(self._model_with_fake_layers()[0].model.language_model)._add_deepstack_rows(
            hidden_states,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, 16),
        )

        self.assertIs(output, hidden_states)

    def test_rejects_deepstack_dtype_and_device_mismatch(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(3, 16)
        positions = torch.arange(3).expand(3, -1)
        rows = torch.tensor([1])

        with self.assertRaisesRegex(TypeError, "dtype"):
            model(
                None,
                positions,
                inputs_embeds=embeddings,
                visual_token_rows=rows,
                deepstack_visual_embeds=tuple(
                    torch.randn(1, 16, dtype=torch.float64) for _ in range(3)
                ),
            )
        with self.assertRaisesRegex(ValueError, "device"):
            model(
                None,
                positions,
                inputs_embeds=embeddings,
                visual_token_rows=rows,
                deepstack_visual_embeds=tuple(
                    torch.empty(1, 16, device="meta") for _ in range(3)
                ),
            )

    def test_rejects_invalid_positions(self):
        model, _ = self._model_with_fake_layers()
        embeddings = torch.randn(3, 16)
        for positions in (
            torch.arange(3),
            torch.arange(3).expand(2, -1),
            torch.arange(4).expand(3, -1),
            torch.arange(3, dtype=torch.float32).expand(3, -1),
        ):
            with self.subTest(shape=tuple(positions.shape), dtype=positions.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    model(None, positions, inputs_embeds=embeddings)


if __name__ == "__main__":
    unittest.main()
