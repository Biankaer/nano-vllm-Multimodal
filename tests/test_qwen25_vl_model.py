import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)

from nanovllm.layers.multimodal_rotary_embedding import (
    MultimodalRotaryEmbedding,
    apply_multimodal_rotary_emb,
)
from nanovllm.models.qwen25_vl import Qwen25VLForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM


def tiny_qwen25_config():
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        intermediate_size=512,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
        vocab_size=256,
        tie_word_embeddings=True,
    )


class MultimodalRotaryEmbeddingTests(unittest.TestCase):
    def test_rotation_matches_transformers(self):
        torch.manual_seed(7)
        positions = torch.tensor(
            (
                (0, 1, 1, 2, 2),
                (0, 1, 1, 2, 2),
                (0, 1, 2, 1, 2),
            )
        )
        query = torch.randn(5, 2, 128)
        key = torch.randn(5, 1, 128)
        sections = (16, 24, 24)
        rotary = MultimodalRotaryEmbedding(
            head_dim=128,
            base=1_000_000.0,
            sections=sections,
        )
        cos, sin = rotary.cos_sin(positions, query.dtype)

        actual_query, actual_key = apply_multimodal_rotary_emb(
            query,
            key,
            cos,
            sin,
            sections,
        )
        expected_query, expected_key = apply_multimodal_rotary_pos_emb(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            cos.unsqueeze(1),
            sin.unsqueeze(1),
            list(sections),
        )

        torch.testing.assert_close(
            actual_query,
            expected_query.squeeze(0).transpose(0, 1),
        )
        torch.testing.assert_close(
            actual_key,
            expected_key.squeeze(0).transpose(0, 1),
        )

    def test_rejects_invalid_position_shape(self):
        rotary = MultimodalRotaryEmbedding(
            head_dim=128,
            base=1_000_000.0,
            sections=(16, 24, 24),
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            rotary.cos_sin(torch.arange(4), torch.float32)


class Qwen25VLModelTests(unittest.TestCase):
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_model_uses_qwen25_attention_and_checkpoint_names(self, *_):
        model = Qwen25VLForCausalLM(tiny_qwen25_config())
        parameters = dict(model.named_parameters())

        self.assertIn("model.layers.0.self_attn.qkv_proj.bias", parameters)
        self.assertNotIn("model.layers.0.self_attn.q_norm.weight", parameters)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(model.position_axes, 3)

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_model_accepts_prepared_embeddings(self, *_):
        model = Qwen25VLForCausalLM(tiny_qwen25_config())
        model.model.layers = nn.ModuleList()
        model.model.norm = nn.Identity()
        embeddings = torch.randn(3, 256)
        positions = torch.arange(3).expand(3, -1)

        output = model(None, positions, inputs_embeds=embeddings)

        self.assertIs(output, embeddings)

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_model_ignores_empty_deepstack_payload(self, *_):
        model = Qwen25VLForCausalLM(tiny_qwen25_config())
        model.model.layers = nn.ModuleList()
        model.model.norm = nn.Identity()
        embeddings = torch.randn(3, 256)
        positions = torch.arange(3).expand(3, -1)

        output = model(
            None,
            positions,
            inputs_embeds=embeddings,
            visual_token_rows=torch.tensor([1]),
            deepstack_visual_embeds=(),
        )

        self.assertIs(output, embeddings)

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_model_explicitly_rejects_nonempty_deepstack_payload(self, *_):
        model = Qwen25VLForCausalLM(tiny_qwen25_config())
        embeddings = torch.randn(3, 256)
        positions = torch.arange(3).expand(3, -1)

        with self.assertRaisesRegex(NotImplementedError, "DeepStack"):
            model(
                None,
                positions,
                inputs_embeds=embeddings,
                visual_token_rows=torch.tensor([1]),
                deepstack_visual_embeds=(torch.randn(1, 256),),
            )


class Qwen3ModelInterfaceTests(unittest.TestCase):
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_qwen3_supports_the_shared_embedding_interface(self, *_):
        config = tiny_qwen25_config()
        config.attention_bias = False
        config.head_dim = 128
        config.rope_scaling = None
        model = Qwen3ForCausalLM(config)
        model.model.layers = nn.ModuleList()
        model.model.norm = nn.Identity()
        embeddings = torch.randn(3, 256)

        output = model(None, torch.arange(3), inputs_embeds=embeddings)

        self.assertIs(output, embeddings)
        self.assertEqual(model.position_axes, 1)
        self.assertEqual(model.embed(torch.tensor([1, 2])).shape, (2, 256))


if __name__ == "__main__":
    unittest.main()
