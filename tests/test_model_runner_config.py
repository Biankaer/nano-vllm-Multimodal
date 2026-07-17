import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import (
    calculate_num_kvcache_blocks,
    resolve_model_dtype,
)


class ResolveModelDTypeTests(unittest.TestCase):
    def test_uses_transformers_torch_dtype_field(self):
        config = SimpleNamespace(torch_dtype=torch.bfloat16)

        self.assertIs(resolve_model_dtype(config), torch.bfloat16)

    def test_keeps_dtype_field_compatibility(self):
        config = SimpleNamespace(dtype=torch.float16)

        self.assertIs(resolve_model_dtype(config), torch.float16)

class KVCacheBudgetTests(unittest.TestCase):
    def test_reserves_vision_feature_cache_before_allocating_kv_blocks(self):
        blocks = calculate_num_kvcache_blocks(
            total_bytes=1000,
            gpu_memory_utilization=0.9,
            used_bytes=100,
            peak_bytes=100,
            current_bytes=100,
            block_bytes=100,
            reserved_bytes=200,
        )

        self.assertEqual(blocks, 6)


class NativeVisionConfigTests(unittest.TestCase):
    def make_hf_config(self, model_type="qwen2_5_vl"):
        return SimpleNamespace(
            model_type=model_type,
            max_position_embeddings=32768,
        )

    def test_qwen25_vl_native_runtime_rejects_tensor_parallel(self):
        with TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=self.make_hf_config(),
            ):
                with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
                    Config(model_dir, tensor_parallel_size=2)

    def test_vision_feature_cache_budget_must_be_positive(self):
        with TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=self.make_hf_config(),
            ):
                with self.assertRaisesRegex(ValueError, "cache byte budget"):
                    Config(model_dir, vision_feature_cache_bytes=0)

    def test_native_vision_defaults_use_byte_budget_and_sdpa(self):
        with TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=self.make_hf_config(),
            ):
                config = Config(model_dir)

        self.assertEqual(config.vision_feature_cache_bytes, 2 * 1024**3)
        self.assertEqual(config.vision_attn_implementation, "sdpa")

if __name__ == "__main__":
    unittest.main()
