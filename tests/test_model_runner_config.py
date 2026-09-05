import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)

from nanovllm.config import (
    Config,
    effective_text_config,
    resolve_language_attn_implementation,
    resolve_vision_attn_implementation,
)
from nanovllm.engine.model_runner import (
    ModelRunner,
    calculate_num_kvcache_blocks,
    prepare_sampling_inputs,
    resolve_model_dtype,
)


class ResolveModelDTypeTests(unittest.TestCase):
    def test_uses_transformers_torch_dtype_field(self):
        config = SimpleNamespace(torch_dtype=torch.bfloat16)

        self.assertIs(resolve_model_dtype(config), torch.bfloat16)

    def test_keeps_dtype_field_compatibility(self):
        config = SimpleNamespace(dtype=torch.float16)

        self.assertIs(resolve_model_dtype(config), torch.float16)

    def test_qwen3_vl_uses_nested_text_config_dtype(self):
        config = SimpleNamespace(
            model_type="qwen3_vl",
            dtype=torch.float32,
            text_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        )

        self.assertIs(resolve_model_dtype(config), torch.bfloat16)


class EffectiveTextConfigTests(unittest.TestCase):
    def test_runner_warmup_failure_releases_partial_model_and_process_group(self):
        error = RuntimeError("warmup failed")
        config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3", torch_dtype=torch.float16,
            ),
            kvcache_block_size=256, enforce_eager=True,
            tensor_parallel_size=1, model="/model",
        )
        runner = ModelRunner.__new__(ModelRunner)
        with (
            patch("nanovllm.engine.model_runner.dist.init_process_group"),
            patch("nanovllm.engine.model_runner.dist.is_initialized", return_value=True),
            patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy,
            patch("nanovllm.engine.model_runner.torch.cuda.set_device"),
            patch("nanovllm.engine.model_runner.torch.cuda.empty_cache"),
            patch("nanovllm.engine.model_runner.torch.set_default_device"),
            patch("nanovllm.engine.model_runner.torch.set_default_dtype"),
            patch("nanovllm.engine.model_runner.Qwen3ForCausalLM", return_value=object()),
            patch("nanovllm.engine.model_runner.load_model"),
            patch.object(ModelRunner, "warmup_model", side_effect=error),
        ):
            with self.assertRaisesRegex(RuntimeError, "warmup failed") as caught:
                runner.__init__(config, rank=0, event=SimpleNamespace())
        self.assertIs(caught.exception, error)
        destroy.assert_called_once_with()
        self.assertFalse(hasattr(runner, "model"))
        self.assertFalse(hasattr(runner, "sampler"))

    def test_exit_after_failed_initialization_does_not_destroy_twice(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.world_size = 1
        runner.enforce_eager = True
        with (
            patch("nanovllm.engine.model_runner.dist.is_initialized", return_value=False),
            patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy,
            patch("nanovllm.engine.model_runner.torch.cuda.synchronize"),
        ):
            runner.exit()
        destroy.assert_not_called()

    def test_exit_is_idempotent_for_rank_zero_shared_memory(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.world_size = 2
        runner.rank = 0
        runner.enforce_eager = True
        runner.vision_provider = None
        shared_memory = SimpleNamespace(close=Mock(), unlink=Mock())
        runner.shm = shared_memory
        with (
            patch("nanovllm.engine.model_runner.dist.barrier") as barrier,
            patch("nanovllm.engine.model_runner.dist.is_initialized", return_value=True),
            patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy,
            patch("nanovllm.engine.model_runner.torch.cuda.synchronize") as synchronize,
        ):
            runner.exit()
            runner.exit()
        shared_memory.close.assert_called_once_with()
        shared_memory.unlink.assert_called_once_with()
        barrier.assert_called_once_with()
        destroy.assert_called_once_with()
        synchronize.assert_called_once_with()
        self.assertIsNone(runner.shm)

    def test_runner_initialization_failure_restores_globals_and_process_group(self):
        original_dtype = torch.float64
        error = RuntimeError("checkpoint audit failed")
        config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_vl",
                text_config=SimpleNamespace(torch_dtype=torch.bfloat16),
            ),
            kvcache_block_size=256,
            enforce_eager=True,
            tensor_parallel_size=1,
            model="/model",
        )
        runner = ModelRunner.__new__(ModelRunner)
        with (
            patch("nanovllm.engine.model_runner.dist.init_process_group"),
            patch("nanovllm.engine.model_runner.dist.is_initialized", return_value=True),
            patch("nanovllm.engine.model_runner.dist.destroy_process_group") as destroy,
            patch("nanovllm.engine.model_runner.torch.cuda.set_device"),
            patch("nanovllm.engine.model_runner.torch.cuda.empty_cache"),
            patch("nanovllm.engine.model_runner.torch.get_default_dtype", return_value=original_dtype),
            patch("nanovllm.engine.model_runner.torch.get_default_device", return_value="meta"),
            patch("nanovllm.engine.model_runner.torch.set_default_device") as set_device,
            patch("nanovllm.engine.model_runner.torch.set_default_dtype") as set_dtype,
            patch(
                "nanovllm.models.qwen_vl_adapter.audit_qwen3_vl_checkpoint",
                side_effect=error,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "checkpoint audit failed") as caught:
                runner.__init__(config, rank=0, event=SimpleNamespace())
        self.assertIs(caught.exception, error)
        set_device.assert_any_call("meta")
        self.assertEqual(set_dtype.call_args_list[-1].args, (original_dtype,))
        destroy.assert_called_once_with()
        self.assertFalse(hasattr(runner, "vision_provider"))

    def test_qwen3_vl_runner_constructs_native_language_model(self):
        text_config = SimpleNamespace(torch_dtype=torch.bfloat16)
        hf_config = SimpleNamespace(model_type="qwen3_vl", text_config=text_config)
        config = SimpleNamespace(
            hf_config=hf_config, kvcache_block_size=256, enforce_eager=True,
            tensor_parallel_size=1, model="/model",
        )
        runner = ModelRunner.__new__(ModelRunner)
        sentinel = RuntimeError("stop after qwen3-vl model construction")
        with (
            patch("nanovllm.engine.model_runner.dist.init_process_group"),
            patch("nanovllm.engine.model_runner.torch.cuda.set_device"),
            patch("nanovllm.engine.model_runner.torch.set_default_device"),
            patch("nanovllm.engine.model_runner.torch.set_default_dtype"),
            patch("nanovllm.models.qwen_vl_adapter.audit_qwen3_vl_checkpoint"),
            patch("nanovllm.engine.model_runner.Qwen3VLForCausalLM", side_effect=sentinel) as cls,
        ):
            with self.assertRaisesRegex(RuntimeError, str(sentinel)):
                runner.__init__(config, rank=0, event=SimpleNamespace())
        cls.assert_called_once_with(
            text_config,
            attention_backend="flash_attention_2",
        )

    def test_qwen3_vl_returns_nested_text_config(self):
        text_config = SimpleNamespace(hidden_size=2048)
        config = SimpleNamespace(model_type="qwen3_vl", text_config=text_config)

        self.assertIs(effective_text_config(config), text_config)

    def test_qwen25_vl_returns_real_nested_text_config(self):
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "vocab_size": 100,
            }
        )

        self.assertIs(effective_text_config(config), config.text_config)

    def test_qwen3_text_model_returns_the_original_config(self):
        config = SimpleNamespace(model_type="qwen3")

        self.assertIs(effective_text_config(config), config)

    def test_qwen25_runner_constructs_language_model_with_text_config(self):
        text_config = SimpleNamespace(torch_dtype=torch.bfloat16)
        hf_config = SimpleNamespace(
            model_type="qwen2_5_vl",
            text_config=text_config,
        )
        config = SimpleNamespace(
            hf_config=hf_config,
            kvcache_block_size=256,
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        runner = ModelRunner.__new__(ModelRunner)
        sentinel = RuntimeError("stop after language model construction")

        with (
            patch("nanovllm.engine.model_runner.dist.init_process_group"),
            patch("nanovllm.engine.model_runner.torch.cuda.set_device"),
            patch("nanovllm.engine.model_runner.torch.set_default_device"),
            patch("nanovllm.engine.model_runner.torch.set_default_dtype"),
            patch(
                "nanovllm.engine.model_runner.Qwen25VLForCausalLM",
                side_effect=sentinel,
            ) as model_cls,
        ):
            with self.assertRaisesRegex(RuntimeError, str(sentinel)):
                runner.__init__(config, rank=0, event=SimpleNamespace())

        model_cls.assert_called_once_with(text_config)

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

    def test_qwen3_vl_kv_cache_uses_nested_text_config(self):
        text_config = SimpleNamespace(
            num_key_value_heads=2,
            head_dim=64,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=4,
        )
        config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_vl",
                text_config=text_config,
            ),
            gpu_memory_utilization=0.9,
            vision_feature_cache_bytes=1024,
            num_kvcache_blocks=-1,
        )
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = config
        runner.block_size = 256
        runner.world_size = 1
        runner.model_dtype = torch.bfloat16
        runner.vision_provider = None
        runner.model = SimpleNamespace(modules=lambda: [])

        with (
            patch(
                "nanovllm.engine.model_runner.torch.cuda.mem_get_info",
                return_value=(9_000_000, 10_000_000),
            ),
            patch(
                "nanovllm.engine.model_runner.torch.cuda.memory_stats",
                return_value={
                    "allocated_bytes.all.peak": 1000,
                    "allocated_bytes.all.current": 1000,
                },
            ),
            patch("nanovllm.engine.model_runner.torch.empty") as empty,
        ):
            runner.allocate_kv_cache()

        self.assertGreater(config.num_kvcache_blocks, 0)
        empty.assert_called_once_with(
            2,
            4,
            config.num_kvcache_blocks,
            256,
            2,
            64,
        )


class PrepareSamplingInputsTests(unittest.TestCase):
    def test_temperatures_stay_on_cpu_when_default_device_is_non_cpu(self):
        with torch.device("meta"):
            temperatures, _ = prepare_sampling_inputs([0.0, 0.7])

        self.assertEqual(temperatures.device.type, "cpu")

    def test_all_zero_host_temperatures_set_greedy_flag(self):
        temperatures, all_greedy = prepare_sampling_inputs([0.0, 0.0])

        torch.testing.assert_close(temperatures, torch.tensor([0.0, 0.0]))
        self.assertIs(all_greedy, True)

    def test_mixed_host_temperatures_clear_greedy_flag(self):
        temperatures, all_greedy = prepare_sampling_inputs([0.0, 0.7])

        torch.testing.assert_close(temperatures, torch.tensor([0.0, 0.7]))
        self.assertIs(all_greedy, False)


class NativeVisionConfigTests(unittest.TestCase):
    @staticmethod
    def write_local_config(model_dir, model_type):
        Path(model_dir, "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8"
        )

    def test_qwen3_version_preflight_runs_before_auto_config(self):
        hf_config = self.make_hf_config(
            "qwen3_vl", text_config=SimpleNamespace(max_position_embeddings=8192)
        )
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen3_vl")
            with (
                patch("nanovllm.config.require_transformers_version",
                      side_effect=ImportError("transformers too old")) as preflight,
                patch("nanovllm.config.AutoConfig.from_pretrained",
                      return_value=hf_config) as auto_config,
            ):
                with self.assertRaisesRegex(ImportError, "too old"):
                    Config(model_dir)
        preflight.assert_called_once_with()
        auto_config.assert_not_called()

    def test_qwen3_sufficient_version_continues_to_auto_config(self):
        hf_config = self.make_hf_config(
            "qwen3_vl", text_config=SimpleNamespace(max_position_embeddings=8192)
        )
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen3_vl")
            with (
                patch("nanovllm.config.require_transformers_version") as preflight,
                patch("nanovllm.config.AutoConfig.from_pretrained",
                      return_value=hf_config) as auto_config,
            ):
                Config(model_dir)
        preflight.assert_called_once_with()
        auto_config.assert_called_once_with(model_dir)

    def test_invalid_local_config_has_clear_directory_error(self):
        with TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid.*config.json"):
                Config(model_dir)

    def make_hf_config(self, model_type="qwen2_5_vl", **overrides):
        values = dict(
            model_type=model_type,
            max_position_embeddings=32768,
        )
        values.update(overrides)
        return SimpleNamespace(
            **values,
        )

    def test_qwen25_vl_native_runtime_rejects_tensor_parallel(self):
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen2_5_vl")
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=self.make_hf_config(),
            ):
                with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
                    Config(model_dir, tensor_parallel_size=2)

    def test_vision_feature_cache_budget_must_be_positive(self):
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen2_5_vl")
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=self.make_hf_config(),
            ):
                with self.assertRaisesRegex(ValueError, "cache byte budget"):
                    Config(model_dir, vision_feature_cache_bytes=0)

    def test_qwen25_auto_resolves_to_sdpa(self):
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen2_5_vl")
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=self.make_hf_config(),
            ):
                config = Config(model_dir)

        self.assertEqual(config.vision_feature_cache_bytes, 2 * 1024**3)
        self.assertEqual(config.vision_attn_implementation, "sdpa")

    def test_qwen3_vl_auto_resolves_to_flash_attention_2(self):
        hf_config = self.make_hf_config(
            "qwen3_vl",
            text_config=SimpleNamespace(max_position_embeddings=131072),
        )
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen3_vl")
            with (
                patch("nanovllm.config.require_transformers_version"),
                patch("nanovllm.config.AutoConfig.from_pretrained", return_value=hf_config),
            ):
                config = Config(model_dir)

        self.assertEqual(config.vision_attn_implementation, "flash_attention_2")
        self.assertEqual(config.language_attn_implementation, "flash_attention_2")
        self.assertEqual(config.max_model_len, 4096)

    def test_explicit_sdpa_language_backend_is_preserved_for_qwen3_vl(self):
        self.assertEqual(
            resolve_language_attn_implementation("qwen3_vl", "sdpa"),
            "sdpa",
        )

    def test_language_backend_auto_keeps_existing_models_on_flash(self):
        self.assertEqual(
            resolve_language_attn_implementation("qwen2_5_vl", "auto"),
            "flash_attention_2",
        )

    def test_unsupported_language_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "language_attn_implementation"):
            resolve_language_attn_implementation("qwen3_vl", "eager")

    def test_qwen3_vl_max_model_len_uses_nested_text_config(self):
        hf_config = self.make_hf_config(
            "qwen3_vl",
            max_position_embeddings=1024,
            text_config=SimpleNamespace(max_position_embeddings=8192),
        )
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen3_vl")
            with (
                patch("nanovllm.config.require_transformers_version"),
                patch("nanovllm.config.AutoConfig.from_pretrained", return_value=hf_config),
            ):
                config = Config(model_dir, max_model_len=4096)

        self.assertEqual(config.max_model_len, 4096)

    def test_qwen3_vl_native_runtime_rejects_tensor_parallel(self):
        hf_config = self.make_hf_config(
            "qwen3_vl",
            text_config=SimpleNamespace(max_position_embeddings=131072),
        )
        with TemporaryDirectory() as model_dir:
            self.write_local_config(model_dir, "qwen3_vl")
            with (
                patch("nanovllm.config.require_transformers_version"),
                patch("nanovllm.config.AutoConfig.from_pretrained", return_value=hf_config),
            ):
                with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
                    Config(model_dir, tensor_parallel_size=2)

    def test_explicit_supported_vision_backend_is_preserved(self):
        self.assertEqual(
            resolve_vision_attn_implementation("qwen3_vl", "sdpa"),
            "sdpa",
        )

    def test_unsupported_vision_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vision_attn_implementation"):
            resolve_vision_attn_implementation("qwen3_vl", "eager")


class DependencyConstraintTests(unittest.TestCase):
    def test_transformers_is_pinned_below_version_five(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

        self.assertIn('"transformers>=4.57.6,<5"', pyproject)

if __name__ == "__main__":
    unittest.main()
