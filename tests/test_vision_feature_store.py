import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.multimodal.feature_store import VisionFeatureStore
from nanovllm.multimodal.qwen25_vl_processor import VisionInput
from nanovllm.multimodal.runtime import VisionFeatureBundle
from nanovllm.multimodal.vision_provider import (
    VisionFeatureProvider,
    _move_module_preserving_float32_buffers,
    _probe_qwen3_flash_attention,
)


class FakeVisionTower(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.hidden_size = hidden_size
        self.calls = 0
        self.encoded_images = 0

    def forward(self, pixel_values, *, grid_thw):
        self.calls += 1
        self.encoded_images += grid_thw.shape[0]
        outputs = []
        for index, (grid_t, grid_h, grid_w) in enumerate(grid_thw.tolist()):
            token_count = grid_t * (grid_h // 2) * (grid_w // 2)
            outputs.append(
                torch.full(
                    (token_count, self.hidden_size),
                    float(index + 1),
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )
            )
        return torch.cat(outputs)


class FakeBundleVisionTower(FakeVisionTower):
    def forward(self, pixel_values, *, grid_thw):
        final = super().forward(pixel_values, grid_thw=grid_thw)
        return VisionFeatureBundle(final, (final + 10, final + 20, final + 30))


class VisionFeatureStoreTests(unittest.TestCase):
    def test_flash_probe_runs_minimal_varlen_kernel_with_requested_shape(self):
        fake_qkv = [object(), object(), object()]
        fake_cu = object()
        with (
            patch("nanovllm.multimodal.vision_provider.torch.empty", side_effect=fake_qkv) as empty,
            patch("nanovllm.multimodal.vision_provider.torch.tensor", return_value=fake_cu),
            patch("nanovllm.multimodal.vision_provider._resolve_flash_attn_varlen_func") as resolve,
            patch("nanovllm.multimodal.vision_provider.torch.cuda.current_stream") as stream,
        ):
            _probe_qwen3_flash_attention("cuda:2", torch.bfloat16, head_dim=64)
        self.assertEqual(empty.call_count, 3)
        empty.assert_any_call((2, 1, 64), device=torch.device("cuda:2"), dtype=torch.bfloat16)
        resolve.return_value.assert_called_once_with(
            fake_qkv[0], fake_qkv[1], fake_qkv[2],
            cu_seqlens_q=fake_cu, cu_seqlens_k=fake_cu,
            max_seqlen_q=2, max_seqlen_k=2,
            dropout_p=0.0, causal=False,
        )
        stream.assert_called_once_with(torch.device("cuda:2"))
        stream.return_value.synchronize.assert_called_once_with()

    def test_flash_probe_wraps_kernel_failure_with_sdpa_guidance(self):
        with (
            patch("nanovllm.multimodal.vision_provider.torch.empty", return_value=object()),
            patch("nanovllm.multimodal.vision_provider.torch.tensor", return_value=object()),
            patch("nanovllm.multimodal.vision_provider._resolve_flash_attn_varlen_func") as resolve,
        ):
            resolve.return_value.side_effect = RuntimeError("unsupported gpu")
            with self.assertRaisesRegex(RuntimeError, "probe failed.*sdpa"):
                _probe_qwen3_flash_attention("cuda:0", torch.float16, head_dim=64)

    def test_flash_probe_rejects_non_half_dtype_before_allocating(self):
        with patch("nanovllm.multimodal.vision_provider.torch.empty") as empty:
            with self.assertRaisesRegex(RuntimeError, "float16 or bfloat16.*sdpa"):
                _probe_qwen3_flash_attention("cuda:0", torch.float32, head_dim=64)
        empty.assert_not_called()

    def test_accounts_for_all_bundle_layers(self):
        store = VisionFeatureStore(byte_budget=48)
        bundle = VisionFeatureBundle(
            torch.zeros(2, 2),
            (torch.ones(2, 2), torch.full((2, 2), 2.0)),
        )

        store.put("a", bundle)

        self.assertIs(store.peek("a"), store.get("a"))
        self.assertEqual(store.resident_bytes, 48)

    def test_single_put_failure_is_atomic_even_if_eviction_would_be_needed(self):
        store = VisionFeatureStore(byte_budget=32)
        store.put("pinned", torch.zeros(4))
        store.put("other", torch.ones(4))

        with store.pin(("pinned",)):
            with self.assertRaisesRegex(MemoryError, "budget=32"):
                store.put("large", torch.zeros(8))

        self.assertIsNotNone(store.peek("pinned"))
        self.assertIsNotNone(store.peek("other"))
        self.assertIsNone(store.peek("large"))
        self.assertEqual(store.stats().evictions, 0)

    def test_dtype_move_preserves_explicit_float32_buffers(self):
        module = nn.Linear(2, 2, bias=False)
        module.register_buffer("inv_freq", torch.tensor([1.0, 0.1234567]))
        expected = module.inv_freq.clone()

        _move_module_preserving_float32_buffers(
            module,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        self.assertEqual(module.weight.dtype, torch.bfloat16)
        self.assertEqual(module.inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(module.inv_freq, expected))

    def test_provider_splits_final_and_deepstack_identically_per_image(self):
        provider = VisionFeatureProvider(
            FakeBundleVisionTower(),
            spatial_merge_size=2,
            expected_hidden_size=4,
            feature_store=VisionFeatureStore(4096),
        )
        inputs = (
            VisionInput("a", torch.zeros(4, 8), (1, 2, 2)),
            VisionInput("b", torch.zeros(8, 8), (1, 2, 4)),
        )
        bundles = provider._encode(inputs)
        self.assertEqual(bundles["a"].final_embedding.shape, (1, 4))
        self.assertEqual(bundles["b"].final_embedding.shape, (2, 4))
        self.assertTrue(torch.equal(
            bundles["b"].deepstack_embeddings[2],
            bundles["b"].final_embedding + 30,
        ))

    def test_provider_rejects_deepstack_row_mismatch(self):
        tower = FakeBundleVisionTower()
        tower.forward = lambda pixel_values, grid_thw: VisionFeatureBundle(
            torch.zeros(1, 4), (torch.zeros(2, 4),)
        )
        provider = VisionFeatureProvider(
            tower, spatial_merge_size=2, expected_hidden_size=4,
            feature_store=VisionFeatureStore(4096),
        )
        with self.assertRaisesRegex(ValueError, "same token and hidden dimensions"):
            provider._encode((VisionInput("a", torch.zeros(4, 8), (1, 2, 2)),))

    def test_evicts_lru_entry_by_tensor_bytes(self):
        store = VisionFeatureStore(byte_budget=32)
        store.put("a", torch.zeros(4))
        store.put("b", torch.ones(4))
        self.assertIsNotNone(store.get("a"))

        store.put("c", torch.full((4,), 2.0))

        self.assertIsNone(store.peek("b"))
        self.assertIsNotNone(store.peek("a"))
        self.assertIsNotNone(store.peek("c"))
        self.assertEqual(store.resident_bytes, 32)
        self.assertEqual(store.stats().evictions, 1)

    def test_tracks_hits_and_misses(self):
        store = VisionFeatureStore(byte_budget=32)
        store.put("a", torch.zeros(4))

        self.assertIsNotNone(store.get("a"))
        self.assertIsNone(store.get("missing"))

        stats = store.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)

    def test_pinned_entry_cannot_be_evicted(self):
        store = VisionFeatureStore(byte_budget=16)
        store.put("a", torch.zeros(4))

        with store.pin(("a",)) as pinned:
            self.assertIs(pinned["a"], store.peek("a"))
            with self.assertRaisesRegex(MemoryError, "required=32.*budget=16"):
                store.put("b", torch.ones(4))

        store.put("b", torch.ones(4))
        self.assertIsNone(store.peek("a"))
        self.assertIsNotNone(store.peek("b"))

    def test_rejects_single_entry_larger_than_budget(self):
        store = VisionFeatureStore(byte_budget=8)

        with self.assertRaisesRegex(MemoryError, "required=16.*budget=8"):
            store.put("large", torch.zeros(4))

    def test_put_many_keeps_the_complete_working_set(self):
        store = VisionFeatureStore(byte_budget=32)
        store.put("old", torch.zeros(4))

        store.put_many(
            {
                "a": torch.ones(4),
                "b": torch.full((4,), 2.0),
            }
        )

        self.assertIsNone(store.peek("old"))
        self.assertIsNotNone(store.peek("a"))
        self.assertIsNotNone(store.peek("b"))

    def test_put_many_fails_without_partial_insert(self):
        store = VisionFeatureStore(byte_budget=16)
        store.put("old", torch.zeros(4))

        with self.assertRaisesRegex(MemoryError, "required=32.*budget=16"):
            store.put_many(
                {
                    "a": torch.ones(4),
                    "b": torch.full((4,), 2.0),
                }
            )

        self.assertIsNotNone(store.peek("old"))
        self.assertIsNone(store.peek("a"))
        self.assertIsNone(store.peek("b"))


class VisionFeatureProviderTests(unittest.TestCase):
    def test_qwen3_provider_loads_only_model_visual_weights(self):
        config = SimpleNamespace(
            model_type="qwen3_vl",
            vision_config=SimpleNamespace(spatial_merge_size=2, out_hidden_size=4),
        )
        tower = FakeVisionTower()
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=config),
            patch(
                "nanovllm.models.qwen3_vl_vision.Qwen3VLVisionModel",
                return_value=tower,
            ) as tower_cls,
            patch(
                "nanovllm.multimodal.vision_provider._move_module_preserving_float32_buffers"
            ) as preserve_buffers,
            patch("nanovllm.multimodal.vision_provider._probe_qwen3_flash_attention") as probe,
            patch("nanovllm.multimodal.vision_provider.load_model") as loader,
        ):
            provider = VisionFeatureProvider.from_pretrained(
                "/model", byte_budget=1024, dtype=torch.float32,
                device="cpu", attn_implementation="sdpa",
            )
        tower_cls.assert_called_once_with(config.vision_config, attention_backend="sdpa")
        loader.assert_called_once_with(
            tower, "/model", include_prefixes=("model.visual.",),
            strip_prefix="model.visual.",
        )
        self.assertIs(provider.vision_tower, tower)
        preserve_buffers.assert_called_once_with(
            tower, device="cpu", dtype=torch.float32,
        )
        probe.assert_not_called()

    def test_qwen3_flash_fails_during_initialization_without_cuda(self):
        config = SimpleNamespace(model_type="qwen3_vl", vision_config=object())
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=config),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA.*sdpa"):
                VisionFeatureProvider.from_pretrained(
                    "/model", byte_budget=1024, dtype=torch.bfloat16,
                    device="cpu", attn_implementation="flash_attention_2",
                )

    def setUp(self):
        self.tower = FakeVisionTower()
        self.store = VisionFeatureStore(byte_budget=1024)
        self.provider = VisionFeatureProvider(
            self.tower,
            spatial_merge_size=2,
            expected_hidden_size=4,
            feature_store=self.store,
        )
        self.input_a = VisionInput("a", torch.zeros(4, 8), (1, 2, 2))
        self.input_b = VisionInput("b", torch.ones(8, 8), (1, 2, 4))
        self.provider.register_inputs((self.input_a, self.input_b))

    def test_deduplicates_misses_and_reuses_cached_features(self):
        resolved = self.provider.resolve(("a", "a", "b"))

        self.assertEqual(self.tower.calls, 1)
        self.assertEqual(self.tower.encoded_images, 2)
        self.assertIsInstance(resolved["a"], VisionFeatureBundle)
        self.assertEqual(resolved["a"].final_embedding.shape, (1, 4))
        self.assertEqual(resolved["b"].final_embedding.shape, (2, 4))
        self.assertEqual(resolved["a"].deepstack_embeddings, ())

        cached = self.provider.resolve(("a", "b"))

        self.assertEqual(self.tower.calls, 1)
        self.assertIs(cached["a"], self.store.peek("a"))
        stats = self.provider.stats()
        self.assertEqual(stats.cache_hits, 2)
        self.assertEqual(stats.cache_misses, 2)
        self.assertEqual(stats.encoder_batches, 1)
        self.assertEqual(stats.encoded_images, 2)

    def test_batched_outputs_are_cached_in_independent_exact_size_storages(self):
        resolved = self.provider.resolve(("a", "b"))
        final_a = resolved["a"].final_embedding
        final_b = resolved["b"].final_embedding

        self.assertNotEqual(final_a.data_ptr(), final_b.data_ptr())
        self.assertNotEqual(
            final_a.untyped_storage().data_ptr(),
            final_b.untyped_storage().data_ptr(),
        )
        for final in (final_a, final_b):
            self.assertEqual(final.untyped_storage().nbytes(), final.nbytes)

    def test_resolve_pinned_keeps_complete_batch_resident(self):
        with self.provider.resolve_pinned(("a", "b")) as resolved:
            self.assertEqual(set(resolved), {"a", "b"})
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                self.store.put("a", torch.zeros(1, 4))

    def test_rejects_unknown_feature_key(self):
        with self.assertRaisesRegex(KeyError, "missing"):
            self.provider.resolve(("missing",))

    def test_release_removes_host_inputs_but_keeps_gpu_cache(self):
        self.provider.resolve(("a",))

        self.provider.release_inputs(("a",))

        self.assertEqual(self.provider.stats().registered_inputs, 1)
        self.assertIsNotNone(self.store.peek("a"))

    def test_shared_input_registration_is_reference_counted(self):
        self.provider.register_inputs((self.input_a, self.input_a))

        self.provider.release_inputs(("a",))
        self.assertEqual(self.provider.stats().registered_inputs, 2)

        self.provider.release_inputs(("a",))
        self.assertEqual(self.provider.stats().registered_inputs, 1)

    def test_rejects_conflicting_registration(self):
        conflicting = VisionInput("a", torch.ones(4, 8), (1, 2, 2))

        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.provider.register_inputs((conflicting,))


if __name__ == "__main__":
    unittest.main()
