import unittest

import torch
from torch import nn

from nanovllm.multimodal.feature_store import VisionFeatureStore
from nanovllm.multimodal.qwen25_vl_processor import VisionInput
from nanovllm.multimodal.vision_provider import (
    VisionFeatureProvider,
    _move_module_preserving_float32_buffers,
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


class VisionFeatureStoreTests(unittest.TestCase):
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
        self.assertEqual(resolved["a"].shape, (1, 4))
        self.assertEqual(resolved["b"].shape, (2, 4))

        cached = self.provider.resolve(("a", "b"))

        self.assertEqual(self.tower.calls, 1)
        self.assertIs(cached["a"], self.store.peek("a"))
        stats = self.provider.stats()
        self.assertEqual(stats.cache_hits, 2)
        self.assertEqual(stats.cache_misses, 2)
        self.assertEqual(stats.encoder_batches, 1)
        self.assertEqual(stats.encoded_images, 2)

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
