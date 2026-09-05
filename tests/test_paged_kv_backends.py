import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.layers.paged_kv import (
    create_paged_kv_backend,
    index_value_validation_enabled,
    resolve_paged_kv_backend_name,
    validate_gather_inputs,
    validate_store_inputs,
)


def store_inputs():
    key = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    key_cache = torch.full((3, 4, 2, 4), -1.0)
    value_cache = torch.full_like(key_cache, -1.0)
    slots = torch.tensor([0, 5, -1], dtype=torch.int32)
    return key, value, key_cache, value_cache, slots


class PagedKVValidationTests(unittest.TestCase):
    def test_gpu_index_value_validation_is_debug_opt_in(self):
        cuda_tensor = SimpleNamespace(is_cuda=True)
        cpu_tensor = SimpleNamespace(is_cuda=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOVLLM_PAGED_KV_VALIDATE_INDICES", None)
            self.assertFalse(index_value_validation_enabled(cuda_tensor))
            self.assertTrue(index_value_validation_enabled(cpu_tensor))
        with patch.dict(
            os.environ,
            {"NANOVLLM_PAGED_KV_VALIDATE_INDICES": "1"},
        ):
            self.assertTrue(index_value_validation_enabled(cuda_tensor))

    def test_valid_store_contract_returns_dimensions(self):
        dimensions = validate_store_inputs(*store_inputs())

        self.assertEqual(dimensions.num_tokens, 3)
        self.assertEqual(dimensions.num_kv_heads, 2)
        self.assertEqual(dimensions.head_dim, 4)
        self.assertEqual(dimensions.num_slots, 12)

    def test_store_rejects_shape_dtype_and_layout_mismatches(self):
        key, value, key_cache, value_cache, slots = store_inputs()
        cases = [
            ((key[:2], value, key_cache, value_cache, slots), "matching shape"),
            ((key, value, key_cache[:, :, :1], value_cache, slots), "cache shape"),
            ((key, value, key_cache, value_cache, slots.to(torch.int64)), "int32"),
            ((key.transpose(1, 2), value.transpose(1, 2), key_cache, value_cache, slots), "inner dimensions"),
        ]

        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_store_inputs(*arguments)

    def test_store_rejects_out_of_range_slots_but_accepts_minus_one(self):
        key, value, key_cache, value_cache, slots = store_inputs()
        validate_store_inputs(key, value, key_cache, value_cache, slots)
        slots[1] = 12

        with self.assertRaisesRegex(ValueError, "slot"):
            validate_store_inputs(key, value, key_cache, value_cache, slots)

    def test_gather_validates_length_table_and_dtype(self):
        cache = torch.zeros(3, 4, 2, 4)
        table = torch.tensor([2, 0], dtype=torch.int32)

        dimensions = validate_gather_inputs(cache, table, 6)

        self.assertEqual(dimensions.required_blocks, 2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_gather_inputs(cache, table, -1)
        with self.assertRaisesRegex(ValueError, "int32"):
            validate_gather_inputs(cache, table.to(torch.int64), 6)
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_gather_inputs(cache, table[:1], 6)
        with self.assertRaisesRegex(ValueError, "physical block"):
            validate_gather_inputs(cache, torch.tensor([3, 0], dtype=torch.int32), 6)


class BackendResolutionTests(unittest.TestCase):
    def test_explicit_backends_are_preserved(self):
        for name in ("pytorch", "triton", "cuda"):
            with self.subTest(name=name):
                self.assertEqual(
                    resolve_paged_kv_backend_name(
                        name,
                        cuda_available=True,
                        cuda_extension_available=True,
                    ),
                    name,
                )

    def test_auto_prefers_cuda_extension_then_triton(self):
        self.assertEqual(
            resolve_paged_kv_backend_name(
                "auto", cuda_available=True, cuda_extension_available=True,
            ),
            "cuda",
        )
        self.assertEqual(
            resolve_paged_kv_backend_name(
                "auto", cuda_available=True, cuda_extension_available=False,
            ),
            "triton",
        )
        self.assertEqual(
            resolve_paged_kv_backend_name(
                "auto", cuda_available=False, cuda_extension_available=False,
            ),
            "pytorch",
        )

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "paged_kv_backend"):
            resolve_paged_kv_backend_name(
                "flash", cuda_available=True, cuda_extension_available=True,
            )


class PyTorchBackendTests(unittest.TestCase):
    def test_store_scatters_and_skips_minus_one(self):
        key, value, key_cache, value_cache, slots = store_inputs()
        backend = create_paged_kv_backend("pytorch")

        backend.store(key, value, key_cache, value_cache, slots)

        self.assertTrue(torch.equal(key_cache.flatten(0, 1)[0], key[0]))
        self.assertTrue(torch.equal(key_cache.flatten(0, 1)[5], key[1]))
        self.assertTrue(torch.equal(value_cache.flatten(0, 1)[5], value[1]))
        self.assertTrue(torch.all(key_cache.flatten(0, 1)[1] == -1))

    def test_gather_follows_nonsequential_physical_blocks(self):
        cache = torch.arange(3 * 4 * 2 * 4, dtype=torch.float32).reshape(3, 4, 2, 4)
        table = torch.tensor([2, 0], dtype=torch.int32)
        backend = create_paged_kv_backend("pytorch")

        actual = backend.gather(cache, table, 6)
        expected = torch.cat((cache[2], cache[0]), dim=0)[:6]

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
