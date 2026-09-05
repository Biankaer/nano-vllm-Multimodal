import gc
import unittest

import torch

from nanovllm.layers.paged_kv.pytorch_backend import PyTorchPagedKVBackend
from nanovllm.layers.paged_kv.triton_backend import TritonPagedKVBackend
from nanovllm.layers.paged_kv.cuda_backend import CUDAPagedKVBackend


class TritonBackendConstructionTests(unittest.TestCase):
    def test_backend_has_stable_name_without_compiling_kernel(self):
        self.assertEqual(TritonPagedKVBackend().name, "triton")


class CUDABackendConstructionTests(unittest.TestCase):
    def test_backend_name_does_not_eagerly_compile(self):
        self.assertEqual(CUDAPagedKVBackend().name, "cuda")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonPagedKVParityTests(unittest.TestCase):
    def test_store_and_gather_match_pytorch_reference(self):
        torch.manual_seed(7)
        reference = PyTorchPagedKVBackend()
        triton_backend = TritonPagedKVBackend()
        for dtype in (torch.float16, torch.bfloat16):
            for heads, head_dim in ((1, 64), (2, 128), (8, 64)):
                with self.subTest(dtype=dtype, heads=heads, head_dim=head_dim):
                    key = torch.randn(9, heads, head_dim, device="cuda", dtype=dtype)
                    value = torch.randn_like(key)
                    slots = torch.tensor([0, 17, -1, 4, 31, 8, 23, 2, 15], device="cuda", dtype=torch.int32)
                    reference_k = torch.full((4, 8, heads, head_dim), -3, device="cuda", dtype=dtype)
                    reference_v = torch.full_like(reference_k, -3)
                    actual_k = reference_k.clone()
                    actual_v = reference_v.clone()

                    reference.store(key, value, reference_k, reference_v, slots)
                    triton_backend.store(key, value, actual_k, actual_v, slots)

                    self.assertTrue(torch.equal(actual_k, reference_k))
                    self.assertTrue(torch.equal(actual_v, reference_v))
                    table = torch.tensor([3, 1, 0], device="cuda", dtype=torch.int32)
                    expected = reference.gather(reference_k, table, 19)
                    actual = triton_backend.gather(actual_k, table, 19)
                    self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CUDAPagedKVParityTests(unittest.TestCase):
    def test_store_accepts_fused_qkv_noncontiguous_token_stride(self):
        torch.manual_seed(29)
        backend = CUDAPagedKVBackend()
        reference = PyTorchPagedKVBackend()
        packed_qkv = torch.randn(
            9, 8, 64, dtype=torch.float16, device="cuda"
        )
        key = packed_qkv[:, 4:6]
        value = packed_qkv[:, 6:8]
        self.assertGreater(key.stride(0), key.shape[1] * key.shape[2])
        slots = torch.tensor(
            [0, 17, 4, 31, 8, 23, 2, 15, 9],
            dtype=torch.int32,
            device="cuda",
        )
        expected_k = torch.full(
            (4, 8, 2, 64), -3, dtype=torch.float16, device="cuda"
        )
        expected_v = torch.full_like(expected_k, -3)
        actual_k = expected_k.clone()
        actual_v = expected_v.clone()

        reference.store(key, value, expected_k, expected_v, slots)
        backend.store(key, value, actual_k, actual_v, slots)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual_k, expected_k))
        self.assertTrue(torch.equal(actual_v, expected_v))

    def test_store_gather_full_parity_matrix(self):
        torch.manual_seed(23)
        backend = CUDAPagedKVBackend()
        reference = PyTorchPagedKVBackend()
        num_blocks = 41
        block_size = 16
        num_slots = num_blocks * block_size

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for heads in (1, 2, 8):
                for head_dim in (64, 80, 128, 256):
                    for tokens in (1, 8, 257):
                        for mapping_kind in ("contiguous", "random_with_skips"):
                            with self.subTest(
                                dtype=dtype,
                                heads=heads,
                                head_dim=head_dim,
                                tokens=tokens,
                                mapping=mapping_kind,
                            ):
                                key = torch.randn(
                                    tokens,
                                    heads,
                                    head_dim,
                                    device="cuda",
                                    dtype=dtype,
                                )
                                value = torch.randn_like(key)
                                if mapping_kind == "contiguous":
                                    slots = torch.arange(
                                        tokens, device="cuda", dtype=torch.int32
                                    )
                                else:
                                    slots = torch.randperm(
                                        num_slots, device="cuda", dtype=torch.int32
                                    )[:tokens]
                                    slots[::11] = -1
                                expected_k = torch.full(
                                    (num_blocks, block_size, heads, head_dim),
                                    -3,
                                    device="cuda",
                                    dtype=dtype,
                                )
                                expected_v = torch.full_like(expected_k, -3)
                                actual_k = expected_k.clone()
                                actual_v = expected_v.clone()

                                stream = torch.cuda.Stream()
                                stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):
                                    reference.store(
                                        key, value, expected_k, expected_v, slots
                                    )
                                    backend.store(
                                        key, value, actual_k, actual_v, slots
                                    )
                                stream.synchronize()

                                self.assertTrue(torch.equal(actual_k, expected_k))
                                self.assertTrue(torch.equal(actual_v, expected_v))
                                table = torch.randperm(
                                    num_blocks, device="cuda", dtype=torch.int32
                                )
                                sequence_length = min(tokens, num_slots)
                                self.assertTrue(
                                    torch.equal(
                                        backend.gather(
                                            actual_k, table, sequence_length
                                        ),
                                        reference.gather(
                                            expected_k, table, sequence_length
                                        ),
                                    )
                                )

    def test_runtime_allocation_store_and_gather_match_reference(self):
        torch.manual_seed(11)
        backend = CUDAPagedKVBackend()
        reference = PyTorchPagedKVBackend()
        allocation = backend.allocate(
            (2, 5, 8, 2, 128), dtype=torch.bfloat16, device="cuda"
        )
        self.assertIsNotNone(allocation.owner)
        self.assertEqual(allocation.tensor.shape, (2, 5, 8, 2, 128))
        allocation.tensor.fill_(-5)
        key_cache, value_cache = allocation.tensor[0], allocation.tensor[1]
        expected_k = key_cache.clone()
        expected_v = value_cache.clone()
        key = torch.randn(10, 2, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        slots = torch.tensor(
            [0, 9, 17, -1, 31, 2, 7, 20, 39, 12],
            device="cuda",
            dtype=torch.int32,
        )

        reference.store(key, value, expected_k, expected_v, slots)
        backend.store(key, value, key_cache, value_cache, slots)

        self.assertTrue(torch.equal(key_cache, expected_k))
        self.assertTrue(torch.equal(value_cache, expected_v))
        table = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)
        self.assertTrue(torch.equal(
            backend.gather(key_cache, table, 19),
            reference.gather(expected_k, table, 19),
        ))

    def test_runtime_owner_async_copies_pinned_slot_mapping(self):
        backend = CUDAPagedKVBackend()
        allocation = backend.allocate(
            (2, 1, 8, 1, 64), dtype=torch.float16, device="cuda"
        )
        host_slots = torch.tensor([0, 3, 7], dtype=torch.int32).pin_memory()

        with torch.cuda.stream(torch.cuda.Stream()):
            device_slots = backend.copy_slots_async(allocation, host_slots)
            torch.cuda.current_stream().synchronize()

        self.assertEqual(device_slots.device.type, "cuda")
        self.assertTrue(torch.equal(device_slots.cpu(), host_slots))

    def test_runtime_owner_async_copy_accepts_empty_slot_mapping(self):
        backend = CUDAPagedKVBackend()
        allocation = backend.allocate(
            (2, 1, 8, 1, 64), dtype=torch.float16, device="cuda"
        )
        host_slots = torch.empty(0, dtype=torch.int32, pin_memory=True)

        device_slots = backend.copy_slots_async(allocation, host_slots)

        self.assertEqual(device_slots.device.type, "cuda")
        self.assertEqual(device_slots.dtype, torch.int32)
        self.assertEqual(device_slots.numel(), 0)

    def test_runtime_owner_reuses_grown_slot_buffer_safely(self):
        backend = CUDAPagedKVBackend()
        allocation = backend.allocate(
            (2, 1, 8, 1, 64), dtype=torch.float16, device="cuda"
        )
        first_host = torch.tensor([0, 3, 7, 2], dtype=torch.int32).pin_memory()
        second_host = torch.tensor([6, 1], dtype=torch.int32).pin_memory()

        first = backend.copy_slots_async(allocation, first_host)
        torch.cuda.current_stream().synchronize()
        second = backend.copy_slots_async(allocation, second_host)
        torch.cuda.current_stream().synchronize()

        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.equal(second.cpu(), second_host))

    def test_async_copy_retains_ephemeral_pinned_host_tensor(self):
        backend = CUDAPagedKVBackend()
        allocation = backend.allocate(
            (2, 1, 8, 1, 64), dtype=torch.float16, device="cuda"
        )
        host_slots = torch.tensor([0, 3, 7], dtype=torch.int32).pin_memory()

        device_slots = backend.copy_slots_async(allocation, host_slots)
        self.assertEqual(allocation.owner.pending_host_copy_count(), 1)
        del host_slots
        gc.collect()

        torch.cuda.current_stream().synchronize()
        self.assertEqual(device_slots.tolist(), [0, 3, 7])

    def test_repeated_async_slot_copy_and_store_matches_decode_sequence(self):
        backend = CUDAPagedKVBackend()
        allocation = backend.allocate(
            (2, 1, 16, 1, 64), dtype=torch.float16, device="cuda"
        )
        allocation.tensor.fill_(-1)
        key_cache, value_cache = allocation.tensor[0], allocation.tensor[1]

        for step in range(64):
            key = torch.full(
                (1, 1, 64), step, dtype=torch.float16, device="cuda"
            )
            value = -key
            host_slot = torch.tensor(
                [step % 16], dtype=torch.int32, pin_memory=True
            )
            device_slot = backend.copy_slots_async(allocation, host_slot)
            del host_slot
            backend.store(
                key, value, key_cache, value_cache, device_slot
            )

        torch.cuda.synchronize()
        expected = torch.arange(48, 64, dtype=torch.float16, device="cuda")
        self.assertTrue(
            torch.equal(key_cache[0, :, 0, 0], expected),
            msg=f"actual key slots: {key_cache[0, :, 0, 0].tolist()}",
        )
        self.assertTrue(torch.equal(value_cache[0, :, 0, 0], -expected))


if __name__ == "__main__":
    unittest.main()
