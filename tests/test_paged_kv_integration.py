import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.layers.attention import Attention, SDPAAttention, store_kvcache
from nanovllm.layers.paged_kv import PagedKVAllocation
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import reset_context, set_context


class RecordingBackend:
    name = "recording"

    def __init__(self):
        self.store_calls = []
        self.gather_calls = []

    def store(self, key, value, key_cache, value_cache, slot_mapping):
        self.store_calls.append((key, value, key_cache, value_cache, slot_mapping))

    def gather(self, cache, block_table, sequence_length):
        self.gather_calls.append((cache, block_table, sequence_length))
        return cache.index_select(0, block_table[:1].to(torch.long)).flatten(0, 1)[:sequence_length]


class AttentionBackendIntegrationTests(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_legacy_store_wrapper_delegates_to_triton_backend(self):
        key = torch.randn(2, 1, 4)
        value = torch.randn_like(key)
        key_cache = torch.zeros(2, 4, 1, 4)
        value_cache = torch.zeros_like(key_cache)
        slots = torch.tensor([0, 1], dtype=torch.int32)

        with patch(
            "nanovllm.layers.attention.TritonPagedKVBackend"
        ) as backend_type:
            store_kvcache(key, value, key_cache, value_cache, slots)

        backend_type.return_value.store.assert_called_once_with(
            key, value, key_cache, value_cache, slots
        )

    def test_flash_attention_routes_store_through_bound_backend(self):
        backend = RecordingBackend()
        attention = Attention(2, 4, 0.5, 1)
        attention.paged_kv_backend = backend
        attention.k_cache = torch.zeros(2, 4, 1, 4)
        attention.v_cache = torch.zeros_like(attention.k_cache)
        query = torch.randn(2, 2, 4)
        key = torch.randn(2, 1, 4)
        value = torch.randn_like(key)
        cu = torch.tensor([0, 2], dtype=torch.int32)
        slots = torch.tensor([0, 1], dtype=torch.int32)
        set_context(True, cu, cu, 2, 2, slot_mapping=slots)

        with patch(
            "nanovllm.layers.attention.flash_attn_varlen_func",
            return_value=torch.zeros_like(query),
        ):
            attention(query, key, value)

        self.assertEqual(len(backend.store_calls), 1)
        self.assertIs(backend.store_calls[0][-1], slots)

    def test_sdpa_routes_store_and_gather_through_bound_backend(self):
        backend = RecordingBackend()
        attention = SDPAAttention(2, 4, 0.5, 1)
        attention.paged_kv_backend = backend
        attention.k_cache = torch.randn(2, 4, 1, 4)
        attention.v_cache = torch.randn_like(attention.k_cache)
        query = torch.randn(1, 2, 4)
        key = torch.randn(1, 1, 4)
        value = torch.randn_like(key)
        slots = torch.tensor([4], dtype=torch.int32)
        table = torch.tensor([[1]], dtype=torch.int32)
        set_context(
            False,
            slot_mapping=slots,
            context_lens=torch.tensor([1], dtype=torch.int32),
            block_tables=table,
        )

        output = attention(query, key, value)

        self.assertEqual(tuple(output.shape), (1, 2, 4))
        self.assertEqual(len(backend.store_calls), 1)
        self.assertEqual(len(backend.gather_calls), 2)
        self.assertTrue(all(call[2] == 1 for call in backend.gather_calls))


class ModelRunnerPagedKVTests(unittest.TestCase):
    def test_allocation_retains_owner_and_binds_backend_to_attention(self):
        owner = object()

        class AllocationBackend:
            name = "allocation"

            def __init__(self):
                self.shapes = []

            def allocate(self, shape, *, dtype, device):
                self.shapes.append((tuple(shape), dtype, torch.device(device)))
                return PagedKVAllocation(
                    torch.empty(tuple(shape), dtype=dtype), owner=owner,
                )

        backend = AllocationBackend()
        attention = Attention(2, 2, 1.0, 1)
        text_config = SimpleNamespace(
            num_key_value_heads=1,
            head_dim=2,
            hidden_size=4,
            num_attention_heads=2,
            num_hidden_layers=1,
        )
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3", **text_config.__dict__),
            gpu_memory_utilization=0.9,
            vision_feature_cache_bytes=0,
            num_kvcache_blocks=-1,
        )
        runner.block_size = 4
        runner.world_size = 1
        runner.rank = 0
        runner.model_dtype = torch.float32
        runner.vision_provider = None
        runner.model = SimpleNamespace(modules=lambda: [attention])
        runner.paged_kv_backend = backend

        with (
            patch(
                "nanovllm.engine.model_runner.torch.cuda.mem_get_info",
                return_value=(900, 1000),
            ),
            patch(
                "nanovllm.engine.model_runner.torch.cuda.memory_stats",
                return_value={
                    "allocated_bytes.all.peak": 100,
                    "allocated_bytes.all.current": 100,
                },
            ),
        ):
            runner.allocate_kv_cache()

        self.assertIs(runner.kv_cache_allocation.owner, owner)
        self.assertIs(runner.paged_kv_cache_owner, owner)
        self.assertIs(attention.paged_kv_backend, backend)
        self.assertEqual(attention.k_cache.data_ptr(), runner.kv_cache[0, 0].data_ptr())

    def test_cuda_backend_uses_native_async_slot_copy(self):
        copied = torch.tensor([7], dtype=torch.int32)

        class PinnedSlots:
            dtype = torch.int32

            @staticmethod
            def is_pinned():
                return True

            @staticmethod
            def tolist():
                return [3, -1, 9]

        host_slots = PinnedSlots()

        class Backend:
            name = "cuda"

            def __init__(self):
                self.calls = []

            def copy_slots_async(self, allocation, host_slots):
                self.calls.append((allocation, host_slots))
                return copied

        runner = ModelRunner.__new__(ModelRunner)
        runner.paged_kv_backend = Backend()
        runner.kv_cache_allocation = PagedKVAllocation(torch.empty(0), owner=object())

        with patch(
            "nanovllm.engine.model_runner.torch.tensor",
            return_value=host_slots,
        ) as make_tensor:
            actual = runner.copy_slot_mapping_to_device([3, -1, 9])

        self.assertIs(actual, copied)
        make_tensor.assert_called_once_with(
            [3, -1, 9],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        allocation, host = runner.paged_kv_backend.calls[0]
        self.assertIs(allocation, runner.kv_cache_allocation)
        self.assertEqual(host.dtype, torch.int32)
        self.assertTrue(host.is_pinned())
        self.assertEqual(host.tolist(), [3, -1, 9])

    def test_cuda_warmup_slot_copy_works_before_native_allocation(self):
        copied = torch.empty(0, dtype=torch.int32)

        class HostSlots:
            def cuda(self, *, non_blocking):
                self.non_blocking = non_blocking
                return copied

        host_slots = HostSlots()
        runner = ModelRunner.__new__(ModelRunner)
        runner.paged_kv_backend = SimpleNamespace(name="cuda")

        with patch(
            "nanovllm.engine.model_runner.torch.tensor",
            return_value=host_slots,
        ):
            actual = runner.copy_slot_mapping_to_device([])

        self.assertIs(actual, copied)
        self.assertTrue(host_slots.non_blocking)


if __name__ == "__main__":
    unittest.main()
