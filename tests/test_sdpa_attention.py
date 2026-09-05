import unittest

import torch

from nanovllm.layers.attention import SDPAAttention
from nanovllm.utils.context import reset_context, set_context


class SDPAAttentionTests(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_packed_prefill_keeps_requests_isolated(self):
        attention = SDPAAttention(num_heads=2, head_dim=4, scale=0.5, num_kv_heads=1)
        torch.manual_seed(7)
        query = torch.randn(5, 2, 4)
        key = torch.randn(5, 1, 4)
        value = torch.randn(5, 1, 4)
        cu = torch.tensor([0, 2, 5], dtype=torch.int32)
        set_context(True, cu, cu, 3, 3, torch.empty(0, dtype=torch.int32))

        actual = attention(query, key, value)
        expected = torch.cat(
            [attention._sdpa(query[start:end], key[start:end], value[start:end], causal=True)
             for start, end in ((0, 2), (2, 5))],
            dim=0,
        )

        torch.testing.assert_close(actual, expected)

    def test_gather_paged_kv_respects_block_table_order_and_length(self):
        attention = SDPAAttention(num_heads=2, head_dim=2, scale=1.0, num_kv_heads=1)
        cache = torch.arange(3 * 4 * 1 * 2, dtype=torch.float32).reshape(3, 4, 1, 2)
        block_table = torch.tensor([2, 0], dtype=torch.int32)

        actual = attention._gather_paged(cache, block_table, sequence_length=6)
        expected = torch.cat((cache[2], cache[0]), dim=0)[:6]

        torch.testing.assert_close(actual, expected)

    def test_prefix_prefill_mask_offsets_queries_after_cached_tokens(self):
        mask = SDPAAttention._prefix_causal_mask(
            query_length=2,
            key_length=5,
            device=torch.device("cpu"),
        )

        expected = torch.tensor(
            [[True, True, True, True, False],
             [True, True, True, True, True]],
        )
        torch.testing.assert_close(mask, expected)

    def test_chunked_prefill_calls_sdpa_with_full_prompt_shape(self):
        attention = SDPAAttention(num_heads=2, head_dim=2, scale=1.0, num_kv_heads=1)
        attention.k_cache = torch.zeros(2, 4, 1, 2)
        attention.v_cache = torch.zeros_like(attention.k_cache)
        query = torch.randn(2, 2, 2)
        key = torch.randn(2, 1, 2)
        value = torch.randn(2, 1, 2)
        cu_q = torch.tensor([0, 2], dtype=torch.int32)
        cu_k = torch.tensor([0, 5], dtype=torch.int32)
        set_context(
            True,
            cu_q,
            cu_k,
            2,
            5,
            slot_mapping=torch.tensor([3, 4], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
            full_context_lens=torch.tensor([8], dtype=torch.int32),
            query_start_lens=torch.tensor([3], dtype=torch.int32),
        )

        with unittest.mock.patch.object(
            attention,
            "_sdpa",
            wraps=attention._sdpa,
        ) as sdpa, unittest.mock.patch(
            "nanovllm.layers.attention.store_kvcache"
        ):
            output = attention(query, key, value)

        padded_query, padded_key, padded_value = sdpa.call_args.args
        self.assertEqual(tuple(padded_query.shape), (8, 2, 2))
        self.assertEqual(tuple(padded_key.shape), (8, 1, 2))
        self.assertEqual(tuple(padded_value.shape), (8, 1, 2))
        self.assertEqual(tuple(output.shape), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
