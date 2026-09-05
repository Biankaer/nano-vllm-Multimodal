import torch
import torch.nn.functional as F
from torch import nn

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.paged_kv.triton_backend import TritonPagedKVBackend
from nanovllm.utils.context import get_context


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """把本轮新产生的 K/V 写入 Paged KV Cache。

    参数约定：
    - key/value: [N, num_kv_heads, head_dim]，N 是本轮需要写 Cache 的 token 数。
    - k_cache/v_cache: 物理 KV block；Kernel 将其 slot 维逻辑展平。
    - slot_mapping: [N]，第 i 项给出第 i 个输入 token 的物理 slot，-1 表示跳过。

    该函数只负责数据面写入。slot 是否越界、不同 token 是否错误地映射到同一
    slot，必须由上游 Block Manager 和 Model Runner 保证。
    """
    TritonPagedKVBackend().store(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
    )


class Attention(nn.Module):
    """连接 QKV、Paged KV Cache 与 FlashAttention 的推理 Attention 层。

    这个模块不负责 Q/K/V 线性投影，而负责：
    1. 把本轮新 K/V 写入调度器分配的物理 Cache slot；
    2. 根据 context 选择 Prefill 或 Decode Kernel；
    3. 通过 cu_seqlens、context_lens 和 block_tables 描述变长请求及物理页映射。

    常见输入布局：
    - q: [num_tokens, num_heads, head_dim]
    - k/v: [num_tokens, num_kv_heads, head_dim]
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()

        # Query Head 数。MHA 中它等于 num_kv_heads；GQA/MQA 中通常更大。
        self.num_heads = num_heads

        # 单个 Attention Head 的维度，例如 128。
        self.head_dim = head_dim

        # softmax 前 QK^T 的缩放系数，通常为 1 / sqrt(head_dim)。
        self.scale = scale

        # Key/Value Head 数，直接影响 KV Cache shape 和显存占用。
        self.num_kv_heads = num_kv_heads

        # 空 Tensor 只是占位符。Model Runner 会在模型初始化/运行前，把每一层
        # 对应的预分配物理 KV Cache Tensor 赋给这两个属性。
        self.k_cache = self.v_cache = torch.tensor([])
        self.paged_kv_backend = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # context 由 Scheduler/Block Manager/Model Runner 在调用模型前构造。
        # 它描述当前是 Prefill 还是 Decode，以及 token/请求到物理 Cache 的映射。
        context = get_context()

        # 局部变量只增加 Tensor 引用，不会复制整块 KV Cache。
        k_cache, v_cache = self.k_cache, self.v_cache

        # Prefill 和 Decode 都会产生新的 K/V：
        # - Prefill 写入 Prompt token，供后续 Decode 使用；
        # - Decode 每轮写入新 token，供下一轮使用。
        # 因此 Cache Store 是两个分支共享的前置步骤。
        if k_cache.numel() and v_cache.numel():
            if self.paged_kv_backend is None:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                self.paged_kv_backend.store(
                    k, v, k_cache, v_cache, context.slot_mapping
                )

        if context.is_prefill:
            # 普通首次 Prefill 可以直接使用本轮连续的 k/v。
            # block_tables 非空时，通常存在 Prefix Cache、Chunked Prefill 或历史 KV，
            # Query 需要通过物理 Block Table 同时读取旧 Cache 和本轮刚写入的新 KV。
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            # Varlen FlashAttention 避免把不同长度请求 padding 到同一长度：
            # cu_seqlens_q/k 给出各请求在 packed Q/K/V 中的累计边界；
            # max_seqlen_q/k 用于 Kernel 规划；block_table 映射逻辑块到物理块。
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            # Decode 时每个请求本轮通常只有一个 Query token。
            # q.unsqueeze(1) 将 [batch, heads, dim] 变成 [batch, 1, heads, dim]。
            # context_lens 指明每个请求已有多少有效 KV token；block_tables 指明
            # 这些逻辑 token 分布在哪些物理 Cache blocks 中。
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )

        # Prefill 输出对应本轮所有 Query token；Decode 输出对应每个请求的新 Query。
        return o


class SDPAAttention(nn.Module):
    """HF-aligned SDPA with NanoInfer's paged KV storage contract.

    This is a numerical-reference backend for Qwen3-VL. It keeps paged KV as
    the source of truth, but gathers each scheduled sequence into contiguous
    K/V before calling PyTorch SDPA with the same GQA and causal semantics as
    Transformers 4.57.6.
    """

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.paged_kv_backend = None

    @staticmethod
    def _prefix_causal_mask(query_length: int, key_length: int, device) -> torch.Tensor:
        prefix_length = key_length - query_length
        if prefix_length < 0:
            raise ValueError("key length cannot be shorter than query length")
        query_positions = torch.arange(query_length, device=device) + prefix_length
        key_positions = torch.arange(key_length, device=device)
        return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    @staticmethod
    def _gather_paged(
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        if sequence_length < 0:
            raise ValueError("sequence length must be non-negative")
        block_size = cache.shape[1]
        required_blocks = (sequence_length + block_size - 1) // block_size
        if required_blocks == 0:
            return cache.new_empty((0, *cache.shape[2:]))
        block_ids = block_table[:required_blocks].to(device=cache.device, dtype=torch.long)
        if not block_ids.is_cuda and torch.any(block_ids < 0):
            raise ValueError("block table does not cover the requested sequence")
        return cache.index_select(0, block_ids).flatten(0, 1)[:sequence_length]

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
            is_causal=causal,
            enable_gqa=True,
        )
        return output.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

    def _gather_cache(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        if self.paged_kv_backend is None:
            return self._gather_paged(cache, block_table, sequence_length)
        return self.paged_kv_backend.gather(
            cache, block_table, sequence_length
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.paged_kv_backend is None:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                self.paged_kv_backend.store(
                    k, v, k_cache, v_cache, context.slot_mapping
                )

        outputs = []
        if context.is_prefill:
            request_count = context.cu_seqlens_q.numel() - 1
            for request_index in range(request_count):
                q_start = int(context.cu_seqlens_q[request_index])
                q_end = int(context.cu_seqlens_q[request_index + 1])
                k_start = int(context.cu_seqlens_k[request_index])
                k_end = int(context.cu_seqlens_k[request_index + 1])
                query = q[q_start:q_end]
                query_length = q_end - q_start
                key_length = k_end - k_start
                full_length = (
                    int(context.full_context_lens[request_index])
                    if context.full_context_lens is not None
                    else key_length
                )
                query_start = (
                    int(context.query_start_lens[request_index])
                    if context.query_start_lens is not None
                    else key_length - query_length
                )
                if context.block_tables is None:
                    key = k[k_start:k_end]
                    value = v[k_start:k_end]
                else:
                    key = self._gather_cache(
                        k_cache, context.block_tables[request_index], key_length
                    )
                    value = self._gather_cache(
                        v_cache, context.block_tables[request_index], key_length
                    )
                query_suffix = full_length - query_start - query_length
                key_suffix = full_length - key_length
                if min(query_start, query_suffix, key_suffix) < 0:
                    raise ValueError("SDPA received invalid full-context metadata")
                query = torch.cat(
                    (
                        query.new_zeros(query_start, self.num_heads, self.head_dim),
                        query,
                        query.new_zeros(query_suffix, self.num_heads, self.head_dim),
                    ),
                    dim=0,
                )
                key = torch.cat(
                    (key, key.new_zeros(key_suffix, self.num_kv_heads, self.head_dim)),
                    dim=0,
                )
                value = torch.cat(
                    (value, value.new_zeros(key_suffix, self.num_kv_heads, self.head_dim)),
                    dim=0,
                )
                outputs.append(
                    self._sdpa(query, key, value, causal=full_length > 1)
                    [query_start:query_start + query_length]
                )
        else:
            for request_index in range(q.shape[0]):
                key_length = int(context.context_lens[request_index])
                key = self._gather_cache(
                    k_cache, context.block_tables[request_index], key_length
                )
                value = self._gather_cache(
                    v_cache, context.block_tables[request_index], key_length
                )
                outputs.append(
                    self._sdpa(
                        q[request_index:request_index + 1],
                        key,
                        value,
                        causal=False,
                    )
                )
        return torch.cat(outputs, dim=0)
