import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # launch grid 是 (N,)，因此一个 Triton program 负责一个输入 token。
    # 注意：Triton program 不等于单个 CUDA thread；program 内会并行处理 D 个元素。
    idx = tl.program_id(0)

    # slot_mapping[idx] 把“本轮第 idx 个 token”映射到 KV Cache 的物理 slot。
    # 典型计算来源是：physical_block_id * block_size + block_offset。
    slot = tl.load(slot_mapping_ptr + idx)

    # -1 是无效 slot 的哨兵值，常用于 padding、被过滤 token 或未分配位置。
    # 该 token 不应该修改 KV Cache。
    if slot == -1: return

    # key/value 的逻辑 shape 是 [N, num_kv_heads, head_dim]。
    # D = num_kv_heads * head_dim，所以每个 token 的后两维被视为连续的一维向量。
    # idx * key_stride 定位第 idx 个 token；arange(0, D) 枚举其全部 KV 元素。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 从本轮 QKV projection 的输出中读取当前 token 的全部 K/V。
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # Cache 被逻辑展平为 [num_slots, D]；slot * D 是目标物理 slot 的起点。
    # 若物理布局原本是 [num_blocks, block_size, D]，前两维也可展平成 num_slots。
    cache_offsets = slot * D + tl.arange(0, D)

    # 将当前 token 的 K/V 写入相同物理 slot，供后续 Prefill/Decode Attention 读取。
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


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
    # 这里的 num_kv_heads 比 num_heads 更准确：GQA/MQA 中 Q Head 数可能更多。
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim

    # Kernel 使用连续的 tl.arange(0, D) 搬运一个 token 的全部 KV，因此要求：
    # 1. head_dim 内连续；2. 相邻 KV Head 紧密排列。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim

    # 该检查隐含 Cache 布局近似为 [num_blocks, block_size, D]：
    # 相邻 slot 在物理内存中相隔 D 个元素。
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    # 每个输入 token 必须恰好有一个物理 slot 映射。
    assert slot_mapping.numel() == N

    # 启动 N 个 Triton program：一个 program 负责一个 token 的 K/V Cache 写入。
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
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
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

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
