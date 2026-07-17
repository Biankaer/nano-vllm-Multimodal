# pickle 用来把方法名和参数序列化到共享内存，供 tensor parallel 子进程读取。
import pickle
from dataclasses import dataclass

# torch 是模型执行、张量创建、CUDA graph、显存查询的核心依赖。
import torch

# torch.distributed 用于 tensor parallel 进程组、barrier、all_reduce/gather 等通信。
import torch.distributed as dist

# Event 类型用于标注进程间同步事件。
from multiprocessing.synchronize import Event

# SharedMemory 用于 rank 0 和其它 tensor parallel rank 之间传递命令。
from multiprocessing.shared_memory import SharedMemory

# Config 保存模型路径、并行规模、KV cache 参数等全局配置。
from nanovllm.config import Config

# Sequence 是单条请求的内部状态，ModelRunner 会读取它的 token、block_table 等。
from nanovllm.engine.sequence import Sequence

# 原生语言模型实现；Qwen2.5-VL 的视觉塔由独立 provider 持有。
from nanovllm.models.qwen25_vl import Qwen25VLForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.multimodal.qwen25_vl_processor import VisionInput
from nanovllm.multimodal.vision_provider import VisionFeatureProvider

# Sampler 根据 logits 和 temperature 采样下一个 token。
from nanovllm.layers.sampler import Sampler

# context 是一个轻量全局上下文，attention/lm_head 会从这里读取本轮执行信息。
from nanovllm.utils.context import set_context, get_context, reset_context

# load_model 负责从 safetensors 文件加载权重，并处理 QKV/gate_up 等合并权重。
from nanovllm.utils.loader import load_model


def resolve_model_dtype(hf_config) -> torch.dtype:
    dtype = getattr(hf_config, "dtype", None)
    if dtype is None:
        dtype = getattr(hf_config, "torch_dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"model config must define a torch dtype, got {dtype!r}")
    return dtype


def calculate_num_kvcache_blocks(
    *,
    total_bytes: int,
    gpu_memory_utilization: float,
    used_bytes: int,
    peak_bytes: int,
    current_bytes: int,
    block_bytes: int,
    reserved_bytes: int = 0,
) -> int:
    if block_bytes <= 0:
        raise ValueError("KV cache block bytes must be positive")
    if reserved_bytes < 0:
        raise ValueError("reserved bytes cannot be negative")
    available_bytes = (
        int(total_bytes * gpu_memory_utilization)
        - used_bytes
        - peak_bytes
        + current_bytes
        - reserved_bytes
    )
    return available_bytes // block_bytes


@dataclass(frozen=True, slots=True)
class VisualReplacement:
    output_row: int
    feature_key: str
    feature_row: int


@dataclass(frozen=True, slots=True)
class VisualReplacementPlan:
    replacements: tuple[VisualReplacement, ...]
    required_feature_keys: tuple[str, ...]
    prefix_skipped_features: int


def plan_visual_replacements(seqs: list[Sequence]) -> VisualReplacementPlan:
    replacements = []
    required_feature_keys = []
    skipped_feature_keys = set()
    packed_row_start = 0

    for seq in seqs:
        scheduled_start = seq.num_cached_tokens
        scheduled_end = scheduled_start + seq.num_scheduled_tokens
        metadata = seq.multimodal_metadata
        if metadata is not None:
            for span in metadata.visual_spans:
                span_end = span.start + span.length
                if span_end <= scheduled_start:
                    skipped_feature_keys.add(span.feature_key)
                    continue
                overlap_start = max(span.start, scheduled_start)
                overlap_end = min(span_end, scheduled_end)
                if overlap_start >= overlap_end:
                    continue
                if span.feature_key not in required_feature_keys:
                    required_feature_keys.append(span.feature_key)
                for token_index in range(overlap_start, overlap_end):
                    replacements.append(
                        VisualReplacement(
                            output_row=packed_row_start + token_index - scheduled_start,
                            feature_key=span.feature_key,
                            feature_row=token_index - span.start,
                        )
                    )
        packed_row_start += seq.num_scheduled_tokens

    return VisualReplacementPlan(
        replacements=tuple(replacements),
        required_feature_keys=tuple(required_feature_keys),
        prefix_skipped_features=len(skipped_feature_keys),
    )


def merge_planned_visual_embeddings(
    token_embeddings: torch.Tensor,
    plan: VisualReplacementPlan,
    features: dict[str, torch.Tensor],
) -> torch.Tensor:
    if token_embeddings.ndim != 2:
        raise ValueError("token embeddings must have shape [tokens, hidden_size]")
    if not plan.replacements:
        return token_embeddings

    token_count, hidden_size = token_embeddings.shape
    for replacement in plan.replacements:
        feature = features.get(replacement.feature_key)
        if feature is None:
            raise KeyError(f"missing visual feature: {replacement.feature_key}")
        if feature.ndim != 2 or feature.shape[1] != hidden_size:
            raise ValueError(
                "visual feature hidden size mismatch: "
                f"feature={tuple(feature.shape)}, hidden_size={hidden_size}"
            )
        if not 0 <= replacement.output_row < token_count:
            raise IndexError(f"visual output row is out of range: {replacement.output_row}")
        if not 0 <= replacement.feature_row < feature.shape[0]:
            raise IndexError(
                "visual feature row is out of range: "
                f"key={replacement.feature_key}, row={replacement.feature_row}"
            )

    merged = token_embeddings.clone()
    for replacement in plan.replacements:
        merged[replacement.output_row] = features[replacement.feature_key][
            replacement.feature_row
        ].to(device=merged.device, dtype=merged.dtype)
    return merged


def build_prefill_position_rows(
    seqs: list[Sequence],
    *,
    position_axes: int,
) -> tuple[tuple[int, ...], ...]:
    if position_axes not in (1, 3):
        raise ValueError("position_axes must be one or three")
    rows: list[list[int]] = [[] for _ in range(position_axes)]
    for seq in seqs:
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        metadata = seq.multimodal_metadata
        if position_axes == 3 and metadata is not None:
            prompt_length = len(metadata.position_ids[0])
            prompt_end = min(end, prompt_length)
            for axis, positions in enumerate(metadata.position_ids):
                if start < prompt_end:
                    rows[axis].extend(positions[start:prompt_end])
            continuation_start = max(start, prompt_length)
            if continuation_start < end:
                continuation_positions = range(
                    continuation_start + seq.mrope_position_delta,
                    end + seq.mrope_position_delta,
                )
                for row in rows:
                    row.extend(continuation_positions)
        else:
            positions = range(start, end)
            for row in rows:
                row.extend(positions)
    return tuple(tuple(row) for row in rows)


def build_decode_position_rows(
    seqs: list[Sequence],
    *,
    position_axes: int,
) -> tuple[tuple[int, ...], ...]:
    if position_axes not in (1, 3):
        raise ValueError("position_axes must be one or three")
    positions = tuple(
        len(seq) - 1 + (seq.mrope_position_delta if position_axes == 3 else 0)
        for seq in seqs
    )
    return tuple(positions for _ in range(position_axes))

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 保存全局配置对象。
        self.config = config

        # HuggingFace config 里有模型结构信息，例如层数、头数、dtype。
        hf_config = config.hf_config
        self.model_dtype = resolve_model_dtype(hf_config)

        # KV cache 的 block 大小。
        self.block_size = config.kvcache_block_size

        # 是否强制 eager；False 时 decode 可以使用 CUDA graph。
        self.enforce_eager = config.enforce_eager

        # tensor parallel 总进程数/总 GPU 数。
        self.world_size = config.tensor_parallel_size

        # 当前进程的 rank，rank 0 是主 rank。
        self.rank = rank

        # rank 0 持有 Event 列表，其它 rank 持有自己的 Event。
        self.event = event

        # 初始化 NCCL 进程组；所有 rank 都连接到同一个 TCP 地址。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # 当前 rank 使用同编号 CUDA 设备。
        torch.cuda.set_device(rank)

        # 保存原始默认 dtype，初始化结束后恢复。
        default_dtype = torch.get_default_dtype()

        # 将默认 dtype 切到模型 config 的 dtype，后续创建参数会用这个 dtype。
        torch.set_default_dtype(self.model_dtype)

        # 将默认设备切到 cuda，后续创建模型参数会直接在 GPU 上。
        torch.set_default_device("cuda")

        model_type = getattr(hf_config, "model_type", None)
        self.vision_provider = None
        if model_type == "qwen3":
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)
        elif model_type == "qwen2_5_vl":
            self.model = Qwen25VLForCausalLM(hf_config)
            load_model(self.model, config.model, include_prefixes=("model.",))
            self.vision_provider = VisionFeatureProvider.from_pretrained(
                config.model,
                byte_budget=config.vision_feature_cache_bytes,
                dtype=self.model_dtype,
                device=torch.device("cuda", rank),
                attn_implementation=config.vision_attn_implementation,
            )
        else:
            raise ValueError(f"unsupported native model type: {model_type!r}")

        # 创建采样器，只有 rank 0 会真正用采样结果。
        self.sampler = Sampler()

        # 先跑一次 warmup，触发 CUDA kernel 初始化并统计峰值显存。
        self.warmup_model()

        # 根据剩余显存分配 KV cache，并把每层 Attention 的 k_cache/v_cache 指向对应切片。
        self.allocate_kv_cache()

        # 如果不强制 eager，就为 decode 阶段捕获一组不同 batch size 的 CUDA graph。
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 初始化完成后把默认设备恢复到 CPU，避免外部不小心创建 CUDA 张量。
        torch.set_default_device("cpu")

        # 恢复默认 dtype。
        torch.set_default_dtype(default_dtype)

        # tensor_parallel_size > 1 时，需要建立共享内存通信。
        if self.world_size > 1:
            # rank 0 创建共享内存，负责写入要调用的方法和参数。
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()

            # 非 rank 0 打开同名共享内存，然后进入 loop 等待 rank 0 命令。
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.vision_provider is not None:
            del self.vision_provider
            self.vision_provider = None

        # 多卡时需要关闭共享内存。
        if self.world_size > 1:
            # 当前进程关闭共享内存句柄。
            self.shm.close()

            # 等待所有 rank 都关闭句柄。
            dist.barrier()

            # 只有创建者 rank 0 负责 unlink 共享内存对象。
            if self.rank == 0:
                self.shm.unlink()

        # 如果捕获过 CUDA graph，显式删除相关对象。
        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        # 等待当前设备所有 CUDA 工作完成。
        torch.cuda.synchronize()

        # 销毁 NCCL 进程组。
        dist.destroy_process_group()

    def loop(self):
        # 非 rank 0 子进程常驻循环，等待 rank 0 通过共享内存发命令。
        while True:
            # 从共享内存读出方法名和参数。
            method_name, args = self.read_shm()

            # 在当前 rank 上执行同名方法。
            self.call(method_name, *args)

            # 收到 exit 命令后跳出循环，子进程结束。
            if method_name == "exit":
                break

    def read_shm(self):
        # 只有 tensor parallel 子 rank 会读共享内存。
        assert self.world_size > 1 and self.rank > 0

        # 等待 rank 0 写入数据并 set event。
        self.event.wait()

        # 共享内存前 4 字节保存 payload 长度。
        n = int.from_bytes(self.shm.buf[0:4], "little")

        # 后续 n 字节是 pickle 序列化后的 [method_name, *args]。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        # 当前命令已读取，清除 event，等待下一次命令。
        self.event.clear()

        # 返回方法名和参数列表。
        return method_name, args

    def write_shm(self, method_name, *args):
        # 只有 rank 0 会写共享内存。
        assert self.world_size > 1 and self.rank == 0

        # 把方法名和参数打包成 pickle bytes。
        data = pickle.dumps([method_name, *args])

        # 记录 payload 长度。
        n = len(data)

        # 前 4 字节写长度。
        self.shm.buf[0:4] = n.to_bytes(4, "little")

        # 后面写真正的 payload。
        self.shm.buf[4:n+4] = data

        # 唤醒每个子 rank，让它们读取并执行命令。
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # 多卡时，rank 0 先把同一方法调用广播给其它 rank。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        # 通过方法名找到当前对象上的方法。
        method = getattr(self, method_name, None)

        # 在当前 rank 执行该方法，并返回结果；非 rank 0 的 run 返回一般不会被外部使用。
        return method(*args)

    def warmup_model(self):
        # 清理 PyTorch CUDA cache，让后续显存统计更稳定。
        torch.cuda.empty_cache()

        # 重置 peak memory stats，后面 allocate_kv_cache 会读取 warmup 峰值。
        torch.cuda.reset_peak_memory_stats()

        # 读取最大 batch token 数和最大模型长度。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len

        # warmup 的单序列长度取二者较小值。
        seq_len = min(max_num_batched_tokens, max_model_len)

        # warmup 序列数尽量填满 max_num_batched_tokens，但不超过 max_num_seqs。
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)

        # 构造若干条全 0 token 的假 Sequence。
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]

        # 标记每条 warmup 序列本轮要 prefill seq_len 个 token。
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len

        # 执行一次 prefill forward，触发 kernel 编译/初始化并制造峰值显存。
        self.run(seqs, True)

        # 清掉 warmup 后可释放的临时缓存。
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        # 读取配置对象。
        config = self.config

        # 读取模型结构配置。
        hf_config = config.hf_config

        # 查询当前 GPU 空闲显存和总显存。
        free, total = torch.cuda.mem_get_info()

        # 当前已使用显存。
        used = total - free

        # warmup 期间记录到的峰值已分配显存。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        # 当前仍在使用的已分配显存。
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 当前 rank 持有的 KV heads 数；tensor parallel 会按 rank 切分 KV heads。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        # head_dim 可能在 config 中显式给出，否则用 hidden_size / num_attention_heads。
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # 一个 KV block 的字节数：
        # 2 表示 K 和 V；每层都有一份；每块 block_size 个 token；每 token 有 kv_heads * head_dim。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.model_dtype.itemsize

        # 估算能分配多少 KV blocks。
        # total * utilization 是目标可用显存；减去当前使用和 warmup 额外峰值，再除以单块大小。
        reserved_bytes = (
            config.vision_feature_cache_bytes
            if self.vision_provider is not None
            else 0
        )
        config.num_kvcache_blocks = calculate_num_kvcache_blocks(
            total_bytes=total,
            gpu_memory_utilization=config.gpu_memory_utilization,
            used_bytes=used,
            peak_bytes=peak,
            current_bytes=current,
            block_bytes=block_bytes,
            reserved_bytes=reserved_bytes,
        )

        # 至少要有一个 KV block，否则无法推理。
        assert config.num_kvcache_blocks > 0

        # 分配总 KV cache 张量，形状：
        # [2(K/V), layers, num_blocks, block_size, local_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # layer_id 用来把大 KV cache 的每层切片挂到对应 Attention 模块。
        layer_id = 0

        # 遍历模型所有模块，找到拥有 k_cache/v_cache 属性的 Attention。
        for module in self.model.modules():
            # Attention 模块初始化时有 k_cache 和 v_cache 占位张量。
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 当前层的 K cache 指向总 cache 的对应层切片。
                module.k_cache = self.kv_cache[0, layer_id]

                # 当前层的 V cache 指向总 cache 的对应层切片。
                module.v_cache = self.kv_cache[1, layer_id]

                # 下一个 Attention 对应下一层。
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 找到本 batch 中最长的 block_table 长度。
        max_len = max(len(seq.block_table) for seq in seqs)

        # 将每条序列的 block_table padding 到相同长度；-1 表示无效 block。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]

        # 转成 int32 CUDA 张量；pin_memory + non_blocking 加速 CPU 到 GPU 拷贝。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 返回形状 [batch, max_num_blocks] 的 block table。
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 拼接后的输入 token ids，多个序列会被展平成一维。
        input_ids = []

        position_rows = build_prefill_position_rows(
            seqs,
            position_axes=self.model.position_axes,
        )
        replacement_plan = plan_visual_replacements(seqs)

        # cu_seqlens_q 是 FlashAttention varlen 输入的 query 累计长度。
        cu_seqlens_q = [0]

        # cu_seqlens_k 是 key/value 的累计长度；有 prefix cache 时 K 长度可能大于 Q 长度。
        cu_seqlens_k = [0]

        # 本 batch 中最长 query 长度。
        max_seqlen_q = 0

        # 本 batch 中最长 key/value 上下文长度。
        max_seqlen_k = 0

        # slot_mapping 记录每个新算 token 的 K/V 应写入 KV cache 的哪个物理 slot。
        slot_mapping = []

        # prefix cache 场景需要 block_tables；普通 prefill 不需要。
        block_tables = None

        # 遍历本轮调度的所有序列。
        for seq in seqs:
            # start 是本轮要开始计算的 token 下标，等于已缓存 token 数。
            start = seq.num_cached_tokens

            # seqlen_q 是本轮需要新算的 token 数。
            seqlen_q = seq.num_scheduled_tokens

            # end 是本轮新算 token 的结束位置。
            end = start + seqlen_q

            # seqlen_k 是当前 token 能看到的上下文长度；prefill 中等于 end。
            seqlen_k = end

            # 加入本轮真正要送入模型的 token ids。
            input_ids.extend(seq[start:end])

            # 累加 query 长度，供 flash_attn_varlen_func 切分 batch。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)

            # 累加 key/value 长度；prefix cache 时它包含已缓存前缀。
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # 更新最大 query 长度。
            max_seqlen_q = max(seqlen_q, max_seqlen_q)

            # 更新最大 key/value 长度。
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # warmup 阶段的 Sequence 没有 block_table，因此不需要写 KV cache slot。
            if not seq.block_table:
                continue

            # 本轮新算 token 覆盖的起始逻辑 block。
            start_block = start // self.block_size

            # 本轮新算 token 覆盖的结束逻辑 block，向上取整。
            end_block = (end + self.block_size - 1) // self.block_size

            # 遍历覆盖到的逻辑 blocks，生成每个 token 的物理 KV slot。
            for i in range(start_block, end_block):
                # 逻辑 block i 对应的物理 block 起始 slot。
                slot_start = seq.block_table[i] * self.block_size

                # 如果是第一个 block，需要加上 start 在 block 内的偏移。
                if i == start_block:
                    slot_start += start % self.block_size

                # 如果不是最后一个 block，slot_end 到当前物理 block 末尾。
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size

                # 最后一个 block 只到本轮 end 对应的位置。
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size

                # 将这一段连续物理 slots 加入 slot_mapping。
                slot_mapping.extend(range(slot_start, slot_end))

        # 如果 K 总长度大于 Q 总长度，说明有已缓存前缀参与 attention，需要传 block_tables。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 将 input_ids 拷贝到 GPU。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        # 将 positions 拷贝到 GPU；Qwen3 是 [tokens]，Qwen2.5-VL 是 [3, tokens]。
        positions = torch.tensor(position_rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if self.model.position_axes == 1:
            positions = positions[0]

        inputs_embeds = None
        if replacement_plan.prefix_skipped_features:
            if self.vision_provider is None:
                raise RuntimeError("multimodal prefix metadata requires a vision provider")
            self.vision_provider.record_prefix_skipped_features(
                replacement_plan.prefix_skipped_features
            )
        if replacement_plan.replacements:
            if self.vision_provider is None:
                raise RuntimeError("visual token replacement requires a vision provider")
            token_embeddings = self.model.embed(input_ids)
            with self.vision_provider.resolve_pinned(
                replacement_plan.required_feature_keys
            ) as features:
                inputs_embeds = merge_planned_visual_embeddings(
                    token_embeddings,
                    replacement_plan,
                    features,
                )

        # 将 query 累计长度拷贝到 GPU。
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 将 key/value 累计长度拷贝到 GPU。
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 将 KV cache 写入位置拷贝到 GPU。
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置全局上下文，Attention 和 LM head 会从 get_context() 读取这些信息。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)

        return input_ids, positions, inputs_embeds

    def prepare_decode(self, seqs: list[Sequence]):
        # decode 阶段每条序列只输入一个 last_token。
        input_ids = []

        position_rows = build_decode_position_rows(
            seqs,
            position_axes=self.model.position_axes,
        )

        # 每个 last_token 的 K/V 要写入 KV cache 的物理 slot。
        slot_mapping = []

        # 每条序列当前上下文长度，FlashAttention decode kernel 需要。
        context_lens = []

        # 遍历本轮 decode 的所有序列。
        for seq in seqs:
            # decode 输入就是上一轮生成的 token，或 prefill 前的 prompt 最后一个 token。
            input_ids.append(seq.last_token)

            # attention 可见的上下文长度等于当前序列长度。
            context_lens.append(len(seq))

            # 计算当前 token 对应的物理 KV slot。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # 将 input_ids 拷贝到 GPU。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        positions = torch.tensor(position_rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if self.model.position_axes == 1:
            positions = positions[0]

        # 将 slot_mapping 拷贝到 GPU。
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 将上下文长度拷贝到 GPU。
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # decode 必须依赖 block_tables 来读取历史 KV cache。
        block_tables = self.prepare_block_tables(seqs)

        # 设置 decode 上下文。
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

        # 返回模型 forward 需要的 input_ids 和 positions。
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 取出每条序列自己的 temperature。
        temperatures = [seq.temperature for seq in seqs]

        # 转成 GPU 张量，Sampler 会按 batch 维度逐行缩放 logits。
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

        # 返回形状 [batch] 的 temperature 张量。
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        inputs_embeds: torch.Tensor | None = None,
    ):
        # prefill 长度变化大，直接 eager 跑；enforce_eager=True 时 decode 也直接 eager 跑。
        # decode batch size > 512 时也不用 CUDA graph，因为本实现只捕获到 512。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(
                None if inputs_embeds is not None else input_ids,
                positions,
                inputs_embeds=inputs_embeds,
            )
            return self.model.compute_logits(hidden_states)

        # decode 且允许 CUDA graph 时，走 graph replay。
        else:
            # 当前 decode batch size。
            bs = input_ids.size(0)

            # 读取 prepare_decode 刚写入的全局上下文。
            context = get_context()

            # 选择一个 >= 当前 bs 的已捕获 graph。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

            # graph_vars 是捕获 CUDA graph 时使用的固定地址输入/输出缓冲。
            graph_vars = self.graph_vars

            # 将当前 input_ids 拷贝到 graph 缓冲前 bs 个位置。
            graph_vars["input_ids"][:bs] = input_ids

            # 将当前 positions 拷贝到 graph 缓冲。
            graph_vars["positions"][..., :bs] = positions

            # 先把 slot_mapping 填成 -1，避免 padding 部分写入 KV cache。
            graph_vars["slot_mapping"].fill_(-1)

            # 写入当前 batch 的有效 slot_mapping。
            graph_vars["slot_mapping"][:bs] = context.slot_mapping

            # 清空 context_lens。
            graph_vars["context_lens"].zero_()

            # 写入当前 batch 的上下文长度。
            graph_vars["context_lens"][:bs] = context.context_lens

            # 写入当前 batch 的 block_tables；后面列数可能小于捕获时的最大列数。
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放 CUDA graph。
            graph.replay()

            # graph 输出是 hidden states，这里再计算 logits。
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 根据阶段准备 input_ids、positions 和 attention 所需上下文。
        if is_prefill:
            input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            inputs_embeds = None

        # 只有 rank 0 需要 temperature，因为只有 rank 0 最终采样完整 logits。
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 执行模型 forward，得到 logits。
        logits = self.run_model(input_ids, positions, is_prefill, inputs_embeds)

        # rank 0 做采样并转成 Python list；其它 rank 返回 None。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 清空本轮全局上下文，避免影响下一轮。
        reset_context()

        # 返回采样结果。
        return token_ids

    def register_vision_inputs(self, inputs: tuple[VisionInput, ...]) -> None:
        if self.vision_provider is None:
            if inputs:
                raise RuntimeError("the loaded model has no native vision provider")
            return
        self.vision_provider.register_inputs(inputs)

    def release_vision_inputs(self, keys: tuple[str, ...]) -> None:
        if self.vision_provider is not None:
            self.vision_provider.release_inputs(keys)

    def vision_stats(self):
        return self.vision_provider.stats() if self.vision_provider is not None else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        # 读取配置。
        config = self.config

        # 读取模型结构配置。
        hf_config = config.hf_config

        # CUDA graph 最大只捕获到 512 条 decode 序列。
        max_bs = min(self.config.max_num_seqs, 512)

        # 单条序列最多需要多少个 block。
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 固定地址 input_ids 缓冲，CUDA graph replay 时只能改内容不能改地址。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)

        # 固定地址 positions 缓冲。
        position_shape = (max_bs,) if self.model.position_axes == 1 else (self.model.position_axes, max_bs)
        positions = torch.zeros(position_shape, dtype=torch.int64)

        # 固定地址 slot_mapping 缓冲。
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)

        # 固定地址 context_lens 缓冲。
        context_lens = torch.zeros(max_bs, dtype=torch.int32)

        # 固定地址 block_tables 缓冲。
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)

        # 固定地址模型输出 hidden states 缓冲。
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 预先捕获的 batch sizes：小 batch 精细一些，大 batch 每 16 一个档。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        # batch size -> CUDAGraph 对象。
        self.graphs = {}

        # graph_pool 让多个 graph 共享内存池，降低显存碎片。
        self.graph_pool = None

        # 从大到小捕获 graph。
        for bs in reversed(self.graph_bs):
            # 新建 CUDA graph 对象。
            graph = torch.cuda.CUDAGraph()

            # 设置 decode 上下文，使用当前 bs 的固定缓冲切片。
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # 捕获前先 warmup 一次，确保相关 kernel/内存准备好。
            graph_positions = positions[..., :bs]
            outputs[:bs] = self.model(input_ids[:bs], graph_positions)

            # 捕获 model forward 到 CUDA graph。
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], graph_positions)

            # 第一个 graph 捕获后拿到内存池，后续 graph 复用。
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            # 保存该 batch size 对应的 graph。
            self.graphs[bs] = graph

            # 等待捕获相关 CUDA 工作完成。
            torch.cuda.synchronize()

            # 清空上下文，准备下一次捕获。
            reset_context()

        # 保存所有固定地址缓冲，run_model replay 前会往这些缓冲写新内容。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
