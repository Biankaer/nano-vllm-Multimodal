# atexit 用来注册进程退出时的清理逻辑，避免子进程和 NCCL 资源残留。
import atexit

# fields 可以读取 dataclass Config 的字段名，用于从 kwargs 中筛选合法配置项。
from dataclasses import fields

# perf_counter 是高精度计时器，用来估算 prefill/decode 吞吐量。
from time import perf_counter

# tqdm 用来展示批量生成进度条和简单吞吐指标。
from tqdm.auto import tqdm

# AutoTokenizer 负责文本和 token ids 的互转，也提供 eos_token_id。
from transformers import AutoTokenizer

# Config 保存引擎级别配置，如 batch 上限、KV cache 块大小、并行规模等。
from nanovllm.config import Config

# SamplingParams 保存请求级别采样参数，如 temperature、max_tokens、ignore_eos。
from nanovllm.sampling_params import SamplingParams

# Sequence 表示一条推理请求在引擎内部的状态，包括 token、block table、生成进度等。
from nanovllm.engine.sequence import Sequence

# EngineCore 负责请求生命周期、调度和执行编排。
from nanovllm.engine.engine_core import EngineCore
from nanovllm.engine.metrics import EngineStepStats, RequestTimingStats, SchedulerStats
from nanovllm.multimodal.qwen25_vl_processor import (
    Qwen25VLPromptProcessor,
    VisionInput,
)
from nanovllm.multimodal.runtime import MultimodalPromptMetadata


class LLMEngine:
    """
    LLM 推理引擎的核心类，负责：
    1. 初始化模型、分词器、调度器
    2. 接收用户请求（prompt）
    3. 调度执行循环：prefill -> decode -> 采样 -> 直到所有序列完成
    4. 返回生成的文本
    """

    def __init__(self, model, **kwargs):
        # 读取 Config dataclass 的所有字段名。
        config_fields = {field.name for field in fields(Config)}

        # 只保留 Config 支持的 kwargs；这样 LLM(...) 可以忽略传给其它层的无关参数。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        # 创建引擎配置；Config.__post_init__ 会读取 HuggingFace config。
        config = Config(model, **config_kwargs)
        self.config = config

        # Sequence.block_size 是类变量，所有请求共享同一个 KV cache block 粒度。
        Sequence.block_size = config.kvcache_block_size

        self.multimodal_processor = None
        if getattr(config.hf_config, "model_type", None) == "qwen2_5_vl":
            self.multimodal_processor = Qwen25VLPromptProcessor.from_pretrained(config.model)
            self.tokenizer = self.multimodal_processor.tokenizer
        else:
            # 加载 tokenizer；use_fast=True 使用 Rust 实现的 fast tokenizer。
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # 把 EOS token id 写回 config，Scheduler 用它判断生成是否结束。
        config.eos = self.tokenizer.eos_token_id

        # 创建 EngineCore。EngineCore 会初始化 Executor，再创建 Scheduler。
        # Executor 内部创建 GPUWorker/ModelRunner，并填充 config.num_kvcache_blocks。
        self.core = EngineCore(config)

        # 注册退出清理，用户即使没有显式调用 exit，也尽量释放分布式/共享内存资源。
        atexit.register(self.exit)

    def exit(self):
        """程序退出时的清理：通知子进程退出，等待子进程结束"""
        # EngineCore 会向 Executor/GPUWorker 转发退出逻辑，并等待子进程结束。
        if hasattr(self, "core"):
            self.core.shutdown()
            del self.core

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        multimodal_metadata: MultimodalPromptMetadata | None = None,
        vision_inputs: tuple[VisionInput, ...] = (),
    ) -> Sequence:
        """
        向引擎添加一个新的推理请求。
        参数：
            prompt: 提示词，可以是字符串（自动分词）或已编码的 token id 列表
            sampling_params: 采样参数（温度、最大生成长度等）
        """
        # 如果用户传入字符串，就用 tokenizer 编码成 token ids。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # Sequence 是内部请求状态对象，会保存 prompt token、生成 token、KV block table 等。
        seq = Sequence(prompt, sampling_params, multimodal_metadata)

        # 新请求先进入 EngineCore 内部 Scheduler.waiting 队列，等待 prefill。
        self.core.add_sequence(seq, vision_inputs)
        return seq

    def step(self):
        """
        执行一步推理调度。
        这是推理引擎的核心循环单元：
        1. 调度器决定本轮执行哪些序列（prefill 或 decode）
        2. 模型执行器运行模型前向传播
        3. 调度器后处理（更新序列状态、检测 EOS 等）
        返回：
            outputs: 已完成的序列列表（seq_id, 生成的 token_ids）
            num_tokens: 正数=prefill 处理的 token 数，负数=-decode 序列数（用于吞吐量计算）
        """
        # 具体的 schedule -> execute -> postprocess 已经下沉到 EngineCore。
        return self.core.step()

    def is_finished(self):
        """所有请求是否都已处理完成"""
        # EngineCore 内部 Scheduler 同时没有 waiting 和 running 请求时，整个批次就结束了。
        return self.core.is_finished()

    def scheduler_stats(self) -> SchedulerStats:
        # 暴露调度器快照，供 serving metrics 和 benchmark 使用。
        return self.core.scheduler_stats()

    def step_stats(self) -> EngineStepStats | None:
        # 返回最近一次 step 的性能快照。
        return self.core.latest_step_stats

    def request_stats(self, seq_id: int) -> RequestTimingStats | None:
        # 返回已完成请求的生命周期指标。
        return self.core.request_stats(seq_id)

    def vision_stats(self):
        return self.core.vision_stats()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本的主方法，对用户暴露的核心 API。
        参数：
            prompts: 提示词列表（字符串或 token id 列表）
            sampling_params: 采样参数，可为单个（应用到所有 prompt）或列表（每个 prompt 一个）
            use_tqdm: 是否显示进度条
        返回：
            包含生成的文本和 token_ids 的字典列表
        """
        # 如果传的是单个 SamplingParams，就让所有 prompt 共用同一份采样配置。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("prompts and sampling_params must have the same length")

        # 把所有请求转换成 Sequence 并加入 waiting 队列。
        sequence_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            sequence_ids.append(self.add_request(prompt, sp).seq_id)

        return self._drain_sequences(sequence_ids, use_tqdm=use_tqdm)

    def generate_multimodal(
        self,
        messages_batch: list[list[dict[str, object]]],
        images_batch: list[list[object]],
        details_batch: list[list[str]] | None,
        sampling_params: SamplingParams | list[SamplingParams],
        *,
        use_tqdm: bool = True,
    ) -> list[dict]:
        if self.multimodal_processor is None:
            raise RuntimeError("the loaded model does not support native multimodal prompts")
        if len(messages_batch) != len(images_batch):
            raise ValueError("messages_batch and images_batch must have the same length")
        if details_batch is None:
            details_batch = [["auto"] * len(images) for images in images_batch]
        if len(details_batch) != len(messages_batch):
            raise ValueError("details_batch and messages_batch must have the same length")
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(messages_batch)
        if len(sampling_params) != len(messages_batch):
            raise ValueError("requests and sampling_params must have the same length")

        materialized_prompts = [
            self.multimodal_processor.materialize(messages, images, details)
            for messages, images, details in zip(
                messages_batch,
                images_batch,
                details_batch,
            )
        ]
        sequence_ids = []
        for prompt, request_sampling_params in zip(
            materialized_prompts,
            sampling_params,
        ):
            sequence = self.add_request(
                list(prompt.token_ids),
                request_sampling_params,
                multimodal_metadata=prompt.metadata,
                vision_inputs=prompt.vision_inputs,
            )
            sequence_ids.append(sequence.seq_id)
        return self._drain_sequences(sequence_ids, use_tqdm=use_tqdm)

    def _drain_sequences(
        self,
        sequence_ids: list[int],
        *,
        use_tqdm: bool,
    ) -> list[dict]:
        pbar = tqdm(total=len(sequence_ids), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # outputs 暂存已完成请求：key 是 seq_id，value 是 completion token ids。
        outputs = {}
        remaining = set(sequence_ids)

        # 分别记录最近一次 prefill 和 decode 的吞吐。
        prefill_throughput = decode_throughput = 0.

        # 主推理循环：只要 Scheduler 里还有请求，就持续 step。
        while remaining:
            # 记录本轮开始时间。
            t = perf_counter()

            # 执行一次调度 + 模型运行 + 后处理。
            output, num_tokens = self.step()

            # num_tokens > 0 表示本轮是 prefill，吞吐单位是 tokens/s。
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)

            # num_tokens < 0 表示本轮是 decode，吞吐单位这里写成 seqs/s。
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            # 更新进度条右侧的即时吞吐信息。
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # output 只包含本轮新完成的请求。
            for seq_id, token_ids in output:
                if seq_id not in remaining:
                    continue
                # 保存 completion token ids。
                outputs[seq_id] = token_ids
                remaining.remove(seq_id)

                # 一条请求完成，进度条加 1。
                pbar.update(1)

        # 所有请求完成后关闭进度条。
        pbar.close()

        # Sequence.seq_id 按请求加入顺序递增；排序后恢复到用户输入顺序。
        outputs = [outputs[seq_id] for seq_id in sequence_ids]

        # 把 completion token ids 解码成文本，同时保留 token_ids 方便调试。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 返回 List[dict]，每个 dict 对应一条 prompt。
        return outputs
