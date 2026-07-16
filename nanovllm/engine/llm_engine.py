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

# torch.multiprocessing 用于启动 tensor parallel 的多个 ModelRunner 进程。
import torch.multiprocessing as mp

# Config 保存引擎级别配置，如 batch 上限、KV cache 块大小、并行规模等。
from nanovllm.config import Config

# SamplingParams 保存请求级别采样参数，如 temperature、max_tokens、ignore_eos。
from nanovllm.sampling_params import SamplingParams

# Sequence 表示一条推理请求在引擎内部的状态，包括 token、block table、生成进度等。
from nanovllm.engine.sequence import Sequence

# Scheduler 决定每一步执行哪些 Sequence，以及它们处于 prefill 还是 decode。
from nanovllm.engine.scheduler import Scheduler

# ModelRunner 真正持有模型、KV cache，并在 GPU 上执行 forward 和采样。
from nanovllm.engine.model_runner import ModelRunner


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

        # Sequence.block_size 是类变量，所有请求共享同一个 KV cache block 粒度。
        Sequence.block_size = config.kvcache_block_size

        # 保存 tensor parallel 子进程对象，后续退出时需要 join。
        self.ps = []

        # 保存主进程通知子进程执行某个方法的 Event。
        self.events = []

        # 使用 spawn 启动子进程。CUDA 多进程一般避免 fork，spawn 更干净。
        ctx = mp.get_context("spawn")

        # rank 0 留在当前主进程，rank 1 到 N-1 用子进程启动。
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程对应一个 Event，主进程写共享内存后 set 这个 Event。
            event = ctx.Event()

            # 子进程入口就是 ModelRunner(config, rank, event)。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))

            # 真正启动子进程；子进程初始化模型后会进入 ModelRunner.loop() 等待命令。
            process.start()

            # 记录子进程对象，exit() 中需要等待它退出。
            self.ps.append(process)

            # 记录通信 Event，rank 0 的 ModelRunner.call() 会用这些 Event 唤醒子进程。
            self.events.append(event)

        # rank 0 的 ModelRunner 在主进程里创建；event 列表用于通知其它 rank。
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载 tokenizer；use_fast=True 使用 Rust 实现的 fast tokenizer。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # 把 EOS token id 写回 config，Scheduler 用它判断生成是否结束。
        config.eos = self.tokenizer.eos_token_id

        # 创建调度器。注意此时 config.num_kvcache_blocks 已经被 ModelRunner 填好了。
        self.scheduler = Scheduler(config)

        # 注册退出清理，用户即使没有显式调用 exit，也尽量释放分布式/共享内存资源。
        atexit.register(self.exit)

    def exit(self):
        """程序退出时的清理：通知子进程退出，等待子进程结束"""
        # 通过 ModelRunner.call("exit") 同时通知 rank 0 和所有子 rank 执行退出逻辑。
        self.model_runner.call("exit")

        # 删除主进程 ModelRunner，触发 Python 对其持有对象的释放。
        del self.model_runner

        # 等待所有 tensor parallel 子进程结束，避免僵尸进程。
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
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
        seq = Sequence(prompt, sampling_params)

        # 新请求先进入 Scheduler.waiting 队列，等待 prefill。
        self.scheduler.add(seq)

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
        # 让调度器选出本轮要运行的序列，并告诉我们本轮是 prefill 还是 decode。
        seqs, is_prefill = self.scheduler.schedule()

        # prefill 一次可能处理很多 token；decode 每条序列只处理 1 个 token。
        # 这里用正数表示 prefill token 数，用负数表示 decode 序列数，方便进度条区分。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # 调用模型执行器：准备张量、跑模型、采样，最后返回每条序列的新 token。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 调度器后处理：更新 KV cache 进度、append token、检查 EOS/max_tokens。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 只收集本轮刚完成的序列；未完成的会继续留在 running 队列中。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 返回完成结果和本轮吞吐统计所需的 token/sequence 数。
        return outputs, num_tokens

    def is_finished(self):
        """所有请求是否都已处理完成"""
        # Scheduler 同时没有 waiting 和 running 请求时，整个批次就结束了。
        return self.scheduler.is_finished()

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
        # 创建进度条，total 是请求条数；每完成一条请求 update(1)。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 如果传的是单个 SamplingParams，就让所有 prompt 共用同一份采样配置。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 把所有请求转换成 Sequence 并加入 waiting 队列。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # outputs 暂存已完成请求：key 是 seq_id，value 是 completion token ids。
        outputs = {}

        # 分别记录最近一次 prefill 和 decode 的吞吐。
        prefill_throughput = decode_throughput = 0.

        # 主推理循环：只要 Scheduler 里还有请求，就持续 step。
        while not self.is_finished():
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
                # 保存 completion token ids。
                outputs[seq_id] = token_ids

                # 一条请求完成，进度条加 1。
                pbar.update(1)

        # 所有请求完成后关闭进度条。
        pbar.close()

        # Sequence.seq_id 按请求加入顺序递增；排序后恢复到用户输入顺序。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 把 completion token ids 解码成文本，同时保留 token_ids 方便调试。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 返回 List[dict]，每个 dict 对应一条 prompt。
        return outputs
