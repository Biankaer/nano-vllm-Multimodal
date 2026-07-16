# copy 用来复制传入的 token_ids，避免外部列表后续修改影响内部状态。
from copy import copy

# Enum/auto 用来定义请求状态枚举，状态含义比普通字符串更明确。
from enum import Enum, auto

# count 提供一个全局递增计数器，用来给每条 Sequence 分配 seq_id。
from itertools import count

# SamplingParams 提供 temperature、max_tokens、ignore_eos 等生成参数。
from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # 请求刚加入，还在等待 prefill。
    WAITING = auto()

    # 请求已经完成 prompt prefill，正在 decode。
    RUNNING = auto()

    # 请求已经生成结束。
    FINISHED = auto()


class Sequence:
    # 类级别 block_size，由 LLMEngine 根据 Config.kvcache_block_size 设置。
    block_size = 256

    # 全局递增 id 生成器，保证每条请求都有唯一 seq_id。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        # 给当前请求分配唯一 id；generate 最后用它恢复输出顺序。
        self.seq_id = next(Sequence.counter)

        # 新建请求默认处于 waiting，等待 Scheduler 做 prefill。
        self.status = SequenceStatus.WAITING

        # 保存 prompt token ids 的副本；后续生成 token 会 append 到同一个列表。
        self.token_ids = copy(token_ids)

        # last_token 是 decode 阶段输入给模型的单个 token。
        self.last_token = token_ids[-1]

        # 当前序列总 token 数 = prompt token 数 + 已生成 token 数。
        self.num_tokens = len(self.token_ids)

        # prompt token 数固定不变，用它区分 prompt 和 completion。
        self.num_prompt_tokens = len(token_ids)

        # 已经写入或复用到 KV cache 的 token 数。
        self.num_cached_tokens = 0

        # 本轮被 Scheduler 安排要执行的 token 数。
        self.num_scheduled_tokens = 0

        # True 表示当前序列处在 prefill 语义下；抢占后也会重新变成 True。
        self.is_prefill = True

        # 逻辑 block -> 物理 KV cache block id 的映射表。
        self.block_table = []

        # 当前请求的采样温度。
        self.temperature = sampling_params.temperature

        # 当前请求最多生成的新 token 数。
        self.max_tokens = sampling_params.max_tokens

        # 是否忽略 EOS token。
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        # len(seq) 返回当前序列总 token 数。
        return self.num_tokens

    def __getitem__(self, key):
        # 支持 seq[i] 或 seq[start:end]，方便 ModelRunner 切 prompt token。
        return self.token_ids[key]

    @property
    def is_finished(self):
        # 判断请求是否已经结束。
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 已生成 token 数 = 当前总 token 数 - prompt token 数。
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 返回 prompt 部分 token ids。
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # 返回 completion 部分 token ids。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 当前序列需要多少个逻辑 KV blocks，向上取整。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个逻辑 block 中当前有多少 token。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # block 下标必须在当前逻辑 block 范围内。
        assert 0 <= i < self.num_blocks

        # 返回第 i 个逻辑 block 覆盖的 token ids。
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 把新采样的 token 追加到序列末尾。
        self.token_ids.append(token_id)

        # 更新 decode 阶段下一轮要输入的 last_token。
        self.last_token = token_id

        # 当前序列总长度加 1。
        self.num_tokens += 1

    def __getstate__(self):
        # 多进程通信会 pickle Sequence；prefill 需要完整 token_ids，decode 只需要 last_token。
        last_state = self.last_token if not self.is_prefill else self.token_ids

        # 返回精简状态，减少 rank 0 向其它 rank 传输的数据量。
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # 从 pickle 状态恢复 Sequence 的核心字段。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state

        # prefill 状态传来的是完整 token_ids。
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]

        # decode 状态只传 last_token，避免复制整条长序列。
        else:
            self.token_ids = []
            self.last_token = last_state
