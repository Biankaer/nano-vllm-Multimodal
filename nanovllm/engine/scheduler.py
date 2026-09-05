# deque 适合做双端队列：waiting/running 都需要从两端插入或弹出。
from collections import deque

# Config 提供调度器需要的全局限制和 KV cache 参数。
from nanovllm.config import Config

# Sequence 是单条请求的内部状态；SequenceStatus 标记 WAITING/RUNNING/FINISHED。
from nanovllm.engine.sequence import Sequence, SequenceStatus

# BlockManager 负责分配、释放、复用 KV cache block。
from nanovllm.engine.block_manager import BlockManager

# SchedulerStats 暴露队列长度和 KV cache 使用情况。
from nanovllm.engine.metrics import SchedulerStats


class Scheduler:

    def __init__(self, config: Config):
        # 每轮最多调度多少条序列，防止 batch 维度过大。
        self.max_num_seqs = config.max_num_seqs

        # prefill 阶段每轮最多处理多少 token，防止一次前向太大。
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # EOS token id，用来判断一条生成是否自然结束。
        self.eos = config.eos

        # KV cache block 大小，必须和 Sequence.block_size、BlockManager 保持一致。
        self.block_size = config.kvcache_block_size

        # 创建 block 管理器。num_kvcache_blocks 已由 ModelRunner 根据显存算出。
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # waiting 保存还没有完成 prefill 的请求。
        self.waiting: deque[Sequence] = deque()

        # running 保存已经完成 prompt prefill、正在逐 token decode 的请求。
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # 两个队列都空，说明没有待 prefill 请求，也没有待 decode 请求。
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新请求先进入 waiting 队列，等待被 prefill。
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> Sequence | None:
        """Remove one sequence and release any allocated KV blocks."""
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id != seq_id:
                    continue
                queue.remove(seq)
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                seq.mark_finished("abort")
                return seq
        return None

    def stats(self) -> SchedulerStats:
        # 返回调度器当前快照，供 serving metrics 和 benchmark 使用。
        return SchedulerStats(
            waiting_seqs=len(self.waiting),
            running_seqs=len(self.running),
            total_kv_blocks=len(self.block_manager.blocks),
            used_kv_blocks=len(self.block_manager.used_block_ids),
            free_kv_blocks=len(self.block_manager.free_block_ids),
            prefix_cache_entries=len(self.block_manager.hash_to_block_id),
        )

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 本轮将要送入 ModelRunner 的序列列表。
        scheduled_seqs = []

        # 本轮 prefill 已累计的 token 数，用于控制 max_num_batched_tokens。
        num_batched_tokens = 0

        # -------------------- prefill 阶段 --------------------
        # 只要 waiting 队列非空，就优先尝试调度 prefill。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 只看队首请求，保持 FIFO；如果队首因资源不足不能调度，后面的也先不越过它。
            seq = self.waiting[0]

            # 本轮还剩多少 token 容量。
            remaining = self.max_num_batched_tokens - num_batched_tokens

            # 容量用完，本轮不能继续添加 prefill token。
            if remaining == 0:
                break

            # 如果 seq 还没有 block_table，说明它是第一次被 prefill 调度。
            if not seq.block_table:
                # can_allocate 会检查 prefix cache 能复用多少块，以及空闲 block 是否足够。
                num_cached_blocks = self.block_manager.can_allocate(seq)

                # -1 表示当前 KV cache 空闲块不足，无法给这个请求分配 block。
                if num_cached_blocks == -1:
                    break

                # 真正需要新算的 token 数 = 总 prompt token - 已复用的完整 prefix blocks。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size

            # 如果 seq 已经有 block_table，说明它之前做过 chunked prefill，还没 prefill 完。
            else:
                # 只需要继续计算尚未缓存的 prompt token。
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 只有本轮第一个请求允许 chunked prefill。
            # 如果当前 batch 已经有别的请求，而这个请求塞不下，就留到下一轮。
            if remaining < num_tokens and scheduled_seqs:
                break

            # 第一次调度该请求时，真正为它分配 block_table。
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 本轮为这条序列安排的 token 数；可能是完整 prompt 剩余部分，也可能是 chunk。
            seq.num_scheduled_tokens = min(num_tokens, remaining)

            # 累计本轮 prefill token 数。
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果本轮之后 prompt 全部进入 KV cache，这条请求就可以转入 decode 阶段。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 标记为 RUNNING，并记录进入 decode 生命周期的时间。
                seq.mark_running()

                # 从 waiting 队首移除。
                self.waiting.popleft()

                # 放入 running 队列，后续 decode 会从这里取。
                self.running.append(seq)

            # 不管是否完整 prefill，本轮都要把它送去 ModelRunner。
            scheduled_seqs.append(seq)

        # 只要本轮成功安排了 prefill，就优先返回 prefill batch，不混入 decode。
        if scheduled_seqs:
            return scheduled_seqs, True

        # -------------------- decode 阶段 --------------------
        # 只有没有任何可执行 prefill 时，才调度 decode。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从 running 队首取一条序列，尝试为它生成下一个 token。
            seq = self.running.popleft()

            # decode 可能需要追加新的 KV block；如果没有空闲块，就抢占一些序列。
            while not self.block_manager.can_append(seq):
                # 优先抢占 running 队尾的其它序列，给当前 seq 腾出 KV cache。
                if self.running:
                    self.preempt(self.running.pop())

                # 如果没有其它序列可抢占，只能抢占当前 seq 自己。
                else:
                    self.preempt(seq)
                    break

            # while 正常结束（没有 break）时执行 else，表示当前 seq 已经可以 append。
            else:
                # decode 每轮每条序列只调度 1 个 token。
                seq.num_scheduled_tokens = 1

                # 标记这条序列本轮不是 prefill；Sequence 序列化时会用到这个状态。
                seq.is_prefill = False

                # 如果新 token 落在新 block 的开头，就提前分配一个新 block。
                self.block_manager.may_append(seq)

                # 加入本轮 decode batch。
                scheduled_seqs.append(seq)

        # 如果能走到 decode 阶段，理论上至少应该调度到一条序列。
        assert scheduled_seqs

        # 被调度过的序列还没生成完成，所以放回 running 队首，等待下一轮继续 decode。
        self.running.extendleft(reversed(scheduled_seqs))

        # 返回 decode batch。
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 抢占后，该序列回到 WAITING，未来需要重新 prefill。
        seq.status = SequenceStatus.WAITING

        # 重新 prefill 时要把 prompt/token 前缀再走一遍；prefix cache 可能会复用已 hash 的块。
        seq.is_prefill = True

        # 释放该序列占用的 KV cache blocks。
        self.block_manager.deallocate(seq)

        # 把被抢占的请求放到 waiting 队首，尽快重新获得调度机会。
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # seqs 和 token_ids 一一对应：每条序列得到一个新采样 token。
        for seq, token_id in zip(seqs, token_ids):
            # 对本轮刚写满的完整 block 计算 hash，以便后续 prefix cache 复用。
            self.block_manager.hash_blocks(seq)

            # 更新已经进入 KV cache 的 token 数。
            seq.num_cached_tokens += seq.num_scheduled_tokens

            # 清空本轮调度数量，等待下一轮重新设置。
            seq.num_scheduled_tokens = 0

            # 如果是 chunked prefill 且 prompt 还没全部 prefill 完，本轮不 append 生成 token。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # prefill 完成后，模型会对 prompt 的最后位置采样第一个生成 token；
            # decode 阶段则采样每轮的下一个生成 token。
            seq.append_token(token_id)

            # 如果遇到 EOS，或者生成长度达到 max_tokens，这条序列结束。
            reached_eos = not seq.ignore_eos and token_id == self.eos
            reached_length = seq.num_completion_tokens == seq.max_tokens
            if reached_eos or reached_length:
                # 标记完成并记录完成时间。
                seq.mark_finished("stop" if reached_eos else "length")

                # 释放它占用的 KV cache blocks。
                self.block_manager.deallocate(seq)

                # 从 running 队列移除，不再参与后续 decode。
                self.running.remove(seq)
