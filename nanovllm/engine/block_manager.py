# deque 用来维护空闲 block id 队列，popleft/append 都是 O(1)。
from collections import deque

# xxhash 提供快速非加密哈希，用来为完整 token block 建立 prefix cache key。
import xxhash

# numpy 用来把 token id 列表转成连续 bytes，方便送进 xxhash。
import numpy as np

# Sequence 提供 block(i)、num_blocks、block_table 等信息。
from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        # 物理 KV cache block 的编号；这个编号会出现在 Sequence.block_table 中。
        self.block_id = block_id

        # 引用计数。多个 Sequence 可以因为 prefix cache 共享同一个 block。
        self.ref_count = 0

        # 当前 block 对应 token 内容的 hash；-1 表示尚未写入可复用 hash。
        self.hash = -1

        # 保存这个 block 对应的 token ids，用来在 hash 命中后再次校验，避免哈希碰撞误复用。
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        # 写入这个 block 的 prefix hash。
        self.hash = hash

        # 保存完整 block 的 token ids，后续 prefix cache 命中时做精确比较。
        self.token_ids = token_ids

    def reset(self):
        # 新分配出去的 block 默认由一个 Sequence 引用。
        self.ref_count = 1

        # 重置 hash，等这个 block 被完整写满后再 hash_blocks。
        self.hash = -1

        # 清空 token ids，避免旧内容被误认为当前内容。
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        # 每个物理 KV block 能容纳多少个 token 的 K/V。
        self.block_size = block_size

        # 创建所有物理 block 的元数据；真正的 K/V tensor 在 ModelRunner.kv_cache 中。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # prefix hash -> block id。用于找到可复用的完整前缀 block。
        self.hash_to_block_id: dict[int, int] = dict()

        # 当前空闲 block id 队列。
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 当前正在被某些 Sequence 引用的 block id 集合。
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 创建 64-bit xxhash 对象。
        h = xxhash.xxh64()

        # 如果有前一个 block 的 hash，就把它纳入当前 hash。
        # 这样第 i 个 block 的 hash 不只取决于本 block，也取决于完整前缀。
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))

        # 将 token ids 转成 bytes 后更新 hash。
        h.update(np.array(token_ids).tobytes())

        # 返回整数形式的 hash 值。
        return h.intdigest()

    def _allocate_block(self) -> int:
        # 从空闲队列头部取一个物理 block。
        block_id = self.free_block_ids.popleft()

        # 取出对应的元数据对象。
        block = self.blocks[block_id]

        # 空闲 block 不应该还有引用。
        assert block.ref_count == 0

        # 如果这个 block 过去留下过 hash 映射，并且映射仍指向它，分配前要删掉旧映射。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        # 重置 block 元数据，并把引用计数设为 1。
        block.reset()

        # 记录它现在属于 used 集合。
        self.used_block_ids.add(block_id)

        # 返回分配到的物理 block id。
        return block_id

    def _deallocate_block(self, block_id: int):
        # 只有引用计数归零的 block 才能释放。
        assert self.blocks[block_id].ref_count == 0

        # 从 used 集合移除。
        self.used_block_ids.remove(block_id)

        # 放回空闲队列尾部，供后续请求复用。
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # h 保存“到当前 block 为止”的滚动 prefix hash。
        h = -1

        # 能复用的完整 prefix block 数。
        num_cached_blocks = 0

        # 默认假设整条序列所有 block 都需要新分配。
        num_new_blocks = seq.num_blocks

        # 只检查到倒数第二个 block：最后一个 block 可能未满，不作为 prefix cache 复用单位。
        for i in range(seq.num_blocks - 1):
            # 取第 i 个逻辑 block 的 token ids。
            token_ids = seq.block(i)

            # 计算带前缀依赖的 hash。
            h = self.compute_hash(token_ids, h)

            # 查找这个 prefix hash 是否已经对应某个物理 block。
            block_id = self.hash_to_block_id.get(h, -1)

            # 没命中，或者 hash 命中但 token ids 不一致，就停止 prefix 复用。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break

            # 成功复用一个完整 prefix block。
            num_cached_blocks += 1

            # 如果这个 block 当前仍在 used 集合，复用时不需要额外空闲 block。
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        # 如果空闲 block 数不足以容纳需要新分配的部分，返回 -1 表示暂时不能调度。
        if len(self.free_block_ids) < num_new_blocks:
            return -1

        # 返回可复用的完整 prefix block 数。
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 只有尚未分配 block_table 的 Sequence 才能走 allocate。
        assert not seq.block_table

        # h 是滚动 prefix hash，必须和 can_allocate 中的计算方式一致。
        h = -1

        # 先为可复用的 prefix blocks 填充 block_table。
        for i in range(num_cached_blocks):
            # 取第 i 个逻辑 block 的 token ids。
            token_ids = seq.block(i)

            # 计算该 block 的 prefix hash。
            h = self.compute_hash(token_ids, h)

            # 找到已有物理 block。
            block_id = self.hash_to_block_id[h]

            # 取出 block 元数据。
            block = self.blocks[block_id]

            # 如果 block 当前正在被其它序列使用，只需要增加引用计数。
            if block_id in self.used_block_ids:
                block.ref_count += 1

            # 如果 block 目前在 free 队列但仍保留 hash 内容，则从 free 中拿回来复用。
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            # 逻辑 block i 映射到这个物理 block。
            seq.block_table.append(block_id)

        # 剩余无法复用的逻辑 block，需要分配新的物理 block。
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        # 已经复用的完整 blocks 等价于已经有这些 token 的 KV cache。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # 反向遍历 block_table，逐个减少引用计数。
        for block_id in reversed(seq.block_table):
            # 取物理 block 元数据。
            block = self.blocks[block_id]

            # 当前序列不再引用这个 block。
            block.ref_count -= 1

            # 如果没有其它序列引用，就释放到 free 队列。
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # 序列不再拥有任何 cached token。
        seq.num_cached_tokens = 0

        # 清空逻辑到物理的映射。
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # decode 追加一个 token 时，只有当新 token 正好落在新 block 开头才需要额外 block。
        # len(seq) % block_size == 1 对应 append 后 last_block_num_tokens 为 1 的场景。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 如果下一个 token 会开启一个新 block，就先分配物理 block 并追加到 block_table。
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # start 是本轮执行前已缓存 token 覆盖到的完整 block 下标。
        start = seq.num_cached_tokens // self.block_size

        # end 是本轮执行后会覆盖到的完整 block 下标。
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size

        # 如果本轮没有新写满任何完整 block，就不需要计算 hash。
        if start == end:
            return

        # 如果不是从第 0 块开始，滚动 hash 要接上前一个 block 的 hash。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1

        # 对本轮新写满的每个完整 block 计算并登记 hash。
        for i in range(start, end):
            # 找到逻辑 block i 对应的物理 block。
            block = self.blocks[seq.block_table[i]]

            # 取逻辑 block i 的 token ids。
            token_ids = seq.block(i)

            # 计算包含完整前缀信息的 hash。
            h = self.compute_hash(token_ids, h)

            # 将 hash 和 token ids 写入 block 元数据。
            block.update(h, token_ids)

            # 建立 hash 到物理 block 的索引，供未来请求做 prefix cache。
            self.hash_to_block_id[h] = block.block_id
