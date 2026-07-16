from __future__ import annotations

from nanovllm.config import Config
from nanovllm.engine.executor import Executor
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class EngineCore:
    """Request lifecycle, scheduling, and execution orchestration.

    This class is intentionally small in Phase 2. It extracts the core serving
    loop from `LLMEngine` while preserving the original scheduler semantics.
    Future phases can add async request admission, per-request timing, streaming,
    and continuous batching here.
    """

    def __init__(self, config: Config):
        self.config = config
        self.executor = Executor(config)
        self.scheduler = Scheduler(config)

    def add_sequence(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.executor.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def shutdown(self) -> None:
        self.executor.shutdown()
