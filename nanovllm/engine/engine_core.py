from __future__ import annotations

from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.executor import Executor
from nanovllm.engine.metrics import EngineStepStats, RequestTimingStats, SchedulerStats
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
        self.step_id = 0
        self.latest_step_stats: EngineStepStats | None = None
        self.finished_request_stats: dict[int, RequestTimingStats] = {}

    def add_sequence(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        started = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.executor.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        for seq in seqs:
            if seq.is_finished:
                self.finished_request_stats[seq.seq_id] = seq.timing_stats()
        self._record_step_stats(
            is_prefill=is_prefill,
            batch_size=len(seqs),
            scheduled_tokens=num_tokens,
            elapsed_sec=perf_counter() - started,
        )
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def scheduler_stats(self) -> SchedulerStats:
        return self.scheduler.stats()

    def request_stats(self, seq_id: int) -> RequestTimingStats | None:
        return self.finished_request_stats.get(seq_id)

    def shutdown(self) -> None:
        self.executor.shutdown()

    def _record_step_stats(self, *, is_prefill: bool, batch_size: int, scheduled_tokens: int, elapsed_sec: float) -> None:
        self.step_id += 1
        prefill_tps = scheduled_tokens / elapsed_sec if is_prefill and elapsed_sec > 0 else 0.0
        decode_sps = -scheduled_tokens / elapsed_sec if not is_prefill and elapsed_sec > 0 else 0.0
        self.latest_step_stats = EngineStepStats(
            step_id=self.step_id,
            phase="prefill" if is_prefill else "decode",
            batch_size=batch_size,
            scheduled_tokens=scheduled_tokens if is_prefill else -scheduled_tokens,
            elapsed_sec=elapsed_sec,
            prefill_tokens_per_sec=prefill_tps,
            decode_seqs_per_sec=decode_sps,
            scheduler=self.scheduler.stats(),
        )
