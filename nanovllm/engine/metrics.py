from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SchedulerStats:
    waiting_seqs: int = 0
    running_seqs: int = 0
    total_kv_blocks: int = 0
    used_kv_blocks: int = 0
    free_kv_blocks: int = 0
    prefix_cache_entries: int = 0

    @property
    def kv_cache_usage(self) -> float:
        return self.used_kv_blocks / self.total_kv_blocks if self.total_kv_blocks else 0.0


@dataclass(slots=True)
class EngineStepStats:
    step_id: int
    phase: str
    batch_size: int
    scheduled_tokens: int
    elapsed_sec: float
    prefill_tokens_per_sec: float = 0.0
    decode_seqs_per_sec: float = 0.0
    scheduler: SchedulerStats | None = None


@dataclass(slots=True)
class RequestTimingStats:
    seq_id: int
    prompt_tokens: int
    completion_tokens: int
    queue_time_sec: float | None
    ttft_sec: float | None
    total_latency_sec: float | None
    tpot_sec: float | None
