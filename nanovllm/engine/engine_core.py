from __future__ import annotations

from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.executor import Executor
from nanovllm.engine.metrics import EngineStepStats, RequestTimingStats, SchedulerStats
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.streaming import TokenEvent
from nanovllm.multimodal.qwen25_vl_processor import VisionInput


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
        self._token_events: list[TokenEvent] = []

    def add_sequence(
        self,
        seq: Sequence,
        vision_inputs: tuple[VisionInput, ...] = (),
    ) -> None:
        feature_keys = tuple(dict.fromkeys(item.feature_key for item in vision_inputs))
        if vision_inputs:
            self.executor.register_vision_inputs(vision_inputs)
        try:
            self.scheduler.add(seq)
        except BaseException:
            if feature_keys:
                self.executor.release_vision_inputs(feature_keys)
            raise

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        started = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        completion_counts = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.executor.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        token_events = getattr(self, "_token_events", None)
        if token_events is None:
            token_events = self._token_events = []
        for seq in seqs:
            first_new = completion_counts[seq.seq_id]
            new_token_ids = seq.completion_token_ids[first_new:]
            for index, token_id in enumerate(new_token_ids):
                is_terminal = seq.is_finished and index == len(new_token_ids) - 1
                token_events.append(TokenEvent(
                    seq_id=seq.seq_id,
                    token_id=token_id,
                    finished=is_terminal,
                    finish_reason=seq.finish_reason if is_terminal else None,
                ))
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        for seq in seqs:
            if seq.is_finished:
                self.finished_request_stats[seq.seq_id] = seq.timing_stats()
                self.executor.release_vision_inputs(self._vision_feature_keys(seq))
        self._record_step_stats(
            is_prefill=is_prefill,
            batch_size=len(seqs),
            scheduled_tokens=num_tokens,
            elapsed_sec=perf_counter() - started,
        )
        return outputs, num_tokens

    def drain_token_events(self) -> list[TokenEvent]:
        events = list(getattr(self, "_token_events", ()))
        self._token_events = []
        return events

    def abort_sequence(self, seq_id: int) -> bool:
        seq = self.scheduler.abort(seq_id)
        if seq is None:
            return False
        self._token_events = [
            event for event in getattr(self, "_token_events", ())
            if event.seq_id != seq_id
        ]
        self.finished_request_stats[seq_id] = seq.timing_stats()
        self.executor.release_vision_inputs(self._vision_feature_keys(seq))
        return True

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def scheduler_stats(self) -> SchedulerStats:
        return self.scheduler.stats()

    def request_stats(self, seq_id: int) -> RequestTimingStats | None:
        return self.finished_request_stats.get(seq_id)

    def vision_stats(self):
        return self.executor.vision_stats()

    def shutdown(self) -> None:
        self.executor.shutdown()

    @staticmethod
    def _vision_feature_keys(seq: Sequence) -> tuple[str, ...]:
        metadata = seq.multimodal_metadata
        if metadata is None:
            return ()
        return tuple(dict.fromkeys(span.feature_key for span in metadata.visual_spans))

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
