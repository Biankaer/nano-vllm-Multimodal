from __future__ import annotations

from dataclasses import dataclass, field
from threading import Condition, Event, Thread
from time import monotonic
from typing import Protocol

from nanovllm.multimodal.batching import MultimodalBatcher
from nanovllm.multimodal.manager import MultimodalManager, MultimodalManagerStats
from nanovllm.multimodal.request import MultimodalGenerationRequest
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


class MultimodalBackend(Protocol):
    def generate_chat_batch(
        self,
        messages_batch: list[list[dict[str, object]]],
        *,
        image_inputs_batch: list[list[object]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        ...


@dataclass(slots=True)
class MultimodalEngineStats:
    waiting_requests: int
    submitted_requests: int
    completed_requests: int
    failed_requests: int
    executed_batches: int
    total_batched_requests: int
    manager: MultimodalManagerStats

    @property
    def average_batch_size(self) -> float:
        return self.total_batched_requests / self.executed_batches if self.executed_batches else 0.0


@dataclass(slots=True)
class _PendingRequest:
    request: MultimodalGenerationRequest
    done: Event = field(default_factory=Event)
    result: str | None = None
    error: BaseException | None = None


class MultimodalEngineCore:
    """Request-level continuous batching core for the Qwen2.5-VL bridge."""

    def __init__(
        self,
        backend: MultimodalBackend,
        *,
        policy: MultimodalBatchPolicy | None = None,
        asset_cache_capacity: int = 256,
        feature_cache_capacity: int = 256,
    ):
        self.backend = backend
        self.policy = policy or MultimodalBatchPolicy()
        self.batcher = MultimodalBatcher(self.policy)
        self.manager = MultimodalManager(
            asset_cache_capacity=asset_cache_capacity,
            feature_cache_capacity=feature_cache_capacity,
        )
        self._condition = Condition()
        self._pending: dict[str, _PendingRequest] = {}
        self._worker: Thread | None = None
        self._shutdown = False
        self._submitted_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._executed_batches = 0
        self._total_batched_requests = 0

    def generate(
        self,
        request: MultimodalRequest,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        timeout: float | None = None,
    ) -> str:
        generation_request = MultimodalGenerationRequest(
            request=request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        pending = _PendingRequest(generation_request)
        with self._condition:
            if self._shutdown:
                raise RuntimeError("multimodal engine is shut down")
            self._ensure_worker_started()
            self.batcher.add(generation_request)
            self._pending[generation_request.internal_id] = pending
            self._submitted_requests += 1
            self._condition.notify()

        if not pending.done.wait(timeout):
            raise TimeoutError(f"multimodal request {generation_request.request_id} timed out")
        if pending.error is not None:
            raise RuntimeError(f"multimodal request {generation_request.request_id} failed") from pending.error
        return pending.result or ""

    def stats(self) -> MultimodalEngineStats:
        with self._condition:
            return MultimodalEngineStats(
                waiting_requests=len(self.batcher),
                submitted_requests=self._submitted_requests,
                completed_requests=self._completed_requests,
                failed_requests=self._failed_requests,
                executed_batches=self._executed_batches,
                total_batched_requests=self._total_batched_requests,
                manager=self.manager.stats(),
            )

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join()

    def _ensure_worker_started(self) -> None:
        if self._worker is not None:
            return
        self._worker = Thread(target=self._run_loop, name="nanoinfer-multimodal", daemon=True)
        self._worker.start()

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown and len(self.batcher) == 0:
                    self._condition.wait()
                if self._shutdown and len(self.batcher) == 0:
                    return

                wait_deadline = monotonic() + self.policy.batch_wait_ms / 1000.0
                while not self._shutdown and len(self.batcher) < self.policy.max_batch_size:
                    remaining = wait_deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                batch = self.batcher.pop_batch()

            try:
                image_inputs_batch = self.manager.prepare_batch(batch)
                first = batch.requests[0]
                outputs = self.backend.generate_chat_batch(
                    [request.request.messages for request in batch.requests],
                    image_inputs_batch=image_inputs_batch,
                    max_new_tokens=first.max_new_tokens,
                    temperature=first.temperature,
                    top_p=first.top_p,
                )
                if len(outputs) != len(batch.requests):
                    raise RuntimeError(
                        f"multimodal backend returned {len(outputs)} outputs for {len(batch.requests)} requests"
                    )
                self._finish_batch(batch.requests, outputs=outputs)
            except BaseException as exc:
                self._finish_batch(batch.requests, error=exc)

    def _finish_batch(
        self,
        requests: list[MultimodalGenerationRequest],
        *,
        outputs: list[str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._condition:
            self._executed_batches += 1
            self._total_batched_requests += len(requests)
            if error is None:
                self._completed_requests += len(requests)
            else:
                self._failed_requests += len(requests)

            for index, request in enumerate(requests):
                pending = self._pending.pop(request.internal_id)
                if error is not None:
                    pending.error = error
                else:
                    pending.result = outputs[index] if outputs is not None else ""
                pending.done.set()
