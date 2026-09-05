from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Condition, Event, Thread
from time import monotonic
from typing import Protocol

from nanovllm.multimodal.batching import MultimodalBatcher
from nanovllm.multimodal.manager import MultimodalManager, MultimodalManagerStats
from nanovllm.multimodal.request import MultimodalGenerationRequest
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest
from nanovllm.engine.streaming import GenerationStreamEvent


class MultimodalBackend(Protocol):
    def generate_chat_batch(
        self,
        messages_batch: list[list[dict[str, object]]],
        *,
        image_inputs_batch: list[list[object]],
        image_details_batch: list[list[str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        ignore_eos: bool,
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
    batch_size_histogram: dict[int, int]
    total_queue_wait_ms: float
    total_coalescing_wait_ms: float
    total_execution_ms: float
    manager: MultimodalManagerStats

    @property
    def average_batch_size(self) -> float:
        return self.total_batched_requests / self.executed_batches if self.executed_batches else 0.0

    @property
    def average_queue_wait_ms(self) -> float:
        return self.total_queue_wait_ms / self.total_batched_requests if self.total_batched_requests else 0.0

    @property
    def average_coalescing_wait_ms(self) -> float:
        return self.total_coalescing_wait_ms / self.executed_batches if self.executed_batches else 0.0

    @property
    def average_execution_ms(self) -> float:
        return self.total_execution_ms / self.executed_batches if self.executed_batches else 0.0


@dataclass(slots=True)
class _PendingRequest:
    request: MultimodalGenerationRequest
    done: Event = field(default_factory=Event)
    result: str | None = None
    error: BaseException | None = None
    events: Queue = field(default_factory=Queue)
    canceled: bool = False


class MultimodalStreamHandle:
    def __init__(self, pending: _PendingRequest, *, timeout: float | None) -> None:
        self._pending = pending
        self._timeout = timeout
        self._finished = False

    def __iter__(self):
        return self

    def __next__(self) -> GenerationStreamEvent:
        if self._finished:
            raise StopIteration
        try:
            item = self._pending.events.get(timeout=self._timeout)
        except Empty as exc:
            self.close()
            raise TimeoutError(
                f"multimodal request {self._pending.request.request_id} timed out"
            ) from exc
        if isinstance(item, BaseException):
            self._finished = True
            raise RuntimeError(
                f"multimodal request {self._pending.request.request_id} failed"
            ) from item
        if not isinstance(item, GenerationStreamEvent):
            self._finished = True
            raise RuntimeError("multimodal stream received an invalid event")
        if item.finished:
            self._finished = True
        return item

    def close(self) -> None:
        self._pending.canceled = True
        self._finished = True


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
        self._batch_size_histogram: dict[int, int] = {}
        self._total_queue_wait_ms = 0.0
        self._total_coalescing_wait_ms = 0.0
        self._total_execution_ms = 0.0

    def generate(
        self,
        request: MultimodalRequest,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
        timeout: float | None = None,
    ) -> str:
        pending = self._enqueue(
            request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            ignore_eos=ignore_eos,
        )
        generation_request = pending.request

        if not pending.done.wait(timeout):
            pending.canceled = True
            with self._condition:
                self._condition.notify_all()
            raise TimeoutError(f"multimodal request {generation_request.request_id} timed out")
        if pending.error is not None:
            raise RuntimeError(f"multimodal request {generation_request.request_id} failed") from pending.error
        return pending.result or ""

    def stream(
        self,
        request: MultimodalRequest,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
        timeout: float | None = None,
    ) -> MultimodalStreamHandle:
        pending = self._enqueue(
            request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            ignore_eos=ignore_eos,
        )
        return MultimodalStreamHandle(pending, timeout=timeout)

    def _enqueue(
        self,
        request: MultimodalRequest,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        ignore_eos: bool,
    ) -> _PendingRequest:
        generation_request = MultimodalGenerationRequest(
            request=request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            ignore_eos=ignore_eos,
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
        return pending

    def stats(self) -> MultimodalEngineStats:
        with self._condition:
            return MultimodalEngineStats(
                waiting_requests=len(self.batcher),
                submitted_requests=self._submitted_requests,
                completed_requests=self._completed_requests,
                failed_requests=self._failed_requests,
                executed_batches=self._executed_batches,
                total_batched_requests=self._total_batched_requests,
                batch_size_histogram=dict(self._batch_size_histogram),
                total_queue_wait_ms=self._total_queue_wait_ms,
                total_coalescing_wait_ms=self._total_coalescing_wait_ms,
                total_execution_ms=self._total_execution_ms,
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
        if all(callable(getattr(self.backend, name, None)) for name in (
            "dynamic_session",
            "admit_chat_batch",
            "step_dynamic",
            "abort_dynamic",
        )):
            self._run_dynamic_loop()
            return
        self._run_static_loop()

    def _run_static_loop(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown and len(self.batcher) == 0:
                    self._condition.wait()
                if self._shutdown and len(self.batcher) == 0:
                    return

                coalescing_started = monotonic()
                quiet_deadline = coalescing_started + self.policy.effective_quiet_ms / 1000.0
                max_deadline = coalescing_started + self.policy.effective_max_wait_ms / 1000.0
                observed_queue_size = len(self.batcher)
                while not self._shutdown and len(self.batcher) < self.policy.max_batch_size:
                    remaining = min(quiet_deadline, max_deadline) - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                    queue_size = len(self.batcher)
                    if queue_size > observed_queue_size:
                        quiet_deadline = monotonic() + self.policy.effective_quiet_ms / 1000.0
                    observed_queue_size = queue_size
                batch = self.batcher.pop_batch()
                batch_started = monotonic()
                queue_wait_ms = sum(
                    max(0.0, batch_started - request.arrival_time) * 1000.0
                    for request in batch.requests
                )
                coalescing_wait_ms = max(0.0, batch_started - coalescing_started) * 1000.0

            execution_started = monotonic()
            try:
                image_inputs_batch = self.manager.prepare_batch(batch)
                image_details_batch = [
                    [image.detail for image in request.request.images]
                    for request in batch.requests
                ]
                first = batch.requests[0]
                arguments = dict(
                    image_inputs_batch=image_inputs_batch,
                    image_details_batch=image_details_batch,
                    max_new_tokens=first.max_new_tokens,
                    temperature=first.temperature,
                    top_p=first.top_p,
                    ignore_eos=first.ignore_eos,
                )
                stream_method = getattr(self.backend, "generate_chat_batch_stream", None)
                if callable(stream_method):
                    outputs = self._consume_stream_batch(
                        batch.requests,
                        stream_method(
                            [request.request.messages for request in batch.requests],
                            **arguments,
                        ),
                    )
                else:
                    outputs = self.backend.generate_chat_batch(
                        [request.request.messages for request in batch.requests],
                        **arguments,
                    )
                if len(outputs) != len(batch.requests):
                    raise RuntimeError(
                        f"multimodal backend returned {len(outputs)} outputs for {len(batch.requests)} requests"
                    )
                self._finish_batch(
                    batch.requests,
                    outputs=outputs,
                    queue_wait_ms=queue_wait_ms,
                    coalescing_wait_ms=coalescing_wait_ms,
                    execution_ms=max(0.0, monotonic() - execution_started) * 1000.0,
                )
            except BaseException as exc:
                self._finish_batch(
                    batch.requests,
                    error=exc,
                    queue_wait_ms=queue_wait_ms,
                    coalescing_wait_ms=coalescing_wait_ms,
                    execution_ms=max(0.0, monotonic() - execution_started) * 1000.0,
                )

    def _run_dynamic_loop(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown and len(self.batcher) == 0:
                    self._condition.wait()
                if self._shutdown and len(self.batcher) == 0:
                    return
                batch, queue_wait_ms, coalescing_wait_ms = self._pop_coalesced_batch()

            active: dict[int, _PendingRequest] = {}
            outputs: dict[int, str] = {}
            with self.backend.dynamic_session():
                while True:
                    if batch is not None:
                        try:
                            sequence_ids = self._admit_dynamic_batch(batch)
                        except BaseException as exc:
                            self._finish_batch(
                                batch.requests,
                                error=exc,
                                queue_wait_ms=queue_wait_ms,
                                coalescing_wait_ms=coalescing_wait_ms,
                            )
                            sequence_ids = []
                        else:
                            self._record_admission_batch(
                                batch.requests,
                                queue_wait_ms=queue_wait_ms,
                                coalescing_wait_ms=coalescing_wait_ms,
                            )
                            for request, seq_id in zip(batch.requests, sequence_ids):
                                pending = self._pending[request.internal_id]
                                active[seq_id] = pending
                                outputs[seq_id] = ""
                        batch = None

                    self._abort_canceled_dynamic(active, outputs)
                    if not active:
                        with self._condition:
                            if len(self.batcher) == 0:
                                break
                            batch, queue_wait_ms, coalescing_wait_ms = self._pop_immediate_batch()
                        continue

                    step_started = monotonic()
                    try:
                        events = self.backend.step_dynamic()
                    except BaseException as exc:
                        self._fail_active_dynamic(active, outputs, exc)
                        break
                    finally:
                        with self._condition:
                            self._total_execution_ms += max(0.0, monotonic() - step_started) * 1000.0

                    try:
                        for event in events:
                            if not isinstance(event, GenerationStreamEvent):
                                raise RuntimeError("dynamic multimodal backend yielded an invalid stream event")
                            pending = active.get(event.seq_id)
                            if pending is None:
                                raise RuntimeError(
                                    f"dynamic multimodal backend yielded unknown sequence {event.seq_id}"
                                )
                            outputs[event.seq_id] += event.text_delta
                            if not pending.canceled:
                                pending.events.put(GenerationStreamEvent(
                                    request_index=0,
                                    seq_id=event.seq_id,
                                    token_id=event.token_id,
                                    text_delta=event.text_delta,
                                    finished=event.finished,
                                    finish_reason=event.finish_reason,
                                    prompt_tokens=event.prompt_tokens,
                                    completion_tokens=event.completion_tokens,
                                ))
                            if event.finished:
                                self._finish_dynamic_request(
                                    pending,
                                    output=outputs.pop(event.seq_id),
                                )
                                del active[event.seq_id]
                    except BaseException as exc:
                        self._fail_active_dynamic(active, outputs, exc)
                        break

                    self._abort_canceled_dynamic(active, outputs)
                    with self._condition:
                        if len(self.batcher):
                            batch, queue_wait_ms, coalescing_wait_ms = self._pop_immediate_batch()
                        elif active:
                            continue
                        else:
                            break

            with self._condition:
                if self._shutdown and len(self.batcher) == 0:
                    return

    def _pop_coalesced_batch(self):
        coalescing_started = monotonic()
        quiet_deadline = coalescing_started + self.policy.effective_quiet_ms / 1000.0
        max_deadline = coalescing_started + self.policy.effective_max_wait_ms / 1000.0
        observed_queue_size = len(self.batcher)
        while not self._shutdown and len(self.batcher) < self.policy.max_batch_size:
            remaining = min(quiet_deadline, max_deadline) - monotonic()
            if remaining <= 0:
                break
            self._condition.wait(remaining)
            queue_size = len(self.batcher)
            if queue_size > observed_queue_size:
                quiet_deadline = monotonic() + self.policy.effective_quiet_ms / 1000.0
            observed_queue_size = queue_size
        batch = self.batcher.pop_batch()
        batch_started = monotonic()
        queue_wait_ms = sum(
            max(0.0, batch_started - request.arrival_time) * 1000.0
            for request in batch.requests
        )
        return (
            batch,
            queue_wait_ms,
            max(0.0, batch_started - coalescing_started) * 1000.0,
        )

    def _pop_immediate_batch(self):
        batch = self.batcher.pop_batch()
        batch_started = monotonic()
        queue_wait_ms = sum(
            max(0.0, batch_started - request.arrival_time) * 1000.0
            for request in batch.requests
        )
        return batch, queue_wait_ms, 0.0

    def _admit_dynamic_batch(self, batch):
        image_inputs_batch = self.manager.prepare_batch(batch)
        image_details_batch = [
            [image.detail for image in request.request.images]
            for request in batch.requests
        ]
        first = batch.requests[0]
        sequence_ids = self.backend.admit_chat_batch(
            [request.request.messages for request in batch.requests],
            image_inputs_batch=image_inputs_batch,
            image_details_batch=image_details_batch,
            max_new_tokens=first.max_new_tokens,
            temperature=first.temperature,
            top_p=first.top_p,
            ignore_eos=first.ignore_eos,
        )
        if len(sequence_ids) != len(batch.requests):
            raise RuntimeError(
                f"dynamic multimodal backend admitted {len(sequence_ids)} sequences "
                f"for {len(batch.requests)} requests"
            )
        if len(set(sequence_ids)) != len(sequence_ids):
            raise RuntimeError("dynamic multimodal backend returned duplicate sequence ids")
        return sequence_ids

    def _record_admission_batch(
        self,
        requests: list[MultimodalGenerationRequest],
        *,
        queue_wait_ms: float,
        coalescing_wait_ms: float,
    ) -> None:
        with self._condition:
            self._executed_batches += 1
            self._total_batched_requests += len(requests)
            self._batch_size_histogram[len(requests)] = self._batch_size_histogram.get(len(requests), 0) + 1
            self._total_queue_wait_ms += queue_wait_ms
            self._total_coalescing_wait_ms += coalescing_wait_ms

    def _abort_canceled_dynamic(
        self,
        active: dict[int, _PendingRequest],
        outputs: dict[int, str],
    ) -> None:
        for seq_id, pending in list(active.items()):
            if not pending.canceled:
                continue
            self.backend.abort_dynamic(seq_id)
            outputs.pop(seq_id, None)
            self._finish_dynamic_request(pending, output="")
            del active[seq_id]

    def _fail_active_dynamic(
        self,
        active: dict[int, _PendingRequest],
        outputs: dict[int, str],
        error: BaseException,
    ) -> None:
        for seq_id, pending in list(active.items()):
            try:
                self.backend.abort_dynamic(seq_id)
            except BaseException:
                pass
            self._finish_dynamic_request(pending, error=error)
        active.clear()
        outputs.clear()

    def _finish_dynamic_request(
        self,
        pending: _PendingRequest,
        *,
        output: str = "",
        error: BaseException | None = None,
    ) -> None:
        with self._condition:
            self._pending.pop(pending.request.internal_id, None)
            if error is None:
                self._completed_requests += 1
                pending.result = output
            else:
                self._failed_requests += 1
                pending.error = error
                if not pending.canceled:
                    pending.events.put(error)
            pending.done.set()

    def _consume_stream_batch(
        self,
        requests: list[MultimodalGenerationRequest],
        events,
    ) -> list[str]:
        outputs = [""] * len(requests)
        finished: set[int] = set()
        for event in events:
            if not isinstance(event, GenerationStreamEvent):
                raise RuntimeError("multimodal backend yielded an invalid stream event")
            index = event.request_index
            if not 0 <= index < len(requests):
                raise RuntimeError(f"multimodal backend yielded invalid request index {index}")
            outputs[index] += event.text_delta
            pending = self._pending.get(requests[index].internal_id)
            if pending is not None and not pending.canceled:
                pending.events.put(event)
            if event.finished:
                finished.add(index)
        if finished != set(range(len(requests))):
            raise RuntimeError("multimodal backend stream ended before every request finished")
        return outputs

    def _finish_batch(
        self,
        requests: list[MultimodalGenerationRequest],
        *,
        outputs: list[str] | None = None,
        error: BaseException | None = None,
        queue_wait_ms: float = 0.0,
        coalescing_wait_ms: float = 0.0,
        execution_ms: float = 0.0,
    ) -> None:
        with self._condition:
            self._executed_batches += 1
            self._total_batched_requests += len(requests)
            self._batch_size_histogram[len(requests)] = self._batch_size_histogram.get(len(requests), 0) + 1
            self._total_queue_wait_ms += queue_wait_ms
            self._total_coalescing_wait_ms += coalescing_wait_ms
            self._total_execution_ms += execution_ms
            if error is None:
                self._completed_requests += len(requests)
            else:
                self._failed_requests += len(requests)

            for index, request in enumerate(requests):
                pending = self._pending.pop(request.internal_id)
                if error is not None:
                    pending.error = error
                    if not pending.canceled:
                        pending.events.put(error)
                else:
                    pending.result = outputs[index] if outputs is not None else ""
                pending.done.set()
