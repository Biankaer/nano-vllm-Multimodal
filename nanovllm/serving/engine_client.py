from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Callable, ContextManager, Literal

from nanovllm import LLM, SamplingParams
from nanovllm.config import read_local_model_type
from nanovllm.engine.multimodal_engine import MultimodalEngineCore
from nanovllm.engine.streaming import CumulativeTextDecoder, GenerationStreamEvent
from nanovllm.multimodal import Qwen25VLBackend, Qwen25VLConfig
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


class NativeQwenVLBackend:
    def __init__(
        self,
        llm_getter: Callable[[], LLM],
        execution_lock: ContextManager,
    ) -> None:
        self._llm_getter = llm_getter
        self._execution_lock = execution_lock
        self._dynamic_decoders: dict[int, CumulativeTextDecoder] = {}

    def generate_chat_batch(
        self,
        messages_batch: list[list[dict[str, object]]],
        *,
        image_inputs_batch: list[list[object]],
        image_details_batch: list[list[str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
    ) -> list[str]:
        if top_p != 1.0:
            raise ValueError("native multimodal sampling currently requires top_p=1.0")
        if image_details_batch is None:
            image_details_batch = [
                ["auto"] * len(images)
                for images in image_inputs_batch
            ]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
        )
        with self._execution_lock:
            outputs = self._llm_getter().generate_multimodal(
                messages_batch,
                image_inputs_batch,
                image_details_batch,
                sampling_params,
                use_tqdm=False,
            )
        return [output["text"] for output in outputs]

    def generate_chat_batch_stream(
        self,
        messages_batch: list[list[dict[str, object]]],
        *,
        image_inputs_batch: list[list[object]],
        image_details_batch: list[list[str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
    ):
        if top_p != 1.0:
            raise ValueError("native multimodal sampling currently requires top_p=1.0")
        if image_details_batch is None:
            image_details_batch = [
                ["auto"] * len(images)
                for images in image_inputs_batch
            ]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
        )
        with self._execution_lock:
            yield from self._llm_getter().generate_multimodal_stream(
                messages_batch,
                image_inputs_batch,
                image_details_batch,
                sampling_params,
            )

    @contextmanager
    def dynamic_session(self):
        with self._execution_lock:
            yield

    def admit_chat_batch(
        self,
        messages_batch: list[list[dict[str, object]]],
        *,
        image_inputs_batch: list[list[object]],
        image_details_batch: list[list[str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        ignore_eos: bool = False,
    ) -> list[int]:
        if top_p != 1.0:
            raise ValueError("native multimodal sampling currently requires top_p=1.0")
        llm = self._llm_getter()
        sequence_ids = llm.admit_multimodal_requests(
            messages_batch,
            image_inputs_batch,
            image_details_batch,
            SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
            ),
        )
        for seq_id in sequence_ids:
            self._dynamic_decoders[seq_id] = CumulativeTextDecoder(llm.tokenizer)
        return sequence_ids

    def step_dynamic(self) -> list[GenerationStreamEvent]:
        llm = self._llm_getter()
        llm.step()
        events = []
        for event in llm.core.drain_token_events():
            decoder = self._dynamic_decoders.get(event.seq_id)
            if decoder is None:
                raise RuntimeError(f"dynamic stream received unknown sequence {event.seq_id}")
            prompt_tokens = completion_tokens = None
            if event.finished:
                stats = llm.request_stats(event.seq_id)
                if stats is not None:
                    prompt_tokens = stats.prompt_tokens
                    completion_tokens = stats.completion_tokens
            events.append(GenerationStreamEvent(
                request_index=0,
                seq_id=event.seq_id,
                token_id=event.token_id,
                text_delta=decoder.append(event.token_id),
                finished=event.finished,
                finish_reason=event.finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ))
            if event.finished:
                del self._dynamic_decoders[event.seq_id]
        return events

    def abort_dynamic(self, seq_id: int) -> None:
        self._dynamic_decoders.pop(seq_id, None)
        self._llm_getter().abort_sequence(seq_id)


# Compatibility for existing imports and downstream integrations.
NativeQwen25VLBackend = NativeQwenVLBackend


@dataclass(slots=True)
class EngineClientConfig:
    model_path: str
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    multimodal_model_path: str | None = None
    multimodal_backend: Literal["native", "hf"] = "native"
    multimodal_device_map: str = "auto"
    multimodal_torch_dtype: str = "auto"
    multimodal_attn_implementation: str | None = None
    multimodal_max_batch_size: int = 8
    multimodal_max_images_per_batch: int = 8
    multimodal_batch_wait_ms: float | None = None
    multimodal_batch_quiet_ms: float = 3.0
    multimodal_batch_max_wait_ms: float = 20.0
    multimodal_cache_capacity: int = 256
    multimodal_feature_cache_bytes: int = 2 * 1024**3
    multimodal_allow_local_files: bool = False
    gpu_memory_utilization: float | None = None

    def __post_init__(self) -> None:
        if self.multimodal_backend not in {"native", "hf"}:
            raise ValueError("multimodal_backend must be 'native' or 'hf'")
        if self.multimodal_feature_cache_bytes <= 0:
            raise ValueError("multimodal feature cache byte budget must be positive")
        if self.gpu_memory_utilization is not None and not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


class EngineClient:
    """Serving wrapper for unified native generation and the explicit HF fallback."""

    def __init__(self, config: EngineClientConfig):
        self.config = config
        self._llm: LLM | None = None
        self._multimodal_backend: Qwen25VLBackend | None = None
        self._native_backend: NativeQwenVLBackend | None = None
        self._multimodal_core: MultimodalEngineCore | None = None
        self._lock = Lock()
        self._multimodal_init_lock = Lock()

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            model_path = (
                self.config.multimodal_model_path
                if self.config.multimodal_backend == "native" and self.config.multimodal_model_path
                else self.config.model_path
            )
            llm_kwargs = dict(
                enforce_eager=self.config.enforce_eager,
                tensor_parallel_size=self.config.tensor_parallel_size,
                vision_feature_cache_bytes=self.config.multimodal_feature_cache_bytes,
                vision_attn_implementation=self.config.multimodal_attn_implementation or "auto",
            )
            if self.config.gpu_memory_utilization is not None:
                llm_kwargs["gpu_memory_utilization"] = self.config.gpu_memory_utilization
            self._llm = LLM(model_path, **llm_kwargs)
        return self._llm

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        ignore_eos: bool = False,
    ) -> list[dict]:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        with self._lock:
            return self.llm.generate(prompts, sampling_params, use_tqdm=False)

    @property
    def multimodal_backend(self) -> Qwen25VLBackend:
        if not self.config.multimodal_model_path:
            raise RuntimeError("Multimodal model path is not configured. Set NANOINFER_MM_MODEL or pass multimodal_model_path.")
        if self._multimodal_backend is None:
            model_type = read_local_model_type(self.config.multimodal_model_path)
            if model_type == "qwen3_vl":
                raise ValueError("Qwen3-VL Serving requires multimodal_backend='native'")
            self._multimodal_backend = Qwen25VLBackend(
                Qwen25VLConfig(
                    model_path=self.config.multimodal_model_path,
                    device_map=self.config.multimodal_device_map,
                    torch_dtype=self.config.multimodal_torch_dtype,
                    attn_implementation=self.config.multimodal_attn_implementation,
                )
            )
        return self._multimodal_backend

    def generate_multimodal_chat(
        self,
        request: MultimodalRequest,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        ignore_eos: bool = False,
    ) -> str:
        if self.config.multimodal_backend == "native":
            if top_p != 1.0:
                raise ValueError("native multimodal sampling currently requires top_p=1.0")
        return self.multimodal_core.generate(
            request,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            ignore_eos=ignore_eos,
        )

    def stream_multimodal_chat(
        self,
        request: MultimodalRequest,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        ignore_eos: bool = False,
        timeout: float | None = None,
    ):
        if self.config.multimodal_backend != "native":
            raise NotImplementedError("streaming is available only for the native Qwen-VL backend")
        if top_p != 1.0:
            raise ValueError("native multimodal sampling currently requires top_p=1.0")
        return self.multimodal_core.stream(
            request,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            ignore_eos=ignore_eos,
            timeout=timeout,
        )

    @property
    def native_backend(self) -> NativeQwenVLBackend:
        if self._native_backend is None:
            self._native_backend = NativeQwenVLBackend(lambda: self.llm, self._lock)
        return self._native_backend

    @property
    def multimodal_core(self) -> MultimodalEngineCore:
        if self._multimodal_core is None:
            with self._multimodal_init_lock:
                if self._multimodal_core is None:
                    policy_kwargs = dict(
                        max_batch_size=self.config.multimodal_max_batch_size,
                        max_images_per_batch=self.config.multimodal_max_images_per_batch,
                    )
                    if self.config.multimodal_batch_wait_ms is not None:
                        policy_kwargs["batch_wait_ms"] = self.config.multimodal_batch_wait_ms
                    else:
                        policy_kwargs.update(
                            batch_quiet_ms=self.config.multimodal_batch_quiet_ms,
                            batch_max_wait_ms=self.config.multimodal_batch_max_wait_ms,
                        )
                    policy = MultimodalBatchPolicy(**policy_kwargs)
                    backend = (
                        self.native_backend
                        if self.config.multimodal_backend == "native"
                        else self.multimodal_backend
                    )
                    self._multimodal_core = MultimodalEngineCore(
                        backend,
                        policy=policy,
                        asset_cache_capacity=self.config.multimodal_cache_capacity,
                        feature_cache_capacity=self.config.multimodal_cache_capacity,
                    )
        return self._multimodal_core

    def multimodal_stats(self) -> dict | None:
        if self.config.multimodal_backend == "native":
            if self._llm is None:
                return None
            vision_stats = self._llm.vision_stats()
            if self._multimodal_core is None:
                return None
            manager_stats = self._multimodal_core.manager.stats()
            core_stats = self._multimodal_core.stats()
            return {
                "status": "ok",
                "backend": "native",
                "batching": {
                    "waiting_requests": core_stats.waiting_requests,
                    "submitted_requests": core_stats.submitted_requests,
                    "completed_requests": core_stats.completed_requests,
                    "failed_requests": core_stats.failed_requests,
                    "executed_batches": core_stats.executed_batches,
                    "average_batch_size": core_stats.average_batch_size,
                    "batch_size_histogram": core_stats.batch_size_histogram,
                    "average_queue_wait_ms": core_stats.average_queue_wait_ms,
                    "average_coalescing_wait_ms": core_stats.average_coalescing_wait_ms,
                    "average_execution_ms": core_stats.average_execution_ms,
                },
                "image_asset_cache": {
                    "entries": manager_stats.image_asset_cache_entries,
                    "hits": manager_stats.image_asset_cache_hits,
                    "misses": manager_stats.image_asset_cache_misses,
                    "evictions": manager_stats.image_asset_cache_evictions,
                },
                "vision_feature_cache": asdict(vision_stats) if vision_stats is not None else None,
                "scheduler": asdict(self._llm.scheduler_stats()),
            }
        if self._multimodal_core is None:
            return None
        stats = self._multimodal_core.stats()
        return {
            "status": "ok",
            "backend": "hf",
            "waiting_requests": stats.waiting_requests,
            "submitted_requests": stats.submitted_requests,
            "completed_requests": stats.completed_requests,
            "failed_requests": stats.failed_requests,
            "executed_batches": stats.executed_batches,
            "average_batch_size": stats.average_batch_size,
            "batch_size_histogram": stats.batch_size_histogram,
            "average_queue_wait_ms": stats.average_queue_wait_ms,
            "average_coalescing_wait_ms": stats.average_coalescing_wait_ms,
            "average_execution_ms": stats.average_execution_ms,
            "manager": asdict(stats.manager),
        }

    def shutdown(self) -> None:
        if self._multimodal_core is not None:
            self._multimodal_core.shutdown()
        if self._llm is not None:
            self._llm.exit()
