from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Callable, ContextManager, Literal

from nanovllm import LLM, SamplingParams
from nanovllm.engine.multimodal_engine import MultimodalEngineCore
from nanovllm.multimodal import Qwen25VLBackend, Qwen25VLConfig
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


class NativeQwen25VLBackend:
    def __init__(
        self,
        llm_getter: Callable[[], LLM],
        execution_lock: ContextManager,
    ) -> None:
        self._llm_getter = llm_getter
        self._execution_lock = execution_lock

    def generate_chat_batch(
        self,
        messages_batch: list[list[dict[str, object]]],
        *,
        image_inputs_batch: list[list[object]],
        image_details_batch: list[list[str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
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
    multimodal_batch_wait_ms: float = 5.0
    multimodal_cache_capacity: int = 256
    multimodal_feature_cache_bytes: int = 2 * 1024**3
    multimodal_allow_local_files: bool = False

    def __post_init__(self) -> None:
        if self.multimodal_backend not in {"native", "hf"}:
            raise ValueError("multimodal_backend must be 'native' or 'hf'")
        if self.multimodal_feature_cache_bytes <= 0:
            raise ValueError("multimodal feature cache byte budget must be positive")


class EngineClient:
    """Serving wrapper for unified native generation and the explicit HF fallback."""

    def __init__(self, config: EngineClientConfig):
        self.config = config
        self._llm: LLM | None = None
        self._multimodal_backend: Qwen25VLBackend | None = None
        self._native_backend: NativeQwen25VLBackend | None = None
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
            self._llm = LLM(
                model_path,
                enforce_eager=self.config.enforce_eager,
                tensor_parallel_size=self.config.tensor_parallel_size,
                vision_feature_cache_bytes=self.config.multimodal_feature_cache_bytes,
                vision_attn_implementation=self.config.multimodal_attn_implementation or "sdpa",
            )
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
    ) -> str:
        if self.config.multimodal_backend == "native":
            if top_p != 1.0:
                raise ValueError("native multimodal sampling currently requires top_p=1.0")
        return self.multimodal_core.generate(
            request,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
        )

    @property
    def native_backend(self) -> NativeQwen25VLBackend:
        if self._native_backend is None:
            self._native_backend = NativeQwen25VLBackend(lambda: self.llm, self._lock)
        return self._native_backend

    @property
    def multimodal_core(self) -> MultimodalEngineCore:
        if self._multimodal_core is None:
            with self._multimodal_init_lock:
                if self._multimodal_core is None:
                    policy = MultimodalBatchPolicy(
                        max_batch_size=self.config.multimodal_max_batch_size,
                        max_images_per_batch=self.config.multimodal_max_images_per_batch,
                        batch_wait_ms=self.config.multimodal_batch_wait_ms,
                    )
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
            "manager": asdict(stats.manager),
        }

    def shutdown(self) -> None:
        if self._multimodal_core is not None:
            self._multimodal_core.shutdown()
        if self._llm is not None:
            self._llm.exit()
