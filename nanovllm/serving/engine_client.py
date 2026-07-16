from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from nanovllm import LLM, SamplingParams
from nanovllm.engine.multimodal_engine import MultimodalEngineCore, MultimodalEngineStats
from nanovllm.multimodal import Qwen25VLBackend, Qwen25VLConfig
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


@dataclass(slots=True)
class EngineClientConfig:
    model_path: str
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    multimodal_model_path: str | None = None
    multimodal_device_map: str = "auto"
    multimodal_torch_dtype: str = "auto"
    multimodal_attn_implementation: str | None = None
    multimodal_max_batch_size: int = 8
    multimodal_max_images_per_batch: int = 8
    multimodal_batch_wait_ms: float = 5.0
    multimodal_cache_capacity: int = 256
    multimodal_allow_local_files: bool = False


class EngineClient:
    """Thin serving wrapper around the existing offline LLM API.

    The first milestone intentionally keeps execution simple: requests are served
    by the original blocking `LLM.generate()` API. Later phases can replace this
    wrapper with EngineCore + async continuous batching without changing the API
    layer.
    """

    def __init__(self, config: EngineClientConfig):
        self.config = config
        self._llm: LLM | None = None
        self._multimodal_backend: Qwen25VLBackend | None = None
        self._multimodal_core: MultimodalEngineCore | None = None
        self._lock = Lock()
        self._multimodal_init_lock = Lock()

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM(
                self.config.model_path,
                enforce_eager=self.config.enforce_eager,
                tensor_parallel_size=self.config.tensor_parallel_size,
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
        return self.multimodal_core.generate(
            request,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
        )

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
                    self._multimodal_core = MultimodalEngineCore(
                        self.multimodal_backend,
                        policy=policy,
                        asset_cache_capacity=self.config.multimodal_cache_capacity,
                        feature_cache_capacity=self.config.multimodal_cache_capacity,
                    )
        return self._multimodal_core

    def multimodal_stats(self) -> MultimodalEngineStats | None:
        return self._multimodal_core.stats() if self._multimodal_core is not None else None

    def shutdown(self) -> None:
        if self._multimodal_core is not None:
            self._multimodal_core.shutdown()
        if self._llm is not None:
            self._llm.exit()
