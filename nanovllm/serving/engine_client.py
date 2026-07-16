from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from nanovllm import LLM, SamplingParams
from nanovllm.multimodal import Qwen25VLBackend, Qwen25VLConfig


@dataclass(slots=True)
class EngineClientConfig:
    model_path: str
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    multimodal_model_path: str | None = None
    multimodal_device_map: str = "auto"
    multimodal_torch_dtype: str = "auto"
    multimodal_attn_implementation: str | None = None


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
        self._lock = Lock()
        self._multimodal_lock = Lock()

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
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        with self._multimodal_lock:
            return self.multimodal_backend.generate_chat(
                messages,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
