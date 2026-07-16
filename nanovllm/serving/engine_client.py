from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from nanovllm import LLM, SamplingParams


@dataclass(slots=True)
class EngineClientConfig:
    model_path: str
    enforce_eager: bool = True
    tensor_parallel_size: int = 1


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
        self._lock = Lock()

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
