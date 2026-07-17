from __future__ import annotations

from multiprocessing.synchronize import Event

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.qwen25_vl_processor import VisionInput


class GPUWorker:
    """GPU execution worker.

    `ModelRunner` still owns the real CUDA model, KV cache, and tensor-parallel
    shared-memory protocol. This wrapper gives the higher-level executor a stable
    worker abstraction, so later phases can add CPU workers, remote workers, or
    multimodal workers without changing `EngineCore`.
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.rank = rank
        self.runner = ModelRunner(config, rank, event)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return self.runner.call("run", seqs, is_prefill)

    def register_vision_inputs(self, inputs: tuple[VisionInput, ...]) -> None:
        self.runner.call("register_vision_inputs", inputs)

    def release_vision_inputs(self, keys: tuple[str, ...]) -> None:
        self.runner.call("release_vision_inputs", keys)

    def vision_stats(self):
        return self.runner.call("vision_stats")

    def shutdown(self) -> None:
        self.runner.call("exit")


def run_gpu_worker(config: Config, rank: int, event: Event) -> None:
    """Child-process entrypoint for tensor-parallel workers."""
    GPUWorker(config, rank, event)
