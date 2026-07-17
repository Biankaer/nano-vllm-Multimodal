from __future__ import annotations

import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.gpu_worker import GPUWorker, run_gpu_worker
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.qwen25_vl_processor import VisionInput


class Executor:
    """Owns GPU workers and exposes model execution to EngineCore."""

    def __init__(self, config: Config):
        self.config = config
        self.processes = []
        self.events = []

        ctx = mp.get_context("spawn")
        for rank in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=run_gpu_worker, args=(config, rank, event))
            process.start()
            self.processes.append(process)
            self.events.append(event)

        self.driver_worker = GPUWorker(config, 0, self.events)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return self.driver_worker.run(seqs, is_prefill)

    def register_vision_inputs(self, inputs: tuple[VisionInput, ...]) -> None:
        self.driver_worker.register_vision_inputs(inputs)

    def release_vision_inputs(self, keys: tuple[str, ...]) -> None:
        self.driver_worker.release_vision_inputs(keys)

    def vision_stats(self):
        return self.driver_worker.vision_stats()

    def shutdown(self) -> None:
        self.driver_worker.shutdown()
        del self.driver_worker
        for process in self.processes:
            process.join()
