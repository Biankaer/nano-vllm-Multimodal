import unittest
from types import MethodType

import torch

from nanovllm.engine.engine_core import EngineCore
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.multimodal.qwen25_vl_processor import (
    MaterializedQwen25VLPrompt,
    VisionInput,
)
from nanovllm.multimodal.runtime import MultimodalPromptMetadata, VisualTokenSpan
from nanovllm.sampling_params import SamplingParams


def make_multimodal_sequence():
    positions = tuple(range(4))
    metadata = MultimodalPromptMetadata(
        position_ids=(positions, positions, positions),
        rope_delta=0,
        visual_spans=(VisualTokenSpan(1, 2, "vision:a"),),
    )
    return Sequence([10, 11, 11, 12], multimodal_metadata=metadata)


class FakeExecutor:
    def __init__(self):
        self.registered = []
        self.released = []

    def register_vision_inputs(self, inputs):
        self.registered.append(inputs)

    def release_vision_inputs(self, keys):
        self.released.append(keys)

    def run(self, seqs, is_prefill):
        return [42] * len(seqs)

    def vision_stats(self):
        return "vision-stats"


class FakeScheduler:
    def __init__(self):
        self.added = []
        self.scheduled = []

    def add(self, seq):
        self.added.append(seq)

    def schedule(self):
        return self.scheduled, True

    def postprocess(self, seqs, token_ids, is_prefill):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            seq.mark_finished()

    def stats(self):
        return object()


class NativeMultimodalEngineTests(unittest.TestCase):
    def setUp(self):
        self.core = EngineCore.__new__(EngineCore)
        self.core.executor = FakeExecutor()
        self.core.scheduler = FakeScheduler()
        self.core.step_id = 0
        self.core.latest_step_stats = None
        self.core.finished_request_stats = {}

    def test_registers_vision_inputs_before_admitting_sequence(self):
        sequence = make_multimodal_sequence()
        vision_input = VisionInput("vision:a", torch.zeros(4, 8), (1, 2, 2))

        self.core.add_sequence(sequence, (vision_input,))

        self.assertEqual(self.core.executor.registered, [(vision_input,)])
        self.assertEqual(self.core.scheduler.added, [sequence])

    def test_releases_host_inputs_when_sequence_finishes(self):
        sequence = make_multimodal_sequence()
        sequence.num_scheduled_tokens = len(sequence)
        self.core.scheduler.scheduled = [sequence]

        self.core.step()

        self.assertEqual(self.core.executor.released, [("vision:a",)])
        self.assertEqual(self.core.vision_stats(), "vision-stats")


class FakePromptProcessor:
    def __init__(self, materialized):
        self.materialized = materialized
        self.calls = []

    def materialize(self, messages, images, details):
        self.calls.append((messages, images, details))
        return self.materialized


class SequenceAdmissionCore:
    def __init__(self):
        self.admissions = []

    def add_sequence(self, sequence, vision_inputs=()):
        self.admissions.append((sequence, vision_inputs))


class NativeMultimodalOfflineAPITests(unittest.TestCase):
    def test_materialized_prompt_enters_the_normal_sequence_scheduler(self):
        metadata = make_multimodal_sequence().multimodal_metadata
        vision_input = VisionInput("vision:a", torch.zeros(4, 8), (1, 2, 2))
        materialized = MaterializedQwen25VLPrompt(
            token_ids=(10, 11, 11, 12),
            metadata=metadata,
            vision_inputs=(vision_input,),
        )
        engine = LLMEngine.__new__(LLMEngine)
        engine.multimodal_processor = FakePromptProcessor(materialized)
        engine.core = SequenceAdmissionCore()

        def fake_drain(self, sequence_ids, *, use_tqdm):
            return [{"text": "ok", "token_ids": [42]}]

        engine._drain_sequences = MethodType(fake_drain, engine)
        messages = [[{"role": "user", "content": [{"type": "image"}]}]]
        images = [[object()]]

        outputs = engine.generate_multimodal(
            messages,
            images,
            [["auto"]],
            SamplingParams(temperature=0.1, max_tokens=1),
            use_tqdm=False,
        )

        sequence, admitted_inputs = engine.core.admissions[0]
        self.assertEqual(sequence.prompt_token_ids, [10, 11, 11, 12])
        self.assertIs(sequence.multimodal_metadata, metadata)
        self.assertEqual(admitted_inputs, (vision_input,))
        self.assertEqual(outputs[0]["text"], "ok")


if __name__ == "__main__":
    unittest.main()
