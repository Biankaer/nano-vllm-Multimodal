import unittest
from types import MethodType
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def abort(self, seq_id):
        for sequence in self.waiting:
            if sequence.seq_id == seq_id:
                self.waiting.remove(sequence)
                sequence.mark_finished("abort")
                return sequence
        return None


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

    def test_abort_sequence_releases_registered_vision_inputs(self):
        sequence = make_multimodal_sequence()
        vision_input = VisionInput("vision:a", torch.zeros(4, 8), (1, 2, 2))
        self.core.scheduler.waiting = [sequence]
        self.core.abort_sequence(sequence.seq_id)

        self.assertEqual(self.core.executor.released, [("vision:a",)])
        self.assertTrue(sequence.is_finished)


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
    def test_dynamic_admission_materializes_without_draining_engine(self):
        metadata = make_multimodal_sequence().multimodal_metadata
        materialized = MaterializedQwen25VLPrompt(
            token_ids=(10, 11, 11, 12),
            metadata=metadata,
            vision_inputs=(),
        )
        engine = LLMEngine.__new__(LLMEngine)
        engine.multimodal_processor = FakePromptProcessor(materialized)
        engine.core = SequenceAdmissionCore()

        sequence_ids = engine.admit_multimodal_requests(
            [[{"role": "user", "content": "hello"}]],
            [[]],
            [[]],
            SamplingParams(temperature=0, max_tokens=2),
        )

        self.assertEqual(len(sequence_ids), 1)
        self.assertEqual(sequence_ids[0], engine.core.admissions[0][0].seq_id)

    def test_qwen3_engine_selects_adapter_processor_and_tokenizer(self):
        tokenizer = SimpleNamespace(eos_token_id=42)
        processor = SimpleNamespace(tokenizer=tokenizer)
        adapter = Mock()
        adapter.create_prompt_processor.return_value = processor
        config = SimpleNamespace(
            model="/model", hf_config=SimpleNamespace(model_type="qwen3_vl"), eos=-1,
            kvcache_block_size=256,
        )
        with (
            patch("nanovllm.engine.llm_engine.Config", return_value=config),
            patch("nanovllm.engine.llm_engine.fields", return_value=()),
            patch("nanovllm.engine.llm_engine.resolve_qwen_vl_adapter", return_value=adapter),
            patch("nanovllm.engine.llm_engine.EngineCore", return_value=Mock()),
            patch("nanovllm.engine.llm_engine.atexit.register"),
        ):
            engine = LLMEngine("/model")
        adapter.create_prompt_processor.assert_called_once_with("/model")
        self.assertIs(engine.multimodal_processor, processor)
        self.assertIs(engine.tokenizer, tokenizer)
        self.assertEqual(config.eos, 42)

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

    def test_multimodal_stream_materializes_and_uses_stream_drain(self):
        metadata = make_multimodal_sequence().multimodal_metadata
        materialized = MaterializedQwen25VLPrompt(
            token_ids=(10, 11, 11, 12),
            metadata=metadata,
            vision_inputs=(),
        )
        engine = LLMEngine.__new__(LLMEngine)
        engine.multimodal_processor = FakePromptProcessor(materialized)
        engine.core = SequenceAdmissionCore()

        def fake_stream(self, sequence_ids):
            yield (sequence_ids[0], "delta")

        engine._drain_sequences_stream = MethodType(fake_stream, engine)

        events = list(engine.generate_multimodal_stream(
            [[{"role": "user", "content": "hello"}]],
            [[]],
            [[]],
            SamplingParams(temperature=0, max_tokens=2),
        ))

        self.assertEqual(events[0][1], "delta")
        self.assertEqual(len(engine.core.admissions), 1)


if __name__ == "__main__":
    unittest.main()
