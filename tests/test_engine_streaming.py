import unittest

from nanovllm.engine.engine_core import EngineCore
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.metrics import RequestTimingStats
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.streaming import CumulativeTextDecoder, TokenEvent
from nanovllm.sampling_params import SamplingParams


class ScriptedExecutor:
    def __init__(self, token_id=42):
        self.token_id = token_id
        self.released = []

    def run(self, sequences, is_prefill):
        return [self.token_id for _ in sequences]

    def release_vision_inputs(self, keys):
        self.released.append(keys)


class ScriptedScheduler:
    def __init__(self, sequence, *, is_prefill, append_token, finish=False):
        self.sequence = sequence
        self.is_prefill = is_prefill
        self.append_token = append_token
        self.finish = finish

    def schedule(self):
        return [self.sequence], self.is_prefill

    def postprocess(self, sequences, token_ids, is_prefill):
        sequence = sequences[0]
        if self.append_token:
            sequence.append_token(token_ids[0])
        if self.finish:
            sequence.mark_finished("length")

    def stats(self):
        return object()


def make_core(scheduler, executor=None):
    core = EngineCore.__new__(EngineCore)
    core.executor = executor or ScriptedExecutor()
    core.scheduler = scheduler
    core.step_id = 0
    core.latest_step_stats = None
    core.finished_request_stats = {}
    core._token_events = []
    return core


class EngineTokenEventTests(unittest.TestCase):
    def test_chunked_prefill_without_completion_token_emits_no_event(self):
        sequence = Sequence([1, 2, 3])
        sequence.num_scheduled_tokens = 2
        core = make_core(ScriptedScheduler(
            sequence, is_prefill=True, append_token=False,
        ))

        outputs, num_tokens = core.step()

        self.assertEqual(outputs, [])
        self.assertEqual(num_tokens, 2)
        self.assertEqual(core.drain_token_events(), [])

    def test_sampled_completion_emits_event_without_changing_step_signature(self):
        sequence = Sequence([1, 2], SamplingParams(max_tokens=1))
        sequence.num_scheduled_tokens = 1
        core = make_core(ScriptedScheduler(
            sequence, is_prefill=False, append_token=True, finish=True,
        ))

        outputs, num_tokens = core.step()

        self.assertEqual(outputs, [(sequence.seq_id, [42])])
        self.assertEqual(num_tokens, -1)
        self.assertEqual(core.drain_token_events(), [
            TokenEvent(
                seq_id=sequence.seq_id,
                token_id=42,
                finished=True,
                finish_reason="length",
            )
        ])
        self.assertEqual(core.drain_token_events(), [])

    def test_events_keep_sequence_ids_for_routing(self):
        first = TokenEvent(seq_id=11, token_id=101, finished=False, finish_reason=None)
        second = TokenEvent(seq_id=22, token_id=202, finished=True, finish_reason="stop")

        routed = {11: [], 22: []}
        for item in (first, second):
            routed[item.seq_id].append(item.token_id)

        self.assertEqual(routed, {11: [101], 22: [202]})


class CumulativeDecodeTests(unittest.TestCase):
    def test_cumulative_decode_waits_for_complete_utf8_text(self):
        class SplitTokenizer:
            texts = {(): "", (1,): "", (1, 2): "你", (1, 2, 3): "你好"}

            def decode(self, token_ids, **kwargs):
                return self.texts[tuple(token_ids)]

        decoder = CumulativeTextDecoder(SplitTokenizer())

        deltas = [decoder.append(token_id) for token_id in (1, 2, 3)]

        self.assertEqual(deltas, ["", "你", "好"])
        self.assertEqual("".join(deltas), "你好")

    def test_non_prefix_decode_is_rejected_instead_of_corrupting_output(self):
        class UnstableTokenizer:
            def decode(self, token_ids, **kwargs):
                return "one" if len(token_ids) == 1 else "changed"

        decoder = CumulativeTextDecoder(UnstableTokenizer())
        decoder.append(1)

        with self.assertRaisesRegex(RuntimeError, "prefix"):
            decoder.append(2)


class BatchStreamTests(unittest.TestCase):
    def test_batch_stream_routes_interleaved_events_and_terminal_usage(self):
        class Tokenizer:
            def decode(self, token_ids, **kwargs):
                return "".join(chr(token_id) for token_id in token_ids)

        class Core:
            def __init__(self):
                self.steps = [
                    [
                        TokenEvent(11, ord("A"), False, None),
                        TokenEvent(22, ord("X"), False, None),
                    ],
                    [
                        TokenEvent(11, ord("B"), True, "stop"),
                        TokenEvent(22, ord("Y"), True, "length"),
                    ],
                ]

            def step(self):
                return [], -2

            def drain_token_events(self):
                return self.steps.pop(0)

            def request_stats(self, seq_id):
                return RequestTimingStats(
                    seq_id=seq_id,
                    prompt_tokens=3,
                    completion_tokens=2,
                    queue_time_sec=0.1,
                    ttft_sec=0.2,
                    total_latency_sec=0.4,
                    tpot_sec=0.2,
                )

        engine = LLMEngine.__new__(LLMEngine)
        engine.tokenizer = Tokenizer()
        engine.core = Core()

        events = list(engine._drain_sequences_stream([11, 22]))

        self.assertEqual(
            [(event.request_index, event.text_delta) for event in events],
            [(0, "A"), (1, "X"), (0, "B"), (1, "Y")],
        )
        self.assertEqual(events[-2].finish_reason, "stop")
        self.assertEqual(events[-1].finish_reason, "length")
        self.assertEqual(events[-1].prompt_tokens, 3)
        self.assertEqual(events[-1].completion_tokens, 2)

    def test_batch_stream_rejects_event_for_unknown_sequence(self):
        class Core:
            def step(self):
                return [], -1

            def drain_token_events(self):
                return [TokenEvent(999, 1, False, None)]

        engine = LLMEngine.__new__(LLMEngine)
        engine.tokenizer = object()
        engine.core = Core()

        with self.assertRaisesRegex(RuntimeError, "unknown sequence"):
            list(engine._drain_sequences_stream([11]))


if __name__ == "__main__":
    unittest.main()
