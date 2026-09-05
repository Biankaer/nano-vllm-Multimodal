import unittest

from nanovllm.engine.multimodal_engine import MultimodalEngineCore
from nanovllm.engine.streaming import GenerationStreamEvent
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


class ScriptedStreamingBackend:
    def __init__(self):
        self.batch_sizes = []

    def generate_chat_batch_stream(
        self,
        messages_batch,
        *,
        image_inputs_batch,
        image_details_batch,
        max_new_tokens,
        temperature,
        top_p,
        ignore_eos=False,
    ):
        self.batch_sizes.append(len(messages_batch))
        self.ignore_eos = ignore_eos
        for index in range(len(messages_batch)):
            yield GenerationStreamEvent(
                request_index=index,
                seq_id=100 + index,
                token_id=65 + index,
                text_delta=chr(65 + index),
                finished=False,
                finish_reason=None,
            )
        for index in range(len(messages_batch)):
            yield GenerationStreamEvent(
                request_index=index,
                seq_id=100 + index,
                token_id=97 + index,
                text_delta=chr(97 + index),
                finished=True,
                finish_reason="stop",
                prompt_tokens=4,
                completion_tokens=2,
            )


def request(name: str) -> MultimodalRequest:
    return MultimodalRequest(
        text=name,
        request_id=name,
        messages=[{"role": "user", "content": name}],
    )


class MultimodalStreamingTests(unittest.TestCase):
    def setUp(self):
        self.backend = ScriptedStreamingBackend()
        self.core = MultimodalEngineCore(
            self.backend,
            policy=MultimodalBatchPolicy(
                max_batch_size=8,
                max_images_per_batch=8,
                batch_wait_ms=20,
            ),
        )

    def tearDown(self):
        self.core.shutdown()

    def test_stream_routes_interleaved_batch_events_to_each_request(self):
        first = self.core.stream(
            request("first"), max_new_tokens=2, temperature=0, top_p=1.0,
        )
        second = self.core.stream(
            request("second"), max_new_tokens=2, temperature=0, top_p=1.0,
        )

        first_events = list(first)
        second_events = list(second)

        self.assertEqual(self.backend.batch_sizes, [2])
        self.assertEqual("".join(event.text_delta for event in first_events), "Aa")
        self.assertEqual("".join(event.text_delta for event in second_events), "Bb")
        self.assertTrue(first_events[-1].finished)
        self.assertEqual(first_events[-1].completion_tokens, 2)

    def test_non_stream_generate_remains_compatible_with_streaming_backend(self):
        output = self.core.generate(
            request("first"), max_new_tokens=2, temperature=0, top_p=1.0,
        )

        self.assertEqual(output, "Aa")

    def test_core_forwards_ignore_eos_to_stream_backend(self):
        output = self.core.stream(
            request("first"), max_new_tokens=2, temperature=0, top_p=1.0,
            ignore_eos=True,
        )

        list(output)
        self.assertTrue(self.backend.ignore_eos)


if __name__ == "__main__":
    unittest.main()
