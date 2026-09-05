import json
import unittest

from nanovllm.benchmark.sse_client import (
    OpenAIStreamAccumulator,
    SSEDecoder,
    SSEProtocolError,
)


def event(payload) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


ROLE = {
    "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
}
CONTENT_1 = {
    "choices": [{"delta": {"content": "你"}, "finish_reason": None}],
}
CONTENT_2 = {
    "choices": [{"delta": {"content": "好"}, "finish_reason": None}],
}
FINISH = {
    "choices": [{"delta": {}, "finish_reason": "stop"}],
}
USAGE = {
    "choices": [],
    "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
}
TRANSCRIPT = b"".join(
    event(payload) for payload in (ROLE, CONTENT_1, CONTENT_2, FINISH, USAGE, "[DONE]")
)


def parse_chunks(chunks: list[bytes]):
    decoder = SSEDecoder()
    accumulator = OpenAIStreamAccumulator(request_start=10.0)
    event_index = 0
    for chunk in chunks:
        for parsed in decoder.feed(chunk):
            event_index += 1
            accumulator.consume(parsed, received_at=10.0 + event_index * 0.1)
    for parsed in decoder.close():
        event_index += 1
        accumulator.consume(parsed, received_at=10.0 + event_index * 0.1)
    return accumulator.finish()


class SSEDecoderTests(unittest.TestCase):
    def test_every_single_split_boundary_produces_the_same_stream(self):
        expected = parse_chunks([TRANSCRIPT])

        for split in range(len(TRANSCRIPT) + 1):
            with self.subTest(split=split):
                actual = parse_chunks([TRANSCRIPT[:split], TRANSCRIPT[split:]])
                self.assertEqual(actual.output_text, expected.output_text)
                self.assertEqual(actual.finish_reason, expected.finish_reason)
                self.assertEqual(actual.usage, expected.usage)

    def test_utf8_can_be_split_one_byte_at_a_time(self):
        observation = parse_chunks([bytes([byte]) for byte in TRANSCRIPT])

        self.assertEqual(observation.output_text, "你好")

    def test_multiline_data_fields_are_joined(self):
        decoder = SSEDecoder()

        events = decoder.feed(b"data: first\r\ndata: second\r\n\r\n")

        self.assertEqual([item.data for item in events], ["first\nsecond"])

    def test_malformed_json_is_rejected(self):
        decoder = SSEDecoder()
        accumulator = OpenAIStreamAccumulator(request_start=0.0)

        with self.assertRaisesRegex(SSEProtocolError, "JSON"):
            accumulator.consume(decoder.feed(event("not-json"))[0], received_at=0.1)

    def test_openai_error_object_is_rejected(self):
        decoder = SSEDecoder()
        accumulator = OpenAIStreamAccumulator(request_start=0.0)
        parsed = decoder.feed(event({"error": {"message": "model failed"}}))[0]

        with self.assertRaisesRegex(SSEProtocolError, "model failed"):
            accumulator.consume(parsed, received_at=0.1)

    def test_missing_done_is_rejected(self):
        without_done = TRANSCRIPT[: -len(event("[DONE]"))]

        with self.assertRaisesRegex(SSEProtocolError, "DONE"):
            parse_chunks([without_done])


class OpenAIStreamAccumulatorTests(unittest.TestCase):
    def test_role_only_chunk_does_not_set_ttft(self):
        decoder = SSEDecoder()
        accumulator = OpenAIStreamAccumulator(request_start=10.0)

        accumulator.consume(decoder.feed(event(ROLE))[0], received_at=10.1)
        self.assertIsNone(accumulator.first_content_at)
        accumulator.consume(decoder.feed(event(CONTENT_1))[0], received_at=10.4)

        self.assertEqual(accumulator.first_content_at, 10.4)

    def test_finish_returns_content_timing_and_usage(self):
        observation = parse_chunks([TRANSCRIPT])

        self.assertEqual(observation.output_text, "你好")
        self.assertEqual(observation.finish_reason, "stop")
        self.assertEqual(observation.prompt_tokens, 9)
        self.assertEqual(observation.output_tokens, 2)
        self.assertEqual(observation.token_count_source, "usage")
        self.assertEqual(len(observation.content_chunk_times), 2)
        self.assertGreater(observation.stream_done_at, observation.first_content_at)


if __name__ == "__main__":
    unittest.main()
