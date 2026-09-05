import contextlib
import threading
import unittest

from nanovllm.engine.multimodal_engine import MultimodalEngineCore
from nanovllm.engine.streaming import GenerationStreamEvent
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


def _request(name: str) -> MultimodalRequest:
    return MultimodalRequest(
        text=name,
        request_id=name,
        messages=[{"role": "user", "content": name}],
    )


class ControlledDynamicBackend:
    def __init__(self):
        self.next_seq_id = 10
        self.remaining: dict[int, int] = {}
        self.names: dict[int, str] = {}
        self.log: list[tuple[str, str]] = []
        self.first_step_started = threading.Event()
        self.release_first_step = threading.Event()
        self.step_count = 0

    @contextlib.contextmanager
    def dynamic_session(self):
        yield

    def admit_chat_batch(self, messages_batch, **kwargs):
        sequence_ids = []
        for messages in messages_batch:
            name = str(messages[0]["content"])
            seq_id = self.next_seq_id
            self.next_seq_id += 1
            self.remaining[seq_id] = 3 if name == "first" else 1
            self.names[seq_id] = name
            self.log.append(("admit", name))
            sequence_ids.append(seq_id)
        return sequence_ids

    def step_dynamic(self):
        if self.step_count == 0:
            self.first_step_started.set()
            if not self.release_first_step.wait(1):
                raise TimeoutError("test did not release the first dynamic step")
        self.step_count += 1
        events = []
        for seq_id in list(self.remaining):
            self.remaining[seq_id] -= 1
            finished = self.remaining[seq_id] == 0
            name = self.names[seq_id]
            if finished:
                self.log.append(("finish", name))
                del self.remaining[seq_id]
            events.append(GenerationStreamEvent(
                request_index=-1,
                seq_id=seq_id,
                token_id=65 if name == "first" else 66,
                text_delta="A" if name == "first" else "B",
                finished=finished,
                finish_reason="length" if finished else None,
                prompt_tokens=1 if finished else None,
                completion_tokens=(3 if name == "first" else 1) if finished else None,
            ))
        return events

    def abort_dynamic(self, seq_id):
        self.remaining.pop(seq_id, None)


class DynamicContinuousBatchingTests(unittest.TestCase):
    def test_late_request_is_admitted_before_active_decode_finishes(self):
        backend = ControlledDynamicBackend()
        core = MultimodalEngineCore(
            backend,
            policy=MultimodalBatchPolicy(
                max_batch_size=8,
                batch_quiet_ms=0,
                batch_max_wait_ms=0,
            ),
        )
        try:
            first = core.stream(_request("first"), max_new_tokens=3, temperature=0)
            self.assertTrue(backend.first_step_started.wait(1))
            second = core.stream(_request("second"), max_new_tokens=1, temperature=0)
            backend.release_first_step.set()

            first_events = list(first)
            second_events = list(second)
        finally:
            backend.release_first_step.set()
            core.shutdown()

        self.assertEqual("".join(event.text_delta for event in first_events), "AAA")
        self.assertEqual("".join(event.text_delta for event in second_events), "B")
        self.assertLess(
            backend.log.index(("admit", "second")),
            backend.log.index(("finish", "first")),
        )

    def test_unknown_dynamic_event_fails_request_instead_of_killing_worker(self):
        class InvalidEventBackend(ControlledDynamicBackend):
            def step_dynamic(self):
                return [GenerationStreamEvent(
                    request_index=-1,
                    seq_id=999,
                    token_id=65,
                    text_delta="A",
                    finished=False,
                    finish_reason=None,
                )]

        backend = InvalidEventBackend()
        core = MultimodalEngineCore(
            backend,
            policy=MultimodalBatchPolicy(batch_quiet_ms=0, batch_max_wait_ms=0),
        )
        try:
            stream = core.stream(
                _request("first"), max_new_tokens=3, temperature=0, timeout=0.2,
            )
            with self.assertRaisesRegex(RuntimeError, "failed"):
                list(stream)
        finally:
            backend.release_first_step.set()
            core.shutdown()


if __name__ == "__main__":
    unittest.main()
