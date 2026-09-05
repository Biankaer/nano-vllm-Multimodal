import unittest

from nanovllm.engine.multimodal_engine import MultimodalEngineCore
from nanovllm.multimodal.scheduler_policy import MultimodalBatchPolicy
from nanovllm.multimodal.schemas import MultimodalRequest


class _Backend:
    def generate_chat_batch(self, messages_batch, **kwargs):
        return ["ok"] * len(messages_batch)


class MultimodalBatchingPolicyTests(unittest.TestCase):
    def test_explicit_quiet_and_max_wait_are_exposed(self):
        policy = MultimodalBatchPolicy(
            batch_quiet_ms=2.0,
            batch_max_wait_ms=20.0,
        )

        self.assertEqual(policy.effective_quiet_ms, 2.0)
        self.assertEqual(policy.effective_max_wait_ms, 20.0)

    def test_legacy_batch_wait_remains_fallback(self):
        policy = MultimodalBatchPolicy(batch_wait_ms=7.0)

        self.assertEqual(policy.effective_quiet_ms, 7.0)
        self.assertEqual(policy.effective_max_wait_ms, 7.0)

    def test_policy_rejects_quiet_window_larger_than_max_wait(self):
        with self.assertRaises(ValueError):
            MultimodalBatchPolicy(batch_quiet_ms=21.0, batch_max_wait_ms=20.0)

    def test_core_reports_batch_histogram_and_wait_metrics(self):
        core = MultimodalEngineCore(
            _Backend(),
            policy=MultimodalBatchPolicy(
                max_batch_size=2,
                batch_quiet_ms=0.0,
                batch_max_wait_ms=1.0,
            ),
        )
        try:
            core.generate(
                MultimodalRequest(text="hello", request_id="r1"),
                max_new_tokens=1,
                temperature=0,
            )
            stats = core.stats()
        finally:
            core.shutdown()

        self.assertEqual(stats.batch_size_histogram, {1: 1})
        self.assertGreaterEqual(stats.total_queue_wait_ms, 0.0)
        self.assertGreaterEqual(stats.total_coalescing_wait_ms, 0.0)
        self.assertGreaterEqual(stats.total_execution_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
