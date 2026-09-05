import unittest

from nanovllm.benchmark.metrics import (
    SLOThresholds,
    aggregate_results,
    derive_request_metrics,
    linear_percentile,
)
from nanovllm.benchmark.schema import RequestResult


def make_result(
    request_id: str,
    *,
    start: float = 10.0,
    first: float | None = 10.2,
    done: float | None = 11.0,
    output_tokens: int = 5,
    success: bool = True,
) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        tags=("text",),
        http_status=200 if success else 500,
        success=success,
        error=None if success else "server error",
        timed_out=False,
        request_start=start,
        first_content_at=first,
        stream_done_at=done,
        content_chunk_times=() if first is None else (first,),
        prompt_tokens=8,
        output_tokens=output_tokens,
        token_count_source="usage",
        output_sha256="0" * 64,
        finish_reason="stop" if success else None,
    )


class RequestMetricTests(unittest.TestCase):
    def test_derives_ttft_e2e_tpot_and_single_user_tps(self):
        metrics = derive_request_metrics(make_result("request-1"))

        self.assertAlmostEqual(metrics.ttft_sec, 0.2)
        self.assertAlmostEqual(metrics.e2e_sec, 1.0)
        self.assertAlmostEqual(metrics.tpot_sec, 0.2)
        self.assertAlmostEqual(metrics.single_user_output_tps, 5.0)

    def test_one_token_output_has_no_tpot_and_fails_tpot_slo(self):
        metrics = derive_request_metrics(
            make_result("request-1", output_tokens=1),
            slo=SLOThresholds(ttft_sec=1.0, tpot_sec=0.05),
        )

        self.assertIsNone(metrics.tpot_sec)
        self.assertIsNone(metrics.single_user_output_tps)
        self.assertFalse(metrics.slo_compliant)

    def test_rejects_non_monotonic_success_timestamps(self):
        result = make_result("request-1", first=11.1, done=11.0)

        with self.assertRaisesRegex(ValueError, "monotonic"):
            derive_request_metrics(result)

    def test_default_slo_requires_both_ttft_and_tpot(self):
        passing = derive_request_metrics(
            make_result("pass", first=10.5, done=10.66, output_tokens=5)
        )
        slow_ttft = derive_request_metrics(
            make_result("ttft", first=11.01, done=11.17, output_tokens=5)
        )
        slow_tpot = derive_request_metrics(
            make_result("tpot", first=10.5, done=10.71, output_tokens=5)
        )

        self.assertTrue(passing.slo_compliant)
        self.assertFalse(slow_ttft.slo_compliant)
        self.assertFalse(slow_tpot.slo_compliant)


class AggregateMetricTests(unittest.TestCase):
    def test_linear_percentile_uses_linear_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(linear_percentile(values, 0.0), 1.0)
        self.assertEqual(linear_percentile(values, 0.5), 2.5)
        self.assertAlmostEqual(linear_percentile(values, 0.95), 3.85)
        self.assertAlmostEqual(linear_percentile(values, 0.99), 3.97)

    def test_aggregate_excludes_failures_from_latency_but_reports_errors(self):
        results = [
            make_result("ok-1", start=0.0, first=0.1, done=0.5, output_tokens=5),
            make_result("ok-2", start=0.0, first=0.3, done=1.1, output_tokens=5),
            make_result("failed", start=0.0, first=None, done=0.4, success=False),
        ]

        aggregate = aggregate_results(results, wall_time_sec=2.0)

        self.assertEqual(aggregate.total_requests, 3)
        self.assertEqual(aggregate.successful_requests, 2)
        self.assertEqual(aggregate.failed_requests, 1)
        self.assertAlmostEqual(aggregate.error_rate, 1 / 3)
        self.assertEqual(aggregate.ttft_sec.count, 2)
        self.assertEqual(aggregate.e2e_sec.count, 2)
        self.assertEqual(aggregate.tpot_sec.count, 2)
        self.assertAlmostEqual(aggregate.request_throughput_rps, 1.0)
        self.assertAlmostEqual(aggregate.system_output_tps, 5.0)

    def test_goodput_counts_only_slo_compliant_successes(self):
        results = [
            make_result("pass", start=0.0, first=0.5, done=0.66, output_tokens=5),
            make_result("slow", start=0.0, first=1.1, done=1.26, output_tokens=5),
        ]

        aggregate = aggregate_results(results, wall_time_sec=2.0)

        self.assertEqual(aggregate.slo_compliant_requests, 1)
        self.assertAlmostEqual(aggregate.slo_compliance, 0.5)
        self.assertAlmostEqual(aggregate.goodput_rps, 0.5)


if __name__ == "__main__":
    unittest.main()
