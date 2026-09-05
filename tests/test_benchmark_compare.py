import copy
import unittest

from nanovllm.benchmark.compare import ReportIdentityError, compare_reports


def make_report(engine: str, *, scale: float = 1.0):
    return {
        "schema_version": "1.0",
        "engine": {"name": engine, "version": "0.1", "command": f"serve-{engine}"},
        "identity": {
            "workload_sha256": "a" * 64,
            "checkpoint_fingerprint": "b" * 64,
            "dtype": "bfloat16",
            "generation_parameters": {"temperature": 0, "top_p": 1.0},
            "profile": "text_short",
            "concurrency": 4,
            "gpu_uuid": "GPU-123",
            "request_ids": ["r1", "r2"],
        },
        "aggregates": {
            "ttft_sec": {"p50": 0.5 * scale, "p95": 0.8 * scale, "p99": 1.0 * scale},
            "tpot_sec": {"p50": 0.02 * scale, "p95": 0.03 * scale, "p99": 0.04 * scale},
            "e2e_sec": {"p50": 1.0 * scale, "p95": 1.2 * scale, "p99": 1.4 * scale},
            "request_throughput_rps": 8.0 / scale,
            "system_output_tps": 100.0 / scale,
            "goodput_rps": 7.0 / scale,
        },
        "requests": [
            {"request_id": "r1", "output_sha256": "1" * 64},
            {"request_id": "r2", "output_sha256": "2" * 64},
        ],
        "resource_metrics": {"valid": True, "contaminated": False, "peak_memory_mib": 4096},
    }


class ReportIdentityTests(unittest.TestCase):
    def test_all_fairness_identity_fields_must_match(self):
        baseline = make_report("nanoinfer")
        mutations = {
            "schema_version": lambda report: report.update(schema_version="2.0"),
            "workload": lambda report: report["identity"].update(workload_sha256="c" * 64),
            "checkpoint": lambda report: report["identity"].update(checkpoint_fingerprint="d" * 64),
            "dtype": lambda report: report["identity"].update(dtype="float16"),
            "generation": lambda report: report["identity"].update(
                generation_parameters={"temperature": 0.5, "top_p": 1.0}
            ),
            "profile": lambda report: report["identity"].update(profile="text_long"),
            "concurrency": lambda report: report["identity"].update(concurrency=8),
            "gpu": lambda report: report["identity"].update(gpu_uuid="GPU-456"),
            "request ids": lambda report: report["identity"].update(request_ids=["r1", "r3"]),
        }

        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(make_report("vllm"))
                mutate(changed)
                with self.assertRaises(ReportIdentityError):
                    compare_reports(baseline, changed)

    def test_greedy_output_mismatch_is_rejected(self):
        changed = make_report("vllm")
        changed["requests"][1]["output_sha256"] = "9" * 64

        with self.assertRaisesRegex(ReportIdentityError, "output"):
            compare_reports(make_report("nanoinfer"), changed)

    def test_missing_required_metric_is_rejected(self):
        changed = make_report("vllm")
        del changed["aggregates"]["tpot_sec"]["p95"]

        with self.assertRaisesRegex(ReportIdentityError, "metric"):
            compare_reports(make_report("nanoinfer"), changed)


class RatioTests(unittest.TestCase):
    def test_ratios_are_always_nanoinfer_over_vllm(self):
        comparison = compare_reports(
            make_report("nanoinfer", scale=0.5),
            make_report("vllm", scale=1.0),
        )

        self.assertAlmostEqual(comparison.rows["ttft_sec.p50"].ratio, 0.5)
        self.assertEqual(comparison.rows["ttft_sec.p50"].better_when, "lower")
        self.assertAlmostEqual(comparison.rows["system_output_tps"].ratio, 2.0)
        self.assertEqual(comparison.rows["system_output_tps"].better_when, "higher")

    def test_contaminated_resources_are_not_compared(self):
        nano = make_report("nanoinfer")
        nano["resource_metrics"]["contaminated"] = True

        comparison = compare_reports(nano, make_report("vllm"))

        self.assertFalse(comparison.resource_comparison_valid)
        self.assertIn("contaminated", comparison.resource_comparison_reason)


if __name__ == "__main__":
    unittest.main()
