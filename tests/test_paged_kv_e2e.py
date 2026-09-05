import unittest

from compare_paged_kv_e2e import build_parser
from nanovllm.benchmark.paged_kv_e2e import compare_paged_kv_reports


def report(backend, latencies=(1.0, 2.0), tokens=((1, 2), (3, 4, 5))):
    return {
        "stage": "native-sdpa",
        "parameters": {"paged_kv_backend": backend},
        "cases": [
            {
                "name": f"case-{index}",
                "output_token_ids": list(token_ids),
                "benchmark": {
                    "total_latency_sec": latency,
                    "output_tokens": len(token_ids),
                    "peak_gpu_memory_bytes": 100 + index,
                },
            }
            for index, (latency, token_ids) in enumerate(zip(latencies, tokens))
        ],
    }


class PagedKVE2EComparisonTests(unittest.TestCase):
    def test_cli_requires_three_backend_reports(self):
        args = build_parser().parse_args(
            ["--pytorch", "p.json", "--triton", "t.json", "--cuda", "c.json"]
        )

        self.assertEqual(args.cuda.name, "c.json")

    def test_compares_exact_tokens_latency_throughput_and_memory(self):
        result = compare_paged_kv_reports(
            {
                "pytorch": report("pytorch", latencies=(2.0, 4.0)),
                "triton": report("triton", latencies=(1.0, 2.0)),
                "cuda": report("cuda", latencies=(0.5, 1.0)),
            }
        )

        self.assertTrue(result["tokens_exact"])
        self.assertEqual(result["backends"]["pytorch"]["total_latency_sec"], 6.0)
        self.assertAlmostEqual(result["backends"]["cuda"]["output_tokens_per_sec"], 5 / 1.5)
        self.assertEqual(result["backends"]["cuda"]["speedup_vs_pytorch"], 4.0)
        self.assertEqual(result["backends"]["cuda"]["peak_gpu_memory_bytes"], 101)

    def test_rejects_any_backend_token_difference(self):
        changed = report("cuda", tokens=((1, 9), (3, 4, 5)))

        with self.assertRaisesRegex(ValueError, "token mismatch"):
            compare_paged_kv_reports(
                {
                    "pytorch": report("pytorch"),
                    "triton": report("triton"),
                    "cuda": changed,
                }
            )


if __name__ == "__main__":
    unittest.main()
