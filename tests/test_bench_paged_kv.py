import json
import tempfile
import unittest
from pathlib import Path

import torch

from bench_paged_kv import build_parser
from nanovllm.benchmark.paged_kv import (
    PagedKVBenchmarkCase,
    atomic_write_json,
    effective_bytes,
    expand_benchmark_matrix,
    parse_case_selector,
    run_with_correctness_gate,
    summarize_latencies,
)


class PagedKVBenchmarkContractTests(unittest.TestCase):
    def test_cli_accepts_single_case_profiler_replay(self):
        selector = (
            "backend=cuda;operation=store;dtype=float16;tokens=257;"
            "blocks=64;block=16;heads=8;dim=128;mapping=random;seed=17"
        )

        args = build_parser().parse_args(
            ["--case-selector", selector, "--profile-replay", "25"]
        )

        self.assertEqual(args.case_selector, selector)
        self.assertEqual(args.profile_replay, 25)

    def test_matrix_expansion_is_deterministic_and_unique(self):
        first = expand_benchmark_matrix(
            backends=("pytorch", "triton", "cuda"),
            operations=("store", "gather"),
            dtypes=(torch.float16,),
            token_counts=(1, 257),
            kv_shapes=((2, 128),),
            block_size=16,
            num_blocks=32,
        )
        second = expand_benchmark_matrix(
            backends=("pytorch", "triton", "cuda"),
            operations=("store", "gather"),
            dtypes=(torch.float16,),
            token_counts=(1, 257),
            kv_shapes=((2, 128),),
            block_size=16,
            num_blocks=32,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({case.selector for case in first}), len(first))

    def test_effective_bytes_counts_payload_reads_and_writes(self):
        store = PagedKVBenchmarkCase(
            backend="cuda",
            operation="store",
            dtype=torch.float16,
            num_tokens=8,
            num_blocks=4,
            block_size=16,
            num_kv_heads=2,
            head_dim=128,
        )
        gather = PagedKVBenchmarkCase(
            backend="cuda",
            operation="gather",
            dtype=torch.float32,
            num_tokens=9,
            num_blocks=4,
            block_size=16,
            num_kv_heads=1,
            head_dim=80,
        )

        self.assertEqual(effective_bytes(store), 8 * 2 * 128 * 2 * 4)
        self.assertEqual(effective_bytes(gather), 9 * 1 * 80 * 4 * 2)

    def test_latency_summary_includes_p50_p95_and_p99(self):
        summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["p50_us"], 2.5)
        self.assertAlmostEqual(summary["p95_us"], 3.85)
        self.assertAlmostEqual(summary["p99_us"], 3.97)

    def test_correctness_gate_runs_before_warmup_or_timing(self):
        calls = []

        result = run_with_correctness_gate(
            correctness_check=lambda: calls.append("correctness"),
            warmup=lambda: calls.append("warmup"),
            measure=lambda: calls.append("measure") or [1.0, 2.0],
        )

        self.assertEqual(calls, ["correctness", "warmup", "measure"])
        self.assertEqual(result, [1.0, 2.0])

    def test_selector_round_trips_all_case_identity_fields(self):
        case = PagedKVBenchmarkCase(
            backend="triton",
            operation="gather",
            dtype=torch.bfloat16,
            num_tokens=257,
            num_blocks=64,
            block_size=16,
            num_kv_heads=8,
            head_dim=80,
            mapping="random_with_skips",
            seed=19,
        )

        self.assertEqual(parse_case_selector(case.selector), case)

    def test_atomic_json_report_preserves_identity_and_raw_samples(self):
        case = PagedKVBenchmarkCase(
            backend="pytorch",
            operation="store",
            dtype=torch.float16,
            num_tokens=1,
            num_blocks=2,
            block_size=16,
            num_kv_heads=1,
            head_dim=64,
        )
        report = {
            "schema_version": "nanoinfer.paged_kv_benchmark.v1",
            "identity": {"case_selector": case.selector, "gpu_uuid": "GPU-test"},
            "samples_us": [1.25, 1.5],
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            atomic_write_json(destination, report)

            loaded = json.loads(destination.read_text())

        self.assertEqual(loaded, report)


if __name__ == "__main__":
    unittest.main()
