import unittest
from pathlib import Path

from scripts.profile_paged_kv_ncu import artifact_stem, build_parser
from nanovllm.benchmark.ncu import (
    NCU_REQUIRED_SECTIONS,
    analyze_ncu_metrics,
    build_ncu_collection_command,
    parse_ncu_csv,
)


class NCUCommandTests(unittest.TestCase):
    def test_profile_cli_uses_case_identity_for_artifact_name(self):
        selector = (
            "backend=cuda;operation=gather;dtype=bfloat16;tokens=257;"
            "blocks=64;block=16;heads=8;dim=128;mapping=random;seed=17"
        )
        args = build_parser().parse_args(["--case-selector", selector])

        self.assertEqual(args.case_selector, selector)
        self.assertEqual(
            artifact_stem(args.case_selector),
            "cuda-gather-bfloat16-t257-h8-d128",
        )

    def test_collection_command_filters_kernels_and_requests_sections(self):
        command = build_ncu_collection_command(
            ncu="/opt/ncu",
            python="/env/python",
            benchmark_script=Path("bench_paged_kv.py"),
            case_selector=(
                "backend=cuda;operation=store;dtype=float16;tokens=257;"
                "blocks=64;block=16;heads=8;dim=128;mapping=random;seed=17"
            ),
            report_path=Path("profile_output/store.ncu-rep"),
            replay_iterations=20,
            launch_count=10,
        )

        self.assertEqual(command[0], "/opt/ncu")
        for section in NCU_REQUIRED_SECTIONS:
            self.assertIn(section, command)
        self.assertIn("regex:.*(store_kernel|gather_kernel|_store_paged_kv_kernel|_gather_paged_kv_kernel).*", command)
        self.assertIn("--case-selector", command)
        self.assertIn("--profile-replay", command)


class NCUParserTests(unittest.TestCase):
    def test_parses_metrics_by_kernel_and_normalizes_numeric_values(self):
        payload = """==PROF== Connected to process 1\n\"ID\",\"Kernel Name\",\"Section Name\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"\n\"1\",\"store_kernel<half>\",\"GPU Speed Of Light Throughput\",\"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed\",\"%\",\"72.50\"\n\"1\",\"store_kernel<half>\",\"Occupancy\",\"sm__warps_active.avg.pct_of_peak_sustained_active\",\"%\",\"50.00\"\n\"1\",\"store_kernel<half>\",\"Launch Statistics\",\"launch__registers_per_thread\",\"register/thread\",\"32\"\n"""

        parsed = parse_ncu_csv(payload)

        metrics = parsed["kernels"]["store_kernel<half>"]["metrics"]
        self.assertEqual(
            metrics["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"]["value"],
            72.5,
        )
        self.assertEqual(metrics["launch__registers_per_thread"]["value"], 32.0)

    def test_analysis_preserves_missing_metrics_as_none(self):
        parsed = parse_ncu_csv(
            '"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"\n'
            '"1","gather_kernel<float>","Occupancy","sm__warps_active.avg.pct_of_peak_sustained_active","%","61.25"\n'
        )

        analysis = analyze_ncu_metrics(parsed)

        gather = analysis["gather_kernel<float>"]
        self.assertEqual(gather["achieved_occupancy_pct"], 61.25)
        self.assertIsNone(gather["dram_throughput_pct"])
        self.assertIsNone(gather["warp_execution_efficiency_pct"])
        self.assertIn("dram_throughput_pct", gather["missing_metrics"])

    def test_rejects_csv_without_ncu_metric_header(self):
        with self.assertRaisesRegex(ValueError, "Metric Name"):
            parse_ncu_csv("permission denied")


if __name__ == "__main__":
    unittest.main()
