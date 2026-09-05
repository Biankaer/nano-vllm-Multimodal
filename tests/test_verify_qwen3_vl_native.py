from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import verify_qwen3_vl_native as verification_module

from verify_qwen3_vl_native import (
    REQUIRED_COMPONENT_KEYS,
    SCHEMA_VERSION,
    benchmark_matrix,
    build_artifact_header,
    capture_first_sample_logits,
    checkpoint_identity,
    compare_consistency_results,
    compare_result_files,
    execute_benchmark_matrix,
    external_paged_kv_cache_bytes,
    compare_generation_results,
    compare_component_artifacts,
    compute_native_component_first_logits,
    generate_test_cases,
    load_generation_result,
    parse_args,
    select_first_generated_logits,
    validate_component_artifact,
    write_json,
)


def _result(stage: str, outputs: dict[str, list[int]]) -> dict[str, object]:
    backend = {
        "hf": "hf",
        "hf-flash": "flash_attention_2",
        "native-sdpa": "sdpa",
        "native-flash": "flash_attention_2",
    }[stage]
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "model": "/models/qwen3-vl",
        "checkpoint": {"path": "/models/qwen3-vl", "fingerprint_sha256": "a" * 64},
        "parameters": {
            "temperature": 0,
            "max_new_tokens": 64,
            "dtype": "bfloat16",
            "chat_template_marker": "qwen3-vl-chat-template-v1",
            "vision_attn_implementation": backend,
        },
        "cases": [
            {
                "name": name,
                "metadata": {
                    "purpose": "test", "image_count": 0, "image_sizes": [],
                    "details": [], "force_chunked_prefill": False,
                    "repeat_runs": 1, "image_sha256": [],
                },
                "input_token_ids": [1, 2],
                "output_token_ids": token_ids,
                "text": "output",
                "logits_diagnostics": {
                    "top2_token_ids": [1, 2],
                    "top2_logits": [1.0, 0.0],
                    "margin": 1.0,
                },
                "first_logits_values": [1.0, 0.0],
            }
            for name, token_ids in outputs.items()
        ],
    }


def _consistency_result(passed: bool = True) -> dict[str, object]:
    pairs = {
        name: {"exact": passed}
        for name in (
            "cached_vs_uncached", "full_vs_chunked", "prefix_fresh_vs_reused",
            "single_vs_continuous_batch", "offline_vs_serving_backend",
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "native-consistency",
        "model": "/models/qwen3-vl",
        "checkpoint": {"path": "/models/qwen3-vl", "fingerprint_sha256": "a" * 64},
        "parameters": {
            "temperature": 0, "max_new_tokens": 64, "dtype": "bfloat16",
            "chat_template_marker": "qwen3-vl-chat-template-v1",
            "vision_attn_implementation": "sdpa",
            "language_attn_implementation": "sdpa",
        },
        "comparison": {
            "passed": passed,
            "pairs": pairs,
            "gates": {
                "same_image_repeated_outputs_exact": passed,
                "continuous_batch_rows_exact": passed,
            },
        },
    }


def _component_artifact(stage: str, components=None) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "model": "/models/qwen3-vl",
        "checkpoint": {"path": "/models/qwen3-vl", "fingerprint_sha256": "a" * 64},
        "parameters": {
            "temperature": 0, "dtype": "bfloat16",
            "chat_template_marker": "qwen3-vl-chat-template-v1",
            "vision_attn_implementation": "hf" if stage == "components-hf" else "sdpa",
        },
        "case_metadata": {
            "name": "square_image", "purpose": "test", "image_count": 1,
            "image_sizes": [[2, 2]], "details": ["auto"],
            "force_chunked_prefill": False, "repeat_runs": 1,
            "image_sha256": ["b" * 64],
        },
        "input_identity": {"input_token_ids": [1, 2], "image_sha256": ["b" * 64]},
        "components": components or {key: torch.zeros(2) for key in REQUIRED_COMPONENT_KEYS},
    }


class DeterministicCaseTests(unittest.TestCase):
    def test_external_paged_kv_bytes_count_only_native_owner_allocation(self) -> None:
        cache = torch.empty(2, 3, dtype=torch.float16)
        native = SimpleNamespace(
            core=SimpleNamespace(
                executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        runner=SimpleNamespace(
                            paged_kv_cache_owner=object(), kv_cache=cache
                        )
                    )
                )
            )
        )
        pytorch = SimpleNamespace(
            core=SimpleNamespace(
                executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        runner=SimpleNamespace(
                            paged_kv_cache_owner=None, kv_cache=cache
                        )
                    )
                )
            )
        )

        self.assertEqual(external_paged_kv_cache_bytes(native), 12)
        self.assertEqual(external_paged_kv_cache_bytes(pytorch), 0)

    def test_cli_accepts_explicit_paged_kv_backend_for_native_benchmarks(self) -> None:
        args = parse_args(
            ["--stage", "native-sdpa", "--paged-kv-backend", "cuda"]
        )

        self.assertEqual(args.paged_kv_backend, "cuda")

    def test_runtime_consistency_uses_reference_attention_backends(self) -> None:
        self.assertEqual(
            verification_module.consistency_attention_backends(),
            ("sdpa", "sdpa"),
        )

    def test_hf_flash_reference_keeps_text_on_sdpa(self) -> None:
        self.assertEqual(
            verification_module.hf_stage_attention_implementation("sdpa"),
            "sdpa",
        )
        self.assertEqual(
            verification_module.hf_stage_attention_implementation(
                "flash_attention_2"
            ),
            {
                "vision_config": "flash_attention_2",
                "text_config": "sdpa",
            },
        )

    def test_flash_parity_changes_only_the_vision_attention_backend(self) -> None:
        self.assertEqual(
            verification_module.native_stage_attention_backends("sdpa"),
            ("sdpa", "sdpa"),
        )
        self.assertEqual(
            verification_module.native_stage_attention_backends("flash_attention_2"),
            ("flash_attention_2", "sdpa"),
        )

    def test_fixed_corpus_has_ten_named_cases_and_expected_images(self) -> None:
        expected_names = [
            "english_text",
            "chinese_text",
            "square_image",
            "portrait_image",
            "landscape_image",
            "two_different_images",
            "repeated_image",
            "chinese_image_qa",
            "long_prompt_chunked",
            "same_image_cache_reuse",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            first = generate_test_cases(Path(temp_dir))
            second = generate_test_cases(Path(temp_dir))

            self.assertEqual([case.name for case in first], expected_names)
            self.assertEqual(len(first), 10)
            self.assertEqual(first[2].images[0].size, (256, 256))
            self.assertEqual(first[3].images[0].size, (192, 320))
            self.assertEqual(first[4].images[0].size, (320, 192))
            self.assertIs(first[6].images[0], first[6].images[1])
            self.assertIs(first[9].images[0], first[9].images[1])
            self.assertGreater(len(first[8].messages[-1]["content"][-1]["text"]), 4000)
            self.assertEqual(
                [image.tobytes() for case in first for image in case.images],
                [image.tobytes() for case in second for image in case.images],
            )

    def test_case_metadata_matches_placeholders_and_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for case in generate_test_cases(Path(temp_dir)):
                placeholders = sum(
                    part.get("type") == "image"
                    for message in case.messages
                    for part in (
                        message["content"]
                        if isinstance(message["content"], list)
                        else []
                    )
                )
                self.assertEqual(placeholders, len(case.images), case.name)
                self.assertEqual(len(case.details), len(case.images), case.name)


class GenerationComparisonTests(unittest.TestCase):
    def test_flash_uses_matching_hf_flash_reference_when_provided(self) -> None:
        hf = _result("hf", {"a": [10, 11, 12]})
        sdpa = _result("native-sdpa", {"a": [10, 11, 12]})
        hf_flash = _result("hf-flash", {"a": [10, 99, 12]})
        flash = _result("native-flash", {"a": [10, 99, 12]})

        report = compare_generation_results(
            hf,
            sdpa,
            flash,
            flash_reference=hf_flash,
            consistency=_consistency_result(),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["flash_reference_stage"], "hf-flash")
        self.assertTrue(report["flash"]["strict_token_match"])
        self.assertTrue(report["gates"]["flash_all_tokens_exact"])

    def test_identical_sdpa_and_flash_results_pass_all_gates(self) -> None:
        outputs = {"a": [10, 11, 12], "b": [20, 21]}
        report = compare_generation_results(
            _result("hf", outputs),
            _result("native-sdpa", outputs),
            _result("native-flash", outputs),
            consistency=_consistency_result(),
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["sdpa"]["strict_token_match"])
        self.assertEqual(report["flash"]["first_token_matches"], 2)
        self.assertEqual(report["flash"]["token_agreement_ratio"], 1.0)

    def test_sdpa_difference_reports_first_mismatch_and_fails_strict_gate(self) -> None:
        hf = _result("hf", {"a": [10, 11, 12]})
        sdpa = _result("native-sdpa", {"a": [10, 99, 12]})
        flash = _result("native-flash", {"a": [10, 11, 12]})

        report = compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

        self.assertFalse(report["passed"])
        self.assertEqual(report["sdpa"]["cases"][0]["first_difference_index"], 1)
        self.assertEqual(report["sdpa"]["cases"][0]["reference_token"], 11)
        self.assertEqual(report["sdpa"]["cases"][0]["candidate_token"], 99)

    def test_flash_first_token_difference_fails_flash_gate(self) -> None:
        hf = _result("hf", {"a": [10, 11, 12]})
        sdpa = _result("native-sdpa", {"a": [10, 11, 12]})
        flash = _result("native-flash", {"a": [99, 11, 12]})
        for result in (hf, flash):
            result["cases"][0]["logits_diagnostics"] = {
                "top2_token_ids": [10, 99],
                "top2_logits": [1.0, 0.9],
                "margin": 0.1,
            }

        report = compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["flash_all_first_tokens_exact"])

    def test_processor_input_id_difference_fails_sdpa_gate(self) -> None:
        hf = _result("hf", {"a": [10, 11]})
        sdpa = _result("native-sdpa", {"a": [10, 11]})
        flash = _result("native-flash", {"a": [10, 11]})
        sdpa["cases"][0]["input_token_ids"] = [1, 99]

        report = compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

        self.assertFalse(report["passed"])
        self.assertFalse(report["sdpa"]["cases"][0]["input_token_ids_exact"])

    def test_result_schema_is_validated_when_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hf.json"
            write_json(path, _result("hf", {"a": [1]}))
            self.assertEqual(load_generation_result(path)["stage"], "hf")

            path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "stage": "hf"}))
            with self.assertRaisesRegex(ValueError, "cases"):
                load_generation_result(path)

    def test_cli_comparison_rejects_an_incomplete_fixed_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            for stage, filename in (
                ("hf", "hf.json"),
                ("native-sdpa", "native-sdpa.json"),
                ("native-flash", "native-flash.json"),
            ):
                write_json(results_dir / filename, _result(stage, {"only_one": [1]}))

            with self.assertRaisesRegex(ValueError, "fixed 10-case corpus"):
                compare_result_files(results_dir)

    def test_cli_comparison_requires_native_consistency_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            cases = {name: [1] for name in (
                "english_text", "chinese_text", "square_image", "portrait_image",
                "landscape_image", "two_different_images", "repeated_image",
                "chinese_image_qa", "long_prompt_chunked", "same_image_cache_reuse",
            )}
            for stage, filename in (
                ("hf", "hf.json"), ("native-sdpa", "native-sdpa.json"),
                ("native-flash", "native-flash.json"),
            ):
                write_json(results_dir / filename, _result(stage, cases))
            with self.assertRaisesRegex(FileNotFoundError, "native-consistency"):
                compare_result_files(results_dir)

    def test_generation_role_is_strict(self) -> None:
        outputs = {"a": [1]}
        with self.assertRaisesRegex(ValueError, "stage roles"):
            compare_generation_results(
                _result("native-sdpa", outputs), _result("hf", outputs),
                _result("native-flash", outputs), consistency=_consistency_result(),
            )

    def test_same_tokens_with_severely_different_sdpa_logits_fails(self) -> None:
        outputs = {"a": [1]}
        hf = _result("hf", outputs)
        sdpa = _result("native-sdpa", outputs)
        flash = _result("native-flash", outputs)
        hf["cases"][0]["first_logits_values"] = [10.0, 0.0, -2.0]
        sdpa["cases"][0]["first_logits_values"] = [10.0, 9.9, 9.8]
        flash["cases"][0]["first_logits_values"] = [10.0, 0.0, -2.0]

        report = compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

        self.assertFalse(report["passed"])
        self.assertFalse(report["sdpa"]["logits_gate_passed"])

    def test_close_logits_pass_numeric_gates(self) -> None:
        outputs = {"a": [1]}
        hf = _result("hf", outputs)
        sdpa = _result("native-sdpa", outputs)
        flash = _result("native-flash", outputs)
        hf["cases"][0]["first_logits_values"] = [10.0, 1.0, -2.0]
        sdpa["cases"][0]["first_logits_values"] = [10.001, 1.001, -1.999]
        flash["cases"][0]["first_logits_values"] = [9.999, 1.0, -2.001]

        report = compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

        self.assertTrue(report["passed"])
        self.assertTrue(report["sdpa"]["logits_gate_passed"])
        self.assertTrue(report["flash"]["logits_gate_passed"])

    def test_first_sample_capture_ignores_unsampled_chunk_logits(self) -> None:
        class FakeSampler:
            def __call__(self, logits, temperatures, *, all_greedy):
                return logits.argmax(dim=-1)

        sampler = FakeSampler()
        chunk_a = torch.tensor([[9.0, 0.0]])
        sampled_chunk_b = torch.tensor([[0.0, 7.0]])
        with capture_first_sample_logits(sampler) as captured:
            _ = chunk_a  # run_model output from an incomplete chunk: sampler is not called.
            sampler(sampled_chunk_b, torch.tensor([0.0]), all_greedy=True)

        self.assertTrue(torch.equal(captured[0], sampled_chunk_b[0]))

    def test_selects_first_generated_logits_after_discarded_prefill_chunks(self) -> None:
        captured = [
            torch.tensor([9.0, 0.0]),
            torch.tensor([8.0, 1.0]),
            torch.tensor([0.0, 7.0]),
            torch.tensor([2.0, 6.0]),
        ]

        selected = select_first_generated_logits(captured, output_token_count=2)

        self.assertTrue(torch.equal(selected, captured[2]))

    def test_selects_initial_call_for_full_prefill(self) -> None:
        captured = [torch.tensor([0.0, 7.0]), torch.tensor([2.0, 6.0])]

        selected = select_first_generated_logits(captured, output_token_count=2)

        self.assertTrue(torch.equal(selected, captured[0]))


class ComponentComparisonTests(unittest.TestCase):
    def test_first_logits_requires_same_argmax_even_when_tensors_are_close(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_path = Path(temp_dir) / "hf.pt"
            native_path = Path(temp_dir) / "native.pt"
            hf = _component_artifact("components-hf")
            native = _component_artifact("components-native")
            hf["components"]["first_logits"] = torch.tensor([1.0, 1.001])
            native["components"]["first_logits"] = torch.tensor([1.002, 1.0])
            torch.save(hf, hf_path)
            torch.save(native, native_path)

            report = compare_component_artifacts(hf_path, native_path)

            self.assertFalse(report["passed"])
            logits = report["components"][0]
            self.assertFalse(logits["argmax_match"])
            self.assertEqual(logits["reference_argmax"], 1)
            self.assertEqual(logits["candidate_argmax"], 0)

    def test_processor_integer_tensors_require_exact_equality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_path = Path(temp_dir) / "hf.pt"
            native_path = Path(temp_dir) / "native.pt"
            hf = _component_artifact("components-hf")
            native = _component_artifact("components-native")
            hf["components"]["processor.input_ids"] = torch.tensor([1, 2, 3])
            native["components"]["processor.input_ids"] = torch.tensor([1, 2, 4])
            torch.save(hf, hf_path)
            torch.save(native, native_path)

            report = compare_component_artifacts(hf_path, native_path)

            self.assertFalse(report["passed"])
            processor_entry = next(
                entry for entry in report["components"]
                if entry["name"] == "processor.input_ids"
            )
            self.assertFalse(processor_entry["exact"])

    def test_component_artifact_requires_every_parity_tensor(self) -> None:
        complete = _component_artifact("components-hf")
        validate_component_artifact(complete)
        complete["components"].pop("vision.deepstack.2")
        with self.assertRaisesRegex(ValueError, "vision.deepstack.2"):
            validate_component_artifact(complete)

    def test_component_schema_rejects_unknown_and_non_tensor_values(self) -> None:
        artifact = _component_artifact("components-hf")
        artifact["components"]["unknown"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_component_artifact(artifact, expected_stage="components-hf")
        artifact = _component_artifact("components-hf")
        artifact["components"]["vision.final"] = [1.0]
        with self.assertRaisesRegex(TypeError, "vision.final"):
            validate_component_artifact(artifact, expected_stage="components-hf")

    def test_component_identity_rejects_extra_field_and_bad_sha(self) -> None:
        artifact = _component_artifact("components-hf")
        artifact["input_identity"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "input identity"):
            validate_component_artifact(artifact, expected_stage="components-hf")
        artifact = _component_artifact("components-hf")
        artifact["input_identity"]["image_sha256"] = ["not-a-sha"]
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            validate_component_artifact(artifact, expected_stage="components-hf")

    def test_component_case_metadata_rejects_extra_field_and_bad_type(self) -> None:
        artifact = _component_artifact("components-hf")
        artifact["case_metadata"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "case metadata"):
            validate_component_artifact(artifact, expected_stage="components-hf")
        artifact = _component_artifact("components-hf")
        artifact["case_metadata"]["image_count"] = "1"
        with self.assertRaisesRegex(ValueError, "image_count"):
            validate_component_artifact(artifact, expected_stage="components-hf")

    def test_component_artifact_rejects_wrong_stage_or_input_identity(self) -> None:
        hf = _component_artifact("components-native")
        with self.assertRaisesRegex(ValueError, "stage"):
            validate_component_artifact(hf, expected_stage="components-hf")

        native = _component_artifact("components-native")
        native["input_identity"]["input_token_ids"] = [1, 9]
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_path = Path(temp_dir) / "hf.pt"
            native_path = Path(temp_dir) / "native.pt"
            torch.save(_component_artifact("components-hf"), hf_path)
            torch.save(native, native_path)
            with self.assertRaisesRegex(ValueError, "input identity"):
                compare_component_artifacts(hf_path, native_path)

    def test_native_first_logits_uses_last_packed_prompt_row(self) -> None:
        class FakeModel:
            def __call__(self, *_args, **_kwargs):
                return torch.tensor([[1.0], [2.0], [9.0]])

            def compute_logits(self, hidden):
                return torch.cat((hidden, -hidden), dim=-1)

        logits = compute_native_component_first_logits(
            FakeModel(), None, torch.zeros((3, 3), dtype=torch.long),
            inputs_embeds=torch.zeros((3, 1)), visual_token_rows=None,
            deepstack_visual_embeds=(),
        )

        self.assertTrue(torch.equal(logits, torch.tensor([9.0, -9.0])))


class CliAndBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_and_stages(self) -> None:
        args = parse_args(["--stage", "native-sdpa"])
        self.assertEqual(args.model, "/path/to/Qwen3-VL-2B-Instruct")
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.stage, "native-sdpa")

        component_args = parse_args([
            "--stage", "compare-components",
            "--component-hf", "hf.pt",
            "--component-native", "native.pt",
        ])
        self.assertEqual(component_args.component_cosine_threshold, 0.9999)
        self.assertEqual(parse_args(["--stage", "components-hf"]).stage, "components-hf")
        self.assertEqual(parse_args(["--stage", "components-native"]).stage, "components-native")
        self.assertEqual(parse_args(["--stage", "benchmark-matrix"]).stage, "benchmark-matrix")
        self.assertEqual(parse_args(["--stage", "native-consistency"]).stage, "native-consistency")

    def test_benchmark_matrix_covers_required_dimensions(self) -> None:
        matrix = benchmark_matrix()
        self.assertEqual(len(matrix), 48)
        self.assertEqual({item["backend"] for item in matrix}, {"sdpa", "flash_attention_2"})
        self.assertEqual({item["batch_size"] for item in matrix}, {1, 4, 8})
        self.assertEqual({item["image_count"] for item in matrix}, {1, 2})
        self.assertEqual({item["cache_state"] for item in matrix}, {"cold", "warm"})
        self.assertEqual({item["prefill"] for item in matrix}, {"full", "chunked"})

    def test_executable_benchmark_invokes_all_48_cells_and_records_measurements(self) -> None:
        seen = []

        def fake_runner(cell, _args):
            seen.append(dict(cell))
            return {
                "total_latency_sec": 2.0,
                "output_tokens": 4,
                "output_token_throughput_per_sec": 2.0,
                "peak_gpu_memory_bytes": 100,
                "vision_encoding_latency_sec": None,
                "vision_stats": {"cache_hits": 1},
            }

        report = execute_benchmark_matrix(parse_args(["--stage", "benchmark-matrix"]), runner=fake_runner)

        self.assertEqual(len(seen), 48)
        self.assertEqual(len(report["cells"]), 48)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["parameters"]["language_attn_implementation"],
            "sdpa",
        )
        self.assertTrue(all(cell["measurement"]["total_latency_sec"] == 2.0 for cell in report["cells"]))


class ArtifactMetadataTests(unittest.TestCase):
    def test_header_has_absolute_checkpoint_fingerprint_and_parity_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
            (model / "model-1.safetensors").write_bytes(b"weights")
            args = parse_args(["--stage", "hf", "--model", str(model)])

            header = build_artifact_header(args, "hf")

            self.assertEqual(header["checkpoint"]["path"], str(model.resolve()))
            self.assertEqual(len(header["checkpoint"]["fingerprint_sha256"]), 64)
            self.assertEqual(header["parameters"]["temperature"], 0)
            self.assertEqual(header["parameters"]["max_new_tokens"], 64)
            self.assertEqual(header["parameters"]["dtype"], "bfloat16")
            self.assertTrue(header["parameters"]["chat_template_marker"])

            consistency_header = build_artifact_header(args, "native-consistency")
            self.assertEqual(
                consistency_header["parameters"]["language_attn_implementation"],
                "sdpa",
            )

    def test_compare_rejects_different_checkpoint_identity(self) -> None:
        outputs = {"a": [1]}
        hf = _result("hf", outputs)
        sdpa = _result("native-sdpa", outputs)
        flash = _result("native-flash", outputs)
        for result in (hf, sdpa, flash):
            result["model"] = "/m"
            result["checkpoint"] = {"path": "/m", "fingerprint_sha256": "a" * 64}
            result["parameters"].update(dtype="bfloat16", chat_template_marker="qwen3-vl-chat-template-v1")
        flash["checkpoint"]["fingerprint_sha256"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

    def test_full_checkpoint_fingerprint_changes_when_middle_bytes_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir)
            (model / "config.json").write_text("{}")
            weights = model / "model.safetensors"
            weights.write_bytes(b"a" * (3 * 1024 * 1024))
            before = checkpoint_identity(str(model))["fingerprint_sha256"]
            with weights.open("r+b") as handle:
                handle.seek(1536 * 1024)
                handle.write(b"z")
            after = checkpoint_identity(str(model))["fingerprint_sha256"]
            self.assertNotEqual(before, after)

    def test_checkpoint_manifest_includes_processor_template_vocab_and_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"weights")
            for relative in (
                "preprocessor_config.json", "chat_template.jinja", "vocab.json",
                "nested/extra.json",
            ):
                path = model / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("first")
                before = checkpoint_identity(str(model))["fingerprint_sha256"]
                path.write_text("second")
                after = checkpoint_identity(str(model))["fingerprint_sha256"]
                self.assertNotEqual(before, after, relative)
                path.write_text("first")

    def test_generation_schema_rejects_unknown_root_and_case_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hf.json"
            payload = _result("hf", {"a": [1]})
            payload["unexpected"] = True
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_generation_result(path)
            payload = _result("hf", {"a": [1]})
            payload["cases"][0]["unexpected"] = True
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_generation_result(path)

    def test_component_schema_rejects_extra_envelope_field(self) -> None:
        payload = _component_artifact("components-hf")
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_component_artifact(payload, expected_stage="components-hf")


class DiagnosticsAndConsistencyTests(unittest.TestCase):
    def test_flash_mismatch_without_logits_diagnostics_is_rejected_by_schema(self) -> None:
        hf = _result("hf", {"a": [10, 11]})
        sdpa = _result("native-sdpa", {"a": [10, 11]})
        flash = _result("native-flash", {"a": [10, 99]})
        for result in (hf, flash):
            result["cases"][0].pop("logits_diagnostics")
            result["cases"][0].pop("first_logits_values")

        with self.assertRaisesRegex(ValueError, "logits_diagnostics"):
            compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

    def test_repeated_output_inconsistency_fails_generation_gate(self) -> None:
        hf = _result("hf", {"a": [10]})
        sdpa = _result("native-sdpa", {"a": [10]})
        flash = _result("native-flash", {"a": [10]})
        sdpa["cases"][0]["runtime_consistency"] = {"repeated_outputs_exact": False}

        report = compare_generation_results(hf, sdpa, flash, consistency=_consistency_result())

        self.assertFalse(report["passed"])
        self.assertFalse(report["sdpa"]["cases"][0]["runtime_consistency_exact"])

    def test_generation_compare_requires_passing_consistency_artifact(self) -> None:
        outputs = {"a": [1]}
        with self.assertRaisesRegex(ValueError, "consistency"):
            compare_generation_results(
                _result("hf", outputs), _result("native-sdpa", outputs),
                _result("native-flash", outputs),
            )
        report = compare_generation_results(
            _result("hf", outputs), _result("native-sdpa", outputs),
            _result("native-flash", outputs), consistency=_consistency_result(False),
        )
        self.assertFalse(report["passed"])

    def test_generation_compare_rejects_flash_language_consistency_artifact(self) -> None:
        outputs = {"a": [1]}
        consistency = _consistency_result()
        consistency["parameters"]["language_attn_implementation"] = "flash_attention_2"

        with self.assertRaisesRegex(ValueError, "language_attn_implementation"):
            compare_generation_results(
                _result("hf", outputs), _result("native-sdpa", outputs),
                _result("native-flash", outputs), consistency=consistency,
            )

    def test_generation_schema_rejects_empty_or_malformed_diagnostics(self) -> None:
        payload = _result("hf", {"a": [1]})
        payload["cases"][0]["logits_diagnostics"] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hf.json"
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "logits_diagnostics"):
                load_generation_result(path)

    def test_runtime_consistency_difference_fails_main_gate(self) -> None:
        variants = {
            "uncached": [1, 2],
            "cached": [1, 2],
            "full_prefill": [1, 2],
            "chunked_prefill": [1, 9],
            "prefix_fresh": [1, 2],
            "prefix_reused": [1, 2],
            "single": [1, 2],
            "continuous_batch": [1, 2],
            "offline": [1, 2],
            "serving_backend": [1, 2],
        }

        report = compare_consistency_results(variants)

        self.assertFalse(report["passed"])
        self.assertFalse(report["pairs"]["full_vs_chunked"]["exact"])

    def test_readme_documents_qwen3_commands_and_limits(self) -> None:
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        for required in (
            "transformers==4.57.6",
            "Qwen/Qwen3-VL-2B-Instruct",
            "verify_qwen3_vl_native.py",
            "--stage native-sdpa",
            "--stage native-flash",
            "--stage compare",
            "NANOINFER_MM_ATTN_IMPL",
            "Qwen2.5-VL 回归测试",
        ):
            self.assertIn(required, readme)


if __name__ == "__main__":
    unittest.main()
