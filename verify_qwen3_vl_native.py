from __future__ import annotations

"""Reproducible, process-separable Qwen3-VL correctness and performance gates.

The generation stages write CPU JSON artifacts.  ``compare`` only reads those
artifacts, so the Hugging Face and NanoInfer models never need to share GPU
memory.  Component tensors can likewise be captured independently and checked
with :func:`compare_component_artifacts`; the script never invents unavailable
component values.
"""

import argparse
import gc
import hashlib
import itertools
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw


DEFAULT_MODEL = "/path/to/Qwen3-VL-2B-Instruct"
SCHEMA_VERSION = 1
FIXED_CASE_NAMES = (
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
)
STAGES = (
    "hf",
    "hf-flash",
    "native-sdpa",
    "native-flash",
    "components-hf",
    "components-native",
    "benchmark-matrix",
    "native-consistency",
    "compare",
    "compare-components",
    "all",
)
CHAT_TEMPLATE_MARKER = "qwen3-vl-chat-template-v1"
REQUIRED_COMPONENT_KEYS = (
    "processor.input_ids",
    "processor.pixel_values",
    "processor.image_grid_thw",
    "processor.position_ids",
    "vision.final",
    "vision.deepstack.0",
    "vision.deepstack.1",
    "vision.deepstack.2",
    "first_logits",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
GENERATION_ROOT_FIELDS = {
    "schema_version", "stage", "model", "checkpoint", "parameters", "cases",
    "benchmark_matrix",
}
GENERATION_CASE_FIELDS = {
    "name", "metadata", "input_token_ids", "output_token_ids", "text",
    "logits_diagnostics", "first_logits_values", "first_logits_sidecar",
    "timing", "runtime_consistency", "benchmark",
}


@dataclass(frozen=True)
class VerificationCase:
    name: str
    messages: list[dict[str, object]]
    images: tuple[Image.Image, ...]
    details: tuple[str, ...]
    purpose: str
    force_chunked_prefill: bool = False
    repeat_runs: int = 1

    def metadata(self) -> dict[str, object]:
        return {
            "purpose": self.purpose,
            "image_count": len(self.images),
            "image_sizes": [list(image.size) for image in self.images],
            "details": list(self.details),
            "force_chunked_prefill": self.force_chunked_prefill,
            "repeat_runs": self.repeat_runs,
            "image_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in self.images],
        }


def _gradient_image(size: tuple[int, int], seed: int, label: str) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 3 + y + seed * 17) % 256,
                (x + y * 5 + seed * 29) % 256,
                (x * 7 + y * 2 + seed * 11) % 256,
            )
    draw = ImageDraw.Draw(image)
    margin = max(8, min(size) // 12)
    draw.rectangle(
        (margin, margin, width - margin - 1, height - margin - 1),
        outline=(255, 255, 255),
        width=max(2, min(size) // 64),
    )
    draw.ellipse(
        (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
        outline=(0, 0, 0),
        width=max(2, min(size) // 80),
    )
    draw.text((margin + 4, margin + 4), label, fill=(255, 240, 0))
    return image


def _content(text: str, image_count: int = 0, *, interleaved: bool = False) -> list[dict[str, object]]:
    if image_count == 0:
        return [{"type": "text", "text": text}]
    if not interleaved:
        return [
            *({"type": "image"} for _ in range(image_count)),
            {"type": "text", "text": text},
        ]
    parts: list[dict[str, object]] = []
    for index in range(image_count):
        parts.append({"type": "image"})
        parts.append({"type": "text", "text": f"Image {index + 1}. "})
    parts.append({"type": "text", "text": text})
    return parts


def _messages(content: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"role": "user", "content": content}]


def generate_test_cases(output_dir: Path | None = None) -> list[VerificationCase]:
    """Build the fixed ten-case corpus without downloading external data."""
    square = _gradient_image((256, 256), 1, "SQUARE")
    portrait = _gradient_image((192, 320), 2, "PORTRAIT")
    landscape = _gradient_image((320, 192), 3, "LANDSCAPE")
    alternate = _gradient_image((256, 256), 4, "SECOND")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, image in (
            ("square.png", square),
            ("portrait.png", portrait),
            ("landscape.png", landscape),
            ("alternate.png", alternate),
        ):
            image.save(output_dir / name, format="PNG", compress_level=9)

    long_text = " ".join(
        f"Chunk {index}: inspect the supplied image carefully and preserve every stated constraint."
        for index in range(96)
    )
    return [
        VerificationCase(
            "english_text", _messages(_content("Explain why the sky appears blue in two sentences.")), (), (),
            "English text-only greedy generation",
        ),
        VerificationCase(
            "chinese_text", _messages(_content("用中文简要解释什么是连续批处理。")), (), (),
            "Chinese text-only greedy generation",
        ),
        VerificationCase(
            "square_image", _messages(_content("Describe the geometry and colors.", 1)), (square,), ("auto",),
            "Single square image",
        ),
        VerificationCase(
            "portrait_image", _messages(_content("Describe this portrait-oriented image.", 1)), (portrait,), ("auto",),
            "Single portrait image",
        ),
        VerificationCase(
            "landscape_image", _messages(_content("Describe this landscape-oriented image.", 1)), (landscape,), ("auto",),
            "Single landscape image",
        ),
        VerificationCase(
            "two_different_images",
            _messages(_content("Compare their shapes and dominant colors.", 2, interleaved=True)),
            (square, alternate), ("auto", "auto"), "Two different interleaved images",
        ),
        VerificationCase(
            "repeated_image", _messages(_content("Are these two images identical?", 2, interleaved=True)),
            (square, square), ("auto", "auto"), "Repeated image identity within one request",
        ),
        VerificationCase(
            "chinese_image_qa", _messages(_content("请用中文描述图像中的形状、方向和主要颜色。", 1)),
            (portrait,), ("auto",), "Chinese image question answering",
        ),
        VerificationCase(
            "long_prompt_chunked", _messages(_content(long_text, 1)), (landscape,), ("auto",),
            "Long prompt that forces chunked prefill", force_chunked_prefill=True,
        ),
        VerificationCase(
            "same_image_cache_reuse", _messages(_content("Verify whether both copies show the same content.", 2)),
            (alternate, alternate), ("auto", "auto"),
            "Repeated execution checks feature-cache and prefix-cache reuse", repeat_runs=2,
        ),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify native Qwen3-VL without co-resident reference models")
    parser.add_argument("--stage", choices=STAGES, default="compare")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=Path("verification-results/qwen3-vl"))
    parser.add_argument("--output-dir", type=Path, help="Optional persistent directory for generated PNG inputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--full-prefill-tokens", type=int, default=8192)
    parser.add_argument("--chunked-prefill-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vision-feature-cache-bytes", type=int, default=2 * 1024**3)
    parser.add_argument(
        "--paged-kv-backend",
        choices=("auto", "pytorch", "triton", "cuda"),
        default="auto",
    )
    parser.add_argument("--benchmark", action="store_true", help="Record available native latency/cache/memory metrics")
    parser.add_argument("--benchmark-matrix", action="store_true", help="Print the required 48-cell performance matrix")
    parser.add_argument("--component-hf", type=Path, help="CPU .pt artifact from an HF component capture")
    parser.add_argument("--component-native", type=Path, help="CPU .pt artifact from a native component capture")
    parser.add_argument("--component-rtol", type=float, default=2e-2)
    parser.add_argument("--component-atol", type=float, default=2e-2)
    parser.add_argument("--component-cosine-threshold", type=float, default=0.9999)
    args = parser.parse_args(argv)
    if args.max_new_tokens != 64:
        parser.error("parity verification requires --max-new-tokens 64")
    if args.temperature != 0:
        parser.error("parity verification requires --temperature 0")
    if args.stage == "compare-components" and (args.component_hf is None or args.component_native is None):
        parser.error("compare-components requires --component-hf and --component-native")
    return args


def benchmark_matrix() -> list[dict[str, object]]:
    return [
        {
            "backend": backend,
            "batch_size": batch_size,
            "image_count": image_count,
            "cache_state": cache_state,
            "prefill": prefill,
        }
        for backend, batch_size, image_count, cache_state, prefill in itertools.product(
            ("sdpa", "flash_attention_2"),
            (1, 4, 8),
            (1, 2),
            ("cold", "warm"),
            ("full", "chunked"),
        )
    ]


def checkpoint_identity(model_path: str) -> dict[str, str]:
    path = Path(model_path).expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing config.json: {path}")
    digest = hashlib.sha256()
    excluded_directories = {".cache", "__pycache__", "verification-results", "benchmark-results"}
    files = []
    for file in path.rglob("*"):
        relative = file.relative_to(path)
        if any(part in excluded_directories for part in relative.parts):
            continue
        if file.is_symlink():
            raise ValueError(f"checkpoint manifest does not allow symlinks: {relative.as_posix()}")
        if file.is_file() and not file.name.endswith((".lock", ".tmp")):
            files.append(file)
    files.sort(key=lambda file: file.relative_to(path).as_posix())
    if not any(file.suffix == ".safetensors" for file in files):
        raise FileNotFoundError(f"checkpoint is missing safetensors weights: {path}")
    for file in files:
        stat = file.stat()
        name = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(stat.st_size.to_bytes(8, "little"))
        with file.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return {"path": str(path), "fingerprint_sha256": digest.hexdigest()}


def consistency_attention_backends() -> tuple[str, str]:
    # Runtime-consistency gates are reference checks.  Keeping both attention
    # implementations on SDPA makes cache/prefix/chunking comparisons isolate
    # runtime state changes instead of BF16 differences between Flash kernels
    # selected for different prefill shapes.
    return "sdpa", "sdpa"


def build_artifact_header(args: argparse.Namespace, stage: str) -> dict[str, object]:
    backend = {
        "hf": "hf",
        "hf-flash": "flash_attention_2",
        "native-sdpa": "sdpa",
        "native-flash": "flash_attention_2",
        "components-hf": "hf",
        "components-native": "sdpa",
        "native-consistency": "sdpa",
    }.get(stage)
    parameters = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "dtype": "bfloat16",
        "chat_template_marker": CHAT_TEMPLATE_MARKER,
        "vision_attn_implementation": backend,
    }
    if stage == "native-consistency":
        _, language_backend = consistency_attention_backends()
        parameters["language_attn_implementation"] = language_backend
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "model": str(Path(args.model).expanduser().resolve()),
        "checkpoint": checkpoint_identity(args.model),
        "parameters": parameters,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_checkpoint_identity(payload: dict[str, object], *, location: str = "") -> None:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "fingerprint_sha256"}:
        raise ValueError(f"artifact checkpoint identity{location} is invalid")
    checkpoint_path = checkpoint["path"]
    model_path = payload.get("model")
    if (
        not isinstance(checkpoint_path, str)
        or not Path(checkpoint_path).is_absolute()
        or model_path != checkpoint_path
    ):
        raise ValueError(f"artifact model/checkpoint path identity{location} is invalid")
    fingerprint = checkpoint["fingerprint_sha256"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or not set(fingerprint.lower()) <= _HEX_DIGITS
    ):
        raise ValueError(f"artifact checkpoint fingerprint{location} must be 64 hexadecimal characters")


def _validate_common_parameters(
    parameters: object,
    *,
    include_max_tokens: bool,
    expected_backend: str,
    location: str = "",
) -> None:
    if not isinstance(parameters, dict):
        raise ValueError(f"artifact parameters{location} must be an object")
    expected = {
        "temperature": 0,
        "dtype": "bfloat16",
        "chat_template_marker": CHAT_TEMPLATE_MARKER,
    }
    if include_max_tokens:
        expected["max_new_tokens"] = 64
    expected["vision_attn_implementation"] = expected_backend
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f"artifact parameter {key}{location} must be {value!r}")


def _validate_logits_case(case: dict[str, object], *, location: str) -> None:
    diagnostics = case.get("logits_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
        "top2_token_ids", "top2_logits", "margin"
    }:
        raise ValueError(f"case {location} logits_diagnostics schema is invalid")
    top2_ids = diagnostics["top2_token_ids"]
    top2_logits = diagnostics["top2_logits"]
    if not isinstance(top2_ids, list) or len(top2_ids) != 2 or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in top2_ids
    ):
        raise ValueError(f"case {location} logits_diagnostics top2_token_ids is invalid")
    if not isinstance(top2_logits, list) or len(top2_logits) != 2 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in top2_logits
    ):
        raise ValueError(f"case {location} logits_diagnostics top2_logits is invalid")
    if not isinstance(diagnostics["margin"], (int, float)) or isinstance(diagnostics["margin"], bool):
        raise ValueError(f"case {location} logits_diagnostics margin is invalid")
    values = case.get("first_logits_values")
    if not isinstance(values, list) or len(values) <= 1 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
    ):
        raise ValueError(f"case {location} first_logits_values is invalid")
    sidecar = case.get("first_logits_sidecar")
    if sidecar is not None and (not isinstance(sidecar, str) or not sidecar):
        raise ValueError(f"case {location} first_logits_sidecar is invalid")


def _validate_generation_result(payload: object, path: Path | None = None) -> dict[str, object]:
    location = f" in {path}" if path else ""
    if not isinstance(payload, dict):
        raise ValueError(f"generation result{location} must be a JSON object")
    required = {"schema_version", "stage", "model", "checkpoint", "parameters", "cases"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"generation result{location} is missing: {', '.join(missing)}")
    unexpected_root = sorted(set(payload) - GENERATION_ROOT_FIELDS)
    if unexpected_root:
        raise ValueError(f"generation result{location} has unexpected fields: {unexpected_root}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported generation result schema{location}: {payload['schema_version']}")
    if not isinstance(payload["cases"], list):
        raise ValueError(f"generation result cases{location} must be a list")
    _validate_checkpoint_identity(payload, location=location)
    expected_backend = {
        "hf": "hf", "hf-flash": "flash_attention_2",
        "native-sdpa": "sdpa", "native-flash": "flash_attention_2",
    }.get(payload["stage"])
    if expected_backend is None:
        raise ValueError(f"generation result stage{location} is invalid")
    _validate_common_parameters(
        payload["parameters"], include_max_tokens=True,
        expected_backend=expected_backend, location=location,
    )
    seen: set[str] = set()
    for index, case in enumerate(payload["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"case {index}{location} must be an object")
        case_required = {"name", "metadata", "input_token_ids", "output_token_ids", "text"}
        case_missing = sorted(case_required - case.keys())
        if case_missing:
            raise ValueError(f"case {index}{location} is missing: {', '.join(case_missing)}")
        unexpected_case = sorted(set(case) - GENERATION_CASE_FIELDS)
        if unexpected_case:
            raise ValueError(f"case {index}{location} has unexpected fields: {unexpected_case}")
        if case["name"] in seen:
            raise ValueError(f"duplicate case name{location}: {case['name']}")
        seen.add(case["name"])
        for token_key in ("input_token_ids", "output_token_ids"):
            if not isinstance(case[token_key], list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in case[token_key]
            ):
                raise ValueError(f"case {case['name']}{location} contains invalid {token_key}")
        if not isinstance(case["metadata"], dict):
            raise ValueError(f"case {case['name']}{location} metadata must be an object")
        _validate_logits_case(case, location=f"{case['name']}{location}")
    return payload


def load_generation_result(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required generation artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"generation artifact is invalid JSON: {path}") from error
    return _validate_generation_result(payload, path)


def _case_map(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {case["name"]: case for case in result["cases"]}


def _compare_tokens(reference: list[int], candidate: list[int]) -> dict[str, object]:
    first_difference = next(
        (index for index in range(min(len(reference), len(candidate))) if reference[index] != candidate[index]),
        None,
    )
    if first_difference is None and len(reference) != len(candidate):
        first_difference = min(len(reference), len(candidate))
    denominator = max(len(reference), len(candidate))
    matching = sum(left == right for left, right in zip(reference, candidate))
    return {
        "exact": reference == candidate,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "first_token_match": bool(reference and candidate and reference[0] == candidate[0]),
        "matching_token_count": matching,
        "token_agreement_ratio": matching / denominator if denominator else 1.0,
        "first_difference_index": first_difference,
        "reference_token": reference[first_difference] if first_difference is not None and first_difference < len(reference) else None,
        "candidate_token": candidate[first_difference] if first_difference is not None and first_difference < len(candidate) else None,
    }


def _compare_stage(reference: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    import torch
    import torch.nn.functional as F
    reference_cases = _case_map(reference)
    candidate_cases = _case_map(candidate)
    if reference_cases.keys() != candidate_cases.keys():
        missing = sorted(reference_cases.keys() - candidate_cases.keys())
        unexpected = sorted(candidate_cases.keys() - reference_cases.keys())
        raise ValueError(f"case set mismatch: missing={missing}, unexpected={unexpected}")
    cases = []
    for name, reference_case in reference_cases.items():
        candidate_case = candidate_cases[name]
        comparison = {
            "name": name,
            **_compare_tokens(reference_case["output_token_ids"], candidate_case["output_token_ids"]),
        }
        comparison["input_token_ids_exact"] = (
            reference_case["input_token_ids"] == candidate_case["input_token_ids"]
        )
        comparison["metadata_exact"] = reference_case["metadata"] == candidate_case["metadata"]
        runtime_consistency = candidate_case.get("runtime_consistency", {})
        comparison["runtime_consistency_exact"] = runtime_consistency.get(
            "repeated_outputs_exact", True
        ) is True
        reference_diagnostics = reference_case.get("logits_diagnostics")
        candidate_diagnostics = candidate_case.get("logits_diagnostics")
        diagnostics_missing = reference_diagnostics is None or candidate_diagnostics is None
        comparison["diagnostics_missing"] = diagnostics_missing
        if reference_diagnostics is not None:
            comparison["reference_logits_diagnostics"] = reference_diagnostics
        if candidate_diagnostics is not None:
            comparison["candidate_logits_diagnostics"] = candidate_diagnostics
        reference_logits_values = reference_case.get("first_logits_values")
        candidate_logits_values = candidate_case.get("first_logits_values")
        if reference_logits_values is not None and candidate_logits_values is not None:
            reference_logits = torch.tensor(reference_logits_values, dtype=torch.float32)
            candidate_logits = torch.tensor(candidate_logits_values, dtype=torch.float32)
            if reference_logits.shape != candidate_logits.shape:
                raise ValueError(f"first-token logits shape mismatch for {name}")
            difference = (reference_logits - candidate_logits).abs()
            comparison.update({
                "first_logits_max_abs": difference.max().item(),
                "first_logits_allclose": torch.allclose(
                    reference_logits, candidate_logits, rtol=2e-2, atol=2e-2,
                ),
                "first_logits_cosine": F.cosine_similarity(
                    reference_logits, candidate_logits, dim=0,
                ).item(),
                "first_logits_argmax_match": (
                    reference_logits.argmax().item() == candidate_logits.argmax().item()
                ),
            })
        comparison["logits_gate_passed"] = (
            not diagnostics_missing
            and comparison.get("first_logits_cosine", -1.0) >= 0.9999
            and comparison.get("first_logits_argmax_match") is True
            and (
                comparison.get("first_logits_allclose") is True
                if candidate["stage"] == "native-sdpa"
                else True
            )
        )
        cases.append(comparison)
    total_matching = sum(case["matching_token_count"] for case in cases)
    total_tokens = sum(max(case["reference_length"], case["candidate_length"]) for case in cases)
    return {
        "candidate_stage": candidate["stage"],
        "case_count": len(cases),
        "strict_token_match": all(
            case["exact"]
            and case["input_token_ids_exact"]
            and case["metadata_exact"]
            and case["runtime_consistency_exact"]
            for case in cases
        ),
        "first_token_matches": sum(case["first_token_match"] for case in cases),
        "token_agreement_ratio": total_matching / total_tokens if total_tokens else 1.0,
        "mismatches_have_diagnostics": all(not case["diagnostics_missing"] for case in cases),
        "logits_gate_passed": all(case["logits_gate_passed"] for case in cases),
        "cases": cases,
    }


def compare_generation_results(
    hf: dict[str, object],
    native_sdpa: dict[str, object],
    native_flash: dict[str, object],
    *,
    flash_reference: dict[str, object] | None = None,
    consistency: dict[str, object] | None = None,
) -> dict[str, object]:
    if consistency is None:
        raise ValueError("a native consistency artifact is required")
    for result in (hf, native_sdpa, native_flash):
        _validate_generation_result(result)
    if flash_reference is not None:
        _validate_generation_result(flash_reference)
    expected_stages = ("hf", "native-sdpa", "native-flash")
    actual_stages = tuple(result["stage"] for result in (hf, native_sdpa, native_flash))
    if actual_stages != expected_stages:
        raise ValueError(f"generation artifact stage roles are invalid: {actual_stages}")
    if flash_reference is not None and flash_reference["stage"] != "hf-flash":
        raise ValueError(
            "Flash reference artifact stage role is invalid: "
            f"{flash_reference['stage']}"
        )
    consistency_passed = _validate_consistency_artifact(consistency, hf)
    common_parameter_keys = ("temperature", "max_new_tokens", "dtype", "chat_template_marker")
    reference_parameters = {key: hf["parameters"][key] for key in common_parameter_keys}
    for result in (native_sdpa, native_flash, flash_reference):
        if result is None:
            continue
        if result["checkpoint"] != hf["checkpoint"]:
            raise ValueError("generation artifacts use different checkpoint identity")
        if {key: result["parameters"][key] for key in common_parameter_keys} != reference_parameters:
            raise ValueError("generation artifacts use different parity parameters")
    sdpa = _compare_stage(hf, native_sdpa)
    effective_flash_reference = flash_reference or hf
    flash = _compare_stage(effective_flash_reference, native_flash)
    flash_first_tokens_exact = flash["first_token_matches"] == flash["case_count"]
    flash_all_tokens_exact = flash["strict_token_match"]
    flash_diagnostics_complete = flash["mismatches_have_diagnostics"]
    # SDPA is the strict full-sequence gate. Flash must preserve every first
    # token; later BF16 boundary differences remain visible in the report.
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_stage": hf["stage"],
        "flash_reference_stage": effective_flash_reference["stage"],
        "passed": (
            sdpa["strict_token_match"]
            and flash_first_tokens_exact
            and (flash_all_tokens_exact if flash_reference is not None else True)
            and flash_diagnostics_complete
            and sdpa["logits_gate_passed"]
            and flash["logits_gate_passed"]
            and consistency_passed
        ),
        "gates": {
            "sdpa_all_tokens_exact": sdpa["strict_token_match"],
            "flash_all_tokens_exact": flash_all_tokens_exact,
            "flash_all_first_tokens_exact": flash_first_tokens_exact,
            "flash_mismatches_have_diagnostics": flash_diagnostics_complete,
            "runtime_consistency": consistency_passed,
            "sdpa_first_logits": sdpa["logits_gate_passed"],
            "flash_first_logits": flash["logits_gate_passed"],
        },
        "sdpa": sdpa,
        "flash": flash,
    }


def compare_result_files(results_dir: Path) -> dict[str, object]:
    results = (
        load_generation_result(results_dir / "hf.json"),
        load_generation_result(results_dir / "native-sdpa.json"),
        load_generation_result(results_dir / "native-flash.json"),
    )
    flash_reference_path = results_dir / "hf-flash.json"
    flash_reference = (
        load_generation_result(flash_reference_path)
        if flash_reference_path.is_file()
        else None
    )
    expected = set(FIXED_CASE_NAMES)
    for result in (*results, flash_reference):
        if result is None:
            continue
        actual = set(_case_map(result))
        if actual != expected:
            raise ValueError(
                f"{result['stage']} does not contain the fixed 10-case corpus: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
    consistency_path = results_dir / "native-consistency.json"
    if not consistency_path.is_file():
        raise FileNotFoundError(
            f"required native consistency artifact is missing: {consistency_path}"
        )
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    return compare_generation_results(
        *results,
        flash_reference=flash_reference,
        consistency=consistency,
    )


CONSISTENCY_PAIRS = {
    "cached_vs_uncached": ("uncached", "cached"),
    "full_vs_chunked": ("full_prefill", "chunked_prefill"),
    "prefix_fresh_vs_reused": ("prefix_fresh", "prefix_reused"),
    "single_vs_continuous_batch": ("single", "continuous_batch"),
    "offline_vs_serving_backend": ("offline", "serving_backend"),
}


def _validate_consistency_artifact(
    payload: object,
    reference: dict[str, object] | None = None,
) -> bool:
    if not isinstance(payload, dict):
        raise ValueError("native consistency artifact must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("stage") != "native-consistency":
        raise ValueError("native consistency artifact schema/stage is invalid")
    _validate_checkpoint_identity(payload, location=" in native consistency artifact")
    _validate_common_parameters(
        payload.get("parameters"), include_max_tokens=True,
        expected_backend="sdpa",
        location=" in native consistency artifact",
    )
    if payload["parameters"].get("language_attn_implementation") != "sdpa":
        raise ValueError(
            "artifact parameter language_attn_implementation in native "
            "consistency artifact must be 'sdpa'"
        )
    if reference is not None:
        if payload["checkpoint"] != reference["checkpoint"]:
            raise ValueError("native consistency checkpoint identity differs from generation artifacts")
        keys = ("temperature", "max_new_tokens", "dtype", "chat_template_marker")
        if {key: payload["parameters"][key] for key in keys} != {
            key: reference["parameters"][key] for key in keys
        }:
            raise ValueError("native consistency parameters differ from generation artifacts")
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("native consistency comparison is missing")
    pairs = comparison.get("pairs")
    if not isinstance(pairs, dict) or set(pairs) != set(CONSISTENCY_PAIRS):
        raise ValueError("native consistency pair checks are incomplete")
    pairs_exact = all(
        isinstance(value, dict) and value.get("exact") is True
        for value in pairs.values()
    )
    gates = comparison.get("gates")
    gates_exact = isinstance(gates, dict) and gates == {
        "same_image_repeated_outputs_exact": True,
        "continuous_batch_rows_exact": True,
    }
    return comparison.get("passed") is True and pairs_exact and gates_exact


def compare_consistency_results(variants: dict[str, list[int]]) -> dict[str, object]:
    missing = sorted(
        {variant for pair in CONSISTENCY_PAIRS.values() for variant in pair}
        - variants.keys()
    )
    if missing:
        raise ValueError(f"runtime consistency artifact is missing variants: {missing}")
    pairs = {
        name: _compare_tokens(variants[left], variants[right])
        for name, (left, right) in CONSISTENCY_PAIRS.items()
    }
    return {"passed": all(result["exact"] for result in pairs.values()), "pairs": pairs}


def _run_offline_variant(
    args: argparse.Namespace,
    case: VerificationCase,
    *,
    max_num_batched_tokens: int,
    batch_size: int = 1,
    repeat: int = 1,
) -> tuple[list[int], list[list[int]], dict[str, object] | None]:
    from nanovllm import LLM, SamplingParams

    vision_backend, language_backend = consistency_attention_backends()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max(8, batch_size),
        gpu_memory_utilization=args.gpu_memory_utilization,
        vision_feature_cache_bytes=args.vision_feature_cache_bytes,
        vision_attn_implementation=vision_backend,
        language_attn_implementation=language_backend,
        paged_kv_backend=args.paged_kv_backend,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    all_runs = []
    try:
        for _ in range(repeat):
            output = llm.generate_multimodal(
                [case.messages] * batch_size,
                [list(case.images) for _ in range(batch_size)],
                [list(case.details) for _ in range(batch_size)],
                sampling,
                use_tqdm=False,
            )
            all_runs.append([[int(token) for token in item["token_ids"]] for item in output])
        return all_runs[-1][0], all_runs[-1], _stats_dict(llm.vision_stats())
    finally:
        llm.exit()
        del llm
        _cleanup_cuda()


def run_native_consistency_stage(args: argparse.Namespace) -> dict[str, object]:
    corpus = generate_test_cases(args.results_dir / "consistency-images")
    case = next(item for item in corpus if item.name == "long_prompt_chunked")
    uncached, _, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens,
    )
    cached, cached_runs, cache_stats = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens, repeat=2,
    )
    full, _, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens,
    )
    chunked, _, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.chunked_prefill_tokens,
    )
    prefix_fresh, _, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens,
    )
    prefix_reused, prefix_runs, prefix_stats = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens, repeat=2,
    )
    single, _, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens,
    )
    continuous, continuous_outputs, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens, batch_size=4,
    )
    offline, _, _ = _run_offline_variant(
        args, case, max_num_batched_tokens=args.full_prefill_tokens,
    )

    # NativeQwenVLBackend is the exact Serving adapter before HTTP serialization.
    from nanovllm import LLM, SamplingParams
    from nanovllm.serving.engine_client import NativeQwenVLBackend
    from threading import Lock
    vision_backend, language_backend = consistency_attention_backends()
    serving_llm = LLM(
        args.model, enforce_eager=True, tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.full_prefill_tokens,
        vision_attn_implementation=vision_backend,
        language_attn_implementation=language_backend,
        paged_kv_backend=args.paged_kv_backend,
    )
    try:
        backend = NativeQwenVLBackend(lambda: serving_llm, Lock())
        serving_text = backend.generate_chat_batch(
            [case.messages], image_inputs_batch=[list(case.images)],
            image_details_batch=[list(case.details)], max_new_tokens=64,
            temperature=0, top_p=1.0,
        )[0]
        serving_tokens = [int(token) for token in serving_llm.tokenizer.encode(serving_text)]
        offline_text = serving_llm.tokenizer.decode(offline)
        offline_text_tokens = [int(token) for token in serving_llm.tokenizer.encode(offline_text)]
    finally:
        serving_llm.exit()
        del serving_llm
        _cleanup_cuda()
    # NativeQwenVLBackend exposes text only. Re-encoding both decoded completion
    # texts with the same tokenizer provides an honest token comparison at that
    # public boundary; the original offline-token gates remain separate above.
    variants = {
        "uncached": uncached, "cached": cached,
        "full_prefill": full, "chunked_prefill": chunked,
        "prefix_fresh": prefix_fresh, "prefix_reused": prefix_reused,
        "single": single, "continuous_batch": continuous,
        "offline": offline_text_tokens, "serving_backend": serving_tokens,
    }
    comparison = compare_consistency_results(variants)
    repeated_exact = all(run == cached_runs[0] for run in cached_runs) and all(
        run == prefix_runs[0] for run in prefix_runs
    )
    batch_exact = all(output == continuous_outputs[0] for output in continuous_outputs)
    comparison["gates"] = {
        "same_image_repeated_outputs_exact": repeated_exact,
        "continuous_batch_rows_exact": batch_exact,
    }
    comparison["passed"] = comparison["passed"] and repeated_exact and batch_exact
    return {
        **build_artifact_header(args, "native-consistency"),
        "variants": variants,
        "comparison": comparison,
        "cache_stats": cache_stats,
        "prefix_stats": prefix_stats,
        "prefix_cache_note": "No public disable switch exists; fresh-engine output is compared with second-run reuse.",
        "serving_boundary": "NativeQwenVLBackend direct (no HTTP)",
    }


def _move_batch_to_device(batch: Any, device: str) -> dict[str, Any]:
    if hasattr(batch, "to"):
        return dict(batch.to(device))
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _logits_diagnostics(logits: Any) -> dict[str, object]:
    import torch
    values, indices = torch.topk(logits.detach().float().cpu(), 2)
    return {
        "top2_token_ids": [int(value) for value in indices.tolist()],
        "top2_logits": [float(value) for value in values.tolist()],
        "margin": float(values[0] - values[1]),
    }


def _save_logits_sidecar(results_dir: Path, stage: str, case_name: str, logits: Any) -> str:
    import torch
    path = (results_dir / "logits" / stage / f"{case_name}.pt").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits.detach().float().cpu(), path)
    return str(path)


def hf_stage_attention_implementation(vision_backend: str):
    if vision_backend == "sdpa":
        return "sdpa"
    if vision_backend == "flash_attention_2":
        return {
            "vision_config": "flash_attention_2",
            "text_config": "sdpa",
        }
    raise ValueError(f"unsupported HF parity vision backend: {vision_backend}")


def run_hf_stage(
    args: argparse.Namespace,
    cases: Sequence[VerificationCase],
    *,
    vision_backend: str = "sdpa",
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    stage = "hf" if vision_backend == "sdpa" else "hf-flash"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=hf_stage_attention_implementation(vision_backend),
    ).to(args.device).eval()
    outputs: list[dict[str, object]] = []
    try:
        for case in cases:
            prompt = processor.apply_chat_template(case.messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[prompt], images=list(case.images) or None, videos=None,
                padding=False, return_tensors="pt",
            )
            device_inputs = _move_batch_to_device(inputs, args.device)
            input_ids = [int(token) for token in inputs["input_ids"][0].tolist()]
            started = time.perf_counter()
            with torch.inference_mode():
                first_forward = model(**device_inputs, use_cache=False, logits_to_keep=1)
                first_logits = first_forward.logits[0, -1].float()
                generated = model.generate(
                    **device_inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            token_ids = [int(token) for token in generated[0, len(input_ids):].tolist()]
            outputs.append({
                "name": case.name,
                "metadata": case.metadata(),
                "input_token_ids": input_ids,
                "output_token_ids": token_ids,
                "text": processor.tokenizer.decode(token_ids, skip_special_tokens=False),
                "logits_diagnostics": _logits_diagnostics(first_logits),
                "first_logits_values": first_logits.detach().float().cpu().tolist(),
                "first_logits_sidecar": _save_logits_sidecar(
                    args.results_dir, stage, case.name, first_logits
                ),
                "timing": {"total_latency_sec": elapsed},
            })
    finally:
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()
    result = build_artifact_header(args, stage)
    result["parameters"].update({
        "do_sample": False,
        "language_attn_implementation": "sdpa",
    })
    result["cases"] = outputs
    return result


def _stats_dict(stats: object | None) -> dict[str, object] | None:
    if stats is None:
        return None
    try:
        return asdict(stats)
    except TypeError:
        return dict(vars(stats))


def _numeric_delta(before: dict[str, object] | None, after: dict[str, object] | None) -> dict[str, object] | None:
    if before is None or after is None:
        return None
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if isinstance(after[key], (int, float)) and isinstance(before.get(key, 0), (int, float))
    }


def external_paged_kv_cache_bytes(llm: object) -> int:
    runner = llm.core.executor.driver_worker.runner
    if getattr(runner, "paged_kv_cache_owner", None) is None:
        return 0
    cache = runner.kv_cache
    return int(cache.numel() * cache.element_size())


@contextmanager
def capture_first_sample_logits(sampler: Any):
    """Capture every distribution reaching sampling during one generation."""
    captured: list[Any] = []
    original_call = sampler.__class__.__call__

    def instrumented_call(instance, logits, *args, **kwargs):
        if instance is sampler:
            captured.append(logits[0].detach().float().cpu())
        return original_call(instance, logits, *args, **kwargs)

    sampler.__class__.__call__ = instrumented_call
    try:
        yield captured
    finally:
        sampler.__class__.__call__ = original_call


def select_first_generated_logits(
    captured_logits: Sequence[Any],
    *,
    output_token_count: int,
):
    if output_token_count <= 0:
        raise ValueError("output_token_count must be positive")
    if len(captured_logits) < output_token_count:
        raise ValueError(
            "captured sampler calls cannot be fewer than generated output tokens"
        )
    first_generated_index = len(captured_logits) - output_token_count
    return captured_logits[first_generated_index]


def _cell_case(cell: dict[str, object], image_dir: Path) -> VerificationCase:
    corpus = generate_test_cases(image_dir)
    if cell["image_count"] == 1:
        base = next(case for case in corpus if case.name == "long_prompt_chunked")
    else:
        base = next(case for case in corpus if case.name == "two_different_images")
    text = (
        " ".join(["Force chunked prefill while comparing these images carefully."] * 96)
        if cell["prefill"] == "chunked"
        else "Describe the generated image or images precisely."
    )
    return VerificationCase(
        name="benchmark",
        messages=_messages(_content(text, int(cell["image_count"]), interleaved=True)),
        images=base.images[: int(cell["image_count"])],
        details=base.details[: int(cell["image_count"])],
        purpose="performance matrix cell",
        force_chunked_prefill=cell["prefill"] == "chunked",
    )


def run_native_benchmark_cell(cell: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    import torch
    from nanovllm import LLM, SamplingParams

    case = _cell_case(cell, args.results_dir / "benchmark-images")
    batch_size = int(cell["batch_size"])
    prefill_tokens = (
        args.chunked_prefill_tokens if cell["prefill"] == "chunked"
        else args.full_prefill_tokens
    )
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=prefill_tokens,
        max_num_seqs=max(8, batch_size),
        gpu_memory_utilization=args.gpu_memory_utilization,
        vision_feature_cache_bytes=args.vision_feature_cache_bytes,
        vision_attn_implementation=cell["backend"],
        language_attn_implementation="sdpa",
        paged_kv_backend=args.paged_kv_backend,
    )
    messages_batch = [case.messages for _ in range(batch_size)]
    images_batch = [list(case.images) for _ in range(batch_size)]
    details_batch = [list(case.details) for _ in range(batch_size)]
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    try:
        if cell["cache_state"] == "warm":
            llm.generate_multimodal(
                messages_batch, images_batch, details_batch, sampling, use_tqdm=False,
            )
        before = _stats_dict(llm.vision_stats())
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        outputs = llm.generate_multimodal(
            messages_batch, images_batch, details_batch, sampling, use_tqdm=False,
        )
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        after = _stats_dict(llm.vision_stats())
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        return {
            "total_latency_sec": elapsed,
            "output_tokens": output_tokens,
            "output_token_throughput_per_sec": output_tokens / elapsed if elapsed else 0.0,
            "peak_gpu_memory_bytes": int(
                torch.cuda.max_memory_allocated(args.device)
                + external_paged_kv_cache_bytes(llm)
            ),
            "external_paged_kv_cache_bytes": external_paged_kv_cache_bytes(llm),
            "vision_encoding_latency_sec": None,
            "vision_encoding_latency_note": "not exposed by the public runtime",
            "vision_stats": after,
            "vision_stats_delta": _numeric_delta(before, after),
        }
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def execute_benchmark_matrix(
    args: argparse.Namespace,
    *,
    runner=run_native_benchmark_cell,
) -> dict[str, object]:
    cells = []
    failures = []
    for index, cell in enumerate(benchmark_matrix()):
        try:
            measurement = runner(cell, args)
            required = {
                "total_latency_sec", "output_tokens",
                "output_token_throughput_per_sec", "peak_gpu_memory_bytes",
                "vision_encoding_latency_sec", "vision_stats",
            }
            missing = sorted(required - measurement.keys())
            if missing:
                raise ValueError(f"benchmark runner omitted measurements: {missing}")
            cells.append({"index": index, **cell, "measurement": measurement})
        except BaseException as error:
            failures.append({"index": index, **cell, "error": f"{type(error).__name__}: {error}"})
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "benchmark-matrix",
        "model": args.model,
        "parameters": {
            "vision_attn_implementations": ["sdpa", "flash_attention_2"],
            "language_attn_implementation": "sdpa",
        },
        "passed": not failures and len(cells) == 48,
        "cells": cells,
        "failures": failures,
        "execution_note": (
            "Cold cells create a fresh LLM. Warm cells run an unmeasured warmup before measurement. "
            "Each cell is isolated, which is intentionally slow but prevents cache-state leakage."
        ),
    }


def native_stage_attention_backends(vision_backend: str) -> tuple[str, str]:
    if vision_backend not in {"sdpa", "flash_attention_2"}:
        raise ValueError(f"unsupported native parity vision backend: {vision_backend}")
    # The Flash parity stage isolates the native vision Attention kernel.  The
    # language backend stays on the HF-aligned SDPA reference in both stages.
    return vision_backend, "sdpa"


def run_native_stage(
    args: argparse.Namespace,
    cases: Sequence[VerificationCase],
    *,
    backend: str,
) -> dict[str, object]:
    import torch
    from nanovllm import LLM, SamplingParams

    vision_backend, language_backend = native_stage_attention_backends(backend)
    stage = "native-sdpa" if vision_backend == "sdpa" else "native-flash"
    prefill_tokens = (
        args.chunked_prefill_tokens
        if any(case.force_chunked_prefill for case in cases)
        else args.full_prefill_tokens
    )
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=prefill_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vision_feature_cache_bytes=args.vision_feature_cache_bytes,
        vision_attn_implementation=vision_backend,
        language_attn_implementation=language_backend,
        paged_kv_backend=args.paged_kv_backend,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    outputs: list[dict[str, object]] = []
    try:
        for case in cases:
            materialized = llm.multimodal_processor.materialize(case.messages, case.images, case.details)
            before = _stats_dict(llm.vision_stats())
            torch.cuda.reset_peak_memory_stats(args.device)
            run_outputs = []
            run_latencies = []
            first_logits = None
            for run_index in range(case.repeat_runs):
                with capture_first_sample_logits(
                    llm.core.executor.driver_worker.runner.sampler
                ) as captured_first_logits:
                    torch.cuda.synchronize(args.device)
                    started = time.perf_counter()
                    generated = llm.generate_multimodal(
                        [case.messages], [list(case.images)], [list(case.details)],
                        sampling, use_tqdm=False,
                    )[0]
                    torch.cuda.synchronize(args.device)
                if run_index == 0:
                    if not captured_first_logits:
                        raise RuntimeError(
                            f"native first sampled logits were not captured for {case.name}"
                        )
                    first_logits = select_first_generated_logits(
                        captured_first_logits,
                        output_token_count=len(generated["token_ids"]),
                    )
                run_latencies.append(time.perf_counter() - started)
                run_outputs.append(generated)
            after = _stats_dict(llm.vision_stats())
            selected = run_outputs[-1]
            output_tokens = [int(token) for token in selected["token_ids"]]
            if first_logits is None:
                raise RuntimeError(f"native first-token logits were not captured for {case.name}")
            case_output: dict[str, object] = {
                "name": case.name,
                "metadata": case.metadata(),
                "input_token_ids": [int(token) for token in materialized.token_ids],
                "output_token_ids": output_tokens,
                "text": selected["text"],
                "logits_diagnostics": _logits_diagnostics(first_logits),
                "first_logits_values": first_logits.detach().float().cpu().tolist(),
                "first_logits_sidecar": _save_logits_sidecar(
                    args.results_dir, stage, case.name, first_logits
                ),
                "runtime_consistency": {
                    "repeated_outputs_exact": all(item["token_ids"] == run_outputs[0]["token_ids"] for item in run_outputs),
                    "run_count": len(run_outputs),
                },
            }
            if args.benchmark:
                total_latency = sum(run_latencies)
                case_output["benchmark"] = {
                    "run_latency_sec": run_latencies,
                    "total_latency_sec": total_latency,
                    "vision_encoding_latency_sec": None,
                    "vision_encoding_latency_note": "not exposed by the current public runtime metrics",
                    "output_tokens": sum(len(item["token_ids"]) for item in run_outputs),
                    "output_token_throughput_per_sec": (
                        sum(len(item["token_ids"]) for item in run_outputs) / total_latency if total_latency else 0.0
                    ),
                    "peak_gpu_memory_bytes": int(
                        torch.cuda.max_memory_allocated(args.device)
                        + external_paged_kv_cache_bytes(llm)
                    ),
                    "external_paged_kv_cache_bytes": external_paged_kv_cache_bytes(llm),
                    "vision_stats_before": before,
                    "vision_stats_after": after,
                    "vision_stats_delta": _numeric_delta(before, after),
                }
            outputs.append(case_output)
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    result = build_artifact_header(args, stage)
    result["parameters"].update({
        "vision_attn_implementation": vision_backend,
        "language_attn_implementation": language_backend,
        "paged_kv_backend": args.paged_kv_backend,
        "max_num_batched_tokens": prefill_tokens,
    })
    result["cases"] = outputs
    result["benchmark_matrix"] = benchmark_matrix() if args.benchmark_matrix else None
    return result


def _flatten_component_tensors(payload: object, prefix: str = "") -> dict[str, Any]:
    import torch
    if isinstance(payload, torch.Tensor):
        return {prefix: payload.detach().cpu()}
    if isinstance(payload, dict):
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_component_tensors(value, child))
        return flattened
    if isinstance(payload, (tuple, list)):
        flattened = {}
        for index, value in enumerate(payload):
            child = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_component_tensors(value, child))
        return flattened
    return {}


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.lower()) <= _HEX_DIGITS
    )


def _valid_sha_list(value: object, expected_count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_count
        and all(_valid_sha(item) for item in value)
    )


def validate_component_artifact(
    payload: object,
    *,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("component artifact must be an object")
    required = {
        "schema_version", "stage", "model", "checkpoint", "parameters",
        "case_metadata", "input_identity", "components",
    }
    missing_envelope = sorted(required - payload.keys())
    if missing_envelope:
        raise ValueError(f"component artifact is missing envelope fields: {missing_envelope}")
    unexpected_envelope = sorted(set(payload) - required)
    if unexpected_envelope:
        raise ValueError(f"component artifact has unexpected fields: {unexpected_envelope}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("component artifact schema version is invalid")
    if expected_stage is not None and payload["stage"] != expected_stage:
        raise ValueError(
            f"component artifact stage must be {expected_stage}, got {payload['stage']}"
        )
    if payload["stage"] not in {"components-hf", "components-native"}:
        raise ValueError("component artifact stage is invalid")
    _validate_checkpoint_identity(payload, location=" in component artifact")
    _validate_common_parameters(
        payload["parameters"], include_max_tokens=False,
        expected_backend="hf" if payload["stage"] == "components-hf" else "sdpa",
        location=" in component artifact",
    )
    case_metadata = payload["case_metadata"]
    case_metadata_keys = {
        "name", "purpose", "image_count", "image_sizes", "details",
        "force_chunked_prefill", "repeat_runs", "image_sha256",
    }
    if not isinstance(case_metadata, dict) or set(case_metadata) != case_metadata_keys:
        raise ValueError("component artifact case metadata fields are invalid")
    if not isinstance(case_metadata["name"], str) or not case_metadata["name"]:
        raise ValueError("component artifact case metadata name is invalid")
    if not isinstance(case_metadata["purpose"], str):
        raise ValueError("component artifact case metadata purpose is invalid")
    image_count = case_metadata["image_count"]
    if not isinstance(image_count, int) or isinstance(image_count, bool) or image_count < 0:
        raise ValueError("component artifact case metadata image_count is invalid")
    sizes = case_metadata["image_sizes"]
    if not isinstance(sizes, list) or len(sizes) != image_count or not all(
        isinstance(size, list) and len(size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in size)
        for size in sizes
    ):
        raise ValueError("component artifact case metadata image_sizes is invalid")
    details = case_metadata["details"]
    if not isinstance(details, list) or len(details) != image_count or not all(
        isinstance(value, str) for value in details
    ):
        raise ValueError("component artifact case metadata details is invalid")
    if not isinstance(case_metadata["force_chunked_prefill"], bool):
        raise ValueError("component artifact case metadata force_chunked_prefill is invalid")
    if not isinstance(case_metadata["repeat_runs"], int) or isinstance(
        case_metadata["repeat_runs"], bool
    ) or case_metadata["repeat_runs"] < 1:
        raise ValueError("component artifact case metadata repeat_runs is invalid")
    if not _valid_sha_list(case_metadata["image_sha256"], image_count):
        raise ValueError("component artifact case metadata image_sha256 is invalid")
    input_identity = payload["input_identity"]
    if not isinstance(input_identity, dict) or set(input_identity) != {
        "input_token_ids", "image_sha256"
    }:
        raise ValueError("component artifact input identity is invalid")
    token_ids = input_identity["input_token_ids"]
    if not isinstance(token_ids, list) or not token_ids or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in token_ids
    ):
        raise ValueError("component artifact input identity input_token_ids is invalid")
    if not _valid_sha_list(input_identity["image_sha256"], image_count):
        raise ValueError("component artifact input identity image_sha256 is invalid")
    if input_identity["image_sha256"] != case_metadata["image_sha256"]:
        raise ValueError("component artifact input identity image_sha256 differs from metadata")
    components = payload["components"]
    if not isinstance(components, dict):
        raise ValueError("component artifact components must be a flat dictionary")
    missing = sorted(set(REQUIRED_COMPONENT_KEYS) - components.keys())
    unexpected = sorted(components.keys() - set(REQUIRED_COMPONENT_KEYS))
    if missing or unexpected:
        raise ValueError(
            f"component artifact schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    import torch
    for key in REQUIRED_COMPONENT_KEYS:
        if not isinstance(components[key], torch.Tensor):
            raise TypeError(f"component artifact value {key} must be a Tensor")
    return components


def compare_component_artifacts(
    hf_path: Path,
    native_path: Path,
    *,
    rtol: float = 2e-2,
    atol: float = 2e-2,
    cosine_threshold: float = 0.9999,
) -> dict[str, object]:
    """Compare independently captured CPU tensors; this function loads no model."""
    import torch
    import torch.nn.functional as F

    for path in (hf_path, native_path):
        if not path.is_file():
            raise FileNotFoundError(f"required component artifact is missing: {path}")
    hf_artifact = torch.load(hf_path, map_location="cpu", weights_only=True)
    native_artifact = torch.load(native_path, map_location="cpu", weights_only=True)
    hf = validate_component_artifact(hf_artifact, expected_stage="components-hf")
    native = validate_component_artifact(native_artifact, expected_stage="components-native")
    if hf_artifact["checkpoint"] != native_artifact["checkpoint"]:
        raise ValueError("component artifacts use different checkpoint identity")
    common_parameter_keys = ("temperature", "dtype", "chat_template_marker")
    if {
        key: hf_artifact["parameters"][key] for key in common_parameter_keys
    } != {
        key: native_artifact["parameters"][key] for key in common_parameter_keys
    }:
        raise ValueError("component artifacts use different parameters")
    if hf_artifact["case_metadata"] != native_artifact["case_metadata"]:
        raise ValueError("component artifacts use different case metadata")
    if hf_artifact["input_identity"] != native_artifact["input_identity"]:
        raise ValueError("component artifacts use different input identity")
    entries = []
    for name in sorted(hf):
        raw_reference = hf[name]
        raw_candidate = native[name]
        if raw_reference.shape != raw_candidate.shape:
            entries.append({"name": name, "shape_match": False, "passed": False})
            continue
        if not (raw_reference.is_floating_point() or raw_reference.is_complex()):
            exact = torch.equal(raw_reference, raw_candidate)
            entries.append({
                "name": name,
                "shape_match": True,
                "shape": list(raw_reference.shape),
                "exact": exact,
                "passed": exact,
            })
            continue
        reference = raw_reference.float()
        candidate = raw_candidate.float()
        cosine = F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item()
        allclose = torch.allclose(reference, candidate, rtol=rtol, atol=atol)
        entry = {
            "name": name,
            "shape_match": True,
            "shape": list(reference.shape),
            "max_abs": (reference - candidate).abs().max().item() if reference.numel() else 0.0,
            "cosine": cosine,
            "allclose": allclose,
            "passed": allclose and cosine >= cosine_threshold,
        }
        if name.rsplit(".", 1)[-1] in {"first_logits", "logits"}:
            reference_argmax = reference.argmax().item()
            candidate_argmax = candidate.argmax().item()
            entry.update({
                "reference_argmax": reference_argmax,
                "candidate_argmax": candidate_argmax,
                "argmax_match": reference_argmax == candidate_argmax,
            })
            entry["passed"] = entry["passed"] and entry["argmax_match"]
        entries.append(entry)
    return {
        "thresholds": {"rtol": rtol, "atol": atol, "cosine": cosine_threshold},
        "passed": all(entry["passed"] for entry in entries),
        "components": entries,
        "capture_note": (
            "Artifacts must contain real processor/vision/logit tensors captured in separate processes; "
            "this comparator does not synthesize missing measurements."
        ),
    }


def _component_case(args: argparse.Namespace) -> VerificationCase:
    return next(
        case for case in generate_test_cases(args.results_dir / "component-images")
        if case.name == "square_image"
    )


def _component_envelope(
    args: argparse.Namespace,
    stage: str,
    case: VerificationCase,
    input_token_ids: Sequence[int],
    components: dict[str, object],
) -> dict[str, object]:
    header = build_artifact_header(args, stage)
    header["parameters"].pop("max_new_tokens")
    header.update({
        "case_metadata": {"name": case.name, **case.metadata()},
        "input_identity": {
            "input_token_ids": [int(value) for value in input_token_ids],
            "image_sha256": case.metadata()["image_sha256"],
        },
        "components": components,
    })
    return header


def compute_native_component_first_logits(
    model: Any,
    input_ids: Any,
    positions: Any,
    *,
    inputs_embeds: Any,
    visual_token_rows: Any,
    deepstack_visual_embeds: tuple[Any, ...],
):
    hidden = model(
        input_ids,
        positions,
        inputs_embeds=inputs_embeds,
        visual_token_rows=visual_token_rows,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    if hidden.ndim != 2 or hidden.shape[0] < 1:
        raise ValueError("native component hidden states must be [prompt_tokens, hidden_size]")
    # Direct component capture has no ParallelLMHead prefill context. Select the
    # final prompt row explicitly so it matches HF logits[0, -1].
    from nanovllm.utils.context import reset_context
    reset_context()
    return model.compute_logits(hidden[-1:])[0]


def capture_hf_components(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    case = _component_case(args)
    processor = AutoProcessor.from_pretrained(args.model)
    prompt = processor.apply_chat_template(case.messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt], images=list(case.images), videos=None,
        padding=False, return_tensors="pt",
    )
    device_inputs = _move_batch_to_device(inputs, args.device)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(args.device).eval()
    try:
        with torch.inference_mode():
            image_features, deepstack = model.get_image_features(
                device_inputs["pixel_values"], device_inputs["image_grid_thw"]
            )
            forward_output = model(**device_inputs, use_cache=False, logits_to_keep=1)
            position_ids, _ = model.model.get_rope_index(
                input_ids=device_inputs["input_ids"],
                image_grid_thw=device_inputs["image_grid_thw"],
                video_grid_thw=None,
                attention_mask=device_inputs.get("attention_mask"),
            )
        components = {
            "processor.input_ids": inputs["input_ids"].cpu(),
            "processor.pixel_values": inputs["pixel_values"].cpu(),
            "processor.image_grid_thw": inputs["image_grid_thw"].cpu(),
            "processor.position_ids": position_ids.cpu(),
            "vision.final": torch.cat(tuple(image_features), dim=0).cpu(),
            "vision.deepstack.0": deepstack[0].cpu(),
            "vision.deepstack.1": deepstack[1].cpu(),
            "vision.deepstack.2": deepstack[2].cpu(),
            "first_logits": forward_output.logits[0, -1].cpu(),
        }
        artifact = _component_envelope(
            args, "components-hf", case, inputs["input_ids"][0].tolist(), components,
        )
        validate_component_artifact(artifact, expected_stage="components-hf")
        return artifact
    finally:
        del model
        del processor
        _cleanup_cuda()


def capture_native_components(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import torch.distributed as dist
    import tempfile
    from nanovllm.engine.model_runner import (
        build_planned_visual_embeddings,
        plan_visual_replacements,
    )
    from nanovllm.engine.sequence import Sequence
    from nanovllm.models.qwen_vl_adapter import resolve_qwen_vl_adapter
    from nanovllm.multimodal.qwen3_vl_processor import Qwen3VLPromptProcessor
    from nanovllm.utils.context import reset_context, set_context
    from transformers import AutoConfig

    case = _component_case(args)
    processor = Qwen3VLPromptProcessor.from_pretrained(args.model)
    materialized = processor.materialize(case.messages, case.images, case.details)
    config = AutoConfig.from_pretrained(args.model)
    adapter = resolve_qwen_vl_adapter(config)
    rank = torch.device(args.device).index or 0
    handle, rendezvous = tempfile.mkstemp(prefix="nanoinfer-qwen3-component-")
    os.close(handle)
    os.unlink(rendezvous)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            "nccl", init_method=f"file://{rendezvous}", world_size=1, rank=0,
        )
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(args.device)
    provider = None
    model = None
    try:
        model = adapter.create_language_model(attention_backend="sdpa")
        adapter.load_language_weights(model, args.model)
        model.eval()
        provider = adapter.create_vision_provider(
            args.model, byte_budget=args.vision_feature_cache_bytes,
            dtype=torch.bfloat16, device=args.device, attn_implementation="sdpa",
        )
        provider.register_inputs(materialized.vision_inputs)
        feature_keys = tuple(item.feature_key for item in materialized.vision_inputs)
        sequence = Sequence(list(materialized.token_ids), multimodal_metadata=materialized.metadata)
        sequence.num_scheduled_tokens = len(sequence)
        input_ids = torch.tensor(materialized.token_ids, dtype=torch.long, device=args.device)
        positions = torch.tensor(materialized.metadata.position_ids, dtype=torch.long, device=args.device)
        plan = plan_visual_replacements([sequence])
        with torch.inference_mode(), provider.resolve_pinned(feature_keys) as bundles:
            visual = bundles[feature_keys[0]]
            payload = build_planned_visual_embeddings(model.embed(input_ids), plan, bundles)
            token_count = len(sequence)
            cu = torch.tensor([0, token_count], dtype=torch.int32, device=args.device)
            set_context(
                True, cu, cu, token_count, token_count,
                torch.empty(0, dtype=torch.int32, device=args.device),
            )
            first_logits = compute_native_component_first_logits(
                model,
                None, positions, inputs_embeds=payload.final_embedding,
                visual_token_rows=payload.visual_rows,
                deepstack_visual_embeds=payload.deepstack_embeddings,
            )
            reset_context()
        components = {
            "processor.input_ids": input_ids.unsqueeze(0).cpu(),
            "processor.pixel_values": torch.cat(
                [item.pixel_values for item in materialized.vision_inputs]
            ).cpu(),
            "processor.image_grid_thw": torch.tensor(
                [item.grid_thw for item in materialized.vision_inputs], dtype=torch.long
            ),
            "processor.position_ids": positions.unsqueeze(1).cpu(),
            "vision.final": visual.final_embedding.cpu(),
            "vision.deepstack.0": visual.deepstack_embeddings[0].cpu(),
            "vision.deepstack.1": visual.deepstack_embeddings[1].cpu(),
            "vision.deepstack.2": visual.deepstack_embeddings[2].cpu(),
            "first_logits": first_logits.cpu(),
        }
        artifact = _component_envelope(
            args, "components-native", case, materialized.token_ids, components,
        )
        validate_component_artifact(artifact, expected_stage="components-native")
        return artifact
    finally:
        reset_context()
        if provider is not None:
            del provider
        if model is not None:
            del model
        torch.set_default_device(original_device)
        torch.set_default_dtype(original_dtype)
        _cleanup_cuda()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.benchmark_matrix and args.stage != "benchmark-matrix":
        matrix_path = args.results_dir / "benchmark-matrix.json"
        write_json(matrix_path, {"schema_version": SCHEMA_VERSION, "matrix": benchmark_matrix()})

    if args.stage == "benchmark-matrix":
        report = execute_benchmark_matrix(args)
        write_json(args.results_dir / "benchmark-matrix-results.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1

    if args.stage in {"components-hf", "components-native"}:
        import torch
        capture = capture_hf_components if args.stage == "components-hf" else capture_native_components
        artifact = capture(args)
        output = args.results_dir / f"{args.stage}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, output)
        print(json.dumps({"stage": args.stage, "artifact": str(output.resolve())}, indent=2))
        return 0

    if args.stage == "native-consistency":
        report = run_native_consistency_stage(args)
        write_json(args.results_dir / "native-consistency.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["comparison"]["passed"] else 1

    if args.stage == "compare-components":
        report = compare_component_artifacts(
            args.component_hf, args.component_native,
            rtol=args.component_rtol,
            atol=args.component_atol,
            cosine_threshold=args.component_cosine_threshold,
        )
        write_json(args.results_dir / "component-comparison.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1

    image_dir = args.output_dir or args.results_dir / "images"
    cases = generate_test_cases(image_dir)
    if args.stage in {"hf", "all"}:
        write_json(args.results_dir / "hf.json", run_hf_stage(args, cases))
        _cleanup_cuda()
    if args.stage in {"hf-flash", "all"}:
        write_json(
            args.results_dir / "hf-flash.json",
            run_hf_stage(args, cases, vision_backend="flash_attention_2"),
        )
        _cleanup_cuda()
    if args.stage in {"native-sdpa", "all"}:
        write_json(args.results_dir / "native-sdpa.json", run_native_stage(args, cases, backend="sdpa"))
        _cleanup_cuda()
    if args.stage in {"native-flash", "all"}:
        write_json(
            args.results_dir / "native-flash.json",
            run_native_stage(args, cases, backend="flash_attention_2"),
        )
        _cleanup_cuda()
    if args.stage == "all":
        consistency = run_native_consistency_stage(args)
        write_json(args.results_dir / "native-consistency.json", consistency)
        _cleanup_cuda()
    if args.stage in {"compare", "all"}:
        report = compare_result_files(args.results_dir)
        write_json(args.results_dir / "comparison.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
