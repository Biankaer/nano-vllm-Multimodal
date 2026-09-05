from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Sequence

import torch

from nanovllm.benchmark.metrics import linear_percentile
from nanovllm.layers.paged_kv import create_paged_kv_backend
from nanovllm.layers.paged_kv.pytorch_backend import PyTorchPagedKVBackend


_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


@dataclass(frozen=True, slots=True)
class PagedKVBenchmarkCase:
    backend: str
    operation: str
    dtype: torch.dtype
    num_tokens: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    mapping: str = "contiguous"
    seed: int = 17

    def __post_init__(self) -> None:
        if self.backend not in {"pytorch", "triton", "cuda"}:
            raise ValueError(f"unsupported Paged KV backend: {self.backend}")
        if self.operation not in {"store", "gather"}:
            raise ValueError(f"unsupported Paged KV operation: {self.operation}")
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(f"unsupported Paged KV dtype: {self.dtype}")
        if min(
            self.num_tokens,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("Paged KV benchmark dimensions must be positive")
        if self.num_tokens > self.num_blocks * self.block_size:
            raise ValueError("num_tokens exceeds Paged KV cache capacity")
        if self.mapping not in {"contiguous", "random", "random_with_skips"}:
            raise ValueError(f"unsupported slot mapping: {self.mapping}")

    @property
    def selector(self) -> str:
        fields = (
            ("backend", self.backend),
            ("operation", self.operation),
            ("dtype", _DTYPE_NAMES[self.dtype]),
            ("tokens", self.num_tokens),
            ("blocks", self.num_blocks),
            ("block", self.block_size),
            ("heads", self.num_kv_heads),
            ("dim", self.head_dim),
            ("mapping", self.mapping),
            ("seed", self.seed),
        )
        return ";".join(f"{key}={value}" for key, value in fields)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dtype"] = _DTYPE_NAMES[self.dtype]
        payload["selector"] = self.selector
        return payload


@dataclass(slots=True)
class PreparedPagedKVCase:
    case: PagedKVBenchmarkCase
    backend: Any
    reference: PyTorchPagedKVBackend
    allocation: Any
    key: torch.Tensor
    value: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor

    def invoke(self) -> torch.Tensor | None:
        if self.case.operation == "store":
            self.backend.store(
                self.key,
                self.value,
                self.key_cache,
                self.value_cache,
                self.slot_mapping,
            )
            return None
        return self.backend.gather(
            self.key_cache,
            self.block_table,
            self.case.num_tokens,
        )

    def assert_correct(self) -> None:
        if self.case.operation == "store":
            expected_key = self.key_cache.clone()
            expected_value = self.value_cache.clone()
            self.reference.store(
                self.key,
                self.value,
                expected_key,
                expected_value,
                self.slot_mapping,
            )
            self.invoke()
            torch.cuda.synchronize(self.key.device)
            if not torch.equal(self.key_cache, expected_key):
                raise AssertionError("Paged KV Store key cache differs from PyTorch")
            if not torch.equal(self.value_cache, expected_value):
                raise AssertionError("Paged KV Store value cache differs from PyTorch")
            return
        expected = self.reference.gather(
            self.key_cache,
            self.block_table,
            self.case.num_tokens,
        )
        actual = self.invoke()
        torch.cuda.synchronize(self.key_cache.device)
        if actual is None or not torch.equal(actual, expected):
            raise AssertionError("Paged KV Gather differs from PyTorch")


def parse_case_selector(selector: str) -> PagedKVBenchmarkCase:
    try:
        fields = dict(part.split("=", 1) for part in selector.split(";"))
        if len(fields) != 10:
            raise ValueError
        return PagedKVBenchmarkCase(
            backend=fields["backend"],
            operation=fields["operation"],
            dtype=_NAME_TO_DTYPE[fields["dtype"]],
            num_tokens=int(fields["tokens"]),
            num_blocks=int(fields["blocks"]),
            block_size=int(fields["block"]),
            num_kv_heads=int(fields["heads"]),
            head_dim=int(fields["dim"]),
            mapping=fields["mapping"],
            seed=int(fields["seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Paged KV case selector: {selector}") from exc


def expand_benchmark_matrix(
    *,
    backends: Sequence[str],
    operations: Sequence[str],
    dtypes: Sequence[torch.dtype],
    token_counts: Sequence[int],
    kv_shapes: Sequence[tuple[int, int]],
    block_size: int,
    num_blocks: int,
    mapping: str = "random_with_skips",
    seed: int = 17,
) -> tuple[PagedKVBenchmarkCase, ...]:
    return tuple(
        PagedKVBenchmarkCase(
            backend=backend,
            operation=operation,
            dtype=dtype,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=heads,
            head_dim=head_dim,
            mapping=mapping,
            seed=seed,
        )
        for backend in backends
        for operation in operations
        for dtype in dtypes
        for heads, head_dim in kv_shapes
        for num_tokens in token_counts
    )


def effective_bytes(case: PagedKVBenchmarkCase) -> int:
    element_size = torch.empty((), dtype=case.dtype).element_size()
    payload = case.num_tokens * case.num_kv_heads * case.head_dim * element_size
    return payload * (4 if case.operation == "store" else 2)


def summarize_latencies(samples_us: Iterable[float]) -> dict[str, float | int]:
    samples = tuple(float(value) for value in samples_us)
    if not samples or any(value < 0 for value in samples):
        raise ValueError("latency samples must be non-empty and non-negative")
    return {
        "count": len(samples),
        "mean_us": fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "p50_us": linear_percentile(samples, 0.50),
        "p95_us": linear_percentile(samples, 0.95),
        "p99_us": linear_percentile(samples, 0.99),
    }


def run_with_correctness_gate(
    *,
    correctness_check: Callable[[], None],
    warmup: Callable[[], None],
    measure: Callable[[], list[float]],
) -> list[float]:
    correctness_check()
    warmup()
    return measure()


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def prepare_case(
    case: PagedKVBenchmarkCase,
    *,
    device: torch.device | str = "cuda",
) -> PreparedPagedKVCase:
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("Paged KV GPU benchmark requires a CUDA device")
    backend = create_paged_kv_backend(case.backend)
    allocation = backend.allocate(
        (
            2,
            case.num_blocks,
            case.block_size,
            case.num_kv_heads,
            case.head_dim,
        ),
        dtype=case.dtype,
        device=resolved_device,
    )
    cache = allocation.tensor
    cache.normal_()
    key_cache, value_cache = cache[0], cache[1]
    generator = _make_generator(resolved_device, case.seed)
    key = torch.randn(
        case.num_tokens,
        case.num_kv_heads,
        case.head_dim,
        dtype=case.dtype,
        device=resolved_device,
        generator=generator,
    )
    value = torch.randn(
        case.num_tokens,
        case.num_kv_heads,
        case.head_dim,
        dtype=case.dtype,
        device=resolved_device,
        generator=generator,
    )
    capacity = case.num_blocks * case.block_size
    if case.mapping == "contiguous":
        slot_mapping = torch.arange(
            case.num_tokens, dtype=torch.int32, device=resolved_device
        )
    else:
        slot_mapping = torch.randperm(
            capacity,
            dtype=torch.int32,
            device=resolved_device,
            generator=generator,
        )[: case.num_tokens]
        if case.mapping == "random_with_skips":
            slot_mapping[::11] = -1
    block_table = torch.randperm(
        case.num_blocks,
        dtype=torch.int32,
        device=resolved_device,
        generator=generator,
    )
    torch.cuda.synchronize(resolved_device)
    return PreparedPagedKVCase(
        case=case,
        backend=backend,
        reference=PyTorchPagedKVBackend(),
        allocation=allocation,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        block_table=block_table,
    )


def measure_cuda_iterations(
    invoke: Callable[[], Any],
    *,
    iterations: int,
) -> list[float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        invoke()
        end.record()
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def benchmark_case(
    case: PagedKVBenchmarkCase,
    *,
    warmup_iterations: int = 100,
    iterations: int = 1000,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    prepared = prepare_case(case, device=device)

    def warmup() -> None:
        for _ in range(warmup_iterations):
            prepared.invoke()
        torch.cuda.synchronize(device)

    samples = run_with_correctness_gate(
        correctness_check=prepared.assert_correct,
        warmup=warmup,
        measure=lambda: measure_cuda_iterations(
            prepared.invoke, iterations=iterations
        ),
    )
    summary = summarize_latencies(samples)
    byte_count = effective_bytes(case)
    p50_us = float(summary["p50_us"])
    return {
        "case": case.to_dict(),
        "correctness": "pass",
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "effective_bytes": byte_count,
        "effective_gbps_at_p50": 0.0 if p50_us == 0 else byte_count / (p50_us * 1000.0),
        "latency": summary,
        "samples_us": samples,
    }


def environment_identity(device: torch.device | str = "cuda") -> dict[str, Any]:
    resolved = torch.device(device)
    properties = torch.cuda.get_device_properties(resolved)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": str(resolved),
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "hip_status": (
            "pytorch_rocm_available"
            if torch.version.hip is not None
            else "not_tested_no_rocm_toolchain"
        ),
    }


def build_report(
    results: Sequence[dict[str, Any]],
    *,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    return {
        "schema_version": "nanoinfer.paged_kv_benchmark.v1",
        "identity": environment_identity(device),
        "results": list(results),
    }


def atomic_write_json(destination: Path | str, payload: Any) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paged KV Backend Microbenchmark",
        "",
        "| Backend | Operation | Dtype | Tokens | KV shape | P50 (us) | P95 (us) | P99 (us) | GB/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        case = result["case"]
        latency = result["latency"]
        lines.append(
            "| {backend} | {operation} | {dtype} | {num_tokens} | {num_kv_heads}x{head_dim} | "
            "{p50_us:.3f} | {p95_us:.3f} | {p99_us:.3f} | {gbps:.3f} |".format(
                **case,
                **latency,
                gbps=result["effective_gbps_at_p50"],
            )
        )
    return "\n".join(lines) + "\n"


def atomic_write_text(destination: Path | str, text: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
