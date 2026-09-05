from __future__ import annotations

from typing import Any, Mapping

from nanovllm.benchmark.paged_kv import summarize_latencies


_BACKENDS = ("pytorch", "triton", "cuda")


def _validate_report(backend: str, report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if report.get("stage") not in {"native-sdpa", "native-flash"}:
        raise ValueError(f"{backend} report is not a native generation stage")
    parameters = report.get("parameters")
    if not isinstance(parameters, Mapping) or parameters.get("paged_kv_backend") != backend:
        raise ValueError(f"{backend} report has the wrong paged_kv_backend identity")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{backend} report has no benchmark cases")
    return cases


def compare_paged_kv_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reports) != set(_BACKENDS):
        raise ValueError("reports must contain pytorch, triton, and cuda backends")
    cases_by_backend = {
        backend: _validate_report(backend, reports[backend])
        for backend in _BACKENDS
    }
    reference = {
        str(case["name"]): tuple(int(token) for token in case["output_token_ids"])
        for case in cases_by_backend["pytorch"]
    }
    for backend in _BACKENDS[1:]:
        candidate = {
            str(case["name"]): tuple(int(token) for token in case["output_token_ids"])
            for case in cases_by_backend[backend]
        }
        if candidate != reference:
            mismatches = sorted(
                name
                for name in set(reference) | set(candidate)
                if reference.get(name) != candidate.get(name)
            )
            raise ValueError(
                f"token mismatch for backend {backend}: {', '.join(mismatches)}"
            )

    aggregates: dict[str, dict[str, Any]] = {}
    for backend in _BACKENDS:
        latencies = []
        output_tokens = 0
        peak_memory = 0
        for case in cases_by_backend[backend]:
            benchmark = case.get("benchmark")
            if not isinstance(benchmark, Mapping):
                raise ValueError(f"{backend} case {case.get('name')} has no benchmark data")
            latencies.append(float(benchmark["total_latency_sec"]))
            output_tokens += int(benchmark["output_tokens"])
            peak_memory = max(peak_memory, int(benchmark["peak_gpu_memory_bytes"]))
        total_latency = sum(latencies)
        summary_us = summarize_latencies(value * 1_000_000 for value in latencies)
        aggregates[backend] = {
            "case_count": len(latencies),
            "total_latency_sec": total_latency,
            "output_tokens": output_tokens,
            "output_tokens_per_sec": (
                output_tokens / total_latency if total_latency else 0.0
            ),
            "peak_gpu_memory_bytes": peak_memory,
            "case_latency_sec": {
                key.replace("_us", "_sec"): value / 1_000_000
                if key.endswith("_us")
                else value
                for key, value in summary_us.items()
            },
        }
    baseline = aggregates["pytorch"]["total_latency_sec"]
    for backend in _BACKENDS:
        latency = aggregates[backend]["total_latency_sec"]
        aggregates[backend]["speedup_vs_pytorch"] = (
            baseline / latency if latency else None
        )
    return {
        "schema_version": "nanoinfer.paged_kv_e2e_comparison.v1",
        "tokens_exact": True,
        "case_names": list(reference),
        "backends": aggregates,
    }


def render_paged_kv_e2e_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Paged KV End-to-End Comparison",
        "",
        f"Greedy token IDs exact across all backends: `{str(result['tokens_exact']).lower()}`",
        "",
        "| Backend | Cases | Total latency (s) | Output tok/s | P50 case (s) | P95 case (s) | P99 case (s) | Peak GiB | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in _BACKENDS:
        values = result["backends"][backend]
        latency = values["case_latency_sec"]
        lines.append(
            f"| {backend} | {values['case_count']} | {values['total_latency_sec']:.6f} | "
            f"{values['output_tokens_per_sec']:.3f} | {latency['p50_sec']:.6f} | "
            f"{latency['p95_sec']:.6f} | {latency['p99_sec']:.6f} | "
            f"{values['peak_gpu_memory_bytes'] / 1024**3:.3f} | "
            f"{values['speedup_vs_pytorch']:.3f}x |"
        )
    return "\n".join(lines) + "\n"
