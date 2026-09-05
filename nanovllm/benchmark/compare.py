from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ReportIdentityError(ValueError):
    """Reports cannot be fairly or completely compared."""


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    nanoinfer: float
    vllm: float
    difference: float
    ratio: float | None
    better_when: str


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    profile: str
    concurrency: int
    rows: dict[str, ComparisonRow]
    resource_comparison_valid: bool
    resource_comparison_reason: str | None


_IDENTITY_FIELDS = (
    "workload_sha256",
    "checkpoint_fingerprint",
    "dtype",
    "generation_parameters",
    "profile",
    "concurrency",
    "gpu_uuid",
    "request_ids",
)
_LATENCY_METRICS = ("ttft_sec", "tpot_sec", "e2e_sec")
_PERCENTILES = ("p50", "p95", "p99")
_THROUGHPUT_METRICS = (
    "request_throughput_rps",
    "system_output_tps",
    "goodput_rps",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportIdentityError(f"report {label} must be an object")
    return value


def _require_number(container: Mapping[str, Any], key: str) -> float:
    value = container.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReportIdentityError(f"required metric {key} is missing or non-numeric")
    return float(value)


def _validate_identity(nanoinfer: Mapping[str, Any], vllm: Mapping[str, Any]) -> Mapping[str, Any]:
    nano_schema = nanoinfer.get("schema_version")
    vllm_schema = vllm.get("schema_version")
    if nano_schema != vllm_schema or not isinstance(nano_schema, str):
        raise ReportIdentityError("report schema versions do not match")
    nano_identity = _mapping(nanoinfer.get("identity"), "identity")
    vllm_identity = _mapping(vllm.get("identity"), "identity")
    for field in _IDENTITY_FIELDS:
        if field not in nano_identity or field not in vllm_identity:
            raise ReportIdentityError(f"required identity field is missing: {field}")
        if nano_identity[field] != vllm_identity[field]:
            raise ReportIdentityError(f"report identity mismatch: {field}")
    request_ids = nano_identity["request_ids"]
    if not isinstance(request_ids, list) or len(request_ids) != len(set(request_ids)):
        raise ReportIdentityError("request_ids must be a unique list")
    return nano_identity


def _request_outputs(report: Mapping[str, Any]) -> dict[str, str]:
    requests = report.get("requests")
    if not isinstance(requests, list):
        raise ReportIdentityError("report requests must be a list")
    outputs: dict[str, str] = {}
    for request in requests:
        if not isinstance(request, Mapping):
            raise ReportIdentityError("report request must be an object")
        request_id = request.get("request_id")
        output_sha = request.get("output_sha256")
        if not isinstance(request_id, str) or not isinstance(output_sha, str):
            raise ReportIdentityError("request output identity is incomplete")
        if request_id in outputs:
            raise ReportIdentityError(f"duplicate request result: {request_id}")
        outputs[request_id] = output_sha
    return outputs


def _row(nano: float, vllm: float, *, better_when: str) -> ComparisonRow:
    return ComparisonRow(
        nanoinfer=nano,
        vllm=vllm,
        difference=nano - vllm,
        ratio=None if vllm == 0 else nano / vllm,
        better_when=better_when,
    )


def _resource_gate(
    nanoinfer: Mapping[str, Any],
    vllm: Mapping[str, Any],
) -> tuple[bool, str | None]:
    resources = (
        _mapping(nanoinfer.get("resource_metrics"), "resource_metrics"),
        _mapping(vllm.get("resource_metrics"), "resource_metrics"),
    )
    if any(resource.get("contaminated") is True for resource in resources):
        return False, "GPU resource samples are contaminated by external processes"
    if any(resource.get("valid") is not True for resource in resources):
        return False, "GPU resource samples are unavailable or invalid"
    return True, None


def compare_reports(
    nanoinfer: Mapping[str, Any],
    vllm: Mapping[str, Any],
) -> ComparisonResult:
    identity = _validate_identity(nanoinfer, vllm)
    expected_ids = set(identity["request_ids"])
    nano_outputs = _request_outputs(nanoinfer)
    vllm_outputs = _request_outputs(vllm)
    if set(nano_outputs) != expected_ids or set(vllm_outputs) != expected_ids:
        raise ReportIdentityError("request result ids do not match identity request_ids")
    mismatches = [request_id for request_id in sorted(expected_ids) if nano_outputs[request_id] != vllm_outputs[request_id]]
    if mismatches:
        raise ReportIdentityError(f"greedy output mismatch for request ids: {', '.join(mismatches)}")

    nano_metrics = _mapping(nanoinfer.get("aggregates"), "aggregates")
    vllm_metrics = _mapping(vllm.get("aggregates"), "aggregates")
    rows: dict[str, ComparisonRow] = {}
    for metric in _LATENCY_METRICS:
        nano_latency = _mapping(nano_metrics.get(metric), f"metric {metric}")
        vllm_latency = _mapping(vllm_metrics.get(metric), f"metric {metric}")
        for percentile in _PERCENTILES:
            key = f"{metric}.{percentile}"
            rows[key] = _row(
                _require_number(nano_latency, percentile),
                _require_number(vllm_latency, percentile),
                better_when="lower",
            )
    for metric in _THROUGHPUT_METRICS:
        rows[metric] = _row(
            _require_number(nano_metrics, metric),
            _require_number(vllm_metrics, metric),
            better_when="higher",
        )

    resources_valid, resources_reason = _resource_gate(nanoinfer, vllm)
    return ComparisonResult(
        profile=str(identity["profile"]),
        concurrency=int(identity["concurrency"]),
        rows=rows,
        resource_comparison_valid=resources_valid,
        resource_comparison_reason=resources_reason,
    )
