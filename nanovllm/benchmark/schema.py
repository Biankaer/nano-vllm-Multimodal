from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    request_id: str
    messages: tuple[dict[str, Any], ...]
    max_tokens: int
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("workload request id must not be empty")
        if not self.messages:
            raise ValueError("workload messages must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "messages": list(self.messages),
            "max_tokens": self.max_tokens,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class RequestResult:
    """Raw client-observed result for one streaming request."""

    request_id: str
    tags: tuple[str, ...]
    http_status: int | None
    success: bool
    error: str | None
    timed_out: bool
    request_start: float
    first_content_at: float | None
    stream_done_at: float | None
    content_chunk_times: tuple[float, ...]
    prompt_tokens: int | None
    output_tokens: int
    token_count_source: str
    output_sha256: str
    finish_reason: str | None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    request_id: str
    ttft_sec: float | None
    tpot_sec: float | None
    e2e_sec: float | None
    single_user_output_tps: float | None
    chunk_itl_sec: tuple[float, ...]
    slo_compliant: bool


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    p50: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    request_throughput_rps: float
    system_output_tps: float
    goodput_rps: float
    slo_compliant_requests: int
    slo_compliance: float
    ttft_sec: LatencySummary
    tpot_sec: LatencySummary
    e2e_sec: LatencySummary
    chunk_itl_sec: LatencySummary
    failure_latency_sec: LatencySummary
