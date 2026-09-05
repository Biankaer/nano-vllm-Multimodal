from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Iterable, Sequence

from nanovllm.benchmark.schema import (
    AggregateMetrics,
    LatencySummary,
    RequestMetrics,
    RequestResult,
)


@dataclass(frozen=True, slots=True)
class SLOThresholds:
    ttft_sec: float = 1.0
    tpot_sec: float = 0.050

    def __post_init__(self) -> None:
        if self.ttft_sec < 0 or self.tpot_sec < 0:
            raise ValueError("SLO thresholds must be non-negative")


def _elapsed(start: float, end: float, name: str) -> float:
    value = end - start
    if not isfinite(value) or value < 0:
        raise ValueError(f"request timestamps must be monotonic for {name}")
    return value


def derive_request_metrics(
    result: RequestResult,
    *,
    slo: SLOThresholds = SLOThresholds(),
) -> RequestMetrics:
    ttft = None
    e2e = None
    tpot = None
    output_tps = None

    if result.stream_done_at is not None:
        e2e = _elapsed(result.request_start, result.stream_done_at, "E2E")
    if result.first_content_at is not None:
        ttft = _elapsed(result.request_start, result.first_content_at, "TTFT")
        if result.stream_done_at is not None and result.first_content_at > result.stream_done_at:
            raise ValueError("request timestamps must be monotonic from first content to completion")

    if result.success and (ttft is None or e2e is None):
        raise ValueError("successful requests require first-content and stream-done timestamps")

    if result.output_tokens > 1 and ttft is not None and e2e is not None:
        decode_time = e2e - ttft
        if decode_time < 0:
            raise ValueError("request timestamps must be monotonic during decode")
        tpot = decode_time / (result.output_tokens - 1)
        output_tps = None if decode_time == 0 else (result.output_tokens - 1) / decode_time

    chunk_itl = tuple(
        _elapsed(previous, current, "chunk ITL")
        for previous, current in zip(result.content_chunk_times, result.content_chunk_times[1:])
    )
    compliant = bool(
        result.success
        and ttft is not None
        and tpot is not None
        and ttft <= slo.ttft_sec
        and tpot <= slo.tpot_sec
    )
    return RequestMetrics(
        request_id=result.request_id,
        ttft_sec=ttft,
        tpot_sec=tpot,
        e2e_sec=e2e,
        single_user_output_tps=output_tps,
        chunk_itl_sec=chunk_itl,
        slo_compliant=compliant,
    )


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: Iterable[float]) -> LatencySummary:
    materialized = tuple(values)
    if not materialized:
        return LatencySummary(0, None, None, None, None, None, None)
    return LatencySummary(
        count=len(materialized),
        mean=fmean(materialized),
        minimum=min(materialized),
        maximum=max(materialized),
        p50=linear_percentile(materialized, 0.50),
        p95=linear_percentile(materialized, 0.95),
        p99=linear_percentile(materialized, 0.99),
    )


def aggregate_results(
    results: Iterable[RequestResult],
    *,
    wall_time_sec: float,
    slo: SLOThresholds = SLOThresholds(),
) -> AggregateMetrics:
    if not isfinite(wall_time_sec) or wall_time_sec <= 0:
        raise ValueError("wall_time_sec must be positive")
    materialized = tuple(results)
    derived = tuple((result, derive_request_metrics(result, slo=slo)) for result in materialized)
    successes = tuple((result, metric) for result, metric in derived if result.success)
    failures = tuple((result, metric) for result, metric in derived if not result.success)
    compliant = sum(metric.slo_compliant for _, metric in successes)
    success_count = len(successes)

    return AggregateMetrics(
        total_requests=len(materialized),
        successful_requests=success_count,
        failed_requests=len(failures),
        error_rate=(len(failures) / len(materialized)) if materialized else 0.0,
        request_throughput_rps=success_count / wall_time_sec,
        system_output_tps=sum(result.output_tokens for result, _ in successes) / wall_time_sec,
        goodput_rps=compliant / wall_time_sec,
        slo_compliant_requests=compliant,
        slo_compliance=(compliant / success_count) if success_count else 0.0,
        ttft_sec=summarize(metric.ttft_sec for _, metric in successes if metric.ttft_sec is not None),
        tpot_sec=summarize(metric.tpot_sec for _, metric in successes if metric.tpot_sec is not None),
        e2e_sec=summarize(metric.e2e_sec for _, metric in successes if metric.e2e_sec is not None),
        chunk_itl_sec=summarize(
            value for _, metric in successes for value in metric.chunk_itl_sec
        ),
        failure_latency_sec=summarize(
            metric.e2e_sec for _, metric in failures if metric.e2e_sec is not None
        ),
    )
