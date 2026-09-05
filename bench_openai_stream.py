from __future__ import annotations

"""Reproducible streaming benchmark for any OpenAI-compatible VLM endpoint.

The same workload profiles are used by NanoInfer and vLLM.  The client records
TTFT, TPOT, inter-token/chunk latency, E2E latency and aggregate throughput.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from nanovllm.benchmark.metrics import SLOThresholds, aggregate_results, derive_request_metrics
from nanovllm.benchmark.schema import RequestResult
from nanovllm.benchmark.sse_client import OpenAIStreamAccumulator, SSEDecoder, SSEProtocolError
from nanovllm.benchmark.workloads import generate_profile, workload_sha256


def endpoint(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1/chat/completions") else value + "/v1/chat/completions"


def metrics_url(value: str) -> str:
    parsed = urlparse(value)
    return urlunparse(parsed._replace(path="/metrics/multimodal", query="", fragment=""))


def one_request(url: str, model: str, workload: Any, timeout: float) -> RequestResult:
    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": list(workload.messages),
        "max_tokens": workload.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    accumulator = OpenAIStreamAccumulator(request_start=started)
    try:
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            decoder = SSEDecoder()
            # readline() preserves the arrival boundary of each SSE line.  A
            # large read() often coalesces the whole response and would make
            # all token timestamps identical, yielding a bogus TPOT of zero.
            for chunk in iter(response.readline, b""):
                received = time.perf_counter()
                for event in decoder.feed(chunk):
                    accumulator.consume(event, received_at=received)
            for event in decoder.close():
                accumulator.consume(event, received_at=time.perf_counter())
            observation = accumulator.finish()
        return RequestResult(
            request_id=workload.request_id,
            tags=workload.tags,
            http_status=response.status,
            success=True,
            error=None,
            timed_out=False,
            request_start=observation.request_start,
            first_content_at=observation.first_content_at,
            stream_done_at=observation.stream_done_at,
            content_chunk_times=observation.content_chunk_times,
            prompt_tokens=observation.prompt_tokens,
            output_tokens=observation.output_tokens or 0,
            token_count_source=observation.token_count_source,
            output_sha256=hashlib.sha256(observation.output_text.encode("utf-8")).hexdigest(),
            finish_reason=observation.finish_reason,
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return failed(workload, started, exc.code, detail)
    except (URLError, TimeoutError, OSError, SSEProtocolError, json.JSONDecodeError) as exc:
        return failed(workload, started, None, str(exc), timed_out=isinstance(exc, TimeoutError))


def failed(workload: Any, started: float, status: int | None, error: str, *, timed_out: bool = False) -> RequestResult:
    return RequestResult(
        request_id=workload.request_id,
        tags=workload.tags,
        http_status=status,
        success=False,
        error=error,
        timed_out=timed_out,
        request_start=started,
        first_content_at=None,
        stream_done_at=None,
        content_chunk_times=(),
        prompt_tokens=None,
        output_tokens=0,
        token_count_source="missing",
        output_sha256="",
        finish_reason=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=("text_short", "text_long", "image_single_unique", "image_single_reused", "image_multi", "mixed"), default="image_single_reused")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slo-ttft", type=float, default=1.0)
    parser.add_argument("--slo-tpot", type=float, default=0.05)
    args = parser.parse_args()
    url = endpoint(args.server_url)
    workloads = generate_profile(args.profile, args.count, args.seed)
    for workload in workloads[: args.warmup]:
        result = one_request(url, args.model, workload, args.timeout)
        if not result.success:
            raise SystemExit(f"warmup failed: {result.error}")
    started = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, url, args.model, workload, args.timeout) for workload in workloads]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - started
    aggregates = aggregate_results(results, wall_time_sec=wall, slo=SLOThresholds(args.slo_ttft, args.slo_tpot))
    aggregate_payload = asdict(aggregates)
    successful = [result for result in results if result.success]
    prompt_tokens = sum(result.prompt_tokens or 0 for result in successful)
    output_tokens = sum(result.output_tokens for result in successful)
    aggregate_payload.update({
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_token_throughput_tps": prompt_tokens / wall,
        "total_token_throughput_tps": (prompt_tokens + output_tokens) / wall,
    })
    report = {
        "schema_version": "1.0",
        "identity": {
            "workload_sha256": workload_sha256(workloads),
            "dtype": "bfloat16",
            "generation_parameters": {"temperature": 0, "top_p": 1, "ignore_eos": True},
            "profile": args.profile,
            "concurrency": args.concurrency,
            "request_ids": [w.request_id for w in workloads],
        },
        "config": vars(args) | {"server_url": url},
        "wall_time_sec": wall,
        "aggregates": aggregate_payload,
        "requests": [asdict(result) | {"metrics": asdict(derive_request_metrics(result))} for result in sorted(results, key=lambda x: x.request_id)],
    }
    try:
        with urlopen(metrics_url(url), timeout=5) as response:
            report["server_metrics"] = json.loads(response.read())
    except Exception:
        report["server_metrics"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "aggregates": report["aggregates"]}, ensure_ascii=False, indent=2))
    return 0 if aggregates.failed_requests == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
