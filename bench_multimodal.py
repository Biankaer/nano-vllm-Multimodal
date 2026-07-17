from __future__ import annotations

"""Benchmark the OpenAI-compatible Qwen2.5-VL endpoint.

The benchmark intentionally uses only the Python standard library so it can be
run on a server without adding a client dependency. Streaming is not enabled by
the current API, so TTFT/TPOT are reported as unavailable until SSE is added.
"""

import argparse
import base64
import json
import mimetypes
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


def main() -> None:
    args = parse_args()
    args.server_url = normalize_endpoint(args.server_url)
    image_urls = [image_to_url(value) for value in args.image]
    payload = build_payload(args, image_urls)

    for _ in range(args.warmup):
        result = send_request(args.server_url, payload, args.timeout)
        if not result["ok"]:
            raise SystemExit(f"warmup request failed: {result['error']}")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_request, args.server_url, payload, args.timeout)
            for _ in range(args.num_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_time = time.perf_counter() - started

    report = make_report(args, results, wall_time)
    report["server_multimodal_metrics"] = fetch_server_metrics(args.server_url, args.timeout)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        write_report(Path(args.output_json), report)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-VL through /v1/chat/completions")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen2.5-vl")
    parser.add_argument("--image", action="append", default=[], help="URL, data URL, or local image path; repeatable")
    parser.add_argument("--prompt", default="Describe the image in detail.")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.num_requests < 1 or args.concurrency < 1:
        parser.error("--num-requests and --concurrency must be positive")
    if args.image and args.image_detail not in {"auto", "low", "high"}:
        parser.error("invalid image detail")
    return args


def build_payload(args: argparse.Namespace, image_urls: list[str]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image, "detail": args.image_detail}}
        for image in image_urls
    ]
    content.append({"type": "text", "text": args.prompt})
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }


def image_to_url(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{data}"


def normalize_endpoint(value: str) -> str:
    return value.rstrip("/") if value.rstrip("/").endswith("/v1/chat/completions") else f"{value.rstrip('/')}/v1/chat/completions"


def fetch_server_metrics(endpoint: str, timeout: float) -> dict[str, Any] | None:
    parsed = urlparse(endpoint)
    metrics_url = urlunparse(parsed._replace(path="/metrics/multimodal", query="", fragment=""))
    try:
        with urlopen(metrics_url, timeout=timeout) as response:
            return json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def send_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
            return {
                "ok": True,
                "status": response.status,
                "latency_sec": time.perf_counter() - started,
                "completion_chars": len(body.get("choices", [{}])[0].get("message", {}).get("content", "")),
            }
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "latency_sec": time.perf_counter() - started, "error": detail}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "status": None, "latency_sec": time.perf_counter() - started, "error": str(exc)}


def make_report(args: argparse.Namespace, results: list[dict[str, Any]], wall_time: float) -> dict[str, Any]:
    latencies = [result["latency_sec"] for result in results]
    successful = [result for result in results if result["ok"]]
    failed = [result for result in results if not result["ok"]]
    return {
        "config": {
            "server_url": args.server_url,
            "model": args.model,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "image_count": len(args.image),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "summary": {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "wall_time_sec": wall_time,
            "request_throughput_per_sec": len(successful) / wall_time if wall_time else 0.0,
            "latency_sec": latency_summary(latencies),
            "ttft_sec": None,
            "tpot_sec": None,
        },
        "errors": [result.get("error") for result in failed[:10]],
    }


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
    return {
        "avg": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


if __name__ == "__main__":
    main()
