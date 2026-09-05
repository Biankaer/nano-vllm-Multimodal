#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanovllm.benchmark.paged_kv import (
    atomic_write_json,
    atomic_write_text,
    benchmark_case,
    build_report,
    expand_benchmark_matrix,
    parse_case_selector,
    prepare_case,
    render_markdown,
)


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return parsed


def _integers(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if min(parsed) <= 0:
        raise argparse.ArgumentTypeError("integer matrix values must be positive")
    return parsed


def _kv_shapes(value: str) -> tuple[tuple[int, int], ...]:
    try:
        parsed = tuple(
            tuple(int(part) for part in item.lower().split("x", 1))
            for item in _csv(value)
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("KV shapes must use HEADSxDIM") from exc
    if any(len(shape) != 2 or min(shape) <= 0 for shape in parsed):
        raise argparse.ArgumentTypeError("KV shapes must use positive HEADSxDIM")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch, Triton, and C++/CUDA Paged KV Store/Gather",
    )
    parser.add_argument("--backends", type=_csv, default=_csv("pytorch,triton,cuda"))
    parser.add_argument("--operations", type=_csv, default=_csv("store,gather"))
    parser.add_argument("--dtypes", type=_csv, default=_csv("float16,bfloat16"))
    parser.add_argument("--tokens", type=_integers, default=_integers("1,8,257"))
    parser.add_argument("--kv-shapes", type=_kv_shapes, default=_kv_shapes("1x64,2x128,8x128"))
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--mapping",
        choices=("contiguous", "random", "random_with_skips"),
        default="random_with_skips",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-selector")
    parser.add_argument(
        "--profile-replay",
        type=int,
        default=0,
        help="replay one --case-selector without CUDA events for Nsight Compute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/paged-kv"),
    )
    return parser


def _resolve_cases(args: argparse.Namespace):
    if args.case_selector:
        return (parse_case_selector(args.case_selector),)
    unknown_dtypes = [name for name in args.dtypes if name not in _DTYPES]
    if unknown_dtypes:
        raise ValueError(f"unsupported dtypes: {', '.join(unknown_dtypes)}")
    return expand_benchmark_matrix(
        backends=args.backends,
        operations=args.operations,
        dtypes=tuple(_DTYPES[name] for name in args.dtypes),
        token_counts=args.tokens,
        kv_shapes=args.kv_shapes,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        mapping=args.mapping,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Paged KV benchmark requires a visible CUDA GPU")
    cases = _resolve_cases(args)
    if args.profile_replay:
        if not args.case_selector:
            raise ValueError("--profile-replay requires exactly one --case-selector")
        if args.profile_replay < 0:
            raise ValueError("--profile-replay must be non-negative")
        prepared = prepare_case(cases[0], device=args.device)
        prepared.assert_correct()
        for _ in range(args.profile_replay):
            prepared.invoke()
        torch.cuda.synchronize(args.device)
        print(f"profile replay complete: {cases[0].selector}")
        return 0

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.selector}", flush=True)
        results.append(
            benchmark_case(
                case,
                warmup_iterations=args.warmup,
                iterations=args.iterations,
                device=args.device,
            )
        )
    report = build_report(results, device=args.device)
    json_path = args.output_dir / "paged-kv-microbenchmark.json"
    markdown_path = args.output_dir / "paged-kv-microbenchmark.md"
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, render_markdown(report))
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
