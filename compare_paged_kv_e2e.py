#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanovllm.benchmark.paged_kv import atomic_write_json, atomic_write_text
from nanovllm.benchmark.paged_kv_e2e import (
    compare_paged_kv_reports,
    render_paged_kv_e2e_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exact-token Qwen3-VL runs across Paged KV backends",
    )
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/paged-kv/e2e-comparison"),
    )
    return parser


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_paged_kv_reports(
        {
            "pytorch": _load(args.pytorch),
            "triton": _load(args.triton),
            "cuda": _load(args.cuda),
        }
    )
    json_path = args.output_dir / "paged-kv-e2e-comparison.json"
    markdown_path = args.output_dir / "paged-kv-e2e-comparison.md"
    atomic_write_json(json_path, result)
    atomic_write_text(markdown_path, render_paged_kv_e2e_markdown(result))
    print(markdown_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
