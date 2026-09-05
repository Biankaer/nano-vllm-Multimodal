#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanovllm.benchmark.ncu import (
    NCU_REQUIRED_SECTIONS,
    analyze_ncu_metrics,
    build_ncu_collection_command,
    build_ncu_import_command,
    parse_ncu_csv,
)
from nanovllm.benchmark.paged_kv import atomic_write_json, atomic_write_text, parse_case_selector


def artifact_stem(case_selector: str) -> str:
    case = parse_case_selector(case_selector)
    return (
        f"{case.backend}-{case.operation}-{case.to_dict()['dtype']}-"
        f"t{case.num_tokens}-h{case.num_kv_heads}-d{case.head_dim}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Nsight Compute evidence for one Paged KV benchmark case",
    )
    parser.add_argument("--case-selector", required=True)
    parser.add_argument("--ncu", default=shutil.which("ncu") or "ncu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--replay-iterations", type=int, default=20)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profile_output/paged_kv"),
    )
    return parser


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]):
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case = parse_case_selector(args.case_selector)
    if args.replay_iterations <= 0 or args.launch_count <= 0:
        raise ValueError("replay iterations and launch count must be positive")
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(args.case_selector)
    report_path = output_dir / f"{stem}.ncu-rep"
    csv_path = output_dir / f"{stem}.csv"
    stderr_path = output_dir / f"{stem}.stderr.log"
    stdout_path = output_dir / f"{stem}.stdout.log"
    command_path = output_dir / f"{stem}.command.json"
    analysis_path = output_dir / f"{stem}.json"
    command = build_ncu_collection_command(
        ncu=args.ncu,
        python=args.python,
        benchmark_script=root / "bench_paged_kv.py",
        case_selector=args.case_selector,
        report_path=report_path,
        replay_iterations=args.replay_iterations,
        launch_count=args.launch_count,
    )
    atomic_write_json(
        command_path,
        {
            "cwd": str(root),
            "command": command,
            "sections": list(NCU_REQUIRED_SECTIONS),
        },
    )
    environment = dict(os.environ)
    environment.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    collected = _run(command, cwd=root, environment=environment)
    atomic_write_text(stdout_path, collected.stdout)
    atomic_write_text(stderr_path, collected.stderr)
    base: dict[str, Any] = {
        "schema_version": "nanoinfer.paged_kv_ncu.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case": case.to_dict(),
        "command": command,
        "sections": list(NCU_REQUIRED_SECTIONS),
        "collection_returncode": collected.returncode,
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "valid": False,
    }
    if collected.returncode != 0:
        base["error"] = "Nsight Compute collection failed; inspect stderr_path"
        base["stderr_path"] = str(stderr_path)
        atomic_write_json(analysis_path, base)
        raise RuntimeError(
            f"Nsight Compute failed with exit code {collected.returncode}; "
            f"see {stderr_path}"
        )
    imported = _run(
        build_ncu_import_command(ncu=args.ncu, report_path=report_path),
        cwd=root,
        environment=environment,
    )
    atomic_write_text(csv_path, imported.stdout)
    if imported.stderr:
        atomic_write_text(output_dir / f"{stem}.import.stderr.log", imported.stderr)
    if imported.returncode != 0:
        base["error"] = "Nsight Compute report import failed"
        base["import_returncode"] = imported.returncode
        atomic_write_json(analysis_path, base)
        raise RuntimeError(
            f"Nsight Compute import failed with exit code {imported.returncode}"
        )
    parsed = parse_ncu_csv(imported.stdout)
    base.update(
        {
            "valid": True,
            "parsed": parsed,
            "analysis": analyze_ncu_metrics(parsed),
        }
    )
    atomic_write_json(analysis_path, base)
    print(json.dumps({"analysis": str(analysis_path), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
