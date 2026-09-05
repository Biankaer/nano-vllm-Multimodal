from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable


NCU_REQUIRED_SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "WarpStateStats",
    "LaunchStats",
)
NCU_KERNEL_REGEX = (
    "regex:.*(store_kernel|gather_kernel|_store_paged_kv_kernel|"
    "_gather_paged_kv_kernel).*"
)


def build_ncu_collection_command(
    *,
    ncu: str,
    python: str,
    benchmark_script: Path,
    case_selector: str,
    report_path: Path,
    replay_iterations: int,
    launch_count: int,
) -> list[str]:
    if replay_iterations <= 0 or launch_count <= 0:
        raise ValueError("NCU replay_iterations and launch_count must be positive")
    command = [
        ncu,
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        NCU_KERNEL_REGEX,
        "--launch-skip",
        "1",
        "--launch-count",
        str(launch_count),
        "--force-overwrite",
        "--export",
        str(report_path),
    ]
    for section in NCU_REQUIRED_SECTIONS:
        command.extend(("--section", section))
    command.extend(
        (
            python,
            str(benchmark_script),
            "--case-selector",
            case_selector,
            "--profile-replay",
            str(replay_iterations),
        )
    )
    return command


def build_ncu_import_command(*, ncu: str, report_path: Path) -> list[str]:
    return [ncu, "--import", str(report_path), "--page", "raw", "--csv"]


def _metric_header_offset(text: str) -> int:
    for offset, line in enumerate(text.splitlines()):
        if "Metric Name" in line and "Kernel Name" in line:
            return offset
    raise ValueError("Nsight Compute CSV does not contain a Metric Name header")


def _numeric(value: str) -> float | None:
    normalized = value.strip().replace(",", "")
    if not normalized or normalized in {"n/a", "N/A", "-"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_ncu_csv(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    offset = _metric_header_offset(text)
    reader = csv.DictReader(io.StringIO("\n".join(lines[offset:])))
    kernels: dict[str, dict[str, Any]] = {}
    rows = 0
    for row in reader:
        kernel_name = (row.get("Kernel Name") or "").strip()
        metric_name = (row.get("Metric Name") or "").strip()
        if not kernel_name or not metric_name:
            continue
        rows += 1
        kernel = kernels.setdefault(kernel_name, {"metrics": {}, "sections": []})
        section = (row.get("Section Name") or "").strip()
        if section and section not in kernel["sections"]:
            kernel["sections"].append(section)
        kernel["metrics"][metric_name] = {
            "value": _numeric(row.get("Metric Value") or ""),
            "unit": (row.get("Metric Unit") or "").strip() or None,
            "raw": (row.get("Metric Value") or "").strip(),
        }
    if not rows:
        raise ValueError("Nsight Compute CSV contains no metric rows")
    return {"kernels": kernels, "metric_rows": rows}


def _first_metric(
    metrics: dict[str, dict[str, Any]], names: Iterable[str]
) -> float | None:
    for name in names:
        entry = metrics.get(name)
        if entry is not None and entry.get("value") is not None:
            return float(entry["value"])
    return None


_ANALYSIS_METRICS = {
    "dram_throughput_pct": (
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "achieved_occupancy_pct": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "registers_per_thread": ("launch__registers_per_thread",),
    "threads_per_block": ("launch__block_size",),
    "warp_execution_efficiency_pct": (
        "smsp__thread_inst_executed_per_inst_executed.pct",
    ),
    "eligible_warps_per_cycle": (
        "smsp__warps_eligible.avg.per_cycle_active",
    ),
    "issue_active_pct": (
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
    ),
    "global_load_sectors_per_request": (
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    ),
    "global_store_sectors_per_request": (
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
    ),
}


def analyze_ncu_metrics(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    analysis: dict[str, dict[str, Any]] = {}
    for kernel_name, kernel in parsed["kernels"].items():
        metrics = kernel["metrics"]
        values = {
            label: _first_metric(metrics, candidates)
            for label, candidates in _ANALYSIS_METRICS.items()
        }
        values["missing_metrics"] = [
            label for label, value in values.items() if value is None
        ]
        values["captured_metric_count"] = len(metrics)
        values["sections"] = list(kernel["sections"])
        analysis[kernel_name] = values
    return analysis
