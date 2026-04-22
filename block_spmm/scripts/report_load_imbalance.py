#!/usr/bin/env python3
"""Report host-side runtime and TFLOP/s for load-imbalance experiments.

Reads the host-side CSV (Tracy profiler output) and sparse log to report:
  - Host-side "Device program Loop" wall-clock time
  - Host-side TFLOP/s  (Total FLOPs / host loop time)
  - Device-side TFLOP/s (from the sparse log, for comparison)

Usage:
    python scripts/report_load_imbalance.py
    python scripts/report_load_imbalance.py --data-dir /path/to/profiles_load_imbalance/csvs
"""

import argparse
import csv
import re
import sys
from pathlib import Path


ALGORITHMS = [
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_load_balanced_new_DM_no_lb",
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_snf_no_lb",
    "bsr_spmm_multicore_snfin0_cdain1",
    "bsr_spmm_multicore_snfin0_cdain1_no_lb",
]

ALGO_LABELS = {
    "bsr_spmm_multicore_load_balanced_new_DM":       "Naive (LB)",
    "bsr_spmm_multicore_load_balanced_new_DM_no_lb": "Naive (no LB)",
    "bsr_spmm_multicore_snf":                        "SnF (LB)",
    "bsr_spmm_multicore_snf_no_lb":                  "SnF (no LB)",
    "bsr_spmm_multicore_snfin0_cdain1":              "CDA (LB)",
    "bsr_spmm_multicore_snfin0_cdain1_no_lb":        "CDA (no LB)",
}

TEST_CASES = [
    {
        "name": "parametric_tril_M8192_N8192_K8192_R256_C256",
        "label": "8192x8192, R=256 (iters_y=4)",
    },
    {
        "name": "parametric_tril_M4096_N4096_K4096_R256_C256",
        "label": "4096x4096, R=256 (iters_y=2)",
    },
]

REGISTRY = "Triangular"
HOST_LOOP_ITERATIONS = 10  # "Device program Loop" zone wraps this many EnqueueProgram calls


def get_host_loop_ns(host_csv: Path) -> float | None:
    """Extract 'Device program Loop' total_ns from the host CSV."""
    if not host_csv.exists():
        return None
    with open(host_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Device program Loop" in row.get("name", ""):
                return float(row["total_ns"])
    return None


def get_total_flops(sparse_log: Path) -> float | None:
    """Extract Total FLOPs from the sparse log."""
    if not sparse_log.exists():
        return None
    content = sparse_log.read_text()
    m = re.search(r"Total FLOPs:\s*([\d.eE+\-]+)", content)
    return float(m.group(1)) if m else None


def get_device_tflops(sparse_log: Path) -> float | None:
    """Extract Device TFLOP/s from the sparse log."""
    if not sparse_log.exists():
        return None
    content = sparse_log.read_text()
    m = re.search(r"Device TFLOP/s:\s*([\d.]+)", content)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Report host-side timing for load-imbalance experiments")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_load_imbalance/csvs",
        help="Root directory containing registry/host_code/ CSV files",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    for tc in TEST_CASES:
        test_name = tc["name"]
        print(f"{'=' * 80}")
        print(f"  {tc['label']}")
        print(f"{'=' * 80}")
        print(f"  {'Algorithm':<20s}  {'Host loop (ms)':>14s}  {'Host TFLOP/s':>13s}  {'Device TFLOP/s':>14s}")
        print(f"  {'-' * 20}  {'-' * 14}  {'-' * 13}  {'-' * 14}")

        for algo in ALGORITHMS:
            label = ALGO_LABELS[algo]
            algo_dir = data_dir / REGISTRY / algo

            host_csv = algo_dir / f"{test_name}.csv"
            sparse_log = algo_dir / f"{test_name}_sparse.log"

            host_ns = get_host_loop_ns(host_csv)
            total_flops = get_total_flops(sparse_log)
            device_tflops = get_device_tflops(sparse_log)

            if host_ns is not None:
                host_ns_per_run = host_ns / HOST_LOOP_ITERATIONS
                host_ms = host_ns_per_run / 1e6
                host_ms_str = f"{host_ms:>14.2f}"
            else:
                host_ns_per_run = None
                host_ms = None
                host_ms_str = f"{'N/A':>14s}"

            if host_ns_per_run is not None and total_flops is not None:
                host_tflops = total_flops / (host_ns_per_run / 1e9) / 1e12
                host_tflops_str = f"{host_tflops:>13.2f}"
            else:
                host_tflops = None
                host_tflops_str = f"{'N/A':>13s}"

            if device_tflops is not None:
                device_tflops_str = f"{device_tflops:>14.2f}"
            else:
                device_tflops_str = f"{'N/A':>14s}"

            print(f"  {label:<20s}  {host_ms_str}  {host_tflops_str}  {device_tflops_str}")

        print()


if __name__ == "__main__":
    main()
