#!/usr/bin/env python3
"""Plot throughput bar charts for the 4 mixed-sweep test cases.

Produces 4 separate bar charts (one per test case), each comparing the
3 base algorithms.

Usage:
    python scripts/plot_mixed_sweep.py
    python scripts/plot_mixed_sweep.py --data-dir /path/to/profiles_mixed_sweep/csvs
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = [
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_snfin0_cdain1",
]

ALGO_LABELS = ["Naive", "SnF", "CDA"]

COLOR_BEST = "#70AD47"
COLOR_REST = "#A0A0A0"

CASES = [
    {
        "registry": "PatternD50",
        "test_name": "parametric_row_M8192_N8192_K8192_R256_C256_dppm500000",
        "title": "Row, R=C=256, d=50%",
    },
    {
        "registry": "PatternD50",
        "test_name": "parametric_multi_diag_M8192_N8192_K8192_R256_C256_dppm500000",
        "title": "Multi-Diag, R=C=256, d=50%",
    },
    {
        "registry": "PatternUltra64_6000",
        "test_name": "parametric_row_M8192_N8192_K8192_R64_C64_dppm6000",
        "title": "Row, R=C=64, d=0.6%",
    },
    {
        "registry": "PatternUltra64_6000",
        "test_name": "parametric_M8192_N8192_K8192_R64_C64_dppm6000",
        "title": "Random, R=C=64, d=0.6%",
    },
]


def extract_tflops(log_path):
    if not log_path.exists():
        return None
    try:
        content = log_path.read_text()
    except Exception as e:
        print(f"  WARNING: Could not read {log_path}: {e}", file=sys.stderr)
        return None
    match = re.search(r"(?:Device\s+)?TFLOP/?s:\s*([\d.]+)", content, re.IGNORECASE)
    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description="Plot mixed-sweep throughput bar charts")
    parser.add_argument(
        "--data-dir",
        default="${TT_METAL_HOME}/profiles_mixed_sweep/csvs",
        help="Root directory containing registry/host_code/ CSV files",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    groups = [
        ("d=50%, R=C=256", CASES[:2], "mixed_sweep_d50.png"),
        ("d=0.6%, R=C=64", CASES[2:], "mixed_sweep_d06.png"),
    ]

    for group_title, group_cases, out_name in groups:
        fig, axes = plt.subplots(1, len(group_cases), figsize=(8 * len(group_cases) // 2 + 4, 4.5))
        if len(group_cases) == 1:
            axes = [axes]

        # Collect all values first to determine shared y-limit
        all_values = []
        for case in group_cases:
            for algo in ALGORITHMS:
                log_path = data_dir / case["registry"] / algo / f"{case['test_name']}_sparse.log"
                tflops = extract_tflops(log_path)
                if tflops is None:
                    tflops = 0.0
                all_values.append(tflops)
        shared_ylim = max(all_values) * 1.25 if max(all_values) > 0 else 1

        for case_idx, case in enumerate(group_cases):
            ax = axes[case_idx]
            values = []
            for algo in ALGORITHMS:
                log_path = data_dir / case["registry"] / algo / f"{case['test_name']}_sparse.log"
                tflops = extract_tflops(log_path)
                if tflops is None:
                    print(f"  WARNING: Missing {log_path}", file=sys.stderr)
                    tflops = 0.0
                values.append(tflops)

            x = np.arange(len(ALGORITHMS))
            best_idx = int(np.argmax(values))
            colors = [COLOR_BEST if i == best_idx else COLOR_REST for i in range(len(values))]
            bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(ALGO_LABELS, fontsize=9)
            ax.set_title(case["title"], fontsize=11)
            ax.set_ylabel("TFLOP/s" if case_idx == 0 else "", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_ylim(0, shared_ylim)

        fig.suptitle(f"Mixed Sweep: {group_title}", fontsize=13, y=1.02)
        fig.tight_layout()

        output_path = figures_dir / out_name
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
