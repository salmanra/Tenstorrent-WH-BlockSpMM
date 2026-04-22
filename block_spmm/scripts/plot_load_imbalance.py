#!/usr/bin/env python3
"""Plot load-imbalance experiment: 4 algorithms on a triangular BSR matrix.

Usage:
    python scripts/plot_load_imbalance.py
    python scripts/plot_load_imbalance.py --data-dir /path/to/profiles_load_imbalance/csvs
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
    ("bsr_spmm_multicore_snfin0_cdain1", "DDA"),
    ("bsr_spmm_multicore_snfin0_cdain1_no_lb", "DDA\n(no LB)"),
]

COLORS = ["#70AD47", "#A9D18E"]

REGISTRY = "Triangular"

TEST_CASES = [
    {
        "name": "parametric_tril_M8192_N8192_K8192_R256_C256",
        "title": "Lower-Triangular BSR\n(M=N=K=8192, R=C=256, 528 blocks, iters_y=4)",
        "suffix": "8192",
    },
    {
        "name": "parametric_tril_M4096_N4096_K4096_R256_C256",
        "title": "Lower-Triangular BSR\n(M=N=K=4096, R=C=256, 136 blocks, iters_y=2)",
        "suffix": "4096",
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


def collect_values(data_dir, test_case):
    test_name = test_case["name"]
    values = []
    for algo_dir, _ in ALGORITHMS:
        log_path = data_dir / REGISTRY / algo_dir / f"{test_name}_sparse.log"
        tflops = extract_tflops(log_path)
        if tflops is None:
            print(f"  WARNING: Missing {log_path}", file=sys.stderr)
            tflops = 0.0
        values.append(tflops)
    return values


def plot_one(ax, values, test_case, ylim):
    """Plot a single test case on the given axes."""
    labels = [label for _, label in ALGORITHMS]
    colors = list(COLORS)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("TFLOP/s", fontsize=11)
    ax.set_title(test_case["title"], fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ylim)


def main():
    parser = argparse.ArgumentParser(description="Plot load-imbalance experiment")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_load_imbalance_V2/csvs",
        help="Root directory containing registry/host_code/ CSV files",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Pre-collect all values to compute a shared y-limit across both subplots
    case_values = [collect_values(data_dir, tc) for tc in TEST_CASES]
    global_max = max((max(v) for v in case_values), default=0)
    shared_ylim = global_max * 1.25 if global_max > 0 else 1

    # Combined figure with side-by-side axes
    fig, axes = plt.subplots(1, len(TEST_CASES), figsize=(7 * len(TEST_CASES), 5))
    if len(TEST_CASES) == 1:
        axes = [axes]

    for ax, tc, vals in zip(axes, TEST_CASES, case_values):
        plot_one(ax, vals, tc, shared_ylim)

    fig.suptitle("Load Imbalance: LB vs No-LB on Triangular Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = figures_dir / "load_imbalance.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
