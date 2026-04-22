#!/usr/bin/env python3
"""Plot load-imbalance experiment: 6 algorithm variants on an upper-triangular BSR matrix.

Usage:
    python scripts/plot_load_imbalance_upper.py
    python scripts/plot_load_imbalance_upper.py --data-dir /path/to/profiles_load_imbalance_upper_V2/csvs
"""

import argparse
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

REGISTRY = "UpperTriangular"

TEST_CASES = [
    {
        "name": "parametric_triu_M8192_N8192_K8192_R256_C256",
        "title": "Upper-Triangular BSR\n(M=N=K=8192, R=C=256, 528 blocks, iters_y=4)",
        "suffix": "8192",
    },
    {
        "name": "parametric_triu_M4096_N4096_K4096_R256_C256",
        "title": "Upper-Triangular BSR\n(M=N=K=4096, R=C=256, 136 blocks, iters_y=2)",
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


def plot_one(ax, data_dir, test_case):
    """Plot a single test case on the given axes."""
    test_name = test_case["name"]
    labels = []
    values = []
    colors = []
    for (algo_dir, label), color in zip(ALGORITHMS, COLORS):
        log_path = data_dir / REGISTRY / algo_dir / f"{test_name}_sparse.log"
        tflops = extract_tflops(log_path)
        if tflops is None:
            print(f"  WARNING: Missing {log_path}", file=sys.stderr)
            tflops = 0.0
        labels.append(label)
        values.append(tflops)
        colors.append(color)

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
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)


def main():
    parser = argparse.ArgumentParser(description="Plot upper-triangular load-imbalance experiment")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_load_imbalance_upper_V2/csvs",
        help="Root directory containing registry/host_code/ CSV files",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Combined figure with side-by-side axes
    fig, axes = plt.subplots(1, len(TEST_CASES), figsize=(7 * len(TEST_CASES), 5))
    if len(TEST_CASES) == 1:
        axes = [axes]

    for ax, tc in zip(axes, TEST_CASES):
        plot_one(ax, data_dir, tc)

    fig.suptitle("Load Imbalance: LB vs No-LB on Upper-Triangular Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = figures_dir / "load_imbalance_upper.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
