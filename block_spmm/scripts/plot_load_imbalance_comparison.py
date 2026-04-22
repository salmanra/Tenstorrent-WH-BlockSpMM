#!/usr/bin/env python3
"""Compare Naive, SnF, and DDA (load-balanced) on lower- vs upper-triangular BSR matrices.

Usage:
    python scripts/plot_load_imbalance_comparison.py
    python scripts/plot_load_imbalance_comparison.py \
        --lower-dir /path/to/lower/csvs --upper-dir /path/to/upper/csvs
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
    ("bsr_spmm_multicore_load_balanced_new_DM", "Naive"),
    ("bsr_spmm_multicore_snf", "SnF"),
    ("bsr_spmm_multicore_snfin0_cdain1", "DDA"),
]

LOWER_REGISTRY = "Triangular"
UPPER_REGISTRY = "UpperTriangular"

TEST_CASES = [
    {
        "lower_name": "parametric_tril_M8192_N8192_K8192_R256_C256",
        "upper_name": "parametric_triu_M8192_N8192_K8192_R256_C256",
        "title": "M=N=K=8192, R=C=256",
        "suffix": "8192",
    },
    {
        "lower_name": "parametric_tril_M4096_N4096_K4096_R256_C256",
        "upper_name": "parametric_triu_M4096_N4096_K4096_R256_C256",
        "title": "M=N=K=4096, R=C=256",
        "suffix": "4096",
    },
]

TRI_COLORS = {"lower": "#4472C4", "upper": "#ED7D31"}


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


def collect_values(lower_dir, upper_dir, test_case):
    lower_vals = []
    upper_vals = []
    for algo_dir, _ in ALGORITHMS:
        lower_path = lower_dir / LOWER_REGISTRY / algo_dir / f"{test_case['lower_name']}_sparse.log"
        upper_path = upper_dir / UPPER_REGISTRY / algo_dir / f"{test_case['upper_name']}_sparse.log"

        lt = extract_tflops(lower_path)
        ut = extract_tflops(upper_path)
        if lt is None:
            print(f"  WARNING: Missing {lower_path}", file=sys.stderr)
            lt = 0.0
        if ut is None:
            print(f"  WARNING: Missing {upper_path}", file=sys.stderr)
            ut = 0.0
        lower_vals.append(lt)
        upper_vals.append(ut)
    return lower_vals, upper_vals


def plot_one(ax, lower_vals, upper_vals, test_case, ylim):
    """Plot one test-case: grouped bars (lower/upper) for each algorithm."""
    n_algos = len(ALGORITHMS)
    bar_width = 0.3
    x = np.arange(n_algos)

    bars_lo = ax.bar(x - bar_width / 2, lower_vals, bar_width,
                     color=TRI_COLORS["lower"], edgecolor="white", linewidth=0.5,
                     label="Lower-Triangular")
    bars_up = ax.bar(x + bar_width / 2, upper_vals, bar_width,
                     color=TRI_COLORS["upper"], edgecolor="white", linewidth=0.5,
                     label="Upper-Triangular")

    for bars in (bars_lo, bars_up):
        for bar in bars:
            val = bar.get_height()
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

    algo_labels = [label for _, label in ALGORITHMS]
    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels, fontsize=11)
    ax.set_ylabel("TFLOP/s", fontsize=11)
    ax.set_title(test_case["title"], fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ylim)
    ax.legend(fontsize=9)


def main():
    parser = argparse.ArgumentParser(
        description="Compare algorithms on lower- vs upper-triangular matrices")
    parser.add_argument(
        "--lower-dir",
        default="/home/user/tt-metal/profiles_load_imbalance_V2/csvs",
        help="Root CSV directory for lower-triangular experiments",
    )
    parser.add_argument(
        "--upper-dir",
        default="/home/user/tt-metal/profiles_load_imbalance_upper_V2/csvs",
        help="Root CSV directory for upper-triangular experiments",
    )
    args = parser.parse_args()
    lower_dir = Path(args.lower_dir)
    upper_dir = Path(args.upper_dir)

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Pre-collect all values to compute a shared y-limit across both subplots
    case_values = [collect_values(lower_dir, upper_dir, tc) for tc in TEST_CASES]
    global_max = max(
        (max(lo + up) for lo, up in case_values),
        default=0,
    )
    shared_ylim = global_max * 1.25 if global_max > 0 else 1

    # Combined figure
    fig, axes = plt.subplots(1, len(TEST_CASES), figsize=(7 * len(TEST_CASES), 5))
    if len(TEST_CASES) == 1:
        axes = [axes]

    for ax, tc, (lo, up) in zip(axes, TEST_CASES, case_values):
        plot_one(ax, lo, up, tc, shared_ylim)

    fig.suptitle("Lower- vs Upper-Triangular: Naive, SnF, DDA (Load-Balanced)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = figures_dir / "load_imbalance_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
