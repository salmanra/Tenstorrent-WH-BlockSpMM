#!/usr/bin/env python3
"""Plot Naive vs SnF vs CDA on multi_diag ultra-sparse (R=C=64, d=0.6%).

Usage:
    python scripts/plot_multi_diag_ultra_sparse.py
    python scripts/plot_multi_diag_ultra_sparse.py --data-dir /path/to/csvs
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
    "bsr_spmm_multicore_load_balanced_new_DM",
    "bsr_spmm_multicore_snf",
    "bsr_spmm_multicore_snfin0_cdain1",
]

ALGO_LABELS = ["Naive", "SnF", "CDA"]

COLOR_BEST = "#70AD47"
COLOR_REST = "#A0A0A0"

REGISTRY = "PatternUltra64_6000"
TEST_NAME = "parametric_multi_diag_M8192_N8192_K8192_R64_C64_dppm6000"


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
    parser = argparse.ArgumentParser(
        description="Plot multi_diag ultra-sparse throughput")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_multi_diag_ultra_sparse/csvs",
        help="Root CSV directory",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    values = []
    for algo in ALGORITHMS:
        log_path = data_dir / REGISTRY / algo / f"{TEST_NAME}_sparse.log"
        tflops = extract_tflops(log_path)
        if tflops is None:
            print(f"  WARNING: Missing {log_path}", file=sys.stderr)
            tflops = 0.0
        values.append(tflops)

    x = np.arange(len(ALGORITHMS))
    best_idx = int(np.argmax(values))
    colors = [COLOR_BEST if i == best_idx else COLOR_REST for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(6, 4.5))
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
    ax.set_xticklabels(ALGO_LABELS, fontsize=10)
    ax.set_ylabel("TFLOP/s", fontsize=11)
    ax.set_title("Multi-Diag, R=C=64, d=0.6%", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)

    fig.tight_layout()
    output_path = figures_dir / "multi_diag_ultra_sparse.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
