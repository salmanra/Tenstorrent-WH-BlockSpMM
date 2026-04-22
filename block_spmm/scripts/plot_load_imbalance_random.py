#!/usr/bin/env python3
"""Plot LB vs No-LB on random 25% density matrices for Naive, SnF, and DDA.

Bars show host-side milliseconds per iteration, extracted from the Tracy
"Device program Loop" zone in the host CSV.

Usage:
    python scripts/plot_load_imbalance_random.py
    python scripts/plot_load_imbalance_random.py --data-dir /path/to/csvs
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (lb_dir, no_lb_dir, label)
ALGORITHMS = [
    ("bsr_spmm_multicore_load_balanced_new_DM",
     "bsr_spmm_multicore_load_balanced_new_DM_no_lb", "Naive"),
    ("bsr_spmm_multicore_snf",
     "bsr_spmm_multicore_snf_no_lb", "SnF"),
    ("bsr_spmm_multicore_snfin0_cdain1",
     "bsr_spmm_multicore_snfin0_cdain1_no_lb", "DDA"),
]

REGISTRY = "PatternD25"
TEST_NAME = "parametric_M8192_N8192_K8192_R256_C256_dppm250000"
HOST_LOOP_ITERATIONS = 10

# Paired colors: solid for LB, lighter for no-LB
COLORS_LB    = ["#4472C4", "#ED7D31", "#70AD47"]
COLORS_NO_LB = ["#8FAADC", "#F4B183", "#A9D18E"]


def extract_host_ms(host_csv_path):
    """Extract per-iteration host ms from 'Device program Loop' zone."""
    if not host_csv_path.exists():
        return None
    with open(host_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Device program Loop" in row.get("name", ""):
                total_ns = float(row["total_ns"])
                return (total_ns / HOST_LOOP_ITERATIONS) / 1e6  # ms
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot LB vs No-LB on random 25% density matrices (runtime ms)")
    parser.add_argument(
        "--data-dir",
        default="/home/user/tt-metal/profiles_load_imbalance_random/csvs",
        help="Root CSV directory",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    n_algos = len(ALGORITHMS)
    bar_width = 0.3
    x = np.arange(n_algos)

    lb_vals = []
    no_lb_vals = []
    for lb_dir, no_lb_dir, _ in ALGORITHMS:
        lb_csv = data_dir / REGISTRY / lb_dir / f"{TEST_NAME}.csv"
        no_lb_csv = data_dir / REGISTRY / no_lb_dir / f"{TEST_NAME}.csv"

        lb_ms = extract_host_ms(lb_csv)
        no_lb_ms = extract_host_ms(no_lb_csv)
        if lb_ms is None:
            print(f"  WARNING: Missing {lb_csv}", file=sys.stderr)
            lb_ms = 0.0
        if no_lb_ms is None:
            print(f"  WARNING: Missing {no_lb_csv}", file=sys.stderr)
            no_lb_ms = 0.0
        lb_vals.append(lb_ms)
        no_lb_vals.append(no_lb_ms)

    algo_labels = [label for _, _, label in ALGORITHMS]

    # ── Speedup ratio plot (no-LB ms / LB ms, baseline = 1.0 = no-LB) ──
    # Ratio > 1 means LB is faster (takes fewer ms).
    ratios = []
    for lb_v, no_lb_v in zip(lb_vals, no_lb_vals):
        ratios.append(no_lb_v / lb_v if lb_v > 0 else 0.0)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars_r = ax2.bar(x, ratios, width=0.5, color=COLORS_LB,
                     edgecolor="white", linewidth=0.5)

    for bar, ratio in zip(bars_r, ratios):
        if ratio > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, ratio + 0.01,
                     f"{ratio:.2f}x", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="No-LB baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algo_labels, fontsize=11)
    ax2.set_ylabel("Speedup (No-LB time / LB time)", fontsize=11)
    ax2.set_title("LB Speedup over No-LB: Random 25% Density\n(M=N=K=8192, R=C=256)",
                  fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(ratios) * 1.25 if max(ratios) > 0 else 1.5)
    ax2.legend(fontsize=9)

    fig2.tight_layout()
    output_path2 = figures_dir / "load_imbalance_random_speedup.png"
    fig2.savefig(output_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {output_path2}")


if __name__ == "__main__":
    main()
