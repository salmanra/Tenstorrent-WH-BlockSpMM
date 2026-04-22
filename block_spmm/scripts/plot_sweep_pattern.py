#!/usr/bin/env python3
"""DDA vs GPU throughput by sparsity pattern, density, and block size."""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

DDA_ROOT = "${TT_METAL_HOME}/profiles_sc26_april5/csvs"
GPU_DIR = "${TT_METAL_HOME}/tt_metal/programming_examples/block_sparse/gpu-normalized"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
DDA_HC = "bsr_spmm_multicore_snfin0_cdain1"

PATTERNS = ["row", "random", "multi_diag", "col"]
PATTERN_LABELS = ["Row", "Random", "Banded", "Col"]
PATTERN_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#2ca02c"]

# Block size → (GPU csv filename, list of (density_label, registry_name))
AXES = [
    (
        "32×32",
        "sweep_pattern_32.csv",
        [
            ("0.003%", "PatternUltra32_30"),
            ("0.03%", "PatternUltra32_300"),
            ("0.3%", "PatternUltra32_3000"),
        ],
    ),
    (
        "64×64",
        "sweep_pattern_64.csv",
        [
            ("0.006%", "PatternUltra64_60"),
            ("0.06%", "PatternUltra64_600"),
            ("0.6%", "PatternUltra64_6000"),
            ("1%", "PatternUltra64_10000"),
        ],
    ),
    (
        "128×128",
        "sweep_pattern_128.csv",
        [
            ("5%", "PatternD5_128"),
            ("10%", "PatternD10_128"),
            ("25%", "PatternD25_128"),
            ("50%", "PatternD50_128"),
        ],
    ),
    (
        "256×256",
        "sweep_pattern_256.csv",
        [
            ("5%", "PatternD5"),
            ("10%", "PatternD10"),
            ("25%", "PatternD25"),
            ("50%", "PatternD50"),
        ],
    ),
]


def classify_pattern(case_name):
    if "_row_" in case_name:
        return "row"
    elif "_col_" in case_name:
        return "col"
    elif "_multi_diag_" in case_name:
        return "multi_diag"
    else:
        return "random"


def parse_dda_tflops(registry_name):
    """Return {pattern: tflops} from DDA sparse logs."""
    d = os.path.join(DDA_ROOT, registry_name, DDA_HC)
    result = {}
    if not os.path.isdir(d):
        return result
    for f in os.listdir(d):
        if not f.endswith("_sparse.log"):
            continue
        pat = classify_pattern(f)
        path = os.path.join(d, f)
        with open(path) as fh:
            for line in fh:
                m = re.search(r"Device TFLOP/s:\s+([\d.]+)", line)
                if m:
                    result[pat] = float(m.group(1))
    return result


def parse_gpu_csv(gpu_csv_path):
    """Return {(registry_idx, pattern): tflops} from GPU CSV."""
    result = {}
    with open(gpu_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg = int(row["Registry"])
            pat = classify_pattern(row["Case"])
            result[(reg, pat)] = float(row["Avg_TFLOPs"])
    return result


def registry_to_gpu_index(registry_name):
    """Map registry name to the index used in the GPU CSV Registry column."""
    # The GPU CSVs use the profiling_suite.hpp registry indices
    name_to_idx = {
        "PatternD5": 2,
        "PatternD10": 3,
        "PatternD25": 4,
        "PatternD50": 5,
        "PatternD5_128": 13,
        "PatternD10_128": 14,
        "PatternD25_128": 15,
        "PatternD50_128": 16,
        "PatternUltra32_30": 17,
        "PatternUltra32_100": 18,
        "PatternUltra32_300": 19,
        "PatternUltra32_1000": 20,
        "PatternUltra32_3000": 21,
        "PatternUltra32_10000": 22,
        "PatternUltra64_60": 23,
        "PatternUltra64_200": 24,
        "PatternUltra64_600": 25,
        "PatternUltra64_2000": 26,
        "PatternUltra64_6000": 27,
        "PatternUltra64_10000": 28,
    }
    return name_to_idx.get(registry_name)


def plot_figure(axes_specs, title, out_name):
    """Group by pattern; bars touch within each group; densities labelled below."""
    n_axes = len(axes_specs)

    BAR_WIDTH = 1.0  # bars touch (width = unit step within group)
    PATTERN_GAP = 4.0  # gap of 4 bar-widths between adjacent pattern groups

    # Width is set by the WIDEST single axis (axes are stacked vertically)
    units_per_axis = []
    for _, _, densities in axes_specs:
        n_d = len(densities)
        units_per_axis.append(len(PATTERNS) * (2 * n_d) + (len(PATTERNS) - 1) * PATTERN_GAP)
    fig_width = max(14.0, max(units_per_axis) * 0.32 + 3.0)
    fig_height = 7.5 * n_axes + 1.5  # ~7.5" per axis + room for suptitle/margins
    fig, axes = plt.subplots(n_axes, 1, figsize=(fig_width, fig_height))
    if n_axes == 1:
        axes = [axes]

    for ax_idx, (block_label, gpu_csv_name, densities) in enumerate(axes_specs):
        ax = axes[ax_idx]
        gpu_data = parse_gpu_csv(os.path.join(GPU_DIR, gpu_csv_name))

        n_densities = len(densities)
        bars_per_group = n_densities * 2
        pattern_spacing = bars_per_group * BAR_WIDTH + PATTERN_GAP

        for p_idx, pat in enumerate(PATTERNS):
            group_left = p_idx * pattern_spacing

            for d_idx, (density_label, registry_name) in enumerate(densities):
                dda_data = parse_dda_tflops(registry_name)
                gpu_reg_idx = registry_to_gpu_index(registry_name)

                dda_x = group_left + (2 * d_idx + 0.5) * BAR_WIDTH
                gpu_x = group_left + (2 * d_idx + 1.5) * BAR_WIDTH

                dda_val = dda_data.get(pat, 0)
                gpu_val = gpu_data.get((gpu_reg_idx, pat), 0) if gpu_reg_idx is not None else 0

                # DDA: white fill, dotted hatch
                ax.bar(
                    dda_x,
                    dda_val,
                    BAR_WIDTH,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    hatch=".",
                    label="DDA" if (p_idx == 0 and d_idx == 0) else None,
                )
                # GPU: white fill, diagonal hatch
                ax.bar(
                    gpu_x,
                    gpu_val,
                    BAR_WIDTH,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    hatch="////",
                    label="GPU" if (p_idx == 0 and d_idx == 0) else None,
                )

                # Density label southwest from the pair center, anchored at top-right
                pair_center = group_left + (2 * d_idx + 1) * BAR_WIDTH
                ax.text(
                    pair_center,
                    -0.01,
                    density_label,
                    ha="right",
                    va="top",
                    rotation=45,
                    rotation_mode="anchor",
                    fontsize=24,
                    transform=ax.get_xaxis_transform(),
                )

        # Pattern labels manually placed just below the density labels
        ax.set_xticks([])
        for p_idx, pattern_label in enumerate(PATTERN_LABELS):
            group_center = p_idx * pattern_spacing + bars_per_group * BAR_WIDTH / 2
            ax.text(
                group_center,
                -0.28,
                pattern_label,
                ha="center",
                va="top",
                fontsize=32,
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

        ax.set_title(f"R=C={block_label}", fontsize=36)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=28)

        ax.set_ylabel("TFLOP/s", fontsize=34)

        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=28, framealpha=0.9)
        ax.set_xlim(-BAR_WIDTH, len(PATTERNS) * pattern_spacing - PATTERN_GAP + BAR_WIDTH)

    fig.subplots_adjust(top=0.88, bottom=0.06, hspace=0.55)
    fig.suptitle(title, fontsize=36, y=0.96)

    out_path = os.path.join(OUT_DIR, out_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_figure(
        AXES[:2],
        "DDA (N150) vs GPU: Sparse (R=C=32, R=C=64)",
        "sweep_pattern_dda_vs_gpu_ultrasparse.png",
    )
    plot_figure(
        AXES[2:],
        "DDA (N150) vs GPU: Semi-Sparse (R=C=128, R=C=256)",
        "sweep_pattern_dda_vs_gpu_standard.png",
    )
