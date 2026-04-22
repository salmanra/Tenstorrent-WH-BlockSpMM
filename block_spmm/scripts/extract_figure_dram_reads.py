#!/usr/bin/env python3
"""Extract DRAM-read test cases from the four dram_reads CSVs that correspond
to bars in the seven SC26 submission figures.

Outputs one row per unique (group, registry, test, host_code) — i.e. one row
per bar — with total in0_reads and in1_reads summed across all cores.

Usage:
    python scripts/extract_figure_dram_reads.py
    python scripts/extract_figure_dram_reads.py --out-dir scripts/figures_dram_reads
"""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Source CSV paths ──
DRAM_CSVS = {
    "load_imbalance":        SCRIPT_DIR / "dram_reads_load_imbalance.csv",
    "load_imbalance_upper":  SCRIPT_DIR / "dram_reads_load_imbalance_upper.csv",
    "load_imbalance_random": SCRIPT_DIR / "dram_reads_load_imbalance_random.csv",
    "multi_diag_ultra_sparse": SCRIPT_DIR / "dram_reads_multi_diag_ultra_sparse.csv",
    "mixed_sweep":           SCRIPT_DIR / "dram_reads_mixed_sweep.csv",
    "sweep_pattern":         SCRIPT_DIR / "dram_reads_sweep_pattern.csv",
}

# ── Figure definitions ──
# Each figure maps to a list of (source_csv_key, filter_dict) pairs.
# filter_dict keys match CSV columns; a value can be a single string or a list.

FIGURES = {
    # Fig 1: load_imbalance.png — DDA ± LB, lower-triangular, both sizes
    "load_imbalance": [
        ("load_imbalance", {"registry": "29", "host_code": ["CDA", "CDA_no_lb"]}),
    ],

    # Fig 2: load_imbalance_upper.png — DDA ± LB, upper-triangular, both sizes
    "load_imbalance_upper": [
        ("load_imbalance_upper", {"registry": "30", "host_code": ["CDA", "CDA_no_lb"]}),
    ],

    # Fig 3: load_imbalance_comparison.png — Naive/SnF/DDA (LB only), both triangles
    "load_imbalance_comparison": [
        ("load_imbalance",       {"registry": "29", "host_code": ["Naive", "SnF", "CDA"]}),
        ("load_imbalance_upper", {"registry": "30", "host_code": ["Naive", "SnF", "CDA"]}),
    ],

    # Fig 4: mixed_sweep_d50.png — 3 algos, PatternD50, row + multi_diag
    "mixed_sweep_d50": [
        ("mixed_sweep", {"registry": "5", "host_code": [
            "Naive_row_R256_d50", "SnF_row_R256_d50", "CDA_row_R256_d50",
            "Naive_multi_diag_R256_d50", "SnF_multi_diag_R256_d50", "CDA_multi_diag_R256_d50",
        ]}),
    ],

    # Fig 5: mixed_sweep_d06.png — 3 algos, PatternUltra64_6000, row + random
    "mixed_sweep_d06": [
        ("mixed_sweep", {"registry": "27", "host_code": [
            "Naive_row_R64_d0.6", "SnF_row_R64_d0.6", "CDA_row_R64_d0.6",
            "Naive_random_R64_d0.6", "SnF_random_R64_d0.6", "CDA_random_R64_d0.6",
        ]}),
    ],

    # Fig 8: load_imbalance_random.png — LB vs no-LB, random 25%, all 3 algos
    "load_imbalance_random": [
        ("load_imbalance_random", {"registry": "4", "host_code": [
            "Naive", "Naive_no_lb", "SnF", "SnF_no_lb", "CDA", "CDA_no_lb",
        ]}),
    ],

    # Fig 9: multi_diag_ultra_sparse.png — 3 algos, multi_diag, R=C=64, d=0.6%
    "multi_diag_ultra_sparse": [
        ("multi_diag_ultra_sparse", {"registry": "27", "host_code": ["Naive", "SnF", "CDA"]}),
    ],

    # Fig 6: sweep_pattern_dda_vs_gpu_ultrasparse.png — CDA only, R=C=32 and R=C=64
    "sweep_pattern_ultrasparse": [
        ("sweep_pattern", {"registry": [str(r) for r in range(17, 22)]}),   # 32×32
        ("sweep_pattern", {"registry": [str(r) for r in range(23, 29)]}),   # 64×64
    ],

    # Fig 7: sweep_pattern_dda_vs_gpu_standard.png — CDA only, R=C=128 and R=C=256
    "sweep_pattern_standard": [
        ("sweep_pattern", {"registry": [str(r) for r in range(13, 17)]}),   # 128×128
        ("sweep_pattern", {"registry": [str(r) for r in range(2, 6)]}),     # 256×256
    ],
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def row_matches(row, filters):
    for col, val in filters.items():
        if isinstance(val, list):
            if row[col] not in val:
                return False
        else:
            if row[col] != val:
                return False
    return True


def aggregate_figure(specs, all_data):
    """Return one row per unique (group, registry, test, host_code),
    with in0_reads and in1_reads summed across cores."""
    agg = OrderedDict()  # preserves insertion order = source file order
    for source_key, filters in specs:
        for row in all_data[source_key]:
            if not row_matches(row, filters):
                continue
            key = (row["group"], row["registry"], row["test"], row["host_code"])
            if key not in agg:
                agg[key] = {"in0_reads": 0, "in1_reads": 0}
            agg[key]["in0_reads"] += int(row["in0_reads"])
            agg[key]["in1_reads"] += int(row["in1_reads"])

    rows = []
    for (group, registry, test, host_code), totals in agg.items():
        rows.append({
            "group": group,
            "registry": registry,
            "test": test,
            "host_code": host_code,
            "total_in0_reads": totals["in0_reads"],
            "total_in1_reads": totals["in1_reads"],
        })
    return rows


FIELDNAMES = ["group", "registry", "test", "host_code", "total_in0_reads", "total_in1_reads"]


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-bar DRAM read totals for each figure")
    parser.add_argument("--out-dir", default=None,
                        help="Write per-figure CSVs here (default: stdout)")
    args = parser.parse_args()

    all_data = {}
    for key, path in DRAM_CSVS.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            all_data[key] = []
        else:
            all_data[key] = load_csv(path)

    for fig_name, specs in FIGURES.items():
        rows = aggregate_figure(specs, all_data)

        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"dram_reads_{fig_name}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {out_path} ({len(rows)} rows)", file=sys.stderr)
        else:
            print(f"\n### {fig_name} ({len(rows)} rows) ###")
            writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
