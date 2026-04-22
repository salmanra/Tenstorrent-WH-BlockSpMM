#!/usr/bin/env python3
"""Aggregate Fig. 3 or Fig. 4 DDA throughput into a single CSV.

Walks <output-dir>/csvs/<registry>/bsr_spmm_multicore_snfin0_cdain1/*_sparse.log,
extracts `Device TFLOP/s:` from each, and writes one row per (registry, pattern).
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

DDA_HC = "bsr_spmm_multicore_snfin0_cdain1"

# (registry_idx, registry_name, block_size, density_label)
FIG3_REGISTRIES = [
    (17, "PatternUltra32_30", 32, "0.003%"),
    (19, "PatternUltra32_300", 32, "0.03%"),
    (21, "PatternUltra32_3000", 32, "0.3%"),
    (23, "PatternUltra64_60", 64, "0.006%"),
    (25, "PatternUltra64_600", 64, "0.06%"),
    (27, "PatternUltra64_6000", 64, "0.6%"),
    (28, "PatternUltra64_10000", 64, "1%"),
]

FIG4_REGISTRIES = [
    (13, "PatternD5_128", 128, "5%"),
    (14, "PatternD10_128", 128, "10%"),
    (15, "PatternD25_128", 128, "25%"),
    (16, "PatternD50_128", 128, "50%"),
    (2, "PatternD5", 256, "5%"),
    (3, "PatternD10", 256, "10%"),
    (4, "PatternD25", 256, "25%"),
    (5, "PatternD50", 256, "50%"),
]


def classify_pattern(filename: str) -> str:
    if "_row_" in filename:
        return "Row"
    if "_col_" in filename:
        return "Col"
    if "_multi_diag_" in filename:
        return "Banded"
    return "Random"


def extract_tflops(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = re.search(r"Device\s+TFLOP/?s:\s*([\d.]+)", log_path.read_text())
    return float(m.group(1)) if m else None


def aggregate(output_dir: Path, registries, out_csv: Path) -> int:
    rows = []
    missing = 0
    for reg_idx, reg_name, block_size, density in registries:
        algo_dir = output_dir / "csvs" / reg_name / DDA_HC
        if not algo_dir.is_dir():
            print(f"  [warn] missing dir: {algo_dir}", file=sys.stderr)
            missing += 4
            continue
        per_pattern = {}
        for f in sorted(os.listdir(algo_dir)):
            if not f.endswith("_sparse.log"):
                continue
            pat = classify_pattern(f)
            tflops = extract_tflops(algo_dir / f)
            if tflops is not None:
                per_pattern[pat] = tflops
        for pat in ("Row", "Col", "Banded", "Random"):
            tflops = per_pattern.get(pat)
            if tflops is None:
                missing += 1
                print(f"  [warn] missing TFLOPs: reg={reg_idx} pattern={pat}", file=sys.stderr)
            rows.append(
                {
                    "registry": reg_idx,
                    "registry_name": reg_name,
                    "block_size": block_size,
                    "density": density,
                    "pattern": pat,
                    "tflops": tflops,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "registry",
                "registry_name",
                "block_size",
                "density",
                "pattern",
                "tflops",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}" + (f" ({missing} missing)" if missing else ""))
    return missing


def main():
    p = argparse.ArgumentParser(description="Aggregate Fig. 3 or Fig. 4 DDA throughput.")
    p.add_argument("--figure", choices=["3", "4"], required=True, help="Which figure")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Raw profiling output root (e.g. $TT_METAL_HOME/profiles_paper_fig3)",
    )
    p.add_argument("--out-csv", type=Path, required=True, help="Aggregated CSV path")
    args = p.parse_args()

    registries = FIG3_REGISTRIES if args.figure == "3" else FIG4_REGISTRIES
    missing = aggregate(args.output_dir, registries, args.out_csv)
    sys.exit(0 if missing == 0 else 1)


if __name__ == "__main__":
    main()
