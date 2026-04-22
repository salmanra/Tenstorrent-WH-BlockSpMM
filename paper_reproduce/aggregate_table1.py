#!/usr/bin/env python3
"""Aggregate Table 1 (DRAM reads + throughput, 6 cases x 3 algorithms) into a single CSV.

Throughput comes from <output-dir>/csvs/<registry>/<algo>/<test>_sparse.log
DRAM reads come from the two CSVs produced by count_dram_reads.py:
  dram_reads_table1_sparse_cases.csv
  dram_reads_table1_triangular_cases.csv
Per-core counts are summed across all 64 Tensix cores per (case, algorithm).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Paper Table 1: (case label, registry_idx, registry_name, test_idx, test_stem, block_size, density_or_pattern)
CASES = [
    ("Row_R256_d25",      4,  "PatternD25",          0, "parametric_row_M8192_N8192_K8192_R256_C256_dppm250000", 256, "d=25%"),
    ("Banded_R256_d50",   5,  "PatternD50",          2, "parametric_multi_diag_M8192_N8192_K8192_R256_C256_dppm500000", 256, "d=50%"),
    ("Row_R64_d0.6",      27, "PatternUltra64_6000", 0, "parametric_row_M8192_N8192_K8192_R64_C64_dppm6000", 64, "d=0.6%"),
    ("Diagonal_R64_d0.6", 27, "PatternUltra64_6000", 2, "parametric_multi_diag_M8192_N8192_K8192_R64_C64_dppm6000", 64, "d=0.6%"),
    ("LowerTri_R256",     29, "Triangular",          0, "parametric_tril_M8192_N8192_K8192_R256_C256", 256, "lower-tri"),
    ("UpperTri_R256",     30, "UpperTriangular",     0, "parametric_triu_M8192_N8192_K8192_R256_C256", 256, "upper-tri"),
]

# Algorithms: (display_label, dram_match_prefix, profiling_index, algo_dir_name)
# NOTE: count_dram_reads.py labels DDA as "CDA_<case>" (historical name); paper uses "DDA".
ALGOS = [
    ("Naive", "Naive", 0, "bsr_spmm_multicore_load_balanced_new_DM"),
    ("SnF",   "SnF",   1, "bsr_spmm_multicore_snf"),
    ("DDA",   "CDA",   2, "bsr_spmm_multicore_snfin0_cdain1"),
]


def extract_tflops(log_path: Path) -> float | None:
    if not log_path.exists():
        print(f"  [warn] missing TFLOPs log: {log_path}", file=sys.stderr)
        return None
    m = re.search(r"Device\s+TFLOP/?s:\s*([\d.]+)", log_path.read_text())
    return float(m.group(1)) if m else None


def sum_dram_reads(csv_paths, registry: int, test: int, algo_label: str) -> tuple[int | None, int | None]:
    """Sum in0/in1 reads across cores for (registry, test, host_code starts with algo_label)."""
    in0_total = 0
    in1_total = 0
    matched = False
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if int(row["registry"]) != registry or int(row["test"]) != test:
                    continue
                if not row["host_code"].startswith(f"{algo_label}_"):
                    continue
                in0_total += int(row["in0_reads"])
                in1_total += int(row["in1_reads"])
                matched = True
    if not matched:
        return None, None
    return in0_total, in1_total


def main():
    p = argparse.ArgumentParser(description="Aggregate Table 1 into a single CSV.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Raw profiling output root (e.g. /home/user/tt-metal/profiles_paper_table1)")
    p.add_argument("--dram-dir", type=Path, required=True,
                   help="Dir containing dram_reads_table1_{sparse,triangular}_cases.csv")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    dram_csvs = [
        args.dram_dir / "dram_reads_table1_sparse_cases.csv",
        args.dram_dir / "dram_reads_table1_triangular_cases.csv",
    ]

    csv_root = args.output_dir / "csvs"
    rows = []
    missing = 0
    for case_label, reg, reg_name, test, stem, block_size, density in CASES:
        for algo_label, dram_prefix, _prof_idx, algo_dir in ALGOS:
            log = csv_root / reg_name / algo_dir / f"{stem}_sparse.log"
            tflops = extract_tflops(log)
            in0, in1 = sum_dram_reads(dram_csvs, reg, test, dram_prefix)
            if tflops is None or in0 is None:
                missing += 1
            rows.append({
                "case":       case_label,
                "registry":   reg,
                "test":       test,
                "block_size": block_size,
                "density":    density,
                "algorithm":  algo_label,
                "in0_reads":  in0,
                "in1_reads":  in1,
                "tflops":     tflops,
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case", "registry", "test", "block_size", "density",
            "algorithm", "in0_reads", "in1_reads", "tflops",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}" +
          (f" ({missing} missing cells)" if missing else ""))
    sys.exit(0 if missing == 0 else 1)


if __name__ == "__main__":
    main()
