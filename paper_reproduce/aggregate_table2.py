#!/usr/bin/env python3
"""Aggregate Table 2 (load-balancing effect) into a single CSV.

Consumes the three sub-script output trees:
  <prefix>_lower/csvs/Triangular/<algo>/<tril_stem>_sparse.log
  <prefix>_upper/csvs/UpperTriangular/<algo>/<triu_stem>_sparse.log
  <prefix>_random/csvs/PatternD25/<algo>/<random_stem>.csv
Triangular rows use TFLOP/s. Random row uses host-side ms/iter extracted from the
`Device program Loop` total_ns / iteration count in the main .csv.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

TRI_ALGOS = [
    ("Naive", "LB", "bsr_spmm_multicore_load_balanced_new_DM"),
    ("Naive", "no-LB", "bsr_spmm_multicore_load_balanced_new_DM_no_lb"),
    ("SnF", "LB", "bsr_spmm_multicore_snf"),
    ("SnF", "no-LB", "bsr_spmm_multicore_snf_no_lb"),
    ("DDA", "LB", "bsr_spmm_multicore_snfin0_cdain1"),
    ("DDA", "no-LB", "bsr_spmm_multicore_snfin0_cdain1_no_lb"),
]

TRI_CASES = [
    # (label, prefix_suffix, registry_dir, stem_prefix, sizes)
    ("Lower", "lower", "Triangular", "parametric_tril", [8192, 4096]),
    ("Upper", "upper", "UpperTriangular", "parametric_triu", [8192, 4096]),
]

RANDOM_REGISTRY = "PatternD25"
RANDOM_TEST_STEM = "parametric_M8192_N8192_K8192_R256_C256_dppm250000"
RANDOM_ITERS = 10  # HOST_LOOP_ITERATIONS in profile_block_spmm.cpp


def extract_tflops(log_path: Path) -> float | None:
    if not log_path.exists():
        print(f"  [warn] missing: {log_path}", file=sys.stderr)
        return None
    m = re.search(r"Device\s+TFLOP/?s:\s*([\d.]+)", log_path.read_text())
    return float(m.group(1)) if m else None


def extract_host_ms(csv_path: Path) -> float | None:
    if not csv_path.exists():
        print(f"  [warn] missing: {csv_path}", file=sys.stderr)
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if "Device program Loop" in row.get("name", ""):
                return float(row["total_ns"]) / RANDOM_ITERS / 1e6
    return None


def main():
    p = argparse.ArgumentParser(description="Aggregate Table 2 into a single CSV.")
    p.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefix used by reproduce_table2.sh " "(e.g. $TT_METAL_HOME/profiles_paper_table2)",
    )
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    rows = []
    missing = 0

    # Triangular cases
    for tri_label, prefix_suf, reg_dir, stem_prefix, sizes in TRI_CASES:
        tri_root = Path(str(args.output_prefix) + f"_{prefix_suf}") / "csvs" / reg_dir
        for algo_label, variant, algo_dir in TRI_ALGOS:
            for size in sizes:
                log = tri_root / algo_dir / f"{stem_prefix}_M{size}_N{size}_K{size}_R256_C256_sparse.log"
                tflops = extract_tflops(log)
                if tflops is None:
                    missing += 1
                rows.append(
                    {
                        "case": f"{tri_label}-Triangular",
                        "matrix_size": size,
                        "algorithm": algo_label,
                        "lb_variant": variant,
                        "tflops": tflops,
                        "host_ms_per_iter": None,
                    }
                )

    # Random 25% case
    random_root = Path(str(args.output_prefix) + "_random") / "csvs" / RANDOM_REGISTRY
    for algo_label, variant, algo_dir in TRI_ALGOS:
        csv_path = random_root / algo_dir / f"{RANDOM_TEST_STEM}.csv"
        host_ms = extract_host_ms(csv_path)
        log = random_root / algo_dir / f"{RANDOM_TEST_STEM}_sparse.log"
        tflops = extract_tflops(log)
        if host_ms is None:
            missing += 1
        rows.append(
            {
                "case": "Random_d25",
                "matrix_size": 8192,
                "algorithm": algo_label,
                "lb_variant": variant,
                "tflops": tflops,
                "host_ms_per_iter": host_ms,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "matrix_size",
                "algorithm",
                "lb_variant",
                "tflops",
                "host_ms_per_iter",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}" + (f" ({missing} missing cells)" if missing else ""))
    sys.exit(0 if missing == 0 else 1)


if __name__ == "__main__":
    main()
