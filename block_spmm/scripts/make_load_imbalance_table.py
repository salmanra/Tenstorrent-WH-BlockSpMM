#!/usr/bin/env python3
"""Extract load-imbalance experiment data into a CSV file.

Replaces three figures:
  - load_imbalance.png        (lower-triangular, DDA)
  - load_imbalance_upper.png  (upper-triangular, DDA)
  - load_imbalance_random_speedup.png (random 25% density, all 3 algos)

Section A: TFLOP/s for {Naive, SnF, DDA} x {LB, no-LB}
           on {Lower-, Upper-Triangular} x {8192, 4096}
Section B: Host-side ms and LB speedup for {Naive, SnF, DDA}
           on a random 25%-density workload (8192).

Usage:
    python scripts/make_load_imbalance_table.py
"""

import csv
import re
import sys
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────
TRI_ALGOS = [
    ("bsr_spmm_multicore_load_balanced_new_DM", "Naive", "LB"),
    ("bsr_spmm_multicore_load_balanced_new_DM_no_lb", "Naive", "no-LB"),
    ("bsr_spmm_multicore_snf", "SnF", "LB"),
    ("bsr_spmm_multicore_snf_no_lb", "SnF", "no-LB"),
    ("bsr_spmm_multicore_snfin0_cdain1", "DDA", "LB"),
    ("bsr_spmm_multicore_snfin0_cdain1_no_lb", "DDA", "no-LB"),
]

TRI_DATASETS = [
    {
        "data_dir": Path("${TT_METAL_HOME}/profiles_load_imbalance_V2/csvs"),
        "registry": "Triangular",
        "prefix": "tril",
        "label": "Lower",
    },
    {
        "data_dir": Path("${TT_METAL_HOME}/profiles_load_imbalance_upper_V2/csvs"),
        "registry": "UpperTriangular",
        "prefix": "triu",
        "label": "Upper",
    },
]

SIZES = ["8192", "4096"]

RANDOM_ALGOS = [
    ("bsr_spmm_multicore_load_balanced_new_DM", "bsr_spmm_multicore_load_balanced_new_DM_no_lb", "Naive"),
    ("bsr_spmm_multicore_snf", "bsr_spmm_multicore_snf_no_lb", "SnF"),
    ("bsr_spmm_multicore_snfin0_cdain1", "bsr_spmm_multicore_snfin0_cdain1_no_lb", "DDA"),
]
RANDOM_DATA_DIR = Path("${TT_METAL_HOME}/profiles_load_imbalance_random/csvs")
RANDOM_REGISTRY = "PatternD25"
RANDOM_TEST = "parametric_M8192_N8192_K8192_R256_C256_dppm250000"
RANDOM_ITERS = 10  # HOST_LOOP_ITERATIONS in profile_block_spmm.cpp


# ── Data extraction ──────────────────────────────────────────────────────────
def extract_tflops(log_path):
    if not log_path.exists():
        print(f"  WARNING: missing {log_path}", file=sys.stderr)
        return None
    m = re.search(r"(?:Device\s+)?TFLOP/?s:\s*([\d.]+)", log_path.read_text(), re.IGNORECASE)
    return float(m.group(1)) if m else None


def extract_host_ms(csv_path):
    if not csv_path.exists():
        print(f"  WARNING: missing {csv_path}", file=sys.stderr)
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if "Device program Loop" in row.get("name", ""):
                return float(row["total_ns"]) / RANDOM_ITERS / 1e6
    return None


# ── Table builders ───────────────────────────────────────────────────────────
def build_section_a():
    """Section A: triangular TFLOP/s + LB-vs-no-LB speedup. Returns list of dict rows."""
    rows = []
    # Per-position TFLOP/s, indexed by (name, variant)
    tflops = {}
    for name in ("Naive", "SnF", "DDA"):
        for variant in ("LB", "no-LB"):
            algo_dir = next(a for a, n, v in TRI_ALGOS if n == name and v == variant)
            entry = {}
            for ds in TRI_DATASETS:
                for sz in SIZES:
                    log = (
                        ds["data_dir"]
                        / ds["registry"]
                        / algo_dir
                        / f"parametric_{ds['prefix']}_M{sz}_N{sz}_K{sz}_R256_C256_sparse.log"
                    )
                    entry[f"{ds['label'].lower()}_{sz}_tflops"] = extract_tflops(log)
            tflops[(name, variant)] = entry

    # Build rows; populate speedup columns on the LB row only
    # speedup = lb_tflops / no_lb_tflops; > 1 means LB is faster (matches Section B convention)
    for name in ("Naive", "SnF", "DDA"):
        for variant in ("LB", "no-LB"):
            row = {"algorithm": name, "variant": variant}
            row.update(tflops[(name, variant)])
            for key in ("lower_8192_tflops", "lower_4096_tflops", "upper_8192_tflops", "upper_4096_tflops"):
                speedup_key = key.replace("_tflops", "_lb_speedup")
                if variant == "LB":
                    lb_v = tflops[(name, "LB")].get(key)
                    no_v = tflops[(name, "no-LB")].get(key)
                    row[speedup_key] = (lb_v / no_v) if (lb_v and no_v and no_v > 0) else None
                else:
                    row[speedup_key] = None
            rows.append(row)
    return rows


def build_section_b():
    """Section B: random workload host-ms + speedup. Returns list of dict rows."""
    rows = []
    for lb_dir, no_lb_dir, name in RANDOM_ALGOS:
        lb_ms = extract_host_ms(RANDOM_DATA_DIR / RANDOM_REGISTRY / lb_dir / f"{RANDOM_TEST}.csv")
        no_ms = extract_host_ms(RANDOM_DATA_DIR / RANDOM_REGISTRY / no_lb_dir / f"{RANDOM_TEST}.csv")
        speedup = (no_ms / lb_ms) if (lb_ms and no_ms and lb_ms > 0) else None
        rows.append(
            {
                "algorithm": name,
                "lb_ms": lb_ms,
                "no_lb_ms": no_ms,
                "speedup": speedup,
            }
        )
    return rows


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    section_a = build_section_a()
    section_b = build_section_b()

    out_path = out_dir / "load_imbalance_table.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)

        # ── Section A ──
        # Speedup convention (matches Section B): lb_tflops / no_lb_tflops.
        # > 1 means LB is faster; populated on the LB row only.
        w.writerow(
            ["# Section A: TFLOP/s on triangular matrices " "(M=N=K, R=C=256). lb_speedup = lb_tflops / no_lb_tflops."]
        )
        w.writerow(
            [
                "algorithm",
                "variant",
                "lower_8192_tflops",
                "lower_4096_tflops",
                "upper_8192_tflops",
                "upper_4096_tflops",
                "lower_8192_lb_speedup",
                "lower_4096_lb_speedup",
                "upper_8192_lb_speedup",
                "upper_4096_lb_speedup",
            ]
        )
        for row in section_a:
            w.writerow(
                [
                    row["algorithm"],
                    row["variant"],
                    row.get("lower_8192_tflops"),
                    row.get("lower_4096_tflops"),
                    row.get("upper_8192_tflops"),
                    row.get("upper_4096_tflops"),
                    row.get("lower_8192_lb_speedup"),
                    row.get("lower_4096_lb_speedup"),
                    row.get("upper_8192_lb_speedup"),
                    row.get("upper_4096_lb_speedup"),
                ]
            )

        w.writerow([])

        # ── Section B ──
        w.writerow(["# Section B: Random 25%-density workload " "(M=N=K=8192, R=C=256), host-side ms / iter"])
        w.writerow(["algorithm", "lb_ms", "no_lb_ms", "speedup_no_lb_over_lb"])
        for row in section_b:
            w.writerow(
                [
                    row["algorithm"],
                    row.get("lb_ms"),
                    row.get("no_lb_ms"),
                    row.get("speedup"),
                ]
            )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
