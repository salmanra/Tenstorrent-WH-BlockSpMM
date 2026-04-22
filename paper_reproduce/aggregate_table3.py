#!/usr/bin/env python3
"""Aggregate Table 3 (SDDMM Naive vs. DDA) into a single CSV.

Walks the hardcoded SDDMM output tree under sddmm_profiles/opt_noc/csvs/ and
extracts `Device TFLOP/s:` from each `<stem>_mask.log` for the Density, N, and
K sweeps. Emits one row per test case with both algorithms and their speedup.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

SDDMM_DATA_DIR = Path("/home/user/tt-metal/sddmm_profiles/opt_noc/csvs")

# (sweep label, registry dir)
SWEEPS = [
    ("Density", "SDDMMSweepDensity"),
    ("N",       "SDDMMSweepN"),
    ("K",       "SDDMMSweepK"),
]

ALGOS = [
    ("naive", "bsr_sddmm_multicore_naive"),
    ("cda",   "bsr_sddmm_multicore_CDA"),
]

PARAMETRIC_RE = re.compile(r"parametric_M(\d+)_N(\d+)_K(\d+)_R(\d+)_C(\d+)_d(\d+)")
TFLOPS_RE = re.compile(r"Device\s+TFLOP/?s:\s*([\d.]+)")
NBLOCKS_RE = re.compile(r"Number of blocks:\s*(\d+)")


def parse_meta(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    out = {}
    m = TFLOPS_RE.search(text)
    if m:
        out["tflops"] = float(m.group(1))
    m = NBLOCKS_RE.search(text)
    if m:
        out["nblocks"] = int(m.group(1))
    return out


def collect_algo(algo_dir: Path) -> dict[tuple, dict]:
    """Return {(M,N,K,R,C,density): {'tflops':..., 'nblocks':...}}."""
    out = {}
    if not algo_dir.is_dir():
        return out
    for csv_path in sorted(algo_dir.glob("*.csv")):
        if csv_path.stem.endswith(".device"):
            continue
        m = PARAMETRIC_RE.match(csv_path.stem)
        if not m:
            continue
        key = tuple(int(x) for x in m.groups())
        meta = parse_meta(csv_path.parent / f"{csv_path.stem}_mask.log")
        if meta:
            out[key] = meta
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate Table 3 into a single CSV.")
    p.add_argument("--data-dir", type=Path, default=SDDMM_DATA_DIR,
                   help=f"SDDMM CSV root (default: {SDDMM_DATA_DIR})")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    rows = []
    missing = 0
    for sweep_label, registry in SWEEPS:
        reg_dir = args.data_dir / registry
        algo_data = {label: collect_algo(reg_dir / d) for label, d in ALGOS}
        keys = sorted(set().union(*(d.keys() for d in algo_data.values())))
        for key in keys:
            M, N, K, R, C, density = key
            naive = algo_data["naive"].get(key, {})
            cda   = algo_data["cda"].get(key, {})
            n_tf = naive.get("tflops")
            c_tf = cda.get("tflops")
            if n_tf is None or c_tf is None:
                missing += 1
            speedup = (c_tf / n_tf) if (n_tf and c_tf) else None
            rows.append({
                "sweep":        sweep_label,
                "M": M, "N": N, "K": K, "R": R, "C": C,
                "density":      density,
                "nblocks":      naive.get("nblocks") or cda.get("nblocks"),
                "naive_tflops": n_tf,
                "cda_tflops":   c_tf,
                "speedup":      speedup,
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sweep", "M", "N", "K", "R", "C", "density", "nblocks",
            "naive_tflops", "cda_tflops", "speedup",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}" +
          (f" ({missing} missing cells)" if missing else ""))
    sys.exit(0 if missing == 0 else 1)


if __name__ == "__main__":
    main()
