#!/usr/bin/env python3
"""Count per-core DRAM reads from DPRINT output for profiling groups.

Runs run_block_spmm with DPRINT enabled for each (test, host_code) combination
in a profiling group, parses the output, and writes a CSV summarizing the
number of in0 and in1 DRAM reads per core.

Usage:
    python scripts/count_dram_reads.py --group load_imbalance
    python scripts/count_dram_reads.py --group load_imbalance_upper
    python scripts/count_dram_reads.py --group sweep_pattern
    python scripts/count_dram_reads.py --group mixed_sweep
    python scripts/count_dram_reads.py --dry-run --group load_imbalance
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/user/tt-metal")
RUN_BIN = REPO_ROOT / "build" / "programming_examples" / "rahmy" / "run_block_spmm"

# ── Group definitions ────────────────────────────────────────────────────────
# Each group has a list of "runs": (registry, test_idx, hc_verbose_idx, label)
# hc_verbose_idx is the index into HostCodeRegistryVerbose used by run_block_spmm:
#   0=Naive(LB), 1=SnF(LB), 2=CDA(LB), 3=CDA_no_lb, 4=Naive_no_lb, 5=SnF_no_lb

def _load_imbalance_runs(registry):
    """6 host codes x 2 tests for a triangular registry."""
    hcs = [(0, "Naive"), (4, "Naive_no_lb"), (1, "SnF"), (5, "SnF_no_lb"),
           (2, "CDA"), (3, "CDA_no_lb")]
    runs = []
    for test in [0, 1]:
        for hc_v, label in hcs:
            runs.append((registry, test, hc_v, label))
    return runs


def _sweep_pattern_runs():
    """20 registries x 4 tests, CDA only (hc_verbose=2)."""
    registries = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 2, 3, 4, 5, 13, 14, 15, 16]
    runs = []
    for reg in registries:
        for test in [0, 1, 2, 3]:
            runs.append((reg, test, 2, "CDA"))
    return runs


def _mixed_sweep_runs():
    """4 cherry-picked cases x 3 host codes."""
    cases = [
        (0, 5, "row_R256_d50"),
        (2, 5, "multi_diag_R256_d50"),
        (0, 27, "row_R64_d0.6"),
        (3, 27, "random_R64_d0.6"),
    ]
    runs = []
    for test, reg, desc in cases:
        for hc_v, hc_label in [(0, "Naive"), (1, "SnF"), (2, "CDA")]:
            runs.append((reg, test, hc_v, f"{hc_label}_{desc}"))
    return runs


def _multi_diag_ultra_sparse_runs():
    """3 host codes x 1 test (multi_diag, registry 27 / PatternUltra64_6000, test 2)."""
    runs = []
    for hc_v, hc_label in [(0, "Naive"), (1, "SnF"), (2, "CDA")]:
        runs.append((27, 2, hc_v, hc_label))
    return runs


def _load_imbalance_random_runs():
    """6 host codes x 1 test (random 25% density, registry 4 / PatternD25, test 3)."""
    hcs = [(0, "Naive"), (4, "Naive_no_lb"), (1, "SnF"), (5, "SnF_no_lb"),
           (2, "CDA"), (3, "CDA_no_lb")]
    return [(4, 3, hc_v, label) for hc_v, label in hcs]


def _table1_sparse_runs():
    """Paper Table 1 sparse-pattern rows: 4 cases x 3 LB algorithms = 12 runs.

    Row, R=C=256, d=25%      -> registry 4  (PatternD25),        test 0 (row)
    Banded, R=C=256, d=50%   -> registry 5  (PatternD50),        test 2 (multi_diag)
    Row, R=C=64, d=0.6%      -> registry 27 (PatternUltra64_6000), test 0 (row)
    Diagonal, R=C=64, d=0.6% -> registry 27 (PatternUltra64_6000), test 2 (multi_diag)
    """
    cases = [
        (4,  0, "Row_R256_d25"),
        (5,  2, "Banded_R256_d50"),
        (27, 0, "Row_R64_d0.6"),
        (27, 2, "Diagonal_R64_d0.6"),
    ]
    hcs = [(0, "Naive"), (1, "SnF"), (2, "CDA")]
    return [(reg, test, hc_v, f"{hc_label}_{desc}")
            for reg, test, desc in cases
            for hc_v, hc_label in hcs]


def _table1_triangular_runs():
    """Paper Table 1 triangular rows: 2 cases x 3 LB algorithms = 6 runs.

    Lower-triangular, R=C=256, M=N=K=8192 -> registry 29 (Triangular),      test 0
    Upper-triangular, R=C=256, M=N=K=8192 -> registry 30 (UpperTriangular), test 0
    """
    cases = [
        (29, 0, "LowerTri_R256"),
        (30, 0, "UpperTri_R256"),
    ]
    hcs = [(0, "Naive"), (1, "SnF"), (2, "CDA")]
    return [(reg, test, hc_v, f"{hc_label}_{desc}")
            for reg, test, desc in cases
            for hc_v, hc_label in hcs]


GROUPS = {
    "load_imbalance": {
        "runs": _load_imbalance_runs(29),
        "output": "dram_reads_load_imbalance.csv",
    },
    "load_imbalance_upper": {
        "runs": _load_imbalance_runs(30),
        "output": "dram_reads_load_imbalance_upper.csv",
    },
    "load_imbalance_random": {
        "runs": _load_imbalance_random_runs(),
        "output": "dram_reads_load_imbalance_random.csv",
    },
    "multi_diag_ultra_sparse": {
        "runs": _multi_diag_ultra_sparse_runs(),
        "output": "dram_reads_multi_diag_ultra_sparse.csv",
    },
    "sweep_pattern": {
        "runs": _sweep_pattern_runs(),
        "output": "dram_reads_sweep_pattern.csv",
    },
    "mixed_sweep": {
        "runs": _mixed_sweep_runs(),
        "output": "dram_reads_mixed_sweep.csv",
    },
    "table1_sparse_cases": {
        "runs": _table1_sparse_runs(),
        "output": "dram_reads_table1_sparse_cases.csv",
    },
    "table1_triangular_cases": {
        "runs": _table1_triangular_runs(),
        "output": "dram_reads_table1_triangular_cases.csv",
    },
}

IN0_RE = re.compile(r"in0 DRAM", re.IGNORECASE)
IN1_RE = re.compile(r"in1 DRAM", re.IGNORECASE)
CORE_RE = re.compile(r"\(x=(\d+),y=(\d+)\)")


def parse_dprint_output(output: str) -> dict:
    """Parse DPRINT lines and return {(x,y): {"in0": count, "in1": count}}."""
    counts = defaultdict(lambda: {"in0": 0, "in1": 0})
    for line in output.splitlines():
        core_match = CORE_RE.search(line)
        if not core_match:
            continue
        x, y = int(core_match.group(1)), int(core_match.group(2))
        if IN0_RE.search(line):
            counts[(x, y)]["in0"] += 1
        elif IN1_RE.search(line):
            counts[(x, y)]["in1"] += 1
    return dict(counts)


def run_dprint(test_idx: int, hc_verbose: int, registry: int) -> str:
    """Run run_block_spmm with DPRINT enabled and return stdout."""
    env = os.environ.copy()
    env["TT_METAL_DPRINT_CORES"] = "worker"
    cmd = [str(RUN_BIN), str(test_idx), str(hc_verbose), str(registry)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    # Filter to only DPRINT lines (BR/NC/TR)
    lines = []
    for line in result.stdout.splitlines():
        if ":BR:" in line or ":NC:" in line or ":TR:" in line:
            lines.append(line)
    return "\n".join(lines)


def reset_board():
    """Reset board with tt-smi."""
    subprocess.run(["tt-smi", "-r"], capture_output=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description="Count per-core DRAM reads from DPRINT")
    parser.add_argument("--group", required=True, choices=GROUPS.keys(),
                        help="Profiling group to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for CSV (default: scripts/)")
    args = parser.parse_args()

    group = GROUPS[args.group]
    runs = group["runs"]

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / group["output"]

    rows = []
    total_runs = len(runs)

    for run_num, (registry, test_idx, hc_verbose, hc_label) in enumerate(runs, 1):
        print(f"[{run_num}/{total_runs}] registry={registry} test={test_idx} "
              f"hc={hc_label} (verbose_idx={hc_verbose})")

        if args.dry_run:
            print(f"  DRY-RUN: TT_METAL_DPRINT_CORES=worker "
                  f"{RUN_BIN} {test_idx} {hc_verbose} {registry}")
            continue

        reset_board()
        output = run_dprint(test_idx, hc_verbose, registry)
        counts = parse_dprint_output(output)

        for (x, y), c in sorted(counts.items()):
            rows.append({
                "group": args.group,
                "registry": registry,
                "test": test_idx,
                "host_code": hc_label,
                "core_x": x,
                "core_y": y,
                "in0_reads": c["in0"],
                "in1_reads": c["in1"],
            })

    if not args.dry_run and rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "group", "registry", "test", "host_code",
                "core_x", "core_y", "in0_reads", "in1_reads",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {output_path} ({len(rows)} rows)")
    elif not args.dry_run:
        print("\nNo data collected.")


if __name__ == "__main__":
    main()
