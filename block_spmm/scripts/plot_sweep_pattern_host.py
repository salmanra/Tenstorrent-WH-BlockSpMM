#!/usr/bin/env python3
"""Ultra-sparse DDA vs GPU throughput, but DDA values are computed from
HOST-side timing instead of device-side.

For the GPU bars and the overall layout, this script reuses the same logic as
plot_sweep_pattern.py. The only difference is how DDA TFLOP/s are extracted:

  Device-side (other script):
      grep 'Device TFLOP/s' from {test}_sparse.log

  Host-side (this script):
      Total FLOPs (from sparse log)
      Host time per iteration = (total_ns of "Device program Loop" zone in
                                  the host {test}.csv) / HOST_LOOP_ITERATIONS
      TFLOP/s = Total FLOPs / per-iter seconds / 1e12

The difference exposes per-iteration host dispatch + op-to-op overhead, which
matters most for ultra-sparse cases where the kernel is short.
"""

import os
import re
import csv

import plot_sweep_pattern
from plot_sweep_pattern import (
    AXES, DDA_HC, classify_pattern, plot_figure,
)

DDA_ROOT = "/home/user/tt-metal/profiles_sc26_april5/csvs"
HOST_LOOP_ITERATIONS = 10


def parse_dda_host_tflops(registry_name):
    """Return {pattern: tflops} computed from the host-side 'Device program Loop' zone."""
    d = os.path.join(DDA_ROOT, registry_name, DDA_HC)
    result = {}
    if not os.path.isdir(d):
        return result

    for f in sorted(os.listdir(d)):
        if not f.endswith("_sparse.log"):
            continue
        pat = classify_pattern(f)
        sparse_log = os.path.join(d, f)
        host_csv = os.path.join(d, f[:-len("_sparse.log")] + ".csv")

        # Total FLOPs from the sparse log
        total_flops = None
        with open(sparse_log) as fh:
            for line in fh:
                m = re.search(r"Total FLOPs:\s+([\d.e+]+)", line)
                if m:
                    total_flops = float(m.group(1))
                    break
        if total_flops is None or not os.path.exists(host_csv):
            continue

        # Host-side per-iteration time from the "Device program Loop" zone
        with open(host_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "Device program Loop" in row.get("name", ""):
                    total_ns = float(row["total_ns"])
                    per_iter_s = (total_ns / HOST_LOOP_ITERATIONS) / 1e9
                    if per_iter_s > 0:
                        result[pat] = total_flops / per_iter_s / 1e12
                    break
    return result


if __name__ == "__main__":
    # Have plot_figure use host-side timing for DDA
    plot_sweep_pattern.parse_dda_tflops = parse_dda_host_tflops

    plot_figure(
        AXES[:2],  # ultra-sparse: R=C=32 and R=C=64
        "DDA (N150, host timing) vs GPU: Ultra-Sparse (R=C=32, R=C=64)",
        "sweep_pattern_dda_vs_gpu_ultrasparse_host.png",
    )
