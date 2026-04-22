# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Read device profiler log from a SDDMM run and print kernel duration / device utilization.

The profiler log is generated when running SDDMM with TT_METAL_DEVICE_PROFILER=1.
Since SDDMM dispatches via tt_metal API (not ttnn), run_host_id is always 0, so the
standard per-op analysis ("across": "ops") produces nothing. This script uses per-core
analysis ("across": "core") with adjacent zone matching instead.

Usage:
    python GEMM_profiling/read_sddmm_profiler.py
    python GEMM_profiling/read_sddmm_profiler.py --M 4096 --K 4096 --N 4096 --fidelity HiFi4
"""

import argparse

import numpy as np
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
from tt_metal.tools.profiler.device_post_proc_config import default_setup
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


# Custom config that adds SDDMM zone analysis using per-core adjacent matching.
# MergeMetaclass merges timerAnalysis from child into parent, so all default analyses remain.
class sddmm_setup(default_setup):
    timerAnalysis = {
        "sddmm_trisc1_kernel_duration": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Read device profiler log from a SDDMM run")
    parser.add_argument("--M", type=int, default=4096, help="Rows of sampling mask")
    parser.add_argument("--K", type=int, default=4096, help="Reduction dimension (cols of C = rows of D)")
    parser.add_argument("--N", type=int, default=4096, help="Cols of sampling mask")
    parser.add_argument("--R", type=int, default=32, help="Block row size (elements)")
    parser.add_argument("--C", type=int, default=32, help="Block col size (elements)")
    parser.add_argument("--nblocks", type=int, required=True, help="Number of nonzero blocks in sampling mask")
    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--grid-y", type=int, default=8)
    parser.add_argument("--fidelity", type=str, default="HiFi4", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"])
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup invocations to skip per core")
    args = parser.parse_args()

    if not profiler_log_path.exists():
        print(f"Profiler log not found: {profiler_log_path}")
        print("Run SDDMM with TT_METAL_DEVICE_PROFILER=1 first, then wait for process to fully exit.")
        return

    print(f"Reading: {profiler_log_path}")
    print(f"  File size: {profiler_log_path.stat().st_size:,} bytes\n")

    setup = sddmm_setup()
    deviceData = import_log_run_stats(setup)

    device_analysis = deviceData["devices"][0]["cores"]["DEVICE"].get("analysis", {})

    if not device_analysis:
        print("No analysis results found. The profiler log may not contain expected zone names.")
        return

    # Compute reference values for utilization
    # SDDMM: A = B . (C x D), each nonzero block computes R x C sub-block of (C x D)
    # Per-block matmul: C_rows(R x K) * D_cols(K x C) => 2*R*C*K FLOPs + R*C (eltwise mul at the end) 
    R, C, K, nblocks = args.R, args.C, args.K, args.nblocks
    total_flops = 2 * nblocks * R * C * K + nblocks * R * C 
    # Ideal cycles: each tile matmul is (R/32) * (C/32) * (K/32) tiles per block,
    # times nblocks, divided across cores
    LoFi_cycle = 16
    cycle_per_tile = {
        "LoFi": LoFi_cycle,
        "HiFi2": LoFi_cycle * 2,
        "HiFi3": LoFi_cycle * 3,
        "HiFi4": LoFi_cycle * 4,
    }[args.fidelity]
    num_cores = args.grid_x * args.grid_y
    total_tile_matmuls = nblocks * (R // 32) * (C // 32) * (K // 32)
    ideal_cycles = total_tile_matmuls * cycle_per_tile / num_cores
    device_freq = deviceData["deviceInfo"]["freq"]

    print(f"  Device freq: {device_freq} MHz")
    print(f"  SDDMM:      M={args.M} K={K} N={args.N}, R={R} C={C}, nblocks={nblocks}")
    print(f"  Total FLOPs: {total_flops:.3e}")
    print(f"  Fidelity:    {args.fidelity} on {args.grid_x}x{args.grid_y} grid")
    print(f"  Ideal cycles: {ideal_cycles:.0f}\n")

    # Collect per-core series data, dropping warmup invocations
    analysis_key = "sddmm_trisc1_kernel_duration"
    all_durations = []
    cores = deviceData["devices"][0]["cores"]
    for core_key, core_data in cores.items():
        if core_key == "DEVICE":
            continue
        for risc_data in core_data.get("riscs", {}).values():
            if "analysis" not in risc_data or analysis_key not in risc_data["analysis"]:
                continue
            series = risc_data["analysis"][analysis_key]["series"]
            # Skip warmup invocations, keep the rest
            for entry in series[args.warmup:]:
                all_durations.append(entry["duration_cycles"])

    if all_durations:
        durations = np.array(all_durations)
        avg_cycles = durations.mean()
        avg_time_ms = avg_cycles / device_freq / 1e3
        device_tflops = total_flops / (avg_cycles / device_freq / 1e6) / 1e12
        utilization = ideal_cycles / avg_cycles

        print(f"  TRISC1 kernel duration (warmup={args.warmup} dropped per core):")
        print(f"    Count:          {len(durations)} (across all cores)")
        print(f"    Avg:            {avg_cycles:.0f} cycles ({avg_time_ms:.3f} ms)")
        print(f"    Min:            {durations.min():.0f} cycles")
        print(f"    Max:            {durations.max():.0f} cycles")
        print(f"    Std:            {durations.std():.0f} cycles")
        print(f"    Device util:    {utilization * 100:.2f}%")
        print(f"    Device TFLOP/s: {device_tflops:.2f}")
        print()
    else:
        print(f"  {analysis_key}: NOT FOUND (no per-core TRISC-KERNEL zones)\n")

    # List all other available analysis keys
    other_keys = sorted(k for k in device_analysis if k != analysis_key)
    if other_keys:
        print(f"  Other available analyses ({len(other_keys)}):")
        for key in other_keys:
            stats = device_analysis[key]["stats"]
            avg_cycles = stats["Average"]
            print(f"    {key}: avg {avg_cycles:.0f} cycles ({avg_cycles / device_freq / 1e3:.3f} ms), count={stats['Count']:.0f}")


if __name__ == "__main__":
    main()
