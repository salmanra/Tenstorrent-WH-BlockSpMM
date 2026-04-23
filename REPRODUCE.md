# Reproducing paper results

Prerequisites: complete [`INSTALL.md`](INSTALL.md) and ensure
`TT_METAL_HOME` is exported in your shell.

Wall times below were measured on the reference environment described
in [`ENVIRONMENT.md`](ENVIRONMENT.md). Smaller hosts will see longer
reproductions in proportion to their build parallelism.

## All figures and tables at once

```bash
cd $TT_METAL_HOME/tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM
bash paper_reproduce/run_all.sh
```

This runs all five reproduction scripts sequentially and produces timing
summaries under `paper_reproduce/logs/SUMMARY.txt`. Expected total wall time
is ≈ 1 h 30 min on a single N150.

## Per-figure / per-table scripts

Each script produces one aggregated CSV under `paper_reproduce/outputs/`.

| Paper artifact | Command | Output CSV | Runs | Expected wall time |
|---|---|---|---|---|
| Fig. 3 (sparse DDA) | `bash paper_reproduce/reproduce_fig3.sh` | `fig3_dda_throughput.csv` | 28 | 13-14 min |
| Fig. 4 (semi-sparse DDA) | `bash paper_reproduce/reproduce_fig4.sh` | `fig4_dda_throughput.csv` | 32 | 23-24 min |
| Table 1 (DRAM reads + throughput) | `bash paper_reproduce/reproduce_table1.sh` | `table1_dram_and_throughput.csv` | 36 | 18-19 min |
| Table 2 (load balancing) | `bash paper_reproduce/reproduce_table2.sh` | `table2_load_balance.csv` | 30 | 21-22 min |
| Table 3 (SDDMM) | `bash paper_reproduce/reproduce_table3.sh` | `table3_sddmm_naive_vs_cda.csv` | 30 | 14-15 min |

All times measured on a single Wormhole N150. The per-experiment rebuild
(`build_metal.sh` called once per run) dominates wall time; the actual
kernels run in seconds.

## Flags

Every reproduction script supports:

```
--dry-run         # Print commands without executing
--export-only     # Skip profiling; re-run CSV export from existing traces
--profile-only    # Skip CSV export; only collect tracy traces
-h, --help        # Script-specific usage
```

`reproduce_table1.sh` also supports `--skip-dram` and `--skip-throughput`
to run either phase independently.

## Aggregating without re-running experiments

Each reproduce script ends by calling a standalone aggregator
(`paper_reproduce/aggregate_*.py`). If you already have profiling traces
from a prior run, you can regenerate the CSV without touching the device:

```bash
python paper_reproduce/aggregate_fig.py --figure 3 \
    --output-dir $TT_METAL_HOME/profiles_paper_fig3 \
    --out-csv paper_reproduce/outputs/fig3_dda_throughput.csv
```

Analogous invocations for `aggregate_table1.py`, `aggregate_table2.py`,
`aggregate_table3.py`; `--help` on each prints its argument set.

## Matching against the paper

Expected values from our reference run (Wormhole N150, BF16/HiFi4,
tt-metal `v0.63.0`) are within ~1-2% of the paper's published numbers.
Spot-check rows from each CSV against the paper's tables/figures:

- **Table 1** DDA Lower-triangular R=C=256: `in0_reads=2112, in1_reads=2560, tflops≈43.3`
- **Table 2** DDA Lower-tri 8192 LB speedup: `43.29 / 29.5 ≈ 1.47×`
- **Table 3** SDDMM Density 25% speedup: `34.54 / 21.84 ≈ 1.58×` (paper: 1.59×)
- **Fig. 4** DDA R=C=256 d=5% Row: `≈ 11.22 TFLOP/s` (paper: 11.5)

Minor variations (< 5%) are expected due to profiling noise, firmware
versions, and thermal state.
