# SC26 Block SpMM Profiling Plan

Self-contained profiling project for 3 block SpMM algorithms on Tenstorrent Wormhole.

---

## Algorithms

| Host code index | Code name | Paper name |
|-----------------|-----------|------------|
| 0 | `bsr_spmm_multicore_load_balanced_new_DM` | **Naive** |
| 1 | `bsr_spmm_multicore_snf` | **SnF in0 naive in1** |
| 2 | `bsr_spmm_multicore_snfin0_cdain1` | **SnF in0 CDA in1** |

Host codes 3-14 are ablation variants (no_a_read, no_b_read, no_compute, no_write) for each algorithm.

---

## Registries (10 total)

| # | Name | Cases | Purpose |
|---|------|-------|---------|
| 0 | MicrobenchD25 | 4 | Block sweep (32-256), random, d=25% |
| 1 | MicrobenchD5 | 4 | Block sweep (32-256), random, d=5% |
| 2 | PatternD5 | 4 | row/col/multi_diag/random, R=C=256, d=5% |
| 3 | PatternD10 | 4 | same, d=10% |
| 4 | PatternD25 | 4 | same, d=25% |
| 5 | PatternD50 | 4 | same, d=50% |
| 6 | SweepN | 5 | N in {512..8192}, R=C=256, d=10% |
| 7 | SweepK | 5 | K in {512..8192}, R=C=256, d=10% |
| 8 | SweepBlockSize | 4 | R=C in {32..256}, d=10% |
| 9 | SweepDensity | 5 | d in {5..75%}, R=C=256 |

All cases use M=K=8192 (except Sweep K/N which vary one of these).

---

## Experiment 1: Microbenchmark (Ablation)

**Goal**: Isolate cost of each operation by measuring runtime savings when skipped.

- Registries 0-1, host codes 0-14
- 3 algos x 4 block sizes x 2 densities x 5 variants = **120 runs**
- Output: stacked bar charts of ablation savings per algorithm and block size

## Experiment 2: Pure Throughput

**Goal**: Show CDA wins across patterns/densities, identify exceptions.

- Registries 2-5, host codes 0-2
- 3 algos x 4 patterns x 4 densities = **48 runs**
- Output: grouped bar chart, 4 densities x 4 patterns x 3 algorithms

## Experiment 3: Scaling Sweeps (CDA only)

**Goal**: Characterize CDA scaling with each parameter.

- Registries 6-9, host code 2 only
- (5+5+4+5) = **19 runs**
- Output: 4 line plots (sweep N, K, block size, density)

**Total: 187 runs**

---

## Running

```bash
# List all registries and host codes
./scripts/run_profiling.sh --list

# Run everything
./scripts/run_profiling.sh --phase all

# Run one experiment
./scripts/run_profiling.sh --phase microbench
./scripts/run_profiling.sh --phase throughput
./scripts/run_profiling.sh --phase scaling

# Dry run (preview commands)
./scripts/run_profiling.sh --dry-run --phase all

# Export only (no profiling, just CSV extraction)
./scripts/run_profiling.sh --export-only --phase all

# Skip rebuild
./scripts/run_profiling.sh --no-build --phase microbench
```

## Plotting

```bash
python scripts/plot_microbench.py
python scripts/plot_throughput.py
python scripts/plot_scaling.py
```

Figures are saved to `scripts/figures/`.

---

## Output Directories

- Tracy traces: `profiles_sc26/bsr/{registry_name}/{host_code_name}/`
- CSV exports: `profiles_sc26/csvs/{registry_name}/{host_code_name}/`
