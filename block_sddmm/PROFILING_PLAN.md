# Block SDDMM Profiling Plan

## Overview

This document describes the profiling infrastructure to be added to the `block_sddmm`
project for **ablation profiling** — isolating individual cost components by skipping
them (sparse B reads, dense C reads, dense D reads, compute, DRAM writes) to get
upper-bound and microbenchmark measurements.

SDDMM has three inputs (sparse mask B, dense C, dense D) and one output (sparse A),
giving five cost components to isolate — one more than SpMM.

---

## Profiling Goal: Ablation Runs

Each algorithm is run five additional times, each skipping one cost component:

| Variant name  | Skipped component    | Kernel flag              | Measures                              |
|---------------|----------------------|--------------------------|---------------------------------------|
| `no_b_read`   | Sparse B mask reads  | `SKIP_SPARSE_DRAM_READ=1`| Upper bound without sparse traffic    |
| `no_c_read`   | Dense C DRAM reads   | `SKIP_C_DRAM_READ=1`     | Upper bound without C-read traffic    |
| `no_d_read`   | Dense D DRAM reads   | `SKIP_D_DRAM_READ=1`     | Upper bound without D-read traffic    |
| `no_compute`  | Matmul + Hadamard    | `SKIP_COMPUTE=1`         | Pure data movement cost               |
| `no_write`    | Output DRAM writes   | `SKIP_DRAM_WRITE=1`      | Upper bound without write traffic     |

The CB handshake protocol is always preserved: `cb_push_back`/`cb_pop_front` still
execute even when the underlying operation is skipped, so no downstream kernel deadlocks.

For CDA variants, skipping a dense read (C or D) also skips the CDA sharing protocol
for that matrix. All cores act as if they're solo (just `cb_reserve_back` +
`cb_push_back`) since there's nothing to share when reads are skipped. The per-slot
barrier still runs correctly.

---

## Registry Layout

2 algorithms (naive, CDA) x 6 configurations (full + 5 skip variants) = 12 entries:

| Index | Algorithm | Variant      | Skip flag                  |
|-------|-----------|--------------|----------------------------|
| 0     | naive     | full         | (none)                     |
| 1     | CDA       | full         | (none)                     |
| 2     | naive     | no_b_read    | `SKIP_SPARSE_DRAM_READ=1`  |
| 3     | CDA       | no_b_read    | `SKIP_SPARSE_DRAM_READ=1`  |
| 4     | naive     | no_c_read    | `SKIP_C_DRAM_READ=1`       |
| 5     | CDA       | no_c_read    | `SKIP_C_DRAM_READ=1`       |
| 6     | naive     | no_d_read    | `SKIP_D_DRAM_READ=1`       |
| 7     | CDA       | no_d_read    | `SKIP_D_DRAM_READ=1`       |
| 8     | naive     | no_compute   | `SKIP_COMPUTE=1`           |
| 9     | CDA       | no_compute   | `SKIP_COMPUTE=1`           |
| 10    | naive     | no_write     | `SKIP_DRAM_WRITE=1`        |
| 11    | CDA       | no_write     | `SKIP_DRAM_WRITE=1`        |

---

## Codebase Changes

### 1. Device Kernel SKIP Guards

Each kernel file needs preprocessor ablation flags with defensive `#ifndef` guards
(default = 0, so normal runs are unaffected):

| Kernel file                          | Flags to add                                     |
|--------------------------------------|--------------------------------------------------|
| `data_movement_sddmm_BC.cpp`        | `SKIP_SPARSE_DRAM_READ`, `SKIP_C_DRAM_READ`      |
| `data_movement_sddmm_D_out.cpp`     | `SKIP_D_DRAM_READ`, `SKIP_DRAM_WRITE`             |
| `data_movement_sddmm_BC_CDA.cpp`    | `SKIP_SPARSE_DRAM_READ`, `SKIP_C_DRAM_READ`       |
| `data_movement_sddmm_D_CDA_out.cpp` | `SKIP_D_DRAM_READ`, `SKIP_DRAM_WRITE`             |
| `sddmm_block_multiply.cpp`          | `SKIP_COMPUTE` *(already exists)*                  |

#### BC kernel skip behavior (`data_movement_sddmm_BC.cpp` and `BC_CDA.cpp`)

**`SKIP_SPARSE_DRAM_READ`**: Replace the sparse mask DRAM read with:
```cpp
cb_reserve_back(cb_sparse, sparse_block_num_tiles);
// no DRAM read
cb_push_back(cb_sparse, sparse_block_num_tiles);
```

**`SKIP_C_DRAM_READ`**: Replace the entire K-loop body (DRAM read or CDA receive +
forward) with:
```cpp
for (uint32_t k = 0; k < num_blocks_k; k++) {
    cb_reserve_back(cb_dense_c, dense_c_block_num_tiles);
    // no DRAM read, no CDA receive/forward
    cb_push_back(cb_dense_c, dense_c_block_num_tiles);
}
```
For the CDA variant, this means share set discovery and the CDA protocol are also
skipped (pointless without actual data). The barrier still runs.

#### D kernel skip behavior (`data_movement_sddmm_D_out.cpp` and `D_CDA_out.cpp`)

**`SKIP_D_DRAM_READ`**: Replace the K-loop body (same pattern as C):
```cpp
for (uint32_t k = 0; k < num_blocks_k; k++) {
    cb_reserve_back(cb_dense_d, dense_d_block_num_tiles);
    // no DRAM read, no CDA receive/forward
    cb_push_back(cb_dense_d, dense_d_block_num_tiles);
}
```

**`SKIP_DRAM_WRITE`**: Replace the output writeback with:
```cpp
cb_wait_front(cb_out, out_block_num_tiles);
// no DRAM write
cb_pop_front(cb_out, out_block_num_tiles);
```

#### Compute kernel skip behavior (`sddmm_block_multiply.cpp`)

**`SKIP_COMPUTE`** *(already implemented)*: Drains `cb_dense_c` and `cb_dense_d` on
every K-step, pushes a dummy output tile on the last K-step, and pops `cb_sparse`.

### 2. Host Code Architecture

Each of the 2 host code `.cpp` files already has the `_impl` + `extra_defines` pattern.
Add 5 skip wrapper functions per algorithm:

```cpp
template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_no_b_read(...) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(..., {{"SKIP_SPARSE_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_no_c_read(...) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(..., {{"SKIP_C_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_no_d_read(...) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(..., {{"SKIP_D_DRAM_READ", "1"}});
}
template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_no_compute(...) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(..., {{"SKIP_COMPUTE", "1"}});
}
template<bool verbose, bool is_profiling>
void bsr_sddmm_multicore_naive_no_write(...) {
    bsr_sddmm_multicore_naive_impl<verbose, is_profiling>(..., {{"SKIP_DRAM_WRITE", "1"}});
}
// (same 5 for CDA)
```

Plus explicit template instantiations for `<false, true>` (profiling mode).

Files modified:
- `inc/host_code/bsr_sddmm_multicore_naive.cpp`
- `inc/host_code/bsr_sddmm_multicore_CDA.cpp`

### 3. `host_code.hpp`

Add a `DECLARE_SDDMM_ABLATION_WRAPPERS(func_name)` macro to declare the 5 skip
wrappers for each of the 2 algorithms (10 new function declarations total).

Extend `HostCodeRegistryProfiling` from 2 entries to 12 entries matching the
registry layout table above.

### 4. Profiling Script

Update `sddmm_scripts/run_sddmm_profiling.sh`:
- Add `--phase ablation|sweep|all` flag (default: `all`)
- Add `--ablation-registry N` flag (which sweep registry to use as the reference
  test case set for ablation runs; default: 0)
- For the ablation phase: iterate over host codes 2-11 (the skip variants) against
  the reference registry
- Update `NUM_HOST_CODES` to 12
- Keep sweep phase unchanged (host codes 0-1 against registries 1-4)

---

## Files to Modify

| File | Change |
|------|--------|
| `kernels/dataflow/data_movement_sddmm_BC.cpp` | Add `SKIP_SPARSE_DRAM_READ`, `SKIP_C_DRAM_READ` guards |
| `kernels/dataflow/data_movement_sddmm_D_out.cpp` | Add `SKIP_D_DRAM_READ`, `SKIP_DRAM_WRITE` guards |
| `kernels/dataflow/data_movement_sddmm_BC_CDA.cpp` | Add `SKIP_SPARSE_DRAM_READ`, `SKIP_C_DRAM_READ` guards |
| `kernels/dataflow/data_movement_sddmm_D_CDA_out.cpp` | Add `SKIP_D_DRAM_READ`, `SKIP_DRAM_WRITE` guards |
| `inc/host_code/bsr_sddmm_multicore_naive.cpp` | Add 5 ablation wrapper functions + instantiations |
| `inc/host_code/bsr_sddmm_multicore_CDA.cpp` | Add 5 ablation wrapper functions + instantiations |
| `inc/host_code.hpp` | Add ablation wrapper declarations, extend registry to 12 entries |
| `sddmm_scripts/run_sddmm_profiling.sh` | Add `--phase ablation` support |

---

## Running the Profiling Plan

### Ablation phase

Runs all 5 skip variants x 2 algorithms against a reference registry:

```bash
./sddmm_scripts/run_sddmm_profiling.sh --phase ablation
```

Use a different reference registry (default 0):

```bash
./sddmm_scripts/run_sddmm_profiling.sh --phase ablation --ablation-registry 1
```

### Restrict to one algorithm

```bash
# Only naive ablations:
./sddmm_scripts/run_sddmm_profiling.sh --phase ablation --host-code 0

# Only CDA ablations:
./sddmm_scripts/run_sddmm_profiling.sh --phase ablation --host-code 1
```

### Sweep phase only (unchanged)

```bash
./sddmm_scripts/run_sddmm_profiling.sh --phase sweep
```

### Skip rebuild / dry run

```bash
./sddmm_scripts/run_sddmm_profiling.sh --no-build --phase ablation
./sddmm_scripts/run_sddmm_profiling.sh --dry-run
```

---

## Key Invariants

- **CB handshake protocol is never broken.** Every `cb_push_back` and `cb_pop_front`
  that would happen in a normal run still happens in ablation runs. Only the actual
  DRAM reads/writes, CDA protocol, or compute instructions are skipped.
- **CDA barriers still run under skip flags.** Even when dense reads are skipped,
  cores still iterate over output slots in lockstep and execute barriers. Only the
  data acquisition and forwarding within the K-loop are bypassed.
- **`zone_defines` merge is non-destructive.** Zone profiling flags from
  `sddmm_zone_config::get_zone_defines()` are applied first, then skip flags are
  inserted. Skip flags and zone flags are orthogonal.
- **Existing registries and host codes are unchanged.** Indices 0-1 in
  `HostCodeRegistryProfiling` remain the full algorithms.
