# SC26 Submission: Block-Sparse Linear Algebra on Tenstorrent

Paper artifacts for "Leveraging Network-on-Chip Architecture for Block-Sparse
Linear Algebra on Tenstorrent" (SC26). This repository contains block-sparse
SpMM and SDDMM kernels for the Tenstorrent Wormhole N150, along with the
scripts used to reproduce the paper's figures and tables.

## Repository layout

```
block_spmm/        BSR SpMM kernels (Naive / SnF / DDA)
block_sddmm/       BSR SDDMM kernels (Naive / DDA)
sparse_common/     Shared BSR matrix / buffer helpers used by both kernels
GEMM_profiling/    SDDMM TFLOP/s profiler (used by export_sddmm)
paper_reproduce/   Bash + Python scripts that reproduce Fig 3/4 and Table 1/2/3
CMakeLists.txt     Entry point; expected to be consumed via add_subdirectory()
```

## Embedded-only submodule

This repository is **embedded-only**: it builds as a submodule of a host
project, and does not build standalone. The submodule brings its own
`sparse_common` target; the only external dependency is the host project's
`tt_metal` target (Tenstorrent's low-level kernel framework) and build glue
(the `CREATE_PGM_EXAMPLES_EXE` macro plus `PROGRAMMING_EXAMPLES_TEST_TARGETS`
variable used by the tt-metal programming-examples system).

## Integrating into a host project

In the host's `CMakeLists.txt`:

```cmake
add_subdirectory(SC26_submission)            # provides the variables below

set(HOST_SRCS
    ${SC26_ENTRY_POINT_SRCS}                 # always-built
)
if(ENABLE_TRACY)
    list(APPEND HOST_SRCS ${SC26_TRACY_SRCS}) # profile_* binaries
endif()

CREATE_PGM_EXAMPLES_EXE("${HOST_SRCS}" "my_target")

foreach(t ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
    if(TARGET ${t})
        target_link_libraries(${t} PRIVATE ${SC26_LINK_LIBS})
    endif()
endforeach()
```

The submodule exposes three CMake variables into the parent scope:

- `SC26_ENTRY_POINT_SRCS` — source files for `run_*`, `test_*`, `export_*` binaries.
- `SC26_TRACY_SRCS` — source files for `profile_*` binaries (add when `ENABLE_TRACY`).
- `SC26_LINK_LIBS` — libraries each binary must link: `sddmm_host_code`,
  `sc26_spmm_host_code`, `sparse_common`.

The `sparse_common` target is defined by the submodule itself; host projects
that need the same helpers for adjacent subprojects can link against it
transitively (it is exported globally once `add_subdirectory(SC26_submission)`
runs).

## Reproducing paper results

After a successful host-project build, run the scripts under `paper_reproduce/`:

```bash
bash paper_reproduce/run_all.sh
```

This produces five aggregated CSVs under `paper_reproduce/outputs/`:

- `fig3_dda_throughput.csv` (paper Fig. 3)
- `fig4_dda_throughput.csv` (paper Fig. 4)
- `table1_dram_and_throughput.csv` (paper Table 1)
- `table2_load_balance.csv` (paper Table 2)
- `table3_sddmm_naive_vs_cda.csv` (paper Table 3)

Each reproduction script prints its total wall time at the end; `run_all.sh`
aggregates timings into `paper_reproduce/logs/SUMMARY.txt`.

Individual scripts assume the host-project binaries live at
`$REPO_ROOT/build/programming_examples/rahmy/`. Override via `BUILD_METAL_CMD`
env var for non-default host build commands.
