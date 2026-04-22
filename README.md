# Tenstorrent-WH-BlockSpMM

Artifact for the paper
**"Leveraging Network-on-Chip Architecture for Block-Sparse Linear Algebra on
Tenstorrent"**.

This repository contains block-sparse SpMM (sparse × dense) and SDDMM (sampled
dense-dense) kernels for the Tenstorrent Wormhole N150, plus scripts that
reproduce every figure and table in the paper that targets the N150.

## Repository layout

```
block_spmm/        BSR SpMM kernels (Naive / SnF / DDA)
block_sddmm/       BSR SDDMM kernels (Naive / DDA)
sparse_common/     Shared BSR matrix + buffer helpers (private to this submodule)
GEMM_profiling/    SDDMM TFLOP/s profiler (used by export_sddmm)
paper_reproduce/   Bash + Python scripts that reproduce Fig 3/4 and Table 1/2/3
host_integration/  Host-side (tt-metal) integration assets:
                     - programming_examples.patch
                     - TODO: tracy-tool build + symlink scripts (Block 4.5)
CMakeLists.txt     Submodule entry point, consumed via add_subdirectory()
```

## Embedded-only submodule

This repository is **embedded-only** — it builds inside a host tt-metal
checkout at tag `v0.63.0`. It does not build standalone. The submodule must
be mounted at exactly:

```
$TT_METAL_HOME/tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/
```

That path is baked into the kernel JIT lookup strings. Anywhere else and the
kernels will not compile at runtime.

## Quick start

See [`INSTALL.md`](INSTALL.md) for full setup, and
[`REPRODUCE.md`](REPRODUCE.md) for per-artifact reproduction commands with
expected wall times.

## Environment

All reproduction scripts and binaries read one environment variable:

```bash
export TT_METAL_HOME=/path/to/your/tt-metal-checkout
```

It is required. Missing it produces a clear error from every entry point
(shell / Python / C++).

## Hardware

- Tenstorrent Wormhole N150. The paper's GPU comparison (cuSPARSE on RTX 4090)
  is not part of this artifact.
