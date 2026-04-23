# Installation

This artifact is an **embedded-only submodule** of tt-metal at tag
`v0.63.0`. These instructions walk through a complete setup from scratch
into `/tmp/tt-metal` as an example path; substitute your own location.

## System prerequisites

- **Hardware**: Tenstorrent Wormhole N150 installed and visible to the host.
- **Driver / SMI**: `tt-smi` on `PATH`. Ships with Tenstorrent's driver bundle.
- **Build tools**: CMake ≥ 3.22, Ninja, a C++20 compiler (GCC 11+ or Clang 14+),
  `patch`, `git`.
- **Python**: 3.10 or newer.
- **Disk**: ~10 GB for the tt-metal build tree.

Verify `tt-smi` works before proceeding:

```bash
tt-smi -i 0   # should show chip info for device 0
```

## Step 1 — Clone tt-metal at `v0.63.0`

```bash
git clone https://github.com/tenstorrent/tt-metal.git /tmp/tt-metal
cd /tmp/tt-metal
git checkout v0.63.0
git submodule update --init --recursive
```

**The version pin is not optional.** Later tt-metal releases have changed
the kernel API and profiler output format; this artifact was validated
only against `v0.63.0`.

## Step 2 — Add this submodule

From the tt-metal root:

```bash
git submodule add <URL-of-this-repo> \
    tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM
git submodule update --init --recursive
```

The mount path is load-bearing: kernel JIT lookups are hardcoded to
`tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/…`. Do not mount
it elsewhere.

## Step 3 — Register the submodule with tt-metal's build

Apply the one-line patch that adds `add_subdirectory(Tenstorrent-WH-BlockSpMM)`
to tt-metal's `programming_examples/CMakeLists.txt`:

```bash
cd /tmp/tt-metal   # i.e. $TT_METAL_HOME
patch -p1 < tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/host_integration/programming_examples.patch
```

Revert with `patch -p1 -R < …` if needed.

## Step 4 — Build tt-metal with profiler enabled

```bash
./build_metal.sh --enable-profiler --build-programming-examples
```

This takes **≈ 7 minutes** on our reference 96-logical-core host (AMD
EPYC 7352, 2 × 24-core) with a cold build tree and empty CPM cache — see
[`ENVIRONMENT.md`](ENVIRONMENT.md) for full system details and timing
notes. Expect longer on smaller hosts (`build_metal.sh -j` uses `nproc`).
It produces the submodule's executables at
`build/programming_examples/block_sparse/`:

```
profile_block_spmm
export_block_spmm_to_csv
run_block_spmm
test_block_spmm
profile_sddmm
export_sddmm
run_sddmm
test_sddmm
```

## Step 5 — Link the Tracy host tools

`build_metal.sh --enable-profiler` (Step 4) already built the Tracy
host-side tools (`capture-release` and `csvexport-release`) under
`tt_metal/third_party/tracy/{capture,csvexport}/build/unix/`. The
profile/export binaries invoke them via relative paths (`./capture-release`,
`./csvexport-release`), so they must be reachable from `$TT_METAL_HOME`.

Run the supplied helper to create the symlinks:

```bash
bash tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM/host_integration/setup_tracy_symlinks.sh
```

Idempotent — safe to re-run. Output:

```
[linked]  capture-release -> tt_metal/third_party/tracy/capture/build/unix/capture-release
[linked]  csvexport-release -> tt_metal/third_party/tracy/csvexport/build/unix/csvexport-release
Tracy symlinks ready at $TT_METAL_HOME.
```

## Step 6 — Install Python dependencies

```bash
cd tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM
pip install -r requirements.txt
```

The aggregators need `pandas`, `matplotlib`, `numpy`. Any recent version
works.

> **Python module path — read this if reproduction runs return empty TFLOP/s columns.**
>
> The export binaries spawn `python … read_spmm_profiler.py` / `read_sddmm_profiler.py`
> as subprocesses (via `std::system`). Those scripts import
> `tt_metal.tools.profiler.*`, which only resolves if tt-metal's own Python
> environment is active — typically by having activated tt-metal's virtualenv
> (`python_env/`) or by having built tt-metal via the standard `build_metal.sh`
> flow, which leaves `tt_metal` importable from the default `python`.
>
> If you invoke the reproduction scripts inside an `env -i` / sanitized shell
> that strips `PYTHONPATH` or virtualenv vars, the TFLOP/s column of every
> aggregated CSV will come out blank (the profile/trace steps still succeed,
> only the TFLOP/s extraction silently fails). Run your normal shell.

## Step 7 — Export `TT_METAL_HOME` and verify

```bash
export TT_METAL_HOME=/tmp/tt-metal   # or wherever your checkout lives
```

Add the export to your shell rc if this will be a persistent workspace.

Smoke-test with a single-figure reproduction (≈ 13 min, 28 runs):

```bash
cd $TT_METAL_HOME/tt_metal/programming_examples/Tenstorrent-WH-BlockSpMM
bash paper_reproduce/reproduce_fig3.sh
head paper_reproduce/outputs/fig3_dda_throughput.csv
```

If the `tflops` column is populated for every row, reproduction is working;
proceed to [`REPRODUCE.md`](REPRODUCE.md) for the full set of paper artifacts.

## Next

See [`REPRODUCE.md`](REPRODUCE.md) for per-artifact reproduction commands
and expected wall times.
