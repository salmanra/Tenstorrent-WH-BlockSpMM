# Reference environment

These are the exact hardware, operating system, and toolchain versions on
which the paper's reference numbers (and the [`REPRODUCE.md`](REPRODUCE.md)
expected values) were measured. Reproduction on a substantially different
setup may yield different numbers — we have not attempted cross-platform
validation.

## Hardware

### Tenstorrent Wormhole N150
- Device: one N150 PCIe card, chip index 0
- Firmware: 19.2.0 (Ethernet firmware 7.2.0)
- Harvesting mask: `0x200` (NoC0 `0x200`, no additional simulated harvesting)
- KMD (kernel mode driver): 2.5.0
- UMD (user mode driver / silicon driver): software version 6.0.0
- TT-SMI: 3.0.27
- Host interface: PCIe, IOMMU disabled, 1 GiB hugepage pinned to the device

### Host
- CPU: 2 × AMD EPYC 7352 (24 physical cores / 48 threads each; 96 logical cores total); 2.3 GHz base clock, Zen 2
- RAM: 512 GiB (`MemTotal: 528,191,716 kB`)
- Interconnect: PCIe to the N150; no other accelerators on the bus

> The paper reports the host as "a dual-socket server equipped with two AMD EPYC 7352 processors (24 cores / 48 threads each, 2.3 GHz base clock, Zen 2 microarchitecture), 504 GiB of system memory, and 256 MiB of shared L3 cache across 16 CCX complexes." This artifact ran on that same machine.

## Operating system

- Distribution: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- Kernel: Linux 5.15.0-164-generic
- Architecture: x86_64

## Toolchain

| Tool | Version |
|---|---|
| GCC | 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.2) |
| G++ | 11.4.0 |
| CMake | 4.1.1 |
| Ninja | 1.10.1 |
| Python | 3.10.12 |
| pip | 21.2.4 (within tt-metal's `python_env/`) |
| `patch` | any recent POSIX `patch` |
| `git` | any recent; `git submodule add` with an HTTPS URL is all we use |

Python dependencies for the aggregators are listed in
[`requirements.txt`](requirements.txt) (`pandas`, `matplotlib`, `numpy`;
versions unpinned).

## Pinned software versions

- **tt-metal**: tag `v0.63.0`, commit `d17adf6f58cab05c994c62d230ea9a75e9efd997`
- **tt-metal internal submodules** (initialized automatically by
  `git submodule update --init --recursive` inside tt-metal):
  - `tt_metal/third_party/tracy` → `f6322ff3e353590df38b1dc881e2a6e181c1177e`
  - `tt_metal/third_party/tt_llk` → `da2f082eba309404a2c00ca9761c11438bff5f1f`
  - `tt_metal/third_party/umd` → `b4cf8ec87dda4dd6e4e44d3e76c858e4cb9e3383`
- **This submodule**: see `git log -1` in the checkout. The reference
  numbers below were produced from commit `f8a9c2e`.

## Build timing

- **Cold `build_metal.sh --enable-profiler --build-programming-examples`**:
  **426 s ≈ 7 minutes** on this 96-logical-core host with both
  `build_Release_tracy/` and `.cpmcache/` wiped (CCache preserved at its
  host default location, populated from prior unrelated builds;
  contribution is modest — ccache stats show ~25 % hit rate across
  tt-metal compilations on this host).
- Linear extrapolation to lower core counts is reasonable: expect
  ~30 min on 16 cores, ~1 hour on 8 cores, since `build_metal.sh -j` uses
  `nproc`.
- Warm incremental rebuilds (per-case build inside reproduction scripts):
  ~5-10 seconds each on this hardware.

## Reproduction wall times (reference)

Measured on this machine via `bash paper_reproduce/run_all.sh` against a
fresh GitHub-cloned tt-metal workspace:

| Script | Wall time |
|---|---|
| `reproduce_fig3.sh` | 13 m 27 s |
| `reproduce_fig4.sh` | 23 m 41 s |
| `reproduce_table1.sh` | 18 m 51 s |
| `reproduce_table2.sh` | 21 m 14 s |
| `reproduce_table3.sh` | 14 m 25 s |
| **Total (`run_all.sh`)** | **1 h 31 m 38 s** |

See [`REPRODUCE.md`](REPRODUCE.md) for per-script command-line usage.

## What matters for reproduction outcome

- **Device (N150) firmware and harvesting mask** — affects kernel
  scheduling and available cores. Different harvesting masks will change
  the exact TFLOP/s numbers (nothing depends on a specific mask, but the
  reference numbers correspond to `0x200`).
- **tt-metal `v0.63.0`** — kernel API and profiler output format changed
  in later releases; the submodule's hardcoded JIT paths and profiler
  parsing assume this exact tag.
- **Host CPU count** — affects build parallelism (`-j` inside
  `build_metal.sh`); does not affect reproduction correctness.
- **Python 3.10.x** — any 3.10+ works; the aggregators use no
  version-specific syntax.
- **Compiler version** — GCC 11.4 is what we used. Any C++20-capable
  compiler accepted by tt-metal should produce equivalent binaries.

## What does NOT need to match

- OS distribution, as long as it's a recent Linux with the tt-metal
  prerequisites (Ubuntu 22.04+ and Debian derivatives are known-good).
- RAM, as long as there is enough for compilation peak (≈ 16 GiB headroom
  should suffice; our host has far more).
- CPU generation, as long as x86_64.

## Cross-platform validation

We have **not** validated reproduction on other hardware (different N150
firmware, non-N150 Wormhole variants), other OS distributions, or other
toolchain versions. Single-machine reproduction on the reference setup
above is what this artifact guarantees.
