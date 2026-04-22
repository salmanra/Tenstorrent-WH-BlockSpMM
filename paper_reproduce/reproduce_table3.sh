#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Reproduce Table 3: SDDMM throughput (Naive vs. DDA) across three parameter
# sweeps. All sweeps use M=8192, R=256, C=256.
#
# Paper sub-tables and their registries (block_sddmm/inc/profiling_suite.hpp):
#   (a) Density sweep: registry 2 (SDDMMSweepDensity)   5 cases (5,10,25,50,75%)
#   (b) N sweep:       registry 1 (SDDMMSweepN)         5 cases (N in 512..8192)
#   (c) K sweep:       registry 3 (SDDMMSweepK)         5 cases (K in 512..8192)
#
# Host codes (HostCodeRegistryProfiling):
#   0 = bsr_sddmm_multicore_naive   (paper "Naive")
#   1 = bsr_sddmm_multicore_CDA     (paper "DDA")
#
# Total: 3 registries x 5 cases x 2 algos = 30 runs.
#
# The SDDMM binaries hardcode their output to
#   /home/user/tt-metal/sddmm_profiles/opt_noc/{traces,csvs}/<registry>/<host_code>/
# so this script does not take an --output-dir. Plots/tables downstream expect
# exactly this path.
#
# This script delegates to block_sddmm/sddmm_scripts/run_sddmm_profiling.sh,
# selecting only sweep phase + registries 1, 2, 3 + host codes 0, 1.
###############################################################################

REPO_ROOT="/home/user/tt-metal"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SC26_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDDMM_SCRIPT="${SC26_ROOT}/block_sddmm/sddmm_scripts/run_sddmm_profiling.sh"

DRY_RUN=0
EXPORT_ONLY=0
PROFILE_ONLY=0
NO_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
        --export-only)   EXPORT_ONLY=1; shift ;;
        --profile-only)  PROFILE_ONLY=1; shift ;;
        --no-build)      NO_BUILD=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --dry-run       Print commands without executing
  --export-only   Skip profiling; re-run CSV export from existing traces
  --profile-only  Skip CSV export; only collect tracy traces
  --no-build      Skip the initial cmake build step
  -h, --help      Show this help

Output (hardcoded in binary):
  \$REPO_ROOT/sddmm_profiles/opt_noc/traces/SDDMMSweep{Density,N,K}/<hc>/
  \$REPO_ROOT/sddmm_profiles/opt_noc/csvs/SDDMMSweep{Density,N,K}/<hc>/
EOF
            exit 0 ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

PASSTHROUGH=(--phase sweep --registry 1 --registry 2 --registry 3 --host-code 0 --host-code 1)
[[ $DRY_RUN      -eq 1 ]] && PASSTHROUGH+=("--dry-run")
[[ $EXPORT_ONLY  -eq 1 ]] && PASSTHROUGH+=("--export-only")
[[ $PROFILE_ONLY -eq 1 ]] && PASSTHROUGH+=("--profile-only")
[[ $NO_BUILD     -eq 1 ]] && PASSTHROUGH+=("--no-build")

START_TIME=$(date +%s)

echo "Table 3 reproduction: SDDMM Naive vs. DDA, 3 sweeps x 5 cases x 2 algos = 30 runs"
echo "Output: \$REPO_ROOT/sddmm_profiles/opt_noc/{traces,csvs}/SDDMMSweep{Density,N,K}/"
echo ""

bash "$SDDMM_SCRIPT" "${PASSTHROUGH[@]}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGG_OUT="${SCRIPT_DIR}/outputs/table3_sddmm_naive_vs_cda.csv"

echo ""
echo "=== Aggregating results into ${AGG_OUT} ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN: python $SCRIPT_DIR/aggregate_table3.py --out-csv $AGG_OUT"
else
    python "$SCRIPT_DIR/aggregate_table3.py" --out-csv "$AGG_OUT" || true
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "=== Table 3 reproduction complete (30 runs) ==="
printf "Total elapsed: %dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
