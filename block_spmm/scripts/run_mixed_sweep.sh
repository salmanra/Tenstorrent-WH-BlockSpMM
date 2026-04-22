#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Mixed Sweep: 4 cherry-picked cases x 3 base host codes = 12 runs
#
# Cases:
#   1. row,        R=C=256, d=50%  — PatternD50 (reg 5),           test 0
#   2. multi_diag, R=C=256, d=50%  — PatternD50 (reg 5),           test 2
#   3. row,        R=C=64,  d=0.6% — PatternUltra64_6000 (reg 27), test 0
#   4. random,     R=C=64,  d=0.6% — PatternUltra64_6000 (reg 27), test 3
#
# Host codes: 0 (Naive), 1 (SnF), 2 (CDA)
###############################################################################

REPO_ROOT="${TT_METAL_HOME:?TT_METAL_HOME must be set — set it to your tt-metal checkout path}"
BUILD_DIR="${REPO_ROOT}/build_Release_tracy/programming_examples/block_sparse"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"

HOST_CODE_NAMES=(
    "bsr_spmm_multicore_load_balanced_new_DM"   # 0  Naive
    "bsr_spmm_multicore_snf"                     # 1  SnF
    "bsr_spmm_multicore_snfin0_cdain1"           # 2  CDA
)

# (test_num, registry_num, description)
CASES=(
    "0  5  row_R256_d50"
    "2  5  multi_diag_R256_d50"
    "0 27  row_R64_d0.6"
    "3 27  random_R64_d0.6"
)

# ── Defaults ─────────────────────────────────────────────────────────────────
OUTPUT_DIR="profiles_mixed_sweep"
DRY_RUN=0
EXPORT_ONLY=0
PROFILE_ONLY=0

# ── CLI parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1;      shift ;;
        --export-only)   EXPORT_ONLY=1;   shift ;;
        --profile-only)  PROFILE_ONLY=1;  shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR  Output folder name under repo root (default: profiles_mixed_sweep)"
            echo "  --dry-run         Print commands without executing"
            echo "  --export-only     Only run the CSV export step (skip profiling)"
            echo "  --profile-only    Only run the profiling step (skip CSV export)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# ── Build helper ─────────────────────────────────────────────────────────────
just_build() {
    if [[ $DRY_RUN -eq 0 ]]; then
        echo "[build] Reset board with tt-smi -r"
        tt-smi -r > /dev/null
        pushd "$REPO_ROOT" > /dev/null
        echo "[build] Building with profiling enabled..."
        ./build_metal.sh --enable-profiler --build-programming-examples > build.log 2> build_err.log
    fi
}

# ── Single run ───────────────────────────────────────────────────────────────
TOTAL_RUNS=0

run_one() {
    local test_num=$1
    local host_code_num=$2
    local registry_num=$3
    local desc=$4
    local label="reg=$registry_num  test=$test_num  hc=$host_code_num (${HOST_CODE_NAMES[$host_code_num]})  [$desc]"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    if [[ $EXPORT_ONLY -eq 0 ]]; then
        echo "[profile] ($TOTAL_RUNS) $label"
        local cmd="TT_METAL_DEVICE_PROFILER=1 $PROFILE_BIN $test_num $host_code_num $registry_num $OUTPUT_DIR"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  DRY-RUN: $cmd"
        else
            just_build
            TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$test_num" "$host_code_num" "$registry_num" "$OUTPUT_DIR"
        fi
    fi

    if [[ $PROFILE_ONLY -eq 0 ]]; then
        echo "[export]  ($TOTAL_RUNS) $label"
        local cmd="$EXPORT_BIN $test_num $host_code_num $registry_num $OUTPUT_DIR"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  DRY-RUN: $cmd"
        else
            "$EXPORT_BIN" "$test_num" "$host_code_num" "$registry_num" "$OUTPUT_DIR"
        fi
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "Mixed Sweep: 4 cases x 3 host codes = 12 runs"
echo ""

for case_spec in "${CASES[@]}"; do
    read -r test_num registry_num desc <<< "$case_spec"
    for host_code_num in 0 1 2; do
        run_one "$test_num" "$host_code_num" "$registry_num" "$desc"
    done
done

echo ""
echo "================================================================"
echo "  Done. Total runs completed: $TOTAL_RUNS"
echo "================================================================"
