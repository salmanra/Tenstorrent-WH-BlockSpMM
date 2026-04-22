#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Multi-Diag Ultra-Sparse Experiment
#
# Runs multi_diag test case (registry 27 / PatternUltra64_6000, test 2) on:
#   hc 0  Naive
#   hc 1  SnF
#   hc 2  CDA
#
# Test 2: multi_diag, M=N=K=8192, R=C=64, d=0.6%
#
# 3 runs total (3 host codes x 1 test case).
###############################################################################

REPO_ROOT="${TT_METAL_HOME:?TT_METAL_HOME must be set — set it to your tt-metal checkout path}"
BUILD_DIR="${REPO_ROOT}/build_Release_tracy/programming_examples/block_sparse"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"

REGISTRY=27  # PatternUltra64_6000
TESTS=(2)    # 2=multi_diag

# (host_code_num, label)
RUNS=(
    "0  Naive"
    "1  SnF"
    "2  CDA"
)

# ── Defaults ─────────────────────────────────────────────────────────────────
OUTPUT_DIR="profiles_multi_diag_ultra_sparse"
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
            echo "  --output-dir DIR  Output folder name under repo root (default: profiles_multi_diag_ultra_sparse)"
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

# ── Main ─────────────────────────────────────────────────────────────────────
echo "Multi-Diag Ultra-Sparse Experiment: reg $REGISTRY test 2 x 3 host codes"
echo "Output dir: $OUTPUT_DIR"
echo ""

TOTAL_RUNS=0

for TEST in "${TESTS[@]}"; do
  for run_spec in "${RUNS[@]}"; do
    read -r hc label <<< "$run_spec"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    if [[ $EXPORT_ONLY -eq 0 ]]; then
        echo "[profile] ($TOTAL_RUNS) reg=$REGISTRY test=$TEST hc=$hc [$label]"
        local_cmd="TT_METAL_DEVICE_PROFILER=1 $PROFILE_BIN $TEST $hc $REGISTRY $OUTPUT_DIR"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  DRY-RUN: $local_cmd"
        else
            just_build
            TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$TEST" "$hc" "$REGISTRY" "$OUTPUT_DIR"
        fi
    fi

    if [[ $PROFILE_ONLY -eq 0 ]]; then
        echo "[export]  ($TOTAL_RUNS) reg=$REGISTRY test=$TEST hc=$hc [$label]"
        local_cmd="$EXPORT_BIN $TEST $hc $REGISTRY $OUTPUT_DIR"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  DRY-RUN: $local_cmd"
        else
            "$EXPORT_BIN" "$TEST" "$hc" "$REGISTRY" "$OUTPUT_DIR"
        fi
    fi
  done
done

echo ""
echo "================================================================"
echo "  Done. Total runs completed: $TOTAL_RUNS"
echo "================================================================"
