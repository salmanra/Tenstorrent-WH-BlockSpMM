#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Load-Imbalance Experiment (Upper Triangular)
#
# Runs upper-triangular test cases (registry 30) on:
#   hc 0  Naive (with LB)     hc 16 Naive (no LB)
#   hc 1  SnF (with LB)       hc 17 SnF (no LB)
#   hc 2  CDA (with LB)       hc 15 CDA (no LB)
#
# Test 0: 8192x8192x8192, 256x256 blocks (32 block rows, num_iters_y=4)
# Test 1: 4096x4096x4096, 256x256 blocks (16 block rows, num_iters_y=2)
#
# 12 runs total (6 host codes x 2 test cases).
###############################################################################

REPO_ROOT="${TT_METAL_HOME:?TT_METAL_HOME must be set — set it to your tt-metal checkout path}"

# capture-release/csvexport-release are resolved relative to CWD
cd "$REPO_ROOT"
BUILD_DIR="${REPO_ROOT}/build_Release_tracy/programming_examples/block_sparse"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"

REGISTRY=30  # UpperTriangular
TESTS=(0 1)  # 0=8192/256, 1=4096/256

# (host_code_num, label)
RUNS=(
    "0  Naive"
    "16 Naive_no_lb"
    "1  SnF"
    "17 SnF_no_lb"
    "2  CDA"
    "15 CDA_no_lb"
)

# ── Defaults ─────────────────────────────────────────────────────────────────
OUTPUT_DIR="profiles_load_imbalance_upper_V2"
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
            echo "  --output-dir DIR  Output folder name under repo root (default: profiles_load_imbalance_upper_V2)"
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
echo "Load-Imbalance Experiment: UpperTriangular (reg $REGISTRY) x 6 host codes x ${#TESTS[@]} tests"
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
