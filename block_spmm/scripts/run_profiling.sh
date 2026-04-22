#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SC26 Block SpMM Profiling Script
#
# Orchestrates three experiment phases:
#   microbench  - Ablation study (registries 0-1, host codes 0-14)
#   throughput  - Pure throughput  (registries 2-5, host codes 0-2)
#   scaling     - CDA scaling      (registries 6-9, host code 2)
###############################################################################

REPO_ROOT="/home/user/tt-metal"
BUILD_DIR="${REPO_ROOT}/build_Release_tracy/programming_examples/rahmy"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"

# ── Registry metadata (indices 0-9) ─────────────────────────────────────────
REGISTRY_NAMES=(
    "MicrobenchD25"       # 0
    "MicrobenchD5"        # 1
    "PatternD5"           # 2
    "PatternD10"          # 3
    "PatternD25"          # 4
    "PatternD50"          # 5
    "SweepN"              # 6
    "SweepK"              # 7
    "SweepBlockSize"      # 8
    "SweepDensity"        # 9
    "UltraLowDensity32"   # 10
    "UltraLowDensity64"   # 11
)

REGISTRY_SIZES=(
    4   # 0: MicrobenchD25
    4   # 1: MicrobenchD5
    4   # 2: PatternD5
    4   # 3: PatternD10
    4   # 4: PatternD25
    4   # 5: PatternD50
    5   # 6: SweepN
    5   # 7: SweepK
    4   # 8: SweepBlockSize
    5   # 9: SweepDensity
    6   # 10: UltraLowDensity32
    6   # 11: UltraLowDensity64
)

# ── Host-code names (indices 0-14) ──────────────────────────────────────────
HOST_CODE_NAMES=(
    "bsr_spmm_multicore_load_balanced_new_DM"               # 0  Naive
    "bsr_spmm_multicore_snf"                                 # 1  SnF
    "bsr_spmm_multicore_snfin0_cdain1"                       # 2  CDA
    "bsr_spmm_multicore_load_balanced_new_DM_no_a_read"      # 3
    "bsr_spmm_multicore_snf_no_a_read"                       # 4
    "bsr_spmm_multicore_snfin0_cdain1_no_a_read"             # 5
    "bsr_spmm_multicore_load_balanced_new_DM_no_b_read"      # 6
    "bsr_spmm_multicore_snf_no_b_read"                       # 7
    "bsr_spmm_multicore_snfin0_cdain1_no_b_read"             # 8
    "bsr_spmm_multicore_load_balanced_new_DM_no_compute"     # 9
    "bsr_spmm_multicore_snf_no_compute"                      # 10
    "bsr_spmm_multicore_snfin0_cdain1_no_compute"            # 11
    "bsr_spmm_multicore_load_balanced_new_DM_no_write"       # 12
    "bsr_spmm_multicore_snf_no_write"                        # 13
    "bsr_spmm_multicore_snfin0_cdain1_no_write"              # 14
)

# ── Defaults ─────────────────────────────────────────────────────────────────
PHASE="all"
OUTPUT_DIR="profiles_sc26_april5"
NO_BUILD=0
DRY_RUN=0
LIST_ONLY=0
EXPORT_ONLY=0
PROFILE_ONLY=0

# ── CLI parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --no-build)
            NO_BUILD=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --list)
            LIST_ONLY=1; shift ;;
        --export-only)
            EXPORT_ONLY=1; shift ;;
        --profile-only)
            PROFILE_ONLY=1; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase microbench|throughput|scaling|ultralowdensity|all   Experiment phase (default: all)"
            echo "  --output-dir DIR  Output folder name under repo root (default: profiles_sc26_april5)"
            echo "  --no-build        Skip the cmake build step"
            echo "  --dry-run         Print commands without executing"
            echo "  --list            List registries, sizes, and host codes, then exit"
            echo "  --export-only     Only run the CSV export step (skip profiling)"
            echo "  --profile-only    Only run the profiling step (skip CSV export)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# Validate phase
case "$PHASE" in
    microbench|throughput|scaling|ultralowdensity|all) ;;
    *) echo "Error: invalid phase '$PHASE'. Must be microbench|throughput|scaling|ultralowdensity|all" >&2; exit 1 ;;
esac

# ── List mode ────────────────────────────────────────────────────────────────
if [[ $LIST_ONLY -eq 1 ]]; then
    echo "=== Registries ==="
    for i in "${!REGISTRY_NAMES[@]}"; do
        printf "  [%2d] %-20s  (%d cases)\n" "$i" "${REGISTRY_NAMES[$i]}" "${REGISTRY_SIZES[$i]}"
    done
    echo ""
    echo "=== Host Codes ==="
    for i in "${!HOST_CODE_NAMES[@]}"; do
        printf "  [%2d] %s\n" "$i" "${HOST_CODE_NAMES[$i]}"
    done
    echo ""
    echo "=== Phase Breakdown ==="
    echo "  microbench       : registries 0-1,   host codes 0-14  => 2 * 4 * 15 = 120 runs"
    echo "  throughput       : registries 2-5,   host codes 0-2   => 4 * 4 * 3  =  48 runs"
    echo "  scaling          : registries 6-9,   host code  2     => (5+5+4+5)  =  19 runs"
    echo "  ultralowdensity  : registries 10-11, host codes 0-2   => 2 * 6 * 3  =  36 runs"
    echo "  all              :                                       total      = 223 runs"
    exit 0
fi

# ── Build ────────────────────────────────────────────────────────────────────
build_once() {
    if [[ $NO_BUILD -eq 1 ]]; then
        echo "[build] Skipped (--no-build)"
        return
    fi
    echo "[build] cmake --build ${REPO_ROOT}/build --target rahmy -j$(nproc)"
    if [[ $DRY_RUN -eq 0 ]]; then
        cmake --build "${REPO_ROOT}/build" --target rahmy -j"$(nproc)"
    fi
}

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
    local label="${REGISTRY_NAMES[$registry_num]}  case=$test_num  hc=$host_code_num (${HOST_CODE_NAMES[$host_code_num]})"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    if [[ $EXPORT_ONLY -eq 0 ]]; then
        echo "[profile] ($TOTAL_RUNS) $label"
        local cmd="TT_METAL_DEVICE_PROFILER=1 $PROFILE_BIN $test_num $host_code_num $registry_num $OUTPUT_DIR"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  DRY-RUN: $cmd"
        else
            just_build # some weirdness with profiling zones not cleaning up means we have to reset the board and build every time
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

# ── Phase runners ────────────────────────────────────────────────────────────
run_microbench() {
    echo ""
    echo "================================================================"
    echo "  Phase: microbench (Experiment 1 - Ablation)"
    echo "  Registries 0-1, Host codes 0-14"
    echo "================================================================"
    for registry_num in 0 1; do
        local num_cases=${REGISTRY_SIZES[$registry_num]}
        for (( test_num=0; test_num<num_cases; test_num++ )); do
            for (( host_code_num=0; host_code_num<15; host_code_num++ )); do
                run_one "$test_num" "$host_code_num" "$registry_num"
            done
        done
    done
}

run_throughput() {
    echo ""
    echo "================================================================"
    echo "  Phase: throughput (Experiment 2 - Pure Throughput)"
    echo "  Registries 2-5, Host codes 0-2"
    echo "================================================================"
    for registry_num in 2 3 4 5; do
        local num_cases=${REGISTRY_SIZES[$registry_num]}
        for (( test_num=0; test_num<num_cases; test_num++ )); do
            for host_code_num in 0 1 2; do
                run_one "$test_num" "$host_code_num" "$registry_num"
            done
        done
    done
}

run_scaling() {
    echo ""
    echo "================================================================"
    echo "  Phase: scaling (Experiment 3 - CDA Scaling)"
    echo "  Registries 6-9, Host code 2"
    echo "================================================================"
    for registry_num in 6 7 8 9; do
        local num_cases=${REGISTRY_SIZES[$registry_num]}
        for (( test_num=0; test_num<num_cases; test_num++ )); do
            run_one "$test_num" 2 "$registry_num"
        done
    done
}

run_ultralowdensity() {
    echo ""
    echo "================================================================"
    echo "  Phase: ultralowdensity (Ultra-Low Density Throughput)"
    echo "  Registries 10-11, Host codes 0-2"
    echo "================================================================"
    for registry_num in 10 11; do
        local num_cases=${REGISTRY_SIZES[$registry_num]}
        for (( test_num=0; test_num<num_cases; test_num++ )); do
            for host_code_num in 0 1 2; do
                run_one "$test_num" "$host_code_num" "$registry_num"
            done
        done
    done
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "SC26 Block SpMM Profiling"
echo "Phase: $PHASE"
echo ""

case "$PHASE" in
    microbench)       run_microbench ;;
    throughput)       run_throughput ;;
    scaling)          run_scaling ;;
    ultralowdensity)  run_ultralowdensity ;;
    all)
        run_microbench
        run_throughput
        run_scaling
        run_ultralowdensity
        ;;
esac

echo ""
echo "================================================================"
echo "  Done. Total runs completed: $TOTAL_RUNS"
echo "================================================================"
