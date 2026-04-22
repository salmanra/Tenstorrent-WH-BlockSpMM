#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/user/tt-metal"
BUILD_DIR="${REPO_ROOT}/build/programming_examples/rahmy"

# Defaults
DO_BUILD=true
DO_PROFILE=true
DO_EXPORT=true
DRY_RUN=false
PHASE="all"   # all, ablation, sweep
REGISTRIES=(1 2 3 4)
ABLATION_REGISTRY=0

# Registry sizes (number of test cases per registry)
declare -A REGISTRY_SIZES=(
    [0]=4   # ProfileCaseRegistry
    [1]=5   # ProfileSweepNRegistry
    [2]=5   # ProfileSweepDensityRegistry
    [3]=5   # ProfileSweepKRegistry
    [4]=4   # ProfileSweepBlockSizeRegistry
)

declare -A REGISTRY_NAMES=(
    [0]="SDDMMProfileSuite"
    [1]="SDDMMSweepN"
    [2]="SDDMMSweepDensity"
    [3]="SDDMMSweepK"
    [4]="SDDMMSweepBlockSize"
)

# Host code indices:
#   [0-1]   full (naive, CDA)
#   [2-3]   no_b_read
#   [4-5]   no_c_read
#   [6-7]   no_d_read
#   [8-9]   no_compute
#   [10-11] no_write
NUM_ALGORITHMS=2
NUM_HOST_CODES=12
HOST_CODES=()  # empty = run all for the selected phase

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run SDDMM profiling across sweep registries and/or ablation variants.
Each test case is profiled in a separate process invocation.

Phases:
  all       Run both sweep and ablation phases (default)
  sweep     Run base algorithms (host codes 0-1) against sweep registries
  ablation  Run all 5 skip variants x 2 algorithms against a reference registry

Registries (for sweep phase):
  0  ProfileCaseRegistry          (4 small cases)
  1  ProfileSweepNRegistry        (5 cases, sweep dense output width N)
  2  ProfileSweepDensityRegistry  (5 cases, sweep sparsity density)
  3  ProfileSweepKRegistry        (5 cases, sweep reduction dimension K)
  4  ProfileSweepBlockSizeRegistry (4 cases, sweep block size R=C)

Ablation host codes (indices 2-11):
  2-3   no_b_read   — skip sparse B mask reads
  4-5   no_c_read   — skip dense C reads
  6-7   no_d_read   — skip dense D reads
  8-9   no_compute  — skip matmul + Hadamard
  10-11 no_write    — skip output DRAM writes

Options:
  --phase PHASE          Phase to run: all, ablation, sweep (default: all)
  --no-build             Skip cmake build step
  --profile-only         Run profiling only (skip CSV export)
  --export-only          Run CSV export only (skip profiling)
  --registry N           Only run sweep registry N (0-4); can be repeated
  --ablation-registry N  Registry to use for ablation test cases (default: 0)
  --host-code N          For sweep: algorithm index (0-1). For ablation: algorithm
                         offset (0=naive, 1=CDA). Can be repeated.
  --dry-run              Print commands without executing
  --list                 List all registries and host codes
  -h, --help             Show this help
EOF
    exit 0
}

list_all() {
    echo "=== Sweep Registries ==="
    for reg in 0 1 2 3 4; do
        echo "  [$reg] ${REGISTRY_NAMES[$reg]} (${REGISTRY_SIZES[$reg]} cases)"
    done
    echo ""
    echo "=== Host Codes (HostCodeRegistryProfiling) ==="
    echo "  [0]  bsr_sddmm_multicore_naive"
    echo "  [1]  bsr_sddmm_multicore_CDA"
    echo "  [2]  bsr_sddmm_multicore_naive_no_b_read"
    echo "  [3]  bsr_sddmm_multicore_CDA_no_b_read"
    echo "  [4]  bsr_sddmm_multicore_naive_no_c_read"
    echo "  [5]  bsr_sddmm_multicore_CDA_no_c_read"
    echo "  [6]  bsr_sddmm_multicore_naive_no_d_read"
    echo "  [7]  bsr_sddmm_multicore_CDA_no_d_read"
    echo "  [8]  bsr_sddmm_multicore_naive_no_compute"
    echo "  [9]  bsr_sddmm_multicore_CDA_no_compute"
    echo "  [10] bsr_sddmm_multicore_naive_no_write"
    echo "  [11] bsr_sddmm_multicore_CDA_no_write"
    exit 0
}

# Parse arguments
CUSTOM_REGISTRIES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)            PHASE="$2"; shift 2 ;;
        --no-build)         DO_BUILD=false; shift ;;
        --profile-only)     DO_EXPORT=false; shift ;;
        --export-only)      DO_PROFILE=false; shift ;;
        --registry)         CUSTOM_REGISTRIES+=("$2"); shift 2 ;;
        --ablation-registry) ABLATION_REGISTRY="$2"; shift 2 ;;
        --host-code)        HOST_CODES+=("$2"); shift 2 ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --list)             list_all ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ${#CUSTOM_REGISTRIES[@]} -gt 0 ]]; then
    REGISTRIES=("${CUSTOM_REGISTRIES[@]}")
fi

run_cmd() {
    echo "+ $*"
    if [[ $DRY_RUN == false ]]; then
        "$@"
    fi
}

just_build() {
    echo "[build] Reset board with tt-smi -r"
    tt-smi -r > /dev/null
    pushd "$REPO_ROOT" > /dev/null
    echo "[build] Building with profiling enabled..."
    ./build_metal.sh --enable-profiler --build-programming-examples > build.log 2> build_err.log
    local rc=$?
    popd > /dev/null
    if [[ $rc -ne 0 ]]; then
        echo "[build] FAILED — see $REPO_ROOT/build_err.log"
        exit 1
    fi
    echo "[build] Succeeded."
}

# Build once up front
if [[ "$DO_BUILD" == true ]]; then
    run_cmd just_build
fi

PROFILE_BIN="${BUILD_DIR}/profile_sddmm"
EXPORT_BIN="${BUILD_DIR}/export_sddmm"

# ── Sweep phase ──────────────────────────────────────────────────────
run_sweep() {
    local sweep_hcs=()
    if [[ ${#HOST_CODES[@]} -gt 0 ]]; then
        sweep_hcs=("${HOST_CODES[@]}")
    else
        for (( i=0; i<NUM_ALGORITHMS; i++ )); do
            sweep_hcs+=("$i")
        done
    fi

    for reg in "${REGISTRIES[@]}"; do
        num_cases=${REGISTRY_SIZES[$reg]}
        echo "=== Sweep: Registry ${reg} (${REGISTRY_NAMES[$reg]}, ${num_cases} cases) ==="
        for (( tc=0; tc<num_cases; tc++ )); do
            for hc in "${sweep_hcs[@]}"; do
                run_cmd just_build
                if [[ "$DO_PROFILE" == true ]]; then
                    run_cmd env TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$tc" "$hc" "$reg"
                fi
                if [[ "$DO_EXPORT" == true ]]; then
                    run_cmd "$EXPORT_BIN" "$tc" "$hc" "$reg"
                fi
            done
        done
        echo ""
    done
}

# ── Ablation phase ───────────────────────────────────────────────────
run_ablation() {
    local alg_offsets=()
    if [[ ${#HOST_CODES[@]} -gt 0 ]]; then
        alg_offsets=("${HOST_CODES[@]}")
    else
        for (( i=0; i<NUM_ALGORITHMS; i++ )); do
            alg_offsets+=("$i")
        done
    fi

    local reg=$ABLATION_REGISTRY
    local num_cases=${REGISTRY_SIZES[$reg]}
    echo "=== Ablation: Registry ${reg} (${REGISTRY_NAMES[$reg]}, ${num_cases} cases) ==="

    # First run the full algorithms as baseline
    for (( tc=0; tc<num_cases; tc++ )); do
        for alg in "${alg_offsets[@]}"; do
            local hc=$alg  # full algorithm = index 0 or 1
            run_cmd just_build
            if [[ "$DO_PROFILE" == true ]]; then
                run_cmd env TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$tc" "$hc" "$reg"
            fi
            if [[ "$DO_EXPORT" == true ]]; then
                run_cmd "$EXPORT_BIN" "$tc" "$hc" "$reg"
            fi
        done
    done

    # Then run ablation variants (5 skip groups x selected algorithms)
    # Ablation groups start at host code index 2
    for (( group=0; group<5; group++ )); do
        local group_base=$(( 2 + group * NUM_ALGORITHMS ))
        for (( tc=0; tc<num_cases; tc++ )); do
            for alg in "${alg_offsets[@]}"; do
                local hc=$(( group_base + alg ))
                run_cmd just_build
                if [[ "$DO_PROFILE" == true ]]; then
                    run_cmd env TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$tc" "$hc" "$reg"
                fi
                if [[ "$DO_EXPORT" == true ]]; then
                    run_cmd "$EXPORT_BIN" "$tc" "$hc" "$reg"
                fi
            done
        done
    done
    echo ""
}

# ── Execute selected phase ───────────────────────────────────────────
case "$PHASE" in
    all)
        run_sweep
        run_ablation
        ;;
    sweep)
        run_sweep
        ;;
    ablation)
        run_ablation
        ;;
    *)
        echo "Unknown phase: $PHASE (expected: all, ablation, sweep)"
        exit 1
        ;;
esac

echo "=== Done ==="
