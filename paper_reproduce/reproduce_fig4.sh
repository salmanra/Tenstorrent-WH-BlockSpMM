#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Reproduce Fig. 4: DDA TFLOP/s on Wormhole N150 for semi-sparse BSR SpMM.
#
# Paper reports DDA-only numbers at 8192x8192x8192, BF16/HiFi4, across four
# sparsity patterns (Row, Col, Banded=multi_diag, Random):
#   R=C=128 block, densities 5%, 10%, 25%, 50%
#   R=C=256 block, densities 5%, 10%, 25%, 50%
#
# Registries (see Tenstorrent-WH-BlockSpMM/block_spmm/inc/profiling_suite.hpp):
#   13 PatternD5_128   14 PatternD10_128   15 PatternD25_128   16 PatternD50_128
#    2 PatternD5        3 PatternD10        4 PatternD25        5 PatternD50
# Each registry has 4 cases: test 0=row, 1=col, 2=multi_diag, 3=random.
# Host code 2 is snfin0_cdain1 (DDA in the paper).
#
# Total: 8 registries x 4 patterns = 32 runs.
###############################################################################

REPO_ROOT="${TT_METAL_HOME:?TT_METAL_HOME must be set — set it to your tt-metal checkout path}"

# Profile binary invokes ./capture-release (Tracy) and ./csvexport-release
# via std::system() relative to CWD. Ensure CWD is REPO_ROOT so symlinks resolve.
cd "$REPO_ROOT"
BUILD_DIR="${REPO_ROOT}/build/programming_examples/block_sparse"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"

REGISTRIES=(13 14 15 16 2 3 4 5)
TESTS=(0 1 2 3)   # row, col, multi_diag, random
HC=2              # DDA

OUTPUT_DIR="profiles_paper_fig4"
DRY_RUN=0
EXPORT_ONLY=0
PROFILE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --export-only)  EXPORT_ONLY=1; shift ;;
        --profile-only) PROFILE_ONLY=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --output-dir DIR  Output subdir under \$REPO_ROOT (default: profiles_paper_fig4)
  --dry-run         Print commands without executing
  --export-only     Skip profiling; re-run CSV export from existing traces
  --profile-only    Skip CSV export; only collect tracy traces
  -h, --help        Show this help
EOF
            exit 0 ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# Build command is overridable via BUILD_METAL_CMD env var (for standalone submodule use).
BUILD_METAL_CMD="${BUILD_METAL_CMD:-./build_metal.sh --enable-profiler --build-programming-examples}"

just_build() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  DRY-RUN: tt-smi -r && $BUILD_METAL_CMD"
        return
    fi
    echo "[build] tt-smi -r"
    tt-smi -r > /dev/null
    pushd "$REPO_ROOT" > /dev/null
    eval "$BUILD_METAL_CMD" > build.log 2> build_err.log
    popd > /dev/null
}

START_TIME=$(date +%s)
TOTAL_RUNS=$(( ${#REGISTRIES[@]} * ${#TESTS[@]} ))
RUN_IDX=0

echo "Fig. 4 reproduction: ${TOTAL_RUNS} runs (DDA only)"
echo "Output dir: \$REPO_ROOT/${OUTPUT_DIR}"
echo ""

for reg in "${REGISTRIES[@]}"; do
    for tc in "${TESTS[@]}"; do
        RUN_IDX=$(( RUN_IDX + 1 ))
        echo "[${RUN_IDX}/${TOTAL_RUNS}] registry=${reg} test=${tc} hc=${HC}"

        if [[ $EXPORT_ONLY -eq 0 ]]; then
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "  DRY-RUN: TT_METAL_DEVICE_PROFILER=1 $PROFILE_BIN $tc $HC $reg $OUTPUT_DIR"
            else
                just_build
                TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$tc" "$HC" "$reg" "$OUTPUT_DIR"
            fi
        fi

        if [[ $PROFILE_ONLY -eq 0 ]]; then
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "  DRY-RUN: $EXPORT_BIN $tc $HC $reg $OUTPUT_DIR"
            else
                "$EXPORT_BIN" "$tc" "$HC" "$reg" "$OUTPUT_DIR"
            fi
        fi
    done
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGG_OUT="${SCRIPT_DIR}/outputs/fig4_dda_throughput.csv"

echo ""
echo "=== Aggregating results into ${AGG_OUT} ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN: python $SCRIPT_DIR/aggregate_fig.py --figure 4 --output-dir $REPO_ROOT/$OUTPUT_DIR --out-csv $AGG_OUT"
else
    python "$SCRIPT_DIR/aggregate_fig.py" \
        --figure 4 \
        --output-dir "$REPO_ROOT/$OUTPUT_DIR" \
        --out-csv "$AGG_OUT" || true
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf "\n=== Fig. 4 reproduction complete (%d runs) ===\n" "$TOTAL_RUNS"
printf "Total elapsed: %dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
