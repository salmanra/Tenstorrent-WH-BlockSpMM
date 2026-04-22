#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Reproduce Fig. 3: DDA TFLOP/s on Wormhole N150 for ultra-sparse BSR SpMM.
#
# Paper reports DDA-only numbers at 8192x8192x8192, BF16/HiFi4, across four
# sparsity patterns (Row, Col, Banded=multi_diag, Random):
#   R=C=32  block, densities 0.003%, 0.03%, 0.3%     (3 density bins)
#   R=C=64  block, densities 0.006%, 0.06%, 0.6%, 1% (4 density bins)
#
# Registries (see SC26_submission/block_spmm/inc/profiling_suite.hpp):
#   17 PatternUltra32_30     (d = 0.003%)
#   19 PatternUltra32_300    (d = 0.03%)
#   21 PatternUltra32_3000   (d = 0.3%)
#   23 PatternUltra64_60     (d = 0.006%)
#   25 PatternUltra64_600    (d = 0.06%)
#   27 PatternUltra64_6000   (d = 0.6%)
#   28 PatternUltra64_10000  (d = 1%)
# Each registry has 4 cases, one per pattern: test 0=row, 1=col, 2=multi_diag, 3=random.
# Host code 2 is snfin0_cdain1 (DDA in the paper).
#
# Total: 7 registries x 4 patterns = 28 runs.
###############################################################################

REPO_ROOT="/home/user/tt-metal"
BUILD_DIR="${REPO_ROOT}/build/programming_examples/rahmy"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"

REGISTRIES=(17 19 21 23 25 27 28)
TESTS=(0 1 2 3)   # row, col, multi_diag, random
HC=2              # DDA

OUTPUT_DIR="profiles_paper_fig3"
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
  --output-dir DIR  Output subdir under \$REPO_ROOT (default: profiles_paper_fig3)
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

echo "Fig. 3 reproduction: ${TOTAL_RUNS} runs (DDA only)"
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
AGG_OUT="${SCRIPT_DIR}/outputs/fig3_dda_throughput.csv"

echo ""
echo "=== Aggregating results into ${AGG_OUT} ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN: python $SCRIPT_DIR/aggregate_fig.py --figure 3 --output-dir $REPO_ROOT/$OUTPUT_DIR --out-csv $AGG_OUT"
else
    python "$SCRIPT_DIR/aggregate_fig.py" \
        --figure 3 \
        --output-dir "$REPO_ROOT/$OUTPUT_DIR" \
        --out-csv "$AGG_OUT" || true
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
printf "\n=== Fig. 3 reproduction complete (%d runs) ===\n" "$TOTAL_RUNS"
printf "Total elapsed: %dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
