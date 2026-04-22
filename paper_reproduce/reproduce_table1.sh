#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Reproduce Table 1: per-case DRAM read counts and throughput on Wormhole N150
# for three SpMM algorithms (Naive, SnF, DDA) on 6 cases:
#
# Sparse-pattern cases:
#   (a) Row,      R=C=256, d=25%   reg  4 test 0
#   (b) Banded,   R=C=256, d=50%   reg  5 test 2 (multi_diag)
#   (c) Row,      R=C=64,  d=0.6%  reg 27 test 0
#   (d) Diagonal, R=C=64,  d=0.6%  reg 27 test 2 (multi_diag)
#
# Triangular cases (M=N=K=8192):
#   (e) Lower-triangular, R=C=256  reg 29 test 0
#   (f) Upper-triangular, R=C=256  reg 30 test 0
#
# Two measurements per row:
#   - TFLOP/s: profile_block_spmm (Tracy, HostCodeRegistryProfiling indices 0/1/2)
#   - DRAM reads: run_block_spmm with DPRINT, parsed by count_dram_reads.py
#                 (HostCodeRegistryVerbose indices 0/1/2 for Naive/SnF/CDA)
#
# Total: 18 throughput runs (6 cases x 3 algos) +
#        12 DPRINT runs for sparse cases (table1_sparse_cases group) +
#         6 DPRINT runs for triangular cases (table1_triangular_cases group).
###############################################################################

REPO_ROOT="${TT_METAL_HOME:?TT_METAL_HOME must be set — set it to your tt-metal checkout path}"

# Profile binary invokes ./capture-release (Tracy) and ./csvexport-release
# via std::system() relative to CWD. Ensure CWD is REPO_ROOT so symlinks resolve.
cd "$REPO_ROOT"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLKSPMM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/programming_examples/block_sparse"
PROFILE_BIN="${BUILD_DIR}/profile_block_spmm"
EXPORT_BIN="${BUILD_DIR}/export_block_spmm_to_csv"
SCRIPTS_DIR="${BLKSPMM_ROOT}/block_spmm/scripts"
COUNT_DRAM="${SCRIPTS_DIR}/count_dram_reads.py"

# 6 Table-1 cases: (registry, test, label)
CASES=(
    "4  0 Row_R256_d25"
    "5  2 Banded_R256_d50"
    "27 0 Row_R64_d0.6"
    "27 2 Diagonal_R64_d0.6"
    "29 0 LowerTri_R256"
    "30 0 UpperTri_R256"
)
# Profiling host-code indices (HostCodeRegistryProfiling): 0=Naive, 1=SnF, 2=CDA(DDA)
ALGOS=(0 1 2)

OUTPUT_DIR="profiles_paper_table1"
DRAM_READS_OUT_DIR="${REPO_ROOT}/profiles_paper_table1/dram_reads"
DRY_RUN=0
EXPORT_ONLY=0
PROFILE_ONLY=0
SKIP_DRAM=0
SKIP_THROUGHPUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --dram-out-dir)     DRAM_READS_OUT_DIR="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --export-only)      EXPORT_ONLY=1; shift ;;
        --profile-only)     PROFILE_ONLY=1; shift ;;
        --skip-dram)        SKIP_DRAM=1; shift ;;
        --skip-throughput)  SKIP_THROUGHPUT=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --output-dir DIR     Throughput output subdir under \$REPO_ROOT
                       (default: profiles_paper_table1)
  --dram-out-dir DIR   DRAM-reads CSV output dir
                       (default: \$REPO_ROOT/profiles_paper_table1/dram_reads)
  --dry-run            Print commands without executing
  --export-only        Skip profiling; re-run CSV export from existing traces
  --profile-only       Skip CSV export; only collect tracy traces
  --skip-dram          Skip DRAM-reads phase (throughput only)
  --skip-throughput    Skip throughput phase (DRAM reads only)
  -h, --help           Show this help
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
THROUGHPUT_RUNS=$(( ${#CASES[@]} * ${#ALGOS[@]} ))
RUN_IDX=0

echo "Table 1 reproduction"
echo "  Throughput runs:   ${THROUGHPUT_RUNS} (6 cases x 3 algos)"
echo "  DRAM-reads runs:   18 (via count_dram_reads.py, two groups)"
echo "  Throughput output: \$REPO_ROOT/${OUTPUT_DIR}"
echo "  DRAM-reads output: ${DRAM_READS_OUT_DIR}"
echo ""

# ── Phase 1: Throughput ───────────────────────────────────────────────────────
if [[ $SKIP_THROUGHPUT -eq 0 ]]; then
    echo "=== Phase 1/2: throughput ==="
    for case_spec in "${CASES[@]}"; do
        read -r reg tc label <<< "$case_spec"
        for hc in "${ALGOS[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            echo "[${RUN_IDX}/${THROUGHPUT_RUNS}] ${label}  reg=${reg} test=${tc} hc=${hc}"

            if [[ $EXPORT_ONLY -eq 0 ]]; then
                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "  DRY-RUN: TT_METAL_DEVICE_PROFILER=1 $PROFILE_BIN $tc $hc $reg $OUTPUT_DIR"
                else
                    just_build
                    TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$tc" "$hc" "$reg" "$OUTPUT_DIR"
                fi
            fi

            if [[ $PROFILE_ONLY -eq 0 ]]; then
                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "  DRY-RUN: $EXPORT_BIN $tc $hc $reg $OUTPUT_DIR"
                else
                    "$EXPORT_BIN" "$tc" "$hc" "$reg" "$OUTPUT_DIR"
                fi
            fi
        done
    done
    echo ""
fi

# ── Phase 2: DRAM reads (DPRINT) ──────────────────────────────────────────────
if [[ $SKIP_DRAM -eq 0 ]]; then
    echo "=== Phase 2/2: DRAM reads (DPRINT) ==="
    mkdir -p "$DRAM_READS_OUT_DIR"
    for group in table1_sparse_cases table1_triangular_cases; do
        echo "[group] ${group}"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  DRY-RUN: python $COUNT_DRAM --group $group --output-dir $DRAM_READS_OUT_DIR"
        else
            python "$COUNT_DRAM" --group "$group" --output-dir "$DRAM_READS_OUT_DIR"
        fi
    done
    echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGG_OUT="${SCRIPT_DIR}/outputs/table1_dram_and_throughput.csv"

echo "=== Aggregating results into ${AGG_OUT} ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN: python $SCRIPT_DIR/aggregate_table1.py --output-dir $REPO_ROOT/$OUTPUT_DIR --dram-dir $DRAM_READS_OUT_DIR --out-csv $AGG_OUT"
else
    python "$SCRIPT_DIR/aggregate_table1.py" \
        --output-dir "$REPO_ROOT/$OUTPUT_DIR" \
        --dram-dir   "$DRAM_READS_OUT_DIR" \
        --out-csv    "$AGG_OUT" || true
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo "=== Table 1 reproduction complete ==="
printf "Total elapsed: %dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
