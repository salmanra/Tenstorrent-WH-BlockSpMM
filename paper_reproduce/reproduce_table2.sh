#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Reproduce Table 2: effect of serpentine load balancing on three SpMM
# algorithms (Naive, SnF, DDA) across:
#
#   (a) Lower-triangular, R=C=256, M=N=K in {8192, 4096}  — reg 29 tests 0, 1
#   (b) Upper-triangular, R=C=256, M=N=K in {8192, 4096}  — reg 30 tests 0, 1
#   (c) Random 25% density, R=C=256, M=N=K=8192           — reg  4 test  3
#
# Each case runs 6 host codes:
#   HostCodeRegistryProfiling indices
#     0 = Naive (LB)       16 = Naive_no_lb
#     1 = SnF   (LB)       17 = SnF_no_lb
#     2 = CDA   (LB)       15 = CDA_no_lb
#
# Total: 12 + 12 + 6 = 30 profiling runs.
#
# This orchestrates the three existing scripts in
# SC26_submission/block_spmm/scripts/:
#   run_load_imbalance.sh         (lower-triangular)
#   run_load_imbalance_upper.sh   (upper-triangular)
#   run_load_imbalance_random.sh  (random 25%)
###############################################################################

REPO_ROOT="/home/user/tt-metal"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SC26_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="${SC26_ROOT}/block_spmm/scripts"

OUTPUT_PREFIX="profiles_paper_table2"
DRY_RUN=0
EXPORT_ONLY=0
PROFILE_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-prefix)  OUTPUT_PREFIX="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --export-only)    EXPORT_ONLY=1; shift ;;
        --profile-only)   PROFILE_ONLY=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --output-prefix STR  Prefix for each sub-script's output dir
                       (default: profiles_paper_table2)
                       Produces: {prefix}_lower, {prefix}_upper, {prefix}_random
  --dry-run            Pass --dry-run through to each sub-script
  --export-only        Pass --export-only through
  --profile-only       Pass --profile-only through
  -h, --help           Show this help
EOF
            exit 0 ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

PASSTHROUGH=()
[[ $DRY_RUN      -eq 1 ]] && PASSTHROUGH+=("--dry-run")
[[ $EXPORT_ONLY  -eq 1 ]] && PASSTHROUGH+=("--export-only")
[[ $PROFILE_ONLY -eq 1 ]] && PASSTHROUGH+=("--profile-only")

START_TIME=$(date +%s)

echo "Table 2 reproduction: 3 sub-phases, 30 total runs"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo ""

echo "=== Sub-phase 1/3: lower-triangular (12 runs) ==="
bash "${SCRIPTS_DIR}/run_load_imbalance.sh" \
    --output-dir "${OUTPUT_PREFIX}_lower" \
    "${PASSTHROUGH[@]}"
echo ""

echo "=== Sub-phase 2/3: upper-triangular (12 runs) ==="
bash "${SCRIPTS_DIR}/run_load_imbalance_upper.sh" \
    --output-dir "${OUTPUT_PREFIX}_upper" \
    "${PASSTHROUGH[@]}"
echo ""

echo "=== Sub-phase 3/3: random 25% density (6 runs) ==="
bash "${SCRIPTS_DIR}/run_load_imbalance_random.sh" \
    --output-dir "${OUTPUT_PREFIX}_random" \
    "${PASSTHROUGH[@]}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGG_OUT="${SCRIPT_DIR}/outputs/table2_load_balance.csv"

echo "=== Aggregating results into ${AGG_OUT} ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN: python $SCRIPT_DIR/aggregate_table2.py --output-prefix $REPO_ROOT/$OUTPUT_PREFIX --out-csv $AGG_OUT"
else
    python "$SCRIPT_DIR/aggregate_table2.py" \
        --output-prefix "$REPO_ROOT/$OUTPUT_PREFIX" \
        --out-csv       "$AGG_OUT" || true
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo "=== Table 2 reproduction complete (30 runs) ==="
printf "Total elapsed: %dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
