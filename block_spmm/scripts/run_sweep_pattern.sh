#!/usr/bin/bash
# run_sweep_pattern.sh - Profile DDA across all pattern-by-density registries
#
# Covers the same registries used in the GPU sweep_pattern CSVs:
#   R=C=256: regs 2-5   (PatternD5/D10/D25/D50)
#   R=C=128: regs 13-16 (PatternD5_128/D10_128/D25_128/D50_128)
#   R=C=32:  regs 17-22 (PatternUltra32_30/100/300/1000/3000/10000)
#   R=C=64:  regs 23-28 (PatternUltra64_60/200/600/2000/6000/10000)
#
# All 4 cases per registry, DDA only (host code 2).
# Total: 20 registries × 4 cases = 80 runs.

set -euo pipefail

TT_METAL_DIR="/home/user/tt-metal"
PROFILE_BIN="$TT_METAL_DIR/build/programming_examples/rahmy/profile_block_spmm"
EXPORT_BIN="$TT_METAL_DIR/build/programming_examples/rahmy/export_block_spmm_to_csv"

OUTPUT_DIR="profiles_sweep_pattern"
DRY_RUN=0
NO_BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --no-build)    NO_BUILD=1; shift ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

REGISTRIES=(17 18 19 20 21 22 23 24 25 26 27 28 2 3 4 5 13 14 15 16)
HC=2  # DDA
COUNT=0

function rebuild {
    if [[ "$NO_BUILD" == "1" ]]; then return; fi
    tt-smi -r > /dev/null
    pushd "$TT_METAL_DIR" > /dev/null
    ./build_metal.sh --enable-profiler --build-programming-examples > build.log 2> build_err.log
    local rc=$?
    popd > /dev/null
    if [[ $rc -ne 0 ]]; then
        echo "[build] FAILED — see $TT_METAL_DIR/build_err.log"
        exit 1
    fi
}

for reg in "${REGISTRIES[@]}"; do
    for pc in 0 1 2 3; do
        COUNT=$(( COUNT + 1 ))
        echo "[$COUNT/80] registry=$reg  case=$pc  hc=$HC"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [dry-run] TT_METAL_DEVICE_PROFILER=1 profile_block_spmm $pc $HC $reg $OUTPUT_DIR"
            echo "  [dry-run] export_block_spmm_to_csv $pc $HC $reg $OUTPUT_DIR"
            continue
        fi
        rebuild
        TT_METAL_DEVICE_PROFILER=1 "$PROFILE_BIN" "$pc" "$HC" "$reg" "$OUTPUT_DIR"
        "$EXPORT_BIN" "$pc" "$HC" "$reg" "$OUTPUT_DIR"
    done
done

echo "=== run_sweep_pattern.sh complete ($COUNT runs) ==="
