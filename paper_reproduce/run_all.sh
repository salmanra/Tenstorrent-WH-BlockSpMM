#!/usr/bin/env bash
# Orchestrator: runs all 5 paper reproduction scripts sequentially.
# Each script's stdout/stderr is captured to logs/<name>.log.
# The summary file records success/failure and wall-time for each.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
SUMMARY="${SCRIPT_DIR}/logs/SUMMARY.txt"
mkdir -p "$LOG_DIR"

SCRIPTS=(
    reproduce_fig3.sh
    reproduce_fig4.sh
    reproduce_table1.sh
    reproduce_table2.sh
    reproduce_table3.sh
)

printf "Paper reproduction run started at %s\n\n" "$(date -Iseconds)" > "$SUMMARY"

OVERALL_START=$(date +%s)

for s in "${SCRIPTS[@]}"; do
    name="${s%.sh}"
    log="${LOG_DIR}/${name}.log"
    start=$(date +%s)
    printf "\n========== %s started at %s ==========\n" "$name" "$(date -Iseconds)" | tee -a "$SUMMARY"
    if bash "${SCRIPT_DIR}/${s}" > "$log" 2>&1; then
        status="OK"
    else
        status="FAIL (exit=$?)"
    fi
    end=$(date +%s)
    elapsed=$(( end - start ))
    printf "[%s]  %s  %dh %02dm %02ds  log=%s\n" \
        "$status" "$name" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "$log" | tee -a "$SUMMARY"
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$(( OVERALL_END - OVERALL_START ))
printf "\n========== ALL DONE at %s ==========\n" "$(date -Iseconds)" | tee -a "$SUMMARY"
printf "Total wall time: %dh %02dm %02ds\n" \
    $((OVERALL_ELAPSED/3600)) $(((OVERALL_ELAPSED%3600)/60)) $((OVERALL_ELAPSED%60)) | tee -a "$SUMMARY"
printf "\nAggregated CSVs:\n" | tee -a "$SUMMARY"
ls -la "${SCRIPT_DIR}/outputs/" 2>/dev/null | tee -a "$SUMMARY" || true
