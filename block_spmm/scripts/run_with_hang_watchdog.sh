#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT" || exit 97

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <test_binary> [args...]" >&2
    exit 64
fi

export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_ROOT="$REPO_ROOT/logs/bspmm"
RUN_ID="$(date +%Y%m%d_%H%M%S).$$"
RUN_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
HOST_LOG="$RUN_DIR/host.log"

CHIP_ID="${TT_METAL_BSPMM_DEVICE_ID:-${TT_METAL_DPRINT_CHIPS:-0}}"
export TT_METAL_BSPMM_DEVICE_ID="$CHIP_ID"
export TT_METAL_DPRINT_CHIPS="${TT_METAL_DPRINT_CHIPS:-$CHIP_ID}"
export TT_METAL_DPRINT_RISCVS="${TT_METAL_DPRINT_RISCVS:-BR+NCRISC+TRISC0+TRISC1+TRISC2}"
export TT_METAL_DPRINT_FILE="${TT_METAL_DPRINT_FILE:-$RUN_DIR/dprint.log}"
export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC="${TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC:-1}"
export TT_METAL_WATCHER="${TT_METAL_WATCHER:-5}"
export TT_METAL_WATCHER_APPEND="${TT_METAL_WATCHER_APPEND:-1}"
export BSPMM_HEARTBEAT="${BSPMM_HEARTBEAT:-1}"

WATCHER_LOG="$REPO_ROOT/generated/watcher/watcher.log"
HANG_SECONDS="${BSPMM_WATCHDOG_HANG_SECONDS:-180}"
POLL_SECONDS="${BSPMM_WATCHDOG_POLL_SECONDS:-5}"
KILL_GRACE_SECONDS="${BSPMM_WATCHDOG_KILL_GRACE_SECONDS:-10}"

log() {
    echo "[watchdog] $*" | tee -a "$HOST_LOG"
}

file_stamp() {
    local path="$1"
    if [[ -e "$path" ]]; then
        stat -c '%s:%Y' "$path" 2>/dev/null || echo "0:0"
    else
        echo "0:0"
    fi
}

collect_evidence() {
    log "collecting hang evidence in $RUN_DIR"
    if [[ -f "$WATCHER_LOG" ]]; then
        cp "$WATCHER_LOG" "$RUN_DIR/watcher.log" 2>/dev/null || true
        log "watcher log copied from $WATCHER_LOG"
    else
        log "watcher log not found at $WATCHER_LOG"
    fi
    if [[ -f "$TT_METAL_DPRINT_FILE" ]]; then
        log "dprint log: $TT_METAL_DPRINT_FILE"
    else
        log "dprint log not found at $TT_METAL_DPRINT_FILE"
    fi
    if [[ -f "$REPO_ROOT/tools/tt-triage.py" ]]; then
        log "running tt-triage while the process is still alive"
        timeout 180 python3 "$REPO_ROOT/tools/tt-triage.py" \
            --skip-version-check \
            --disable-colors \
            --disable-progress \
            --llm-output \
            --llm-output-path "$RUN_DIR/triage.csv" \
            --triage-summary-path "$RUN_DIR/triage_summary.txt" \
            > "$RUN_DIR/triage.stdout" 2>&1
        local triage_rc=$?
        log "tt-triage exit code: $triage_rc (stdout: $RUN_DIR/triage.stdout)"
    fi
}

reset_device() {
    local smi=()
    if command -v tt-smi >/dev/null 2>&1; then
        smi=(tt-smi)
    elif command -v uvx >/dev/null 2>&1; then
        smi=(uvx tt-smi@latest)
    else
        log "tt-smi unavailable; cannot reset device"
        return 1
    fi

    log "tt-smi device list before reset:"
    timeout 120 "${smi[@]}" -ls 2>&1 | tee "$RUN_DIR/tt_smi_before_reset.log" | tee -a "$HOST_LOG" || true

    local dev_path=""
    if [[ -n "${TT_METAL_BSPMM_DEVICE_ID:-}" ]]; then
        dev_path="$(awk -v id="$TT_METAL_BSPMM_DEVICE_ID" '$2 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\/dev\/tenstorrent\//) { print $i; exit } }' "$RUN_DIR/tt_smi_before_reset.log")"
    fi
    if [[ -z "$dev_path" ]]; then
        dev_path="$(grep -oE '/dev/tenstorrent/[0-9]+' "$RUN_DIR/tt_smi_before_reset.log" | head -n 1 || true)"
    fi

    if [[ -z "$dev_path" ]]; then
        log "could not determine a /dev/tenstorrent/<id> reset target; refusing to reset"
        return 1
    fi

    log "resetting $dev_path with ${smi[*]}"
    timeout 180 "${smi[@]}" -r "$dev_path" > "$RUN_DIR/tt_smi_reset.log" 2>&1
    local reset_rc=$?
    cat "$RUN_DIR/tt_smi_reset.log" | tee -a "$HOST_LOG"
    log "tt-smi reset exit code: $reset_rc"

    log "tt-smi device list after reset:"
    timeout 120 "${smi[@]}" -ls 2>&1 | tee "$RUN_DIR/tt_smi_after_reset.log" | tee -a "$HOST_LOG" || true
    return "$reset_rc"
}

log "run dir: $RUN_DIR"
log "command: $*"
log "chip: $CHIP_ID dprint: $TT_METAL_DPRINT_FILE watcher: $WATCHER_LOG"

stdbuf -oL -eL "$@" > >(stdbuf -oL tee "$HOST_LOG") 2>&1 &
TEST_PID=$!

last_host_stamp="$(file_stamp "$HOST_LOG")"
last_dprint_stamp="$(file_stamp "$TT_METAL_DPRINT_FILE")"
last_change_epoch="$(date +%s)"
hang=0

while kill -0 "$TEST_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    host_stamp="$(file_stamp "$HOST_LOG")"
    dprint_stamp="$(file_stamp "$TT_METAL_DPRINT_FILE")"
    now="$(date +%s)"
    if [[ "$host_stamp" != "$last_host_stamp" || "$dprint_stamp" != "$last_dprint_stamp" ]]; then
        last_host_stamp="$host_stamp"
        last_dprint_stamp="$dprint_stamp"
        last_change_epoch="$now"
    elif (( now - last_change_epoch >= HANG_SECONDS )); then
        hang=1
        break
    fi
done

if (( hang )); then
    log "HANG: no host/DPRINT log progress for ${HANG_SECONDS}s (pid $TEST_PID still alive)"
    collect_evidence
    log "sending TERM to pid $TEST_PID"
    kill -TERM "$TEST_PID" 2>/dev/null || true
    grace_deadline=$(( $(date +%s) + KILL_GRACE_SECONDS ))
    while kill -0 "$TEST_PID" 2>/dev/null && (( $(date +%s) < grace_deadline )); do
        sleep 1
    done
    if kill -0 "$TEST_PID" 2>/dev/null; then
        log "sending KILL to pid $TEST_PID"
        kill -KILL "$TEST_PID" 2>/dev/null || true
    fi
    wait "$TEST_PID" 2>/dev/null
    reset_device
    log "exiting 124 after hang handling; artifacts in $RUN_DIR"
    exit 124
fi

wait "$TEST_PID"
rc=$?
if [[ -f "$WATCHER_LOG" ]]; then
    cp "$WATCHER_LOG" "$RUN_DIR/watcher.log" 2>/dev/null || true
fi
log "process exited with code $rc; artifacts in $RUN_DIR"
exit "$rc"
