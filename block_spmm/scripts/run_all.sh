#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/run_profiling.sh" --phase all "$@"

python "$SCRIPT_DIR/plot_microbench.py"
python "$SCRIPT_DIR/plot_throughput.py"
python "$SCRIPT_DIR/plot_scaling.py"

echo "All profiling and plotting complete. Figures in $SCRIPT_DIR/figures/"
