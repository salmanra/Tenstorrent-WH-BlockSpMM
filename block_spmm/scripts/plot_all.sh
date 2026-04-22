#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run all SC26 plotting scripts. Each script writes its figures to
# scripts/figures/.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PLOT_SCRIPTS=(
    plot_load_imbalance.py
    plot_load_imbalance_comparison.py
    plot_load_imbalance_random.py
    plot_load_imbalance_upper.py
    plot_mixed_sweep.py
    plot_multi_diag_ultra_sparse.py
    plot_sweep_pattern.py
)

for script in "${PLOT_SCRIPTS[@]}"; do
    echo ""
    echo "==> $script"
    python3 "$SCRIPT_DIR/$script"
done

echo ""
echo "================================================================"
echo "  All plots generated. See $SCRIPT_DIR/figures/"
echo "================================================================"
