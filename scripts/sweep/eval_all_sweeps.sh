#!/usr/bin/env bash
# =============================================================================
# Post-hoc evaluation: run eval_expr24_table3.py on all sweep/validation runs
# to produce Table 3-compatible metrics (NormCov, KL, JS, Unique_valid, log_pterm)
#
# This replaces the insufficient val/acc + val/diversity + val/loss reporting.
#
# Usage:
#   bash scripts/sweep/eval_all_sweeps.sh <logs_root_dir>
#
# Example:
#   bash scripts/sweep/eval_all_sweeps.sh logs/
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
EVAL_SCRIPT="scripts/sweep/eval_expr24_table3.py"
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
OUT_DIR="results/sweep_table3"

cd /data2/xw3763/gflow/ChemGFN

LOGS_ROOT="${1:-logs}"

mkdir -p "${OUT_DIR}"

echo "===== Evaluating all runs in ${LOGS_ROOT} ====="
echo "Output: ${OUT_DIR}/"
echo ""

# Find all test_samples directories (filter to Expr24 runs only)
found=0
for test_dir in $(find "${LOGS_ROOT}" -type d -name "test_samples" 2>/dev/null | grep -iE "expr24|VarExpr24|AvgPrefix|sweep|full_" | sort); do
    run_dir=$(dirname "${test_dir}")
    run_name=$(basename "${run_dir}")

    # Find latest CSV in test_samples/
    latest_csv=$(ls -t "${test_dir}"/samples_test_*.csv 2>/dev/null | head -1)
    if [[ -z "${latest_csv}" ]]; then
        echo "[SKIP] ${run_name}: no samples CSV found"
        continue
    fi

    out_csv="${OUT_DIR}/${run_name}.csv"
    if [[ -f "${out_csv}" ]]; then
        echo "[CACHED] ${run_name}"
        found=$((found + 1))
        continue
    fi

    echo "[EVAL] ${run_name}: ${latest_csv}"
    ${PYTHON} "${EVAL_SCRIPT}" \
        --csv-path "${latest_csv}" \
        --buffer-path "${BUFFER_PATH}" \
        --output-csv "${out_csv}" \
        2>&1 | tail -12

    echo ""
    found=$((found + 1))
done

echo "===== Done: evaluated ${found} runs ====="
echo ""

# Merge all individual CSVs into one summary
if [[ ${found} -gt 0 ]]; then
    echo "Merging into ${OUT_DIR}/all_table3.csv ..."
    ${PYTHON} -c "
import pandas as pd
from pathlib import Path
dfs = []
for f in sorted(Path('${OUT_DIR}').glob('*.csv')):
    if f.name == 'all_table3.csv':
        continue
    df = pd.read_csv(f)
    df.insert(0, 'run', f.stem)
    dfs.append(df)
if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv('${OUT_DIR}/all_table3.csv', index=False)
    print(merged[['run','Acc','Unique_valid','NormCov','KL(pi->p*)','KL(p*->pi)','JS_tok','log_pterm']].to_string(index=False))
else:
    print('No results to merge.')
"
fi
