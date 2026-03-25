#!/usr/bin/env bash
# =============================================================================
# Evaluate all sweep checkpoints NOW (early stop eval)
# Runs eval.py on each checkpoint to generate test CSVs,
# then computes Table 3 metrics.
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

GPUS=(0 1 2 3 4 5 6 7)
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
RESULTS_DIR="results/rebuttal_sweep"
mkdir -p "${RESULTS_DIR}"

per_batch=${#GPUS[@]}
pids=()
idx=0

timestamp() { date "+%H:%M:%S"; }

wait_batch() {
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  pids=()
}

echo "=== EVAL PHASE: generating test CSVs from checkpoints ==="

# Find all sweep checkpoints and run eval.py
for d in logs/train/sweep_*/train/runs/*/; do
  name=$(echo "$d" | grep -oP 'sweep_[^/]+')
  ckpt="${d}checkpoints/last.ckpt"

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] $name: no checkpoint"
    continue
  fi

  # Skip if test CSV already exists
  existing_csv=$(find "$(dirname $(dirname $d))" -name "samples_test_*.csv" -type f 2>/dev/null | head -1)
  if [[ -n "$existing_csv" ]]; then
    echo "[CACHED] $name: $existing_csv"
    continue
  fi

  config="${d}.hydra/config.yaml"
  gpu=${GPUS[$((idx % per_batch))]}

  echo "[$(timestamp)] EVAL GPU $gpu: $name"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/eval.py \
    --config-path="$(realpath ${d}.hydra)" \
    --config-name=config \
    ckpt_path="$(realpath $ckpt)" \
    test_repeats=1 \
    trainer.devices=1 \
    logger.wandb.offline=true \
    2>&1 | tail -3 &

  pids+=($!)
  idx=$((idx + 1))

  if (( ${#pids[@]} >= per_batch )); then
    echo "  [Waiting for eval batch...]"
    wait_batch
  fi
done

wait_batch
echo ""
echo "=== EVAL PHASE DONE ==="
echo ""

# Now compute Table 3 metrics from all CSVs
echo "=== METRICS PHASE: computing Table 3 metrics ==="

for d in logs/train/sweep_*/; do
  name=$(basename "$d")
  csv=$(find "$d" -name "samples_test_*.csv" -type f 2>/dev/null | sort -t_ -k3 -rn | head -1)

  if [[ -z "$csv" ]]; then
    echo "[SKIP] $name: no test CSV"
    continue
  fi

  if [[ -f "${RESULTS_DIR}/${name}.csv" ]]; then
    echo "[CACHED] $name"
    continue
  fi

  echo "[METRICS] $name"
  ${PYTHON} scripts/sweep/eval_expr24_table3.py \
    --csv-path "$csv" \
    --buffer-path "${BUFFER_PATH}" \
    --max-seq-len 9 \
    --output-csv "${RESULTS_DIR}/${name}.csv" \
    2>&1 | grep -E "Acc:|NormCov:|KL|JS|pterm|Unique"
  echo ""
done

# Merge
echo "=== MERGING RESULTS ==="
${PYTHON} -c "
import pandas as pd
from pathlib import Path
results_dir = Path('${RESULTS_DIR}')
dfs = []
for f in sorted(results_dir.glob('sweep_*.csv')):
    df = pd.read_csv(f)
    df.insert(0, 'run', f.stem)
    name = f.stem
    if '_r' in name and '_b' in name:
        parts = name.split('_')
        for p in parts:
            if p.startswith('b') and p[1:].replace('.','').isdigit():
                df['beta'] = float(p[1:])
            if p.startswith('r') and p[1:].replace('.','').isdigit():
                df['rho'] = float(p[1:])
    dfs.append(df)

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(results_dir / 'all_sweep.csv', index=False)
    cols = ['run','Acc','Unique_valid','NormCov','KL(pi->p*)','KL(p*->pi)','JS_tok','log_pterm']
    avail = [c for c in cols if c in merged.columns]
    print()
    print('=' * 90)
    print('SWEEP RESULTS (early checkpoint)')
    print('=' * 90)
    print(merged[avail].to_string(index=False))
    print()
    print(f'Saved: {results_dir}/all_sweep.csv')
"

echo ""
echo "=== ALL DONE ==="
