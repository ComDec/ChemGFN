#!/usr/bin/env bash
# =============================================================================
# Quick eval: load each sweep checkpoint, generate 6400 samples, compute metrics
# Uses eval_rl_baseline.py which is faster than the full eval.py pipeline
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

GPUS=(0 1 2 3 4 5 6 7)
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
RESULTS_DIR="results/rebuttal_sweep"
N_SAMPLES=6400
BATCH_SIZE=64

mkdir -p "${RESULTS_DIR}"

per_batch=${#GPUS[@]}
pids=()
idx=0

wait_batch() {
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  pids=()
}

echo "=== Quick Eval: ${N_SAMPLES} samples per checkpoint ==="

for d in logs/train/sweep_*/train/runs/*/; do
  name=$(echo "$d" | grep -oP 'sweep_[^/]+')
  ckpt="${d}checkpoints/last.ckpt"

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] $name: no checkpoint"
    continue
  fi

  # Skip if result already exists
  if [[ -f "${RESULTS_DIR}/${name}.csv" ]]; then
    echo "[CACHED] $name"
    continue
  fi

  gpu=${GPUS[$((idx % per_batch))]}
  out_dir="${RESULTS_DIR}/eval_${name}"

  echo "[EVAL] GPU ${gpu}: $name → $out_dir"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} scripts/eval_rl_baseline.py \
    --model_path "$(realpath $ckpt)" \
    --exp_name "${name}" \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --max_len 15 \
    --temperature 1.0 \
    --output_dir "${out_dir}" \
    --seed 42 \
    --test_repeats 3 \
    --scorer hit24_dense \
    --buffer_path "${BUFFER_PATH}" \
    2>&1 | tail -5 &

  pids+=($!)
  idx=$((idx + 1))

  if (( ${#pids[@]} >= per_batch )); then
    echo "  [Waiting...]"
    wait_batch
  fi
done

wait_batch
echo ""
echo "=== Eval Done ==="

# Now compute Table 3 metrics from the generated per-sample CSVs
echo ""
echo "=== Computing Table 3 metrics ==="

for out_dir in ${RESULTS_DIR}/eval_sweep_*/; do
  name=$(basename "$out_dir" | sed 's/eval_//')

  # Find the per-sample CSV
  csv=$(find "$out_dir" -name "per_sample_*.csv" -type f 2>/dev/null | sort | tail -1)
  if [[ -z "$csv" ]]; then
    csv=$(find "$out_dir" -name "*.csv" -type f 2>/dev/null | sort | tail -1)
  fi

  if [[ -z "$csv" ]]; then
    echo "[SKIP] $name: no CSV"
    continue
  fi

  echo "[METRICS] $name: $csv"
  ${PYTHON} scripts/sweep/eval_expr24_table3.py \
    --csv-path "$csv" \
    --buffer-path "${BUFFER_PATH}" \
    --max-seq-len 9 \
    --output-csv "${RESULTS_DIR}/${name}.csv" \
    2>&1 | grep -E "Acc:|NormCov:|KL|JS|pterm|Unique"
  echo ""
done

# Final merge
echo "=== Merging ==="
${PYTHON} -c "
import pandas as pd
from pathlib import Path
results_dir = Path('${RESULTS_DIR}')
dfs = []
for f in sorted(results_dir.glob('sweep_*.csv')):
    df = pd.read_csv(f)
    df.insert(0, 'run', f.stem)
    name = f.stem
    for part in name.split('_'):
        if part.startswith('b') and len(part) > 1 and part[1:].replace('.','').isdigit():
            df['beta'] = float(part[1:])
        if part.startswith('r') and len(part) > 1:
            try:
                df['rho'] = float(part[1:])
            except ValueError:
                pass
    dfs.append(df)

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(results_dir / 'all_sweep.csv', index=False)
    cols = ['run','Acc','Unique_valid','NormCov','KL(pi->p*)','KL(p*->pi)','JS_tok','log_pterm']
    avail = [c for c in cols if c in merged.columns]
    print()
    print('=' * 90)
    print('SWEEP RESULTS')
    print('=' * 90)
    print(merged[avail].to_string(index=False))
    print(f'\nSaved: {results_dir}/all_sweep.csv')
else:
    print('No results found')
"
