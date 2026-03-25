#!/usr/bin/env bash
# =============================================================================
# Test-only: resume each sweep checkpoint to trigger test phase
# Since global_step=5000 == max_steps, training immediately skips to test.
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

GPUS=(0 1 2 3 4 5 6 7)
SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
RESULTS_DIR="results/rebuttal_sweep"
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

# Map run name → (beta, rho) overrides
declare -A OVERRIDES
OVERRIDES[sweep_b1_r0]="model.loss_fn.soft_beta=1 model.loss_fn.soft_rho=0"
OVERRIDES[sweep_b1_r0.1]="model.loss_fn.soft_beta=1 model.loss_fn.soft_rho=0.1"
OVERRIDES[sweep_b1_r0.5]="model.loss_fn.soft_beta=1 model.loss_fn.soft_rho=0.5"
OVERRIDES[sweep_b3_r0]="model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0"
OVERRIDES[sweep_b3_r0.1]="model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.1"
OVERRIDES[sweep_b3_r0.5]="model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5"
OVERRIDES[sweep_b5_r0]="model.loss_fn.soft_beta=5 model.loss_fn.soft_rho=0"
OVERRIDES[sweep_b5_r0.1]="model.loss_fn.soft_beta=5 model.loss_fn.soft_rho=0.1"
OVERRIDES[sweep_b5_r0.5]="model.loss_fn.soft_beta=5 model.loss_fn.soft_rho=0.5"

echo "=== Test-Only Phase: resuming checkpoints for test ==="

for d in logs/train/sweep_b*/train/runs/*/; do
  name=$(echo "$d" | grep -oP 'sweep_b[^/]+')
  ckpt="${d}checkpoints/last.ckpt"

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] $name: no checkpoint"
    continue
  fi

  # Skip if test CSV already exists
  existing=$(find "logs/train/${name}" -name "samples_test_*.csv" -type f 2>/dev/null | head -1)
  if [[ -n "$existing" ]]; then
    echo "[CACHED] $name"
    continue
  fi

  gpu=${GPUS[$((idx % per_batch))]}
  extra="${OVERRIDES[$name]:-}"

  echo "[TEST] GPU ${gpu}: $name (ckpt: $(basename $ckpt))"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/eval.py \
    experiment="${SWEEP_BASE}" \
    ckpt_path="$(realpath $ckpt)" \
    exp_name="${name}_test" \
    test_repeats=1 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=1 \
    model.training_mixed_config.n_samples=64 \
    logger.wandb.offline=true \
    ${extra} \
    2>&1 | tail -3 &

  pids+=($!)
  idx=$((idx + 1))

  if (( ${#pids[@]} >= per_batch )); then
    echo "  [Waiting for test batch...]"
    wait_batch
  fi
done

wait_batch
echo ""
echo "=== Test Phase Done ==="

# Find and eval all CSVs
echo ""
echo "=== Computing Table 3 Metrics ==="

for name_dir in logs/train/sweep_b* logs/eval/sweep_b*; do
  [[ -d "$name_dir" ]] || continue
  name=$(basename "$name_dir")
  name=${name%_test}  # strip _test suffix if present

  csv=$(find "$name_dir" -name "samples_test_*.csv" -type f 2>/dev/null | sort | tail -1)
  [[ -z "$csv" ]] && continue
  [[ -f "${RESULTS_DIR}/${name}.csv" ]] && { echo "[CACHED] $name"; continue; }

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
echo "=== Final Summary ==="
${PYTHON} -c "
import pandas as pd
from pathlib import Path
results_dir = Path('${RESULTS_DIR}')
dfs = []
for f in sorted(results_dir.glob('sweep_b*.csv')):
    df = pd.read_csv(f)
    df.insert(0, 'run', f.stem)
    name = f.stem
    for part in name.split('_'):
        if part.startswith('b') and len(part)>1:
            try: df['beta'] = float(part[1:])
            except: pass
        if part.startswith('r') and len(part)>1:
            try: df['rho'] = float(part[1:])
            except: pass
    dfs.append(df)
if dfs:
    m = pd.concat(dfs, ignore_index=True)
    m.to_csv(results_dir / 'all_sweep.csv', index=False)
    cols = ['run','Acc','Unique_valid','NormCov','KL(pi->p*)','KL(p*->pi)','JS_tok','log_pterm']
    avail = [c for c in cols if c in m.columns]
    print()
    print('='*90)
    print('β × ρ SWEEP RESULTS')
    print('='*90)
    print(m[avail].to_string(index=False))
    print(f'\nSaved: {results_dir}/all_sweep.csv')
else:
    print('No results found')
"
