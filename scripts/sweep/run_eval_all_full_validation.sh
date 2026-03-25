#!/usr/bin/env bash
# =============================================================================
# Batch eval for ALL full validation runs.
# Scans logs/train/full_* for last.ckpt and runs eval.py to compute test metrics.
# =============================================================================
set -uo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

WANDB_PROJECT="ChemGFN"
LOG_BASE="logs/train"

GPUS=(0 1 2 3 4 5 6 7)
pids=()
failures=()
idx=0

wait_batch() {
  echo "  Waiting for ${#pids[@]} eval processes..."
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

# Find all full_* run directories with checkpoints
for run_dir in ${LOG_BASE}/full_*/; do
  name=$(basename "$run_dir")
  ckpt=$(find "$run_dir" -name "last.ckpt" -type f 2>/dev/null | sort | tail -1)

  if [[ -z "$ckpt" ]]; then
    echo "SKIP: ${name} (no checkpoint)"
    continue
  fi

  # Determine experiment config from name
  if [[ "$name" == *smiles* ]]; then
    exp="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"
    kstart=5; kend=2; khor=5000
  elif [[ "$name" == *oracle* ]]; then
    exp="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle"
    kstart=7; kend=3; khor=5000
  elif [[ "$name" == *expr24* ]]; then
    exp="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
    kstart=7; kend=3; khor=5000
  else
    echo "SKIP: ${name} (unknown task)"
    continue
  fi

  # Determine hyperparams from tag in name
  if [[ "$name" == *topA* ]]; then
    beta=3; rho=0.5
  elif [[ "$name" == *topB* ]]; then
    beta=5; rho=0.1
  else
    echo "SKIP: ${name} (unknown config tag)"
    continue
  fi

  # Extract seed
  seed=$(echo "$name" | grep -oP 's\K[0-9]+$')
  if [[ -z "$seed" ]]; then
    echo "SKIP: ${name} (cannot parse seed)"
    continue
  fi

  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}

  echo "[Eval] GPU ${gpu}: ${name} (seed=${seed}, beta=${beta}, rho=${rho})"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/eval.py \
    experiment="${exp}" \
    exp_name="${name}" \
    seed="${seed}" \
    ckpt_path="${ckpt}" \
    trainer.max_steps=5000 \
    model.loss_fn.soft_beta="${beta}" \
    model.loss_fn.soft_rho="${rho}" \
    model.loss_fn.aux_weight=0.25 \
    model.factor_schedulers.k_min.start="${kstart}" \
    model.factor_schedulers.k_min.end="${kend}" \
    model.factor_schedulers.k_min.horizon="${khor}" \
    tags="[full_validation_eval]" \
    logger.wandb.project="${WANDB_PROJECT}" \
    logger.wandb.group="full_validation_eval" &

  pids+=("$!")
  idx=$((idx + 1))

  # Batch by GPU count
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    wait_batch
  fi
done

wait_batch

echo "========================================"
total=$idx
if (( ${#failures[@]} )); then
  echo "Eval done with ${#failures[@]}/${total} failure(s): ${failures[*]}"
  exit 1
else
  echo "All ${total} evals completed successfully."
fi
