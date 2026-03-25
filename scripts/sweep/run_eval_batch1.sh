#!/usr/bin/env bash
# =============================================================================
# Batch eval for Batch 1 runs that were launched without test=True.
# Finds last.ckpt for each run and runs eval.py to compute test metrics.
# =============================================================================
set -uo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

WANDB_PROJECT="ChemGFN"
LOG_BASE="logs/train"

# The 8 Batch-1 runs (GPU 0 original + GPUs 1-7 from run_remaining_17.sh)
BATCH1_RUNS=(
  "full_topA_expr24_rp_s42"
  "full_topB_expr24_rp_s42"
  "full_topA_expr24_rp_s123"
  "full_topB_expr24_rp_s123"
  "full_topA_expr24_rp_s2024"
  "full_topB_expr24_rp_s2024"
  "full_topA_expr24_oracle_s42"
  "full_topB_expr24_oracle_s42"
)

# Map run name -> experiment config
declare -A EXP_MAP
for name in "${BATCH1_RUNS[@]}"; do
  if [[ "$name" == *oracle* ]]; then
    EXP_MAP["$name"]="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle"
  else
    EXP_MAP["$name"]="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
  fi
done

# Map run name -> hyperparams (tag determines config)
declare -A BETA_MAP RHO_MAP SEED_MAP
for name in "${BATCH1_RUNS[@]}"; do
  # Extract seed from name
  seed=$(echo "$name" | grep -oP 's\K[0-9]+$')
  SEED_MAP["$name"]=$seed

  if [[ "$name" == *topA* ]]; then
    BETA_MAP["$name"]=3
    RHO_MAP["$name"]=0.5
  else
    BETA_MAP["$name"]=5
    RHO_MAP["$name"]=0.1
  fi
done

GPUS=(0 1 2 3 4 5 6 7)
pids=()
failures=()
idx=0

wait_all() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

for name in "${BATCH1_RUNS[@]}"; do
  # Find the latest last.ckpt
  ckpt=$(find "${LOG_BASE}/${name}" -name "last.ckpt" -type f 2>/dev/null | sort | tail -1)
  if [[ -z "$ckpt" ]]; then
    echo "WARNING: no checkpoint found for ${name}, skipping"
    continue
  fi

  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  exp=${EXP_MAP["$name"]}
  beta=${BETA_MAP["$name"]}
  rho=${RHO_MAP["$name"]}
  seed=${SEED_MAP["$name"]}

  # Determine k_min params
  kstart=7; kend=3; khor=5000

  echo "[Eval] GPU ${gpu}: ${name} -> ${ckpt}"

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
done

wait_all

echo "========================================"
if (( ${#failures[@]} )); then
  echo "Eval done with ${#failures[@]} failure(s): ${failures[*]}"
  exit 1
else
  echo "All Batch-1 evals completed successfully."
fi
