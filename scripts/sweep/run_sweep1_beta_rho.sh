#!/usr/bin/env bash
# =============================================================================
# Sweep 1: beta x rho joint grid on Expr24 + RP replay (screening)
#
# Grid:  beta in {1, 3, 5}  x  rho in {0, 0.1, 0.5}  =  9 runs
# Budget: 35% of full training (1750 / 5000 steps), 1 seed each
# Base:   VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune
# =============================================================================
set -euo pipefail

# --------------- user config ------------------------------------------------
GPUS=(0 1 2 3 4 5 6 7)          # available CUDA devices
MAX_STEPS=1750                   # screening: ~35% of full 5000 steps
SEED=42
BASE_EXP="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
SWEEP_TAG="sweep1_beta_rho"
WANDB_PROJECT="ChemGFN"         # set to your wandb project
# ---------------------------------------------------------------------------

BETAS=(1 3 5)
RHOS=(0 0.1 0.5)

per_batch=${#GPUS[@]}
pids=()
idx=0
failures=()

wait_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

for beta in "${BETAS[@]}"; do
  for rho in "${RHOS[@]}"; do
    gpu=${GPUS[$((idx % per_batch))]}
    run_name="${SWEEP_TAG}_b${beta}_r${rho}"

    echo "[Sweep 1] GPU ${gpu}: beta=${beta}  rho=${rho}  (${run_name})"

    CUDA_VISIBLE_DEVICES="${gpu}" python chemgfn/train.py \
      experiment="${BASE_EXP}" \
      exp_name="${run_name}" \
      seed="${SEED}" \
      trainer.max_steps="${MAX_STEPS}" \
      model.loss_fn.soft_beta="${beta}" \
      model.loss_fn.soft_rho="${rho}" \
      tags="[${SWEEP_TAG},beta_${beta},rho_${rho}]" \
      logger.wandb.project="${WANDB_PROJECT}" \
      logger.wandb.group="${SWEEP_TAG}" &

    pids+=("$!")
    ((idx++))

    if (( ${#pids[@]} >= per_batch )); then
      wait_batch
    fi
  done
done

wait_batch

echo "========================================"
if (( ${#failures[@]} )); then
  echo "Sweep 1 done with ${#failures[@]} failure(s): ${failures[*]}"
  exit 1
else
  echo "Sweep 1 completed: all 9 beta x rho runs finished."
fi
