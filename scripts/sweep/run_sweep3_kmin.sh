#!/usr/bin/env bash
# =============================================================================
# Sweep 3: k_min schedule on Expr24 + RP replay (screening)
#
# Prerequisite: run AFTER Sweep 1 & 2. Plug in best (beta, rho, eta) below.
# Three variants:
#   A) Fixed low:     k_min = 3 (constant, no schedule)
#   B) Paper default: k_min = 7 -> 3 over 5000 steps (linear)
#   C) Fixed high:    k_min = 7 (constant, no schedule)
# Budget: 35% of full training (1750 / 5000 steps), 1 seed each
# =============================================================================
set -euo pipefail

# --------------- user config ------------------------------------------------
GPUS=(0 1 2)
MAX_STEPS=1750
SEED=42
BASE_EXP="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
SWEEP_TAG="sweep3_kmin"
WANDB_PROJECT="ChemGFN"

# >>> FILL THESE from Sweep 1+2 winners <<<
BEST_BETA=3
BEST_RHO=0.5
BEST_ETA=0.25
# ---------------------------------------------------------------------------

# Common overrides
COMMON="experiment=${BASE_EXP} \
  seed=${SEED} \
  trainer.max_steps=${MAX_STEPS} \
  model.loss_fn.soft_beta=${BEST_BETA} \
  model.loss_fn.soft_rho=${BEST_RHO} \
  model.loss_fn.aux_weight=${BEST_ETA} \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.group=${SWEEP_TAG}"

pids=()
failures=()

wait_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

# --- A) Fixed low: k_min = 3 constant ----------------------------------------
echo "[Sweep 3-A] GPU ${GPUS[0]}: k_min fixed=3"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python chemgfn/train.py \
  ${COMMON} \
  exp_name="${SWEEP_TAG}_fixed_low" \
  model.loss_fn.k_min=3 \
  model.factor_schedulers.k_min.start=3 \
  model.factor_schedulers.k_min.end=3 \
  model.factor_schedulers.k_min.horizon=5000 \
  tags="[${SWEEP_TAG},kmin_fixed_low]" &
pids+=("$!")

# --- B) Paper default: k_min = 7 -> 3 linear ---------------------------------
echo "[Sweep 3-B] GPU ${GPUS[1]}: k_min 7->3 (paper default)"
CUDA_VISIBLE_DEVICES="${GPUS[1]}" python chemgfn/train.py \
  ${COMMON} \
  exp_name="${SWEEP_TAG}_schedule_default" \
  model.loss_fn.k_min=2 \
  model.factor_schedulers.k_min.start=7 \
  model.factor_schedulers.k_min.end=3 \
  model.factor_schedulers.k_min.horizon=5000 \
  tags="[${SWEEP_TAG},kmin_schedule_default]" &
pids+=("$!")

# --- C) Fixed high: k_min = 7 constant ---------------------------------------
echo "[Sweep 3-C] GPU ${GPUS[2]}: k_min fixed=7"
CUDA_VISIBLE_DEVICES="${GPUS[2]}" python chemgfn/train.py \
  ${COMMON} \
  exp_name="${SWEEP_TAG}_fixed_high" \
  model.loss_fn.k_min=7 \
  model.factor_schedulers.k_min.start=7 \
  model.factor_schedulers.k_min.end=7 \
  model.factor_schedulers.k_min.horizon=5000 \
  tags="[${SWEEP_TAG},kmin_fixed_high]" &
pids+=("$!")

wait_batch

echo "========================================"
if (( ${#failures[@]} )); then
  echo "Sweep 3 done with ${#failures[@]} failure(s): ${failures[*]}"
  exit 1
else
  echo "Sweep 3 completed: all 3 k_min variants finished."
fi
