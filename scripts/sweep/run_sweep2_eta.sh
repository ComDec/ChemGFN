#!/usr/bin/env bash
# =============================================================================
# Sweep 2: eta (aux_weight) on Expr24 + RP replay (screening)
#
# Prerequisite: run AFTER Sweep 1 finishes. Plug in best (beta, rho) below.
# Grid:  eta in {0.1, 0.25, 0.5}  =  3 runs
# Budget: 35% of full training (1750 / 5000 steps), 1 seed each
# =============================================================================
set -euo pipefail

# --------------- environment -------------------------------------------------
PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 1 2)                     # 3 GPUs suffice for 3 runs
MAX_STEPS=1750
SEED=42
BASE_EXP="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
SWEEP_TAG="sweep2_eta"
WANDB_PROJECT="ChemGFN"

# >>> FILL THESE from Sweep 1 winner <<<
BEST_BETA=3                      # best beta from sweep 1
BEST_RHO=0.5                     # best rho from sweep 1
# ---------------------------------------------------------------------------

ETAS=(0.1 0.25 0.5)

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

for eta in "${ETAS[@]}"; do
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  run_name="${SWEEP_TAG}_eta${eta}_b${BEST_BETA}_r${BEST_RHO}"

  echo "[Sweep 2] GPU ${gpu}: eta=${eta}  (beta=${BEST_BETA}, rho=${BEST_RHO})"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${BASE_EXP}" \
    exp_name="${run_name}" \
    seed="${SEED}" \
    trainer.max_steps="${MAX_STEPS}" \
    model.loss_fn.soft_beta="${BEST_BETA}" \
    model.loss_fn.soft_rho="${BEST_RHO}" \
    model.loss_fn.aux_weight="${eta}" \
    tags="[${SWEEP_TAG},eta_${eta}]" \
    logger.wandb.project="${WANDB_PROJECT}" \
    logger.wandb.group="${SWEEP_TAG}" \
    +test=True &

  pids+=("$!")
  ((idx++))
done

wait_batch

echo "========================================"
if (( ${#failures[@]} )); then
  echo "Sweep 2 done with ${#failures[@]} failure(s): ${failures[*]}"
  exit 1
else
  echo "Sweep 2 completed: all 3 eta runs finished."
fi
