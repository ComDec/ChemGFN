#!/usr/bin/env bash
# =============================================================================
# SMILES Sweep: beta x rho joint grid (cross-task robustness for cA3o-C2)
#
# Grid:  beta in {1, 5, 10}  x  rho in {0, 0.1, 0.5}  =  9 runs
# Budget: 5000 steps (full training), 1 seed each
# Base:   SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix
# Paper default: beta=5, rho=0.1
# =============================================================================
set -euo pipefail

# --------------- environment -------------------------------------------------
PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 2 4 5 6 7)           # available CUDA devices (1=PPO, 3=stale)
MAX_STEPS=5000                # full training
SEED=42
BASE_EXP="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"
SWEEP_TAG="smiles_sweep_beta_rho"
WANDB_PROJECT="ChemGFN"
# ---------------------------------------------------------------------------

BETAS=(1 5 10)
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

    echo "[SMILES Sweep] GPU ${gpu}: beta=${beta}  rho=${rho}  (${run_name})"

    CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
      experiment="${BASE_EXP}" \
      exp_name="${run_name}" \
      seed="${SEED}" \
      trainer.max_steps="${MAX_STEPS}" \
      model.loss_fn.soft_beta="${beta}" \
      model.loss_fn.soft_rho="${rho}" \
      tags="[${SWEEP_TAG},beta_${beta},rho_${rho}]" \
      logger.wandb.project="${WANDB_PROJECT}" \
      logger.wandb.group="${SWEEP_TAG}" \
      +trainer.limit_test_batches=100 \
      +test=True &

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
  echo "SMILES Sweep done with ${#failures[@]} failure(s): ${failures[*]}"
  exit 1
else
  echo "SMILES Sweep completed: all 9 beta x rho runs finished."
fi
