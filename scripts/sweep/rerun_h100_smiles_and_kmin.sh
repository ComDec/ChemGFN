#!/usr/bin/env bash
# =============================================================================
# H100 Machine: SMILES sweep (9 runs) + Expr24 kmin_fixed7 (1 run) = 10 runs
#
# 4×80GB H100, ~12.5GB/SMILES run, ~20GB/Expr24 run
# Layout: GPU 0-2 get 3 SMILES each (9 total), GPU 3 gets 1 SMILES-sized + kmin
# All fit in 1 batch → ~14h wall (5000 steps)
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

WANDB_PROJECT="ChemGFN"
SEED=42
MAX_STEPS=5000

pids=()
failures=()

wait_all() {
  echo "  Waiting for ${#pids[@]} jobs..."
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures+=("$pid"); fi
  done
  pids=()
}

# =============================================================================
# SMILES β×ρ sweep (9 runs)
# =============================================================================
echo "========================================"
echo "SMILES Sweep (9 runs, paper config)"
echo "========================================"

SMILES_BASE="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"
SMILES_TAG="rerun_smiles_paper"
SMILES_COMMON="trainer.max_steps=${MAX_STEPS} logger.wandb.project=${WANDB_PROJECT} +trainer.limit_test_batches=100 +test=True"

# GPU 0: b1_r0, b1_r0.1, b1_r0.5
# GPU 1: b5_r0, b5_r0.1, b5_r0.5
# GPU 2: b10_r0, b10_r0.1, b10_r0.5
gpu=0
for beta in 1 5 10; do
  for rho in 0 0.1 0.5; do
    name="${SMILES_TAG}_b${beta}_r${rho}"
    echo "[Launch] GPU ${gpu}: ${name}"
    CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
      experiment="${SMILES_BASE}" exp_name="${name}" seed=${SEED} \
      ${SMILES_COMMON} \
      model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
      tags="[${SMILES_TAG},beta_${beta},rho_${rho}]" \
      logger.wandb.group="${SMILES_TAG}_beta_rho" \
      2>&1 | tail -3 &
    pids+=("$!")
  done
  gpu=$((gpu + 1))
done

# =============================================================================
# Expr24 kmin_fixed7 (1 run on GPU 3)
# =============================================================================
echo ""
echo "========================================"
echo "Expr24 kmin_fixed7 (1 run, GPU 3)"
echo "========================================"

EXPR_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
EXPR_TAG="rerun_expr24_paper"
EXPR_COMMON="trainer.max_steps=${MAX_STEPS} trainer.accumulate_grad_batches=4 model.training_mixed_config.n_samples=32 logger.wandb.project=${WANDB_PROJECT} +trainer.limit_test_batches=200 +test=True"

echo "[Launch] GPU 3: ${EXPR_TAG}_kmin_fixed7"
CUDA_VISIBLE_DEVICES=3 ${PYTHON} chemgfn/train.py \
  experiment="${EXPR_BASE}" exp_name="${EXPR_TAG}_kmin_fixed7" seed=${SEED} \
  ${EXPR_COMMON} \
  model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
  model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 \
  model.factor_schedulers.k_min.horizon=5000 \
  tags="[${EXPR_TAG},kmin_fixed7]" \
  logger.wandb.group="${EXPR_TAG}_kmin" \
  2>&1 | tail -3 &
pids+=("$!")

echo ""
echo "Total: ${#pids[@]} runs launched"
wait_all

echo ""
echo "========================================"
if (( ${#failures[@]} )); then
  echo "FAILURES: ${#failures[@]}"
  exit 1
else
  echo "All 10 runs succeeded."
fi
