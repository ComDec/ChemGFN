#!/usr/bin/env bash
# =============================================================================
# Full Validation Runs: top-2 configs x 3 seeds on Expr24 (RP + Oracle) + SMILES
#
# Prerequisite: fill in the best hyperparameters from screening sweeps.
# Budget: full 5000 steps, 3 seeds each
# =============================================================================
set -euo pipefail

# --------------- environment -------------------------------------------------
PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 1 2 3 4 5 6 7)
SEEDS=(42 123 2024)
WANDB_PROJECT="ChemGFN"

# >>> FILL THESE from screening sweeps <<<
# Config A: overall best from screening
A_BETA=3
A_RHO=0.5
A_ETA=0.25
A_KMIN_START=7      # use scheduled or fixed depending on sweep 3 result
A_KMIN_END=3
A_KMIN_HORIZON=5000
A_TAG="topA"

# Config B: second best from screening
B_BETA=5
B_RHO=0.1
B_ETA=0.25
B_KMIN_START=7
B_KMIN_END=3
B_KMIN_HORIZON=5000
B_TAG="topB"
# ---------------------------------------------------------------------------

# Experiment templates
EXPR24_RP="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
EXPR24_ORACLE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle"
SMILES_BASE="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"

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

launch() {
  local gpu=$1 name=$2 exp=$3 beta=$4 rho=$5 eta=$6 kstart=$7 kend=$8 khor=$9 seed=${10} tag=${11}

  # For SMILES, k_min range is different (5->2 paper default); override if needed
  echo "[Full] GPU ${gpu}: ${name}  seed=${seed}"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${exp}" \
    exp_name="${name}" \
    seed="${seed}" \
    trainer.max_steps=5000 \
    model.loss_fn.soft_beta="${beta}" \
    model.loss_fn.soft_rho="${rho}" \
    model.loss_fn.aux_weight="${eta}" \
    model.factor_schedulers.k_min.start="${kstart}" \
    model.factor_schedulers.k_min.end="${kend}" \
    model.factor_schedulers.k_min.horizon="${khor}" \
    tags="[full_validation,${tag}]" \
    logger.wandb.project="${WANDB_PROJECT}" \
    logger.wandb.group="full_validation" \
    +test=True &

  pids+=("$!")
  idx=$((idx + 1))

  if (( ${#pids[@]} >= per_batch )); then
    wait_batch
  fi
}

# SMILES-specific k_min (paper default: 5->2)
SMILES_KMIN_START=5
SMILES_KMIN_END=2
SMILES_KMIN_HORIZON=5000

# =============================================================================
# Launch all 18 runs — natural batching by GPU count (no inter-phase waits)
# =============================================================================
echo "===== Launching all 18 runs (8 GPUs, batched by availability) ====="

# Phase A: Expr24 + RP replay — both configs, 3 seeds
for seed in "${SEEDS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "full_${A_TAG}_expr24_rp_s${seed}" "${EXPR24_RP}" \
    "${A_BETA}" "${A_RHO}" "${A_ETA}" \
    "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" \
    "${seed}" "${A_TAG}"

  launch "${GPUS[$((idx % per_batch))]}" \
    "full_${B_TAG}_expr24_rp_s${seed}" "${EXPR24_RP}" \
    "${B_BETA}" "${B_RHO}" "${B_ETA}" \
    "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" \
    "${seed}" "${B_TAG}"
done

# Phase B: Expr24 + Oracle — both configs, 3 seeds
for seed in "${SEEDS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "full_${A_TAG}_expr24_oracle_s${seed}" "${EXPR24_ORACLE}" \
    "${A_BETA}" "${A_RHO}" "${A_ETA}" \
    "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" \
    "${seed}" "${A_TAG}"

  launch "${GPUS[$((idx % per_batch))]}" \
    "full_${B_TAG}_expr24_oracle_s${seed}" "${EXPR24_ORACLE}" \
    "${B_BETA}" "${B_RHO}" "${B_ETA}" \
    "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" \
    "${seed}" "${B_TAG}"
done

# Phase C: SMILES — both configs, 3 seeds
for seed in "${SEEDS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "full_${A_TAG}_smiles_s${seed}" "${SMILES_BASE}" \
    "${A_BETA}" "${A_RHO}" "${A_ETA}" \
    "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" \
    "${seed}" "${A_TAG}"

  launch "${GPUS[$((idx % per_batch))]}" \
    "full_${B_TAG}_smiles_s${seed}" "${SMILES_BASE}" \
    "${B_BETA}" "${B_RHO}" "${B_ETA}" \
    "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" \
    "${seed}" "${B_TAG}"
done

wait_batch

echo "========================================"
total=$((2 * 3 * 3))  # 2 configs x 3 settings x 3 seeds = 18 runs
if (( ${#failures[@]} )); then
  echo "Full validation done with ${#failures[@]}/${total} failure(s): ${failures[*]}"
  exit 1
else
  echo "Full validation completed: all ${total} runs finished."
fi
