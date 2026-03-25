#!/usr/bin/env bash
# Launch the remaining 17 full validation runs on GPUs 1-7.
# GPU 0 is busy with full_topA_expr24_rp_s42.
# Uses nohup to survive parent shell termination.
set -uo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

WANDB_PROJECT="ChemGFN"

# Config A
A_BETA=3; A_RHO=0.5; A_ETA=0.25
A_KMIN_START=7; A_KMIN_END=3; A_KMIN_HORIZON=5000; A_TAG="topA"

# Config B
B_BETA=5; B_RHO=0.1; B_ETA=0.25
B_KMIN_START=7; B_KMIN_END=3; B_KMIN_HORIZON=5000; B_TAG="topB"

# Experiment templates
EXPR24_RP="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
EXPR24_ORACLE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle"
SMILES_BASE="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"

SMILES_KMIN_START=5; SMILES_KMIN_END=2; SMILES_KMIN_HORIZON=5000

LOGDIR="/data2/xw3763/gflow/ChemGFN/logs/sweep_launcher"
mkdir -p "$LOGDIR"

pids=()
failures=()

launch() {
  local gpu=$1 name=$2 exp=$3 beta=$4 rho=$5 eta=$6 kstart=$7 kend=$8 khor=$9 seed=${10} tag=${11}
  echo "[Launch] GPU ${gpu}: ${name}  seed=${seed}"
  env CUDA_VISIBLE_DEVICES="${gpu}" nohup ${PYTHON} chemgfn/train.py \
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
    +test=True \
    > "${LOGDIR}/${name}.log" 2>&1 &
  pids+=("$!")
}

wait_batch() {
  echo "  Waiting for ${#pids[@]} processes (PIDs: ${pids[*]})..."
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

# ============================================================================
# Batch 1: 7 runs on GPUs 1-7
# ============================================================================
echo "===== Batch 1: 7 runs on GPUs 1-7 ====="

# Expr24 RP: topB s42, topA s123, topB s123, topA s2024, topB s2024
launch 1 "full_${B_TAG}_expr24_rp_s42" "${EXPR24_RP}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" 42 "${B_TAG}"

launch 2 "full_${A_TAG}_expr24_rp_s123" "${EXPR24_RP}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" 123 "${A_TAG}"

launch 3 "full_${B_TAG}_expr24_rp_s123" "${EXPR24_RP}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" 123 "${B_TAG}"

launch 4 "full_${A_TAG}_expr24_rp_s2024" "${EXPR24_RP}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" 2024 "${A_TAG}"

launch 5 "full_${B_TAG}_expr24_rp_s2024" "${EXPR24_RP}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" 2024 "${B_TAG}"

# Expr24 Oracle: topA s42, topB s42
launch 6 "full_${A_TAG}_expr24_oracle_s42" "${EXPR24_ORACLE}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" 42 "${A_TAG}"

launch 7 "full_${B_TAG}_expr24_oracle_s42" "${EXPR24_ORACLE}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" 42 "${B_TAG}"

wait_batch

# ============================================================================
# Batch 2: 7 runs on GPUs 1-7
# ============================================================================
echo "===== Batch 2: 7 runs on GPUs 1-7 ====="

# Expr24 Oracle: topA s123, topB s123, topA s2024, topB s2024
launch 1 "full_${A_TAG}_expr24_oracle_s123" "${EXPR24_ORACLE}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" 123 "${A_TAG}"

launch 2 "full_${B_TAG}_expr24_oracle_s123" "${EXPR24_ORACLE}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" 123 "${B_TAG}"

launch 3 "full_${A_TAG}_expr24_oracle_s2024" "${EXPR24_ORACLE}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${A_KMIN_START}" "${A_KMIN_END}" "${A_KMIN_HORIZON}" 2024 "${A_TAG}"

launch 4 "full_${B_TAG}_expr24_oracle_s2024" "${EXPR24_ORACLE}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${B_KMIN_START}" "${B_KMIN_END}" "${B_KMIN_HORIZON}" 2024 "${B_TAG}"

# SMILES: topA s42, topB s42, topA s123
launch 5 "full_${A_TAG}_smiles_s42" "${SMILES_BASE}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" 42 "${A_TAG}"

launch 6 "full_${B_TAG}_smiles_s42" "${SMILES_BASE}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" 42 "${B_TAG}"

launch 7 "full_${A_TAG}_smiles_s123" "${SMILES_BASE}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" 123 "${A_TAG}"

wait_batch

# ============================================================================
# Batch 3: 3 runs on GPUs 1-3
# ============================================================================
echo "===== Batch 3: 3 runs on GPUs 1-3 ====="

launch 1 "full_${B_TAG}_smiles_s123" "${SMILES_BASE}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" 123 "${B_TAG}"

launch 2 "full_${A_TAG}_smiles_s2024" "${SMILES_BASE}" \
  "${A_BETA}" "${A_RHO}" "${A_ETA}" "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" 2024 "${A_TAG}"

launch 3 "full_${B_TAG}_smiles_s2024" "${SMILES_BASE}" \
  "${B_BETA}" "${B_RHO}" "${B_ETA}" "${SMILES_KMIN_START}" "${SMILES_KMIN_END}" "${SMILES_KMIN_HORIZON}" 2024 "${B_TAG}"

wait_batch

echo "========================================"
total=17
if (( ${#failures[@]} )); then
  echo "Done with ${#failures[@]}/${total} failure(s): ${failures[*]}"
  exit 1
else
  echo "All ${total} remaining runs completed successfully."
fi
