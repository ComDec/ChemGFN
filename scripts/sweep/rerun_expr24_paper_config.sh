#!/usr/bin/env bash
# =============================================================================
# Re-run Expr24 sweep — LOCAL MACHINE
#
# 15 runs: 9 β×ρ + 3 η + 3 k_min
# Paper config: n_samples=32, accum=4, bsz=128, 5000 steps
# VRAM: ~20GB/run → 2 per 48GB GPU
#
# Available GPUs: 0,1,2,4,5,7 (6 GPUs × 2 = 12 slots)
# Batch 1: 12 runs → ~2.2h
# Batch 2: 3 runs  → ~2.2h
# Total wall: ~4.5h
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

GPUS=(0 1 2 4 5 7)
JOBS_PER_GPU=2
SEED=42
MAX_STEPS=5000
N_SAMPLES=32
GRAD_ACCUM=4
WANDB_PROJECT="ChemGFN"
TAG="rerun_expr24_paper"
BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"

COMMON="trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM} \
  model.training_mixed_config.n_samples=${N_SAMPLES} \
  logger.wandb.project=${WANDB_PROJECT} \
  +trainer.limit_test_batches=200 \
  +test=True"

# --------------- infrastructure ----------------------------------------------
gpu_slots=()
for g in "${GPUS[@]}"; do
  for ((j=0; j<JOBS_PER_GPU; j++)); do gpu_slots+=("$g"); done
done
slot_count=${#gpu_slots[@]}

pids=(); idx=0; failures=(); run_count=0

wait_all() {
  echo "  Waiting for ${#pids[@]} jobs..."
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures+=("$pid"); fi
  done
  pids=(); idx=0
}

launch() {
  local name=$1; shift
  local extra="$*"
  local gpu=${gpu_slots[$((idx % slot_count))]}
  echo "[Launch] GPU ${gpu}: ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${BASE}" exp_name="${name}" seed="${SEED}" \
    ${COMMON} ${extra} 2>&1 | tail -3 &
  pids+=("$!"); run_count=$((run_count+1)); idx=$((idx+1))
}

echo "========================================"
echo "Expr24 Sweep (paper config: bsz=${N_SAMPLES}×${GRAD_ACCUM}=$((N_SAMPLES*GRAD_ACCUM)))"
echo "GPUs: ${GPUS[*]}, ${JOBS_PER_GPU}/GPU, ${slot_count} slots"
echo "========================================"

# --- Batch 1: β×ρ 9 + η 3 = 12 runs (fits in 12 slots) ---
echo ""
echo "--- Batch 1: β×ρ (9) + η (3) = 12 runs ---"

for beta in 1 3 5; do
  for rho in 0 0.1 0.5; do
    launch "${TAG}_b${beta}_r${rho}" \
      "model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
       tags=[${TAG},beta_${beta},rho_${rho}] logger.wandb.group=${TAG}_beta_rho"
  done
done

for eta in 0.1 0.25 0.5; do
  launch "${TAG}_eta${eta}" \
    "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
     model.loss_fn.aux_weight=${eta} \
     tags=[${TAG},eta_${eta}] logger.wandb.group=${TAG}_eta"
done

wait_all

# --- Batch 2: k_min (3 runs) ---
echo ""
echo "--- Batch 2: k_min (3 runs) ---"

launch "${TAG}_kmin_fixed3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${TAG},kmin_fixed3] logger.wandb.group=${TAG}_kmin"

launch "${TAG}_kmin_7to3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${TAG},kmin_7to3] logger.wandb.group=${TAG}_kmin"

launch "${TAG}_kmin_fixed7" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${TAG},kmin_fixed7] logger.wandb.group=${TAG}_kmin"

wait_all

echo ""
echo "========================================"
echo "DONE: ${run_count} runs"
(( ${#failures[@]} )) && echo "FAILURES: ${#failures[@]}" && exit 1
echo "All succeeded."
