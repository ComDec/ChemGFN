#!/usr/bin/env bash
# =============================================================================
# Re-run SMILES sweep — H100 MACHINE (4×80GB)
#
# 9 runs: β×ρ grid
# Paper config: n_samples=32, accum=4, bsz=128, 5000 steps (experiment defaults)
# VRAM: ~12.5GB/run → 4 per 80GB H100
#
# 4 GPUs × 4 = 16 slots → all 9 runs fit in 1 batch
# Time: ~4h per run → ~4h wall
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

GPUS=(0 1 2 3)
JOBS_PER_GPU=4
SEED=42
MAX_STEPS=5000
WANDB_PROJECT="ChemGFN"
TAG="rerun_smiles_paper"
BASE="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"

# No n_samples/accum override — use experiment defaults (n_samples=32, accum=4)
COMMON="trainer.max_steps=${MAX_STEPS} \
  logger.wandb.project=${WANDB_PROJECT} \
  +trainer.limit_test_batches=100 \
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
echo "SMILES Sweep (paper config: experiment defaults bsz=128)"
echo "GPUs: ${GPUS[*]}, ${JOBS_PER_GPU}/GPU, ${slot_count} slots"
echo "========================================"

for beta in 1 5 10; do
  for rho in 0 0.1 0.5; do
    launch "${TAG}_b${beta}_r${rho}" \
      "model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
       tags=[${TAG},beta_${beta},rho_${rho}] logger.wandb.group=${TAG}_beta_rho"
  done
done

wait_all

echo ""
echo "========================================"
echo "DONE: ${run_count} runs"
(( ${#failures[@]} )) && echo "FAILURES: ${#failures[@]}" && exit 1
echo "All succeeded."
