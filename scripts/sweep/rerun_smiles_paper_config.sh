#!/usr/bin/env bash
# =============================================================================
# Re-run SMILES sweep with PAPER config — H100 MACHINE (4×80GB)
#
# β×ρ grid: 9 runs, 5000 steps, seed=42
# Paper default: β=5, ρ=0.1
#
# Config: n_samples=32, accum=4 (experiment defaults, matching paper)
# Memory: ~12GB/run → 3 per 80GB H100 → 12 slots on 4 GPUs
# Time:   ~2h per run → all 9 fit in 1 batch
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 1 2 3)         # 4×H100
JOBS_PER_GPU=3         # 3 SMILES runs per 80GB H100
SEED=42
MAX_STEPS=5000
WANDB_PROJECT="ChemGFN"
SWEEP_TAG="rerun_smiles_paper"

# Paper config: n_samples=32, accum=4 (already default in experiment yaml)
SWEEP_BASE="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"

COMMON="trainer.max_steps=${MAX_STEPS} \
  logger.wandb.project=${WANDB_PROJECT} \
  +trainer.limit_test_batches=100 \
  +test=True"
# limit_test_batches=100 × n_samples=32 = 3200 per test

# --------------- infrastructure ----------------------------------------------
pids=()
gpu_slots=()
for g in "${GPUS[@]}"; do
  for ((j=0; j<JOBS_PER_GPU; j++)); do gpu_slots+=("$g"); done
done
slot_count=${#gpu_slots[@]}
idx=0
failures=()
run_count=0

wait_all() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures+=("$pid"); fi
  done
  pids=()
  idx=0
}

launch() {
  local name=$1; shift
  local extra="$*"
  local gpu=${gpu_slots[$((idx % slot_count))]}

  echo "[Launch] GPU ${gpu} (slot $((idx+1))/${slot_count}): ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${SWEEP_BASE}" \
    exp_name="${name}" \
    seed="${SEED}" \
    ${COMMON} \
    ${extra} \
    2>&1 | tail -3 &

  pids+=("$!")
  run_count=$((run_count + 1))
  idx=$((idx + 1))
}

echo "========================================"
echo "SMILES Sweep Re-run (paper config)"
echo "  GPUs: ${GPUS[*]}"
echo "  Jobs/GPU: ${JOBS_PER_GPU} (total slots: ${slot_count})"
echo "  Config: experiment defaults (n_samples=32, accum=4, bsz=128)"
echo "  Steps: ${MAX_STEPS}, Eval: 100×32=3200 samples"
echo "========================================"

BETAS=(1 5 10)
RHOS=(0 0.1 0.5)

for beta in "${BETAS[@]}"; do
  for rho in "${RHOS[@]}"; do
    launch "${SWEEP_TAG}_b${beta}_r${rho}" \
      "model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
       tags=[${SWEEP_TAG},beta_${beta},rho_${rho}] \
       logger.wandb.group=${SWEEP_TAG}_beta_rho"
  done
done

echo ""
echo "All ${run_count} runs launched. Waiting..."
wait_all

echo ""
echo "========================================"
echo "DONE: ${run_count} runs"
if (( ${#failures[@]} )); then
  echo "FAILURES: ${#failures[@]}"
  exit 1
else
  echo "All succeeded."
fi
