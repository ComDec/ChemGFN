#!/usr/bin/env bash
# =============================================================================
# Re-run Expr24 sweep with PAPER config — LOCAL MACHINE (8×48GB)
#
# Phase 1: β×ρ grid (9 runs)
# Phase 2: η sweep (3 runs)
# Phase 3: k_min ablation (3 runs)
# Total: 15 runs, 5000 steps each, seed=42
#
# Config: n_samples=32, accum=4 (effective bsz=128, matching paper)
# Memory: ~8GB/run → 3 per 48GB GPU → 21 slots on 7 GPUs
# Time:   ~3.5h per run → all 15 fit in 1 batch
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 1 2 4 5 6 7)  # skip GPU 3 (stale process)
JOBS_PER_GPU=3         # 3 Expr24 runs per 48GB GPU
SEED=42
MAX_STEPS=5000
N_SAMPLES=32
GRAD_ACCUM=4
WANDB_PROJECT="ChemGFN"
SWEEP_TAG="rerun_expr24_paper"

SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"

COMMON="trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM} \
  model.training_mixed_config.n_samples=${N_SAMPLES} \
  logger.wandb.project=${WANDB_PROJECT} \
  +trainer.limit_test_batches=200 \
  +test=True"
# limit_test_batches=200 × n_samples=32 = 6400 (matches paper eval)

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
echo "Expr24 Sweep Re-run (paper config)"
echo "  GPUs: ${GPUS[*]}"
echo "  Jobs/GPU: ${JOBS_PER_GPU} (total slots: ${slot_count})"
echo "  Config: n_samples=${N_SAMPLES}, accum=${GRAD_ACCUM}, bsz=$((N_SAMPLES*GRAD_ACCUM))"
echo "  Steps: ${MAX_STEPS}, Eval: 200×32=6400 samples"
echo "========================================"

# --- Phase 1: β × ρ (9 runs) ---
BETAS=(1 3 5)
RHOS=(0 0.1 0.5)
for beta in "${BETAS[@]}"; do
  for rho in "${RHOS[@]}"; do
    launch "${SWEEP_TAG}_b${beta}_r${rho}" \
      "model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
       tags=[${SWEEP_TAG},beta_${beta},rho_${rho}] \
       logger.wandb.group=${SWEEP_TAG}_beta_rho"
  done
done

# --- Phase 2: η (3 runs, β=3 ρ=0.5) ---
for eta in 0.1 0.25 0.5; do
  launch "${SWEEP_TAG}_eta${eta}" \
    "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
     model.loss_fn.aux_weight=${eta} \
     tags=[${SWEEP_TAG},eta_${eta}] \
     logger.wandb.group=${SWEEP_TAG}_eta"
done

# --- Phase 3: k_min (3 runs) ---
launch "${SWEEP_TAG}_kmin_fixed3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${SWEEP_TAG},kmin_fixed3] logger.wandb.group=${SWEEP_TAG}_kmin"

launch "${SWEEP_TAG}_kmin_7to3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${SWEEP_TAG},kmin_7to3] logger.wandb.group=${SWEEP_TAG}_kmin"

launch "${SWEEP_TAG}_kmin_fixed7" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${SWEEP_TAG},kmin_fixed7] logger.wandb.group=${SWEEP_TAG}_kmin"

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
