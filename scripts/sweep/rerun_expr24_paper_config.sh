#!/usr/bin/env bash
# =============================================================================
# Re-run Expr24 sweep with PAPER config (n_samples=32, accum=4, bsz=128)
#
# Phase 1: β×ρ grid (9 runs)
# Phase 2: η sweep (3 runs)
# Phase 3: k_min ablation (3 runs)
# Total: 15 runs, 5000 steps each, seed=42
#
# Memory: ~6GB per run (n_samples=32) → can fit 2 per 48GB GPU
# Time: ~3.5h per run → 2 per GPU → ~7h per batch of 8 GPUs
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 1 2 3 4 5 6 7)
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
# limit_test_batches=200 × n_samples=32 = 6400 (matches paper)

# --------------- infrastructure ----------------------------------------------
pids=()
gpu_slots=()  # track 2 jobs per GPU
for g in "${GPUS[@]}"; do gpu_slots+=("$g" "$g"); done
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

  echo "[Launch] GPU ${gpu} slot $((idx % slot_count)): ${name}"
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

# =============================================================================
# PHASE 1: β × ρ Sweep (9 runs) — 2 per GPU = 1 batch of ~16 slots
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 1: β × ρ Sweep (9 runs, paper config)"
echo "  n_samples=${N_SAMPLES}, accum=${GRAD_ACCUM}, steps=${MAX_STEPS}"
echo "========================================"

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

# Wait for phase 1 (9 runs on 16 slots → all fit in 1 batch)
echo "[Waiting for Phase 1 (9 runs)...]"
wait_all
echo "Phase 1 done."

# =============================================================================
# PHASE 2: η Sweep (3 runs) — β=3, ρ=0.5 (paper default)
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 2: η Sweep (3 runs)"
echo "========================================"

ETAS=(0.1 0.25 0.5)
for eta in "${ETAS[@]}"; do
  launch "${SWEEP_TAG}_eta${eta}" \
    "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
     model.loss_fn.aux_weight=${eta} \
     tags=[${SWEEP_TAG},eta_${eta}] \
     logger.wandb.group=${SWEEP_TAG}_eta"
done

# =============================================================================
# PHASE 3: k_min Ablation (3 runs)
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 3: k_min Ablation (3 runs)"
echo "========================================"

# Fixed low: k_min = 3
launch "${SWEEP_TAG}_kmin_fixed3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
   model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${SWEEP_TAG},kmin_fixed3] \
   logger.wandb.group=${SWEEP_TAG}_kmin"

# Schedule: 7 -> 3
launch "${SWEEP_TAG}_kmin_7to3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
   model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${SWEEP_TAG},kmin_7to3] \
   logger.wandb.group=${SWEEP_TAG}_kmin"

# Fixed high: k_min = 7
launch "${SWEEP_TAG}_kmin_fixed7" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
   model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[${SWEEP_TAG},kmin_fixed7] \
   logger.wandb.group=${SWEEP_TAG}_kmin"

echo "[Waiting for Phase 2+3 (6 runs)...]"
wait_all

# =============================================================================
echo ""
echo "========================================"
echo "ALL EXPR24 SWEEP EXPERIMENTS DONE"
echo "  Total: ${run_count} runs"
echo "  Config: n_samples=${N_SAMPLES}, accum=${GRAD_ACCUM} (bsz=${N_SAMPLES}×${GRAD_ACCUM}=$((N_SAMPLES*GRAD_ACCUM)))"
echo "========================================"

if (( ${#failures[@]} )); then
  echo "WARNING: ${#failures[@]}/${run_count} failure(s)"
  exit 1
else
  echo "All ${run_count} runs completed."
fi
