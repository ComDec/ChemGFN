#!/usr/bin/env bash
# =============================================================================
# SWEEP-ONLY REBUTTAL EXPERIMENTS (cA3o-C2)
#
# Phase 2a: β×ρ grid (9 runs)
# Phase 2b: η sweep (3 runs)
# Phase 2c: k_min ablation (3 runs)
# Total: 15 runs on Expr24 RP, seed=42, full 5000 steps
#
# Config: n_samples=64, grad_accum=1, +test=True
# =============================================================================
set -euo pipefail

# --------------- environment -------------------------------------------------
PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config -------------------------------------------------
GPUS=(0 1 2 3 4 5 6 7)
SEED=42
WANDB_PROJECT="ChemGFN"
N_SAMPLES=64
GRAD_ACCUM=1
MAX_STEPS=5000

SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"

COMMON="trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM} \
  model.training_mixed_config.n_samples=${N_SAMPLES} \
  logger.wandb.project=${WANDB_PROJECT} \
  +test=True"

# --------------- infrastructure ----------------------------------------------
per_batch=${#GPUS[@]}
pids=()
idx=0
failures=()
run_count=0

wait_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

launch() {
  local gpu=$1; shift
  local name=$1; shift
  local extra="$*"

  echo "[Launch] GPU ${gpu}: ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${SWEEP_BASE}" \
    exp_name="${name}" \
    seed="${SEED}" \
    ${COMMON} \
    ${extra} &

  pids+=("$!")
  run_count=$((run_count + 1))
  idx=$((idx + 1))

  if (( ${#pids[@]} >= per_batch )); then
    echo "  [Waiting for batch of ${per_batch} to complete...]"
    wait_batch
  fi
}

# =============================================================================
# PHASE 2a: β × ρ Sweep (9 runs)
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 2a: β × ρ Sweep (9 runs)"
echo "========================================"

BETAS=(1 3 5)
RHOS=(0 0.1 0.5)

for beta in "${BETAS[@]}"; do
  for rho in "${RHOS[@]}"; do
    launch "${GPUS[$((idx % per_batch))]}" \
      "sweep_b${beta}_r${rho}" \
      "model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
       tags=[rebuttal_sweep,beta_${beta},rho_${rho}] \
       logger.wandb.group=rebuttal_sweep_beta_rho"
  done
done

wait_batch
echo "Phase 2a done: 9 runs"

# =============================================================================
# PHASE 2b: η Sweep (3 runs) — using paper default β=3, ρ=0.5
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 2b: η Sweep (3 runs)"
echo "========================================"

BEST_BETA=3
BEST_RHO=0.5
ETAS=(0.1 0.25 0.5)

for eta in "${ETAS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "sweep_eta${eta}" \
    "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
     model.loss_fn.aux_weight=${eta} \
     tags=[rebuttal_sweep,eta_${eta}] \
     logger.wandb.group=rebuttal_sweep_eta"
done

# =============================================================================
# PHASE 2c: k_min Ablation (3 runs)
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 2c: k_min Ablation (3 runs)"
echo "========================================"

BEST_ETA=0.25

# Fixed low: k_min = 3
launch "${GPUS[$((idx % per_batch))]}" \
  "sweep_kmin_fixed3" \
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
   model.loss_fn.aux_weight=${BEST_ETA} \
   model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_fixed3] \
   logger.wandb.group=rebuttal_sweep_kmin"

# Paper default: k_min = 7 -> 3
launch "${GPUS[$((idx % per_batch))]}" \
  "sweep_kmin_7to3" \
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
   model.loss_fn.aux_weight=${BEST_ETA} \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_7to3] \
   logger.wandb.group=rebuttal_sweep_kmin"

# Fixed high: k_min = 7
launch "${GPUS[$((idx % per_batch))]}" \
  "sweep_kmin_fixed7" \
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
   model.loss_fn.aux_weight=${BEST_ETA} \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_fixed7] \
   logger.wandb.group=rebuttal_sweep_kmin"

wait_batch

# =============================================================================
echo ""
echo "========================================"
echo "ALL SWEEP EXPERIMENTS DONE"
echo "========================================"
echo "  Phase 2a (β×ρ):    9 runs"
echo "  Phase 2b (η):      3 runs"
echo "  Phase 2c (k_min):  3 runs"
echo "  Total:             ${run_count} runs"
echo ""

if (( ${#failures[@]} )); then
  echo "WARNING: ${#failures[@]}/${run_count} failure(s): ${failures[*]}"
  exit 1
else
  echo "All ${run_count} runs completed."
  echo ""
  echo "Next: bash scripts/sweep/eval_all_sweeps.sh logs/"
fi
