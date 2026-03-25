#!/usr/bin/env bash
# =============================================================================
# MASTER REBUTTAL EXPERIMENT LAUNCH
#
# Runs ALL experiments needed for rebuttal in priority order:
#   Phase 1: AvgPrefixTB baseline comparison (QHmk-C6)
#   Phase 2: β×ρ sweep + η + k_min ablation (cA3o-C2)
#
# Key changes from original scripts:
#   - accumulate_grad_batches=1, n_samples=128 (user-verified: better perf)
#   - +test=True on all runs (produces test CSVs for Table 3 metrics)
#   - Consistent PYTHON path
#
# Usage:
#   bash scripts/sweep/run_all_rebuttal.sh
#
# GPU allocation:
#   GPU 0: reserved for RL eval (do NOT use)
#   GPUs 1-7: available (7 GPUs)
# =============================================================================
set -euo pipefail

# --------------- environment -------------------------------------------------
PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config -------------------------------------------------
GPUS=(1 2 3 4 5 6 7)           # GPU 0 reserved for RL eval
SEEDS=(42 123 2024)
WANDB_PROJECT="ChemGFN"
N_SAMPLES=64                    # max that fits in 48GB (12.5GB per process)
GRAD_ACCUM=1                    # grad_accum=1 per user instruction
MAX_STEPS=5000

# Common overrides for all runs
COMMON_OVERRIDES="trainer.max_steps=${MAX_STEPS} \
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
  local exp=$1; shift
  local seed=$1; shift
  local extra_overrides="$*"

  echo "[Launch] GPU ${gpu}: ${name} (seed=${seed})"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${exp}" \
    exp_name="${name}" \
    seed="${seed}" \
    ${COMMON_OVERRIDES} \
    ${extra_overrides} &

  pids+=("$!")
  run_count=$((run_count + 1))
  idx=$((idx + 1))

  if (( ${#pids[@]} >= per_batch )); then
    echo "  [Waiting for batch of ${per_batch} to complete...]"
    wait_batch
  fi
}

# =============================================================================
# PHASE 1: AvgPrefixTB Baseline Comparison (QHmk-C6)
#
# Methods: TB, AvgPrefixTB, AvgPrefixTB+detach_pterm, RapTB
# Settings: Expr24 RP, Expr24 Oracle
# Seeds: 42, 123, 2024
# Total: 4 methods × 2 settings × 3 seeds = 24 runs
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 1: AvgPrefixTB Baseline Comparison"
echo "========================================"

# --- Expr24 RP ---
for seed in "${SEEDS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_TB_rp_s${seed}" \
    "VarExpr24/VarExpr24_TB_no_data_buffer_hit" \
    "${seed}" \
    "tags=[rebuttal_baseline,TB,RP] logger.wandb.group=rebuttal_baseline"

  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_AvgPrefixTB_rp_s${seed}" \
    "VarExpr24/VarExpr24_AvgPrefixTB" \
    "${seed}" \
    "tags=[rebuttal_baseline,AvgPrefixTB,RP] logger.wandb.group=rebuttal_baseline"

  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_AvgPrefixTB_detach_rp_s${seed}" \
    "VarExpr24/VarExpr24_AvgPrefixTB_detach_pterm" \
    "${seed}" \
    "tags=[rebuttal_baseline,AvgPrefixTB_detach,RP] logger.wandb.group=rebuttal_baseline"

  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_RapTB_rp_s${seed}" \
    "VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune" \
    "${seed}" \
    "tags=[rebuttal_baseline,RapTB,RP] logger.wandb.group=rebuttal_baseline"
done

# --- Expr24 Oracle ---
for seed in "${SEEDS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_TB_oracle_s${seed}" \
    "VarExpr24/VarExpr24_TB_no_data_buffer_hit_oracle" \
    "${seed}" \
    "tags=[rebuttal_baseline,TB,Oracle] logger.wandb.group=rebuttal_baseline"

  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_AvgPrefixTB_oracle_s${seed}" \
    "VarExpr24/VarExpr24_AvgPrefixTB_oracle" \
    "${seed}" \
    "tags=[rebuttal_baseline,AvgPrefixTB,Oracle] logger.wandb.group=rebuttal_baseline"

  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_AvgPrefixTB_detach_oracle_s${seed}" \
    "VarExpr24/VarExpr24_AvgPrefixTB_detach_pterm_oracle" \
    "${seed}" \
    "tags=[rebuttal_baseline,AvgPrefixTB_detach,Oracle] logger.wandb.group=rebuttal_baseline"

  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_RapTB_oracle_s${seed}" \
    "VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle" \
    "${seed}" \
    "tags=[rebuttal_baseline,RapTB,Oracle] logger.wandb.group=rebuttal_baseline"
done

# Drain remaining Phase 1
wait_batch
echo ""
echo "Phase 1 dispatched: 24 runs"

# =============================================================================
# PHASE 2: Hyperparameter Sweep (cA3o-C2)
#
# 2a: β×ρ grid: 3×3 = 9 runs (Expr24 RP, seed=42, full 5000 steps)
# 2b: η sweep: 3 runs (plug in best β,ρ from 2a)
# 2c: k_min ablation: 3 runs (plug in best β,ρ,η)
# Total: 15 runs
#
# NOTE: Sweeps 2b and 2c use paper defaults for now.
#       After 2a completes, update BEST_BETA/BEST_RHO and rerun 2b/2c if needed.
# =============================================================================
echo ""
echo "========================================"
echo "PHASE 2a: β × ρ Sweep (9 runs)"
echo "========================================"

SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
BETAS=(1 3 5)
RHOS=(0 0.1 0.5)

for beta in "${BETAS[@]}"; do
  for rho in "${RHOS[@]}"; do
    launch "${GPUS[$((idx % per_batch))]}" \
      "rebuttal_sweep_b${beta}_r${rho}" \
      "${SWEEP_BASE}" \
      "42" \
      "model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} \
       tags=[rebuttal_sweep,beta_${beta},rho_${rho}] \
       logger.wandb.group=rebuttal_sweep_beta_rho"
  done
done

wait_batch
echo "Phase 2a dispatched: 9 runs"

echo ""
echo "========================================"
echo "PHASE 2b: η Sweep (3 runs)"
echo "========================================"

# Using paper defaults for β,ρ; update after 2a if needed
BEST_BETA=3
BEST_RHO=0.5
ETAS=(0.1 0.25 0.5)

for eta in "${ETAS[@]}"; do
  launch "${GPUS[$((idx % per_batch))]}" \
    "rebuttal_sweep_eta${eta}_b${BEST_BETA}_r${BEST_RHO}" \
    "${SWEEP_BASE}" \
    "42" \
    "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
     model.loss_fn.aux_weight=${eta} \
     tags=[rebuttal_sweep,eta_${eta}] \
     logger.wandb.group=rebuttal_sweep_eta"
done

echo ""
echo "========================================"
echo "PHASE 2c: k_min Ablation (3 runs)"
echo "========================================"

BEST_ETA=0.25

# Fixed low: k_min = 3
launch "${GPUS[$((idx % per_batch))]}" \
  "rebuttal_sweep_kmin_fixed_low" \
  "${SWEEP_BASE}" \
  "42" \
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
   model.loss_fn.aux_weight=${BEST_ETA} \
   model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_fixed_low] \
   logger.wandb.group=rebuttal_sweep_kmin"

# Paper default: k_min = 7 -> 3
launch "${GPUS[$((idx % per_batch))]}" \
  "rebuttal_sweep_kmin_schedule" \
  "${SWEEP_BASE}" \
  "42" \
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
   model.loss_fn.aux_weight=${BEST_ETA} \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_schedule] \
   logger.wandb.group=rebuttal_sweep_kmin"

# Fixed high: k_min = 7
launch "${GPUS[$((idx % per_batch))]}" \
  "rebuttal_sweep_kmin_fixed_high" \
  "${SWEEP_BASE}" \
  "42" \
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} \
   model.loss_fn.aux_weight=${BEST_ETA} \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 \
   model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_fixed_high] \
   logger.wandb.group=rebuttal_sweep_kmin"

# Drain all remaining
wait_batch

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "ALL REBUTTAL EXPERIMENTS DISPATCHED"
echo "========================================"
echo "  Phase 1 (AvgPrefixTB baseline): 24 runs"
echo "  Phase 2a (β×ρ sweep):           9 runs"
echo "  Phase 2b (η sweep):             3 runs"
echo "  Phase 2c (k_min ablation):      3 runs"
echo "  Total:                           39 runs"
echo ""
echo "  GPUs used: ${GPUS[*]} (GPU 0 reserved for RL eval)"
echo "  n_samples=${N_SAMPLES}, grad_accum=${GRAD_ACCUM}, max_steps=${MAX_STEPS}"
echo ""

if (( ${#failures[@]} )); then
  echo "WARNING: ${#failures[@]}/${run_count} run(s) failed: ${failures[*]}"
  exit 1
else
  echo "All ${run_count} runs completed successfully."
  echo ""
  echo "Next steps:"
  echo "  1. Run post-hoc eval:  bash scripts/sweep/eval_all_sweeps.sh logs/"
  echo "  2. Generate heatmap:   python scripts/sweep/analyze_sweep.py --source csv --csv-path results/sweep_table3/all_table3.csv"
  echo "  3. Fill rebuttal placeholders with results"
fi
