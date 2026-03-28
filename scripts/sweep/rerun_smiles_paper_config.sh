#!/usr/bin/env bash
# =============================================================================
# Re-run SMILES sweep with PAPER config (n_samples=32, accum=4, bsz=128)
#
# β×ρ grid: 9 runs, 5000 steps, seed=42
# Paper default: β=5, ρ=0.1
#
# Memory: ~10GB per run (SMILES grammar + longer seqs) → 2 per 48GB GPU
# Time: ~2h per run → ~4h total with 8 GPUs
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config ------------------------------------------------
GPUS=(0 1 2 3 4 5 6 7)
SEED=42
MAX_STEPS=5000
WANDB_PROJECT="ChemGFN"
SWEEP_TAG="rerun_smiles_paper"

# Paper config: n_samples=32, accum=4 (already default in experiment yaml)
# We do NOT override n_samples or accumulate_grad_batches — use experiment defaults
SWEEP_BASE="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix"

COMMON="trainer.max_steps=${MAX_STEPS} \
  logger.wandb.project=${WANDB_PROJECT} \
  +trainer.limit_test_batches=100 \
  +test=True"
# limit_test_batches=100 × n_samples=32 = 3200 per test
# (paper eval uses limit=100, test_repeats=3 for total 9600)

# --------------- infrastructure ----------------------------------------------
pids=()
gpu_slots=()
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
# β × ρ Sweep (9 runs)
# =============================================================================
echo ""
echo "========================================"
echo "SMILES β × ρ Sweep (9 runs, paper config)"
echo "  Using experiment defaults: n_samples=32, accum=4"
echo "  Steps: ${MAX_STEPS}"
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

echo "[Waiting for all 9 runs...]"
wait_all

# =============================================================================
echo ""
echo "========================================"
echo "ALL SMILES SWEEP EXPERIMENTS DONE"
echo "  Total: ${run_count} runs"
echo "  Config: paper defaults (n_samples=32, accum=4, bsz=128)"
echo "========================================"

if (( ${#failures[@]} )); then
  echo "WARNING: ${#failures[@]}/${run_count} failure(s)"
  exit 1
else
  echo "All ${run_count} runs completed."
fi
