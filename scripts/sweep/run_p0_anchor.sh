#!/usr/bin/env bash
# =============================================================================
# P0 ANCHOR + GRID COMPLETION + FULL TEST EVAL
#
# Uses all 8 GPUs in 3 stages:
#
# STAGE 1 (GPUs 0-2 + GPU 3, parallel):
#   A) Paper-exact anchor: β=3,ρ=0.5 × 3 seeds, PAPER CONFIG
#      (n_samples=32, accum=4, limit_train_batches=250, 5000 steps)
#      Purpose: calibrate sweep NormCov against paper Table 3
#   B) Grid completion: β=5,ρ=0.5 at 5000 steps, ROUND-1 CONFIG
#      (n_samples=64, accum=1, limit_train_batches=250, 5000 steps)
#
# STAGE 2 (GPUs 0-7):
#   Test eval ALL checkpoints with +trainer.limit_test_batches=100
#   (~6400 samples, ~2-3 min per run)
#
# STAGE 3: Compute Table 3 metrics + merge + figures
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
RESULTS_DIR="results/rebuttal_sweep"
mkdir -p "${RESULTS_DIR}"

timestamp() { date "+%H:%M:%S"; }

pids=()
failures=()

wait_all() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures+=("$pid"); fi
  done
  pids=()
}

# ─────────────────────────────────────────────────────────
# STAGE 1: Training (4 runs, ~1h)
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STAGE 1: Anchor + Grid Completion (4 runs parallel)   ║"
echo "╚══════════════════════════════════════════════════════════╝"

# A) Paper-exact anchor: 3 seeds on GPUs 0,1,2
# CRITICAL: use PAPER CONFIG exactly (n_samples=32, accum=4, limit_train=250)
for i in 0 1 2; do
  seed=(42 123 2024)
  s=${seed[$i]}
  name="anchor_paper_s${s}"
  echo "[$(timestamp)] TRAIN GPU ${i}: ${name} (paper-exact config)"

  CUDA_VISIBLE_DEVICES="${i}" ${PYTHON} chemgfn/train.py \
    experiment="${SWEEP_BASE}" \
    exp_name="${name}" \
    seed="${s}" \
    trainer.max_steps=5000 \
    logger.wandb.project="ChemGFN-rebuttal" \
    logger.wandb.group="anchor_paper_exact" \
    tags="[anchor,paper_exact,seed_${s}]" \
    +test=True \
    +trainer.limit_test_batches=100 \
    2>&1 | tail -2 &

  pids+=($!)
done

# B) Grid completion: β=5,ρ=0.5 at round-1 config on GPU 3
echo "[$(timestamp)] TRAIN GPU 3: sweep_b5_r0.5_full (5000 steps, round-1 config)"

CUDA_VISIBLE_DEVICES=3 ${PYTHON} chemgfn/train.py \
  experiment="${SWEEP_BASE}" \
  exp_name="sweep_b5_r0.5_full" \
  seed=42 \
  trainer.max_steps=5000 \
  trainer.accumulate_grad_batches=1 \
  model.training_mixed_config.n_samples=64 \
  model.loss_fn.soft_beta=5 \
  model.loss_fn.soft_rho=0.5 \
  logger.wandb.project="ChemGFN-rebuttal" \
  logger.wandb.group="rebuttal_sweep_beta_rho" \
  tags="[rebuttal_sweep,beta_5,rho_0.5,full] " \
  +test=True \
  +trainer.limit_test_batches=100 \
  2>&1 | tail -2 &

pids+=($!)

# Meanwhile, use GPUs 4-7 to test-eval round 1 checkpoints
echo ""
echo "[$(timestamp)] Parallel: test-eval round-1 checkpoints on GPUs 4-7"

eval_pids=()
eval_idx=0
EVAL_GPUS=(4 5 6 7)
eval_per_batch=${#EVAL_GPUS[@]}

for d in logs/train/sweep_b*/train/runs/*/; do
  name=$(echo "$d" | grep -oP 'sweep_b[^/]+')
  ckpt="${d}checkpoints/last.ckpt"
  [[ -f "$ckpt" ]] || continue

  # Skip if test CSV already exists in the run dir
  existing=$(find "logs/train/${name}" -name "samples_test_*.csv" -type f 2>/dev/null | head -1)
  if [[ -n "$existing" ]]; then
    echo "  [CACHED] ${name}"
    continue
  fi

  # Also skip if the eval dir already has a CSV (from round 2 eval)
  existing2=$(find "logs/eval/${name}_test" -name "samples_test_*.csv" -type f 2>/dev/null | head -1)
  if [[ -n "$existing2" ]]; then
    echo "  [CACHED] ${name} (eval dir)"
    continue
  fi

  # Parse overrides from name
  extra_ov=""
  for p in $(echo "$name" | tr '_' ' '); do
    [[ "$p" =~ ^b[0-9] ]] && extra_ov+=" model.loss_fn.soft_beta=${p:1}"
    [[ "$p" =~ ^r[0-9] ]] && extra_ov+=" model.loss_fn.soft_rho=${p:1}"
  done

  gpu=${EVAL_GPUS[$((eval_idx % eval_per_batch))]}
  echo "  [$(timestamp)] TEST-EVAL GPU ${gpu}: ${name}"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/eval.py \
    experiment="${SWEEP_BASE}" \
    ckpt_path="$(realpath $ckpt)" \
    exp_name="${name}_test" \
    test_repeats=1 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=1 \
    model.training_mixed_config.n_samples=64 \
    +trainer.limit_test_batches=100 \
    logger.wandb.offline=true \
    ${extra_ov} \
    2>&1 | tail -2 &

  eval_pids+=($!)
  eval_idx=$((eval_idx + 1))

  if (( ${#eval_pids[@]} >= eval_per_batch )); then
    for pid in "${eval_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    eval_pids=()
  fi
done

# Wait for eval batch
for pid in "${eval_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
echo "[$(timestamp)] Round-1 test-eval done"

# Wait for all STAGE 1 training
echo ""
echo "[$(timestamp)] Waiting for training to finish..."
wait_all
echo "[$(timestamp)] STAGE 1 done. Training failures: ${#failures[@]}"

# ─────────────────────────────────────────────────────────
# STAGE 2: Test eval for anchor + grid completion runs
# (They had +test=True so CSVs should already exist)
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STAGE 2: Compute Table 3 Metrics (all runs)            ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Process ALL available test CSVs
for search_root in logs/train logs/eval; do
  for name_dir in ${search_root}/sweep_b* ${search_root}/sweep_eta* ${search_root}/sweep_kmin* ${search_root}/anchor_*; do
    [[ -d "$name_dir" ]] || continue
    name=$(basename "$name_dir" | sed 's/_test$//')

    csv=$(find "$name_dir" -name "samples_test_*.csv" -type f 2>/dev/null | sort | tail -1)
    [[ -z "$csv" ]] && continue
    [[ -f "${RESULTS_DIR}/${name}.csv" ]] && { echo "[CACHED] $name"; continue; }

    echo "[METRICS] $name"
    ${PYTHON} scripts/sweep/eval_expr24_table3.py \
      --csv-path "$csv" \
      --buffer-path "${BUFFER_PATH}" \
      --max-seq-len 9 \
      --output-csv "${RESULTS_DIR}/${name}.csv" \
      2>&1 | grep -E "Acc:|NormCov:|Unique"
  done
done

# ─────────────────────────────────────────────────────────
# STAGE 3: Merge + anchor comparison + figures
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STAGE 3: Merge & Compare                               ║"
echo "╚══════════════════════════════════════════════════════════╝"

${PYTHON} - <<'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("results/rebuttal_sweep")

# Load all results
dfs = []
for f in sorted(results_dir.glob("*.csv")):
    if f.stem in ("all_sweep", "wandb_beta_rho", "wandb_full_metrics"):
        continue
    df = pd.read_csv(f)
    df.insert(0, "run", f.stem)
    dfs.append(df)

if not dfs:
    print("No results!")
    exit(1)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(results_dir / "all_results_final.csv", index=False)

cols = ["run","Acc","Unique_valid","NormCov","KL(pi->p*)","KL(p*->pi)","JS_tok","log_pterm"]
avail = [c for c in cols if c in merged.columns]

# Anchor results
anchor = merged[merged["run"].str.startswith("anchor_")]
sweep = merged[merged["run"].str.startswith("sweep_b")]
eta = merged[merged["run"].str.startswith("sweep_eta")]
kmin = merged[merged["run"].str.startswith("sweep_kmin")]

print("="*100)
print("ANCHOR: Paper-exact config (β=3, ρ=0.5, n_samples=32, accum=4, 5000 steps)")
print("="*100)
if len(anchor) > 0:
    print(anchor[avail].to_string(index=False))
    print()
    for c in ["Acc","NormCov","KL(pi->p*)","KL(p*->pi)","JS_tok","log_pterm"]:
        if c in anchor.columns:
            vals = anchor[c].dropna()
            if len(vals) > 0:
                print(f"  {c}: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")
    print()
    print("  Paper reference (Table 3, Expr24 RP):")
    print("    RapTB: Acc=0.991  NormCov=0.039  KL(π→p*)=0.561  KL(p*→π)=4.480  JS=0.147")
else:
    print("  No anchor results found (training may still be running)")

print()
print("="*100)
print("β × ρ SWEEP (n_samples=64, accum=1, 5000 steps, seed=42)")
print("="*100)
if len(sweep) > 0:
    print(sweep[avail].to_string(index=False))

print()
print("="*100)
print("η SWEEP (β=3, ρ=0.5, 1750 steps)")
print("="*100)
if len(eta) > 0:
    print(eta[avail].to_string(index=False))

print()
print("="*100)
print("k_min ABLATION (β=3, ρ=0.5, η=0.25, 1750 steps)")
print("="*100)
if len(kmin) > 0:
    print(kmin[avail].to_string(index=False))

print(f"\nAll saved: {results_dir}/all_results_final.csv")
PYEOF

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DONE at $(timestamp)                                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
