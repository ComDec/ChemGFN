#!/usr/bin/env bash
# =============================================================================
# ROUND 2: Remaining sweep experiments + fast test eval
#
# Runs:
#   1. β=5, ρ=0.5 (missing from round 1)
#   2. η sweep: {0.1, 0.25, 0.5}
#   3. k_min ablation: {fixed3, 7→3, fixed7}
#   Total: 7 training runs (1750 steps each, ~15 min/run)
#
# Then: test eval on ALL 15 checkpoints (round1 + round2)
#       with limit_test_batches=100 (~2 min/run)
#
# Then: compute Table 3 metrics + generate figures
# =============================================================================
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

GPUS=(0 1 2 3 4 5 6 7)
SEED=42
WANDB_PROJECT="ChemGFN-rebuttal"
N_SAMPLES=64
GRAD_ACCUM=1
MAX_STEPS=1750   # screening budget: 35% of 5000
SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
RESULTS_DIR="results/rebuttal_sweep"

COMMON="trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM} \
  model.training_mixed_config.n_samples=${N_SAMPLES} \
  logger.wandb.project=${WANDB_PROJECT}"

mkdir -p "${RESULTS_DIR}"

timestamp() { date "+%H:%M:%S"; }
per_batch=${#GPUS[@]}
pids=()
idx=0
failures=()

wait_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failures+=("$pid"); fi
  done
  pids=()
}

launch() {
  local gpu=$1 name=$2; shift 2
  local extra="$*"
  echo "[$(timestamp)] TRAIN GPU ${gpu}: ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${SWEEP_BASE}" \
    exp_name="${name}" \
    seed="${SEED}" \
    ${COMMON} \
    ${extra} &
  pids+=($!)
  idx=$((idx + 1))
  if (( ${#pids[@]} >= per_batch )); then wait_batch; fi
}

# ─────────────────────────────────────────────────────────
# STAGE 1: Train 7 runs in parallel (all fit in 8 GPUs)
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  STAGE 1: Training (7 runs, 1750 steps) ║"
echo "╚══════════════════════════════════════════╝"

# Missing β×ρ cell
launch "${GPUS[$((idx % per_batch))]}" "sweep_b5_r0.5" \
  "model.loss_fn.soft_beta=5 model.loss_fn.soft_rho=0.5 \
   tags=[rebuttal_sweep,beta_5,rho_0.5] logger.wandb.group=rebuttal_sweep_beta_rho"

# η sweep (fix β=3, ρ=0.5 = paper default)
for eta in 0.1 0.25 0.5; do
  launch "${GPUS[$((idx % per_batch))]}" "sweep_eta${eta}" \
    "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 \
     model.loss_fn.aux_weight=${eta} \
     tags=[rebuttal_sweep,eta_${eta}] logger.wandb.group=rebuttal_sweep_eta"
done

# k_min ablation (fix β=3, ρ=0.5, η=0.25)
launch "${GPUS[$((idx % per_batch))]}" "sweep_kmin_fixed3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_fixed3] logger.wandb.group=rebuttal_sweep_kmin"

launch "${GPUS[$((idx % per_batch))]}" "sweep_kmin_7to3" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_7to3] logger.wandb.group=rebuttal_sweep_kmin"

launch "${GPUS[$((idx % per_batch))]}" "sweep_kmin_fixed7" \
  "model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.loss_fn.aux_weight=0.25 \
   model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 model.factor_schedulers.k_min.horizon=5000 \
   tags=[rebuttal_sweep,kmin_fixed7] logger.wandb.group=rebuttal_sweep_kmin"

wait_batch
echo "[$(timestamp)] All 7 training runs done. Failures: ${#failures[@]}"

# ─────────────────────────────────────────────────────────
# STAGE 2: Test eval ALL 15 checkpoints (round1 + round2)
#   Key fix: +trainer.limit_test_batches=100 → 6400 samples
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  STAGE 2: Test Eval (all checkpoints)   ║"
echo "╚══════════════════════════════════════════╝"

pids=()
idx=0

for d in logs/train/sweep_*/train/runs/*/; do
  name=$(echo "$d" | grep -oP 'sweep_[^/]+')
  ckpt="${d}checkpoints/last.ckpt"
  [[ -f "$ckpt" ]] || continue

  # Skip if test CSV already exists
  existing=$(find "logs/train/${name}" -name "samples_test_*.csv" -type f 2>/dev/null | head -1)
  [[ -n "$existing" ]] && { echo "[CACHED] $name"; continue; }

  # Parse beta/rho from name for config overrides
  extra_ov=""
  for p in $(echo "$name" | tr '_' ' '); do
    [[ "$p" =~ ^b[0-9] ]] && extra_ov+=" model.loss_fn.soft_beta=${p:1}"
    [[ "$p" =~ ^r[0-9] ]] && extra_ov+=" model.loss_fn.soft_rho=${p:1}"
  done
  # eta runs
  [[ "$name" =~ eta ]] && {
    eta_val=$(echo "$name" | grep -oP 'eta\K[\d.]+')
    extra_ov+=" model.loss_fn.aux_weight=${eta_val} model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5"
  }
  # kmin runs
  [[ "$name" =~ kmin_fixed3 ]] && extra_ov+=" model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 model.factor_schedulers.k_min.horizon=5000"
  [[ "$name" =~ kmin_7to3 ]] && extra_ov+=" model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 model.factor_schedulers.k_min.horizon=5000"
  [[ "$name" =~ kmin_fixed7 ]] && extra_ov+=" model.loss_fn.soft_beta=3 model.loss_fn.soft_rho=0.5 model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 model.factor_schedulers.k_min.horizon=5000"

  gpu=${GPUS[$((idx % per_batch))]}
  echo "[$(timestamp)] TEST GPU ${gpu}: $name"

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

  pids+=($!)
  idx=$((idx + 1))
  if (( ${#pids[@]} >= per_batch )); then
    echo "  [Waiting for test batch...]"
    wait_batch
  fi
done

wait_batch
echo "[$(timestamp)] All test evals done."

# ─────────────────────────────────────────────────────────
# STAGE 3: Compute Table 3 metrics from test CSVs
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  STAGE 3: Table 3 Metrics               ║"
echo "╚══════════════════════════════════════════╝"

for name_dir in logs/train/sweep_* logs/eval/sweep_*; do
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
    2>&1 | grep -E "Acc:|NormCov:|KL|JS|pterm|Unique"
  echo ""
done

# ─────────────────────────────────────────────────────────
# STAGE 4: Merge + Summary Table + Figures
# ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  STAGE 4: Merge & Figures               ║"
echo "╚══════════════════════════════════════════╝"

${PYTHON} - <<'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("results/rebuttal_sweep")
dfs = []
for f in sorted(results_dir.glob("sweep_*.csv")):
    df = pd.read_csv(f)
    df.insert(0, "run", f.stem)
    name = f.stem
    # Parse params
    for part in name.split("_"):
        if part.startswith("b") and len(part)>1:
            try: df["beta"] = float(part[1:])
            except: pass
        if part.startswith("r") and len(part)>1:
            try: df["rho"] = float(part[1:])
            except: pass
    if "eta" in name:
        try: df["eta"] = float(name.split("eta")[1])
        except: pass
        df["sweep_type"] = "eta"
    elif "kmin" in name:
        df["kmin_variant"] = name.split("kmin_")[1]
        df["sweep_type"] = "kmin"
    else:
        df["sweep_type"] = "beta_rho"
    dfs.append(df)

if not dfs:
    print("No results found!")
    exit(0)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(results_dir / "all_sweep.csv", index=False)

cols = ["run","Acc","Unique_valid","NormCov","KL(pi->p*)","KL(p*->pi)","JS_tok","log_pterm"]
avail = [c for c in cols if c in merged.columns]

# β×ρ table
br = merged[merged["sweep_type"]=="beta_rho"]
if len(br)>0:
    print("\n" + "="*90)
    print("β × ρ SWEEP (Table 3 metrics)")
    print("="*90)
    print(br[avail].to_string(index=False))
    print()
    for m in ["NormCov","log_pterm"]:
        if m in br.columns:
            print(f"Heatmap: {m}")
            print(br.pivot_table(index="rho",columns="beta",values=m).to_string(float_format=lambda x:f"{x:.4f}"))
            print()

# η table
et = merged[merged["sweep_type"]=="eta"]
if len(et)>0:
    print("="*90)
    print("η SWEEP")
    print("="*90)
    print(et[avail].to_string(index=False))
    print()

# kmin table
km = merged[merged["sweep_type"]=="kmin"]
if len(km)>0:
    print("="*90)
    print("k_min ABLATION")
    print("="*90)
    print(km[avail].to_string(index=False))
    print()

# Generate figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Fig 1: β×ρ heatmap
    if len(br)>0 and "NormCov" in br.columns:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title, fmt, cmap in [
            (axes[0], "NormCov", "NormCov ↑", ".3f", "YlGnBu"),
            (axes[1], "log_pterm", "log p_term(τ)", ".2f", "RdYlGn"),
        ]:
            if metric not in br.columns: continue
            pivot = br.pivot_table(index="rho", columns="beta", values=metric)
            pivot = pivot.sort_index()
            pivot.columns = [f"β={c}" for c in pivot.columns]
            pivot.index = [f"ρ={i}" for i in pivot.index]
            sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5)
            ax.set_title(title, fontsize=13)
        fig.suptitle("Expr24 (RP): β × ρ sensitivity", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir/"sweep_beta_rho_heatmap.pdf", bbox_inches="tight", dpi=150)
        print(f"Saved: {fig_dir}/sweep_beta_rho_heatmap.pdf")
        plt.close(fig)

    # Fig 2: kmin barplot
    if len(km)>0:
        metrics_k = [c for c in ["NormCov","Acc","log_pterm"] if c in km.columns]
        if metrics_k:
            fig, axes = plt.subplots(1, len(metrics_k), figsize=(4*len(metrics_k), 4))
            if len(metrics_k)==1: axes=[axes]
            order = ["fixed3","7to3","fixed7"]
            labels_k = {"fixed3":"Fixed\nk=3","7to3":"Schedule\n7→3","fixed7":"Fixed\nk=7"}
            colors = ["#4c72b0","#55a868","#c44e52"]
            for ax, metric in zip(axes, metrics_k):
                vals = [km[km["kmin_variant"]==v][metric].values[0] if len(km[km["kmin_variant"]==v])>0 else 0 for v in order]
                bars = ax.bar([labels_k.get(v,v) for v in order], vals, color=colors, edgecolor="black", linewidth=0.5)
                for bar,val in zip(bars,vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=10)
                ax.set_title(metric, fontsize=12)
            fig.suptitle("Expr24: k_min schedule ablation", fontsize=14, y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir/"sweep_kmin_barplot.pdf", bbox_inches="tight", dpi=150)
            print(f"Saved: {fig_dir}/sweep_kmin_barplot.pdf")
            plt.close(fig)

except Exception as e:
    print(f"Figure generation failed: {e}")

print(f"\nAll results saved to: {results_dir}/all_sweep.csv")
PYEOF

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                       ║"
echo "╚══════════════════════════════════════════╝"
echo "  Finished at: $(timestamp)"
if (( ${#failures[@]} )); then
  echo "  Training failures: ${#failures[@]}"
fi
