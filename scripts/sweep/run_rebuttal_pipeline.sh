#!/usr/bin/env bash
# =============================================================================
# REBUTTAL PIPELINE: Train → Eval → Analyze (fully automated)
#
# Runs all sweep experiments sequentially (1 per GPU, 8 parallel),
# then evaluates each with Table 3 metrics, then generates figures.
#
# Usage:
#   nohup bash scripts/sweep/run_rebuttal_pipeline.sh > logs/pipeline.log 2>&1 &
# =============================================================================
set -euo pipefail

# --------------- environment -------------------------------------------------
PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
cd /data2/xw3763/gflow/ChemGFN

# --------------- user config -------------------------------------------------
GPUS=(0 1 2 3 4 5 6 7)
SEED=42
WANDB_PROJECT="ChemGFN-rebuttal"
N_SAMPLES=64
GRAD_ACCUM=1
MAX_STEPS=5000

SWEEP_BASE="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune"
BUFFER_PATH="data/24_points/buffer_24_non_zero.pt"
RESULTS_DIR="results/rebuttal_sweep"

COMMON="trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM} \
  model.training_mixed_config.n_samples=${N_SAMPLES} \
  logger.wandb.project=${WANDB_PROJECT} \
  +test=True"

mkdir -p "${RESULTS_DIR}"
mkdir -p figures

# --------------- infrastructure ----------------------------------------------
per_batch=${#GPUS[@]}
total_failures=0
total_runs=0

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Run a single training job, wait for it, then immediately eval
run_one() {
  local gpu=$1 name=$2; shift 2
  local extra="$*"

  local log_dir="logs/train/${name}/train/runs"
  echo "[$(timestamp)] START  GPU ${gpu}: ${name}"

  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} chemgfn/train.py \
    experiment="${SWEEP_BASE}" \
    exp_name="${name}" \
    seed="${SEED}" \
    ${COMMON} \
    ${extra} 2>&1 | tail -3

  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(timestamp)] FAIL   GPU ${gpu}: ${name} (exit ${rc})"
    return 1
  fi

  echo "[$(timestamp)] TRAIN DONE  ${name}"

  # Find the test CSV and eval immediately
  local csv_path
  csv_path=$(find "logs/train/${name}" -name "samples_test_*.csv" -type f 2>/dev/null | sort -t_ -k3 -rn | head -1)
  if [[ -n "${csv_path}" ]]; then
    echo "[$(timestamp)] EVAL   ${name}: ${csv_path}"
    ${PYTHON} scripts/sweep/eval_expr24_table3.py \
      --csv-path "${csv_path}" \
      --buffer-path "${BUFFER_PATH}" \
      --max-seq-len 9 \
      --output-csv "${RESULTS_DIR}/${name}.csv" 2>&1 | grep -E "Acc:|Unique|NormCov|KL|JS|pterm"
  else
    echo "[$(timestamp)] WARN   ${name}: no test CSV found, skipping eval"
  fi

  echo "[$(timestamp)] DONE   ${name}"
  return 0
}

# Run a batch of jobs in parallel (1 per GPU), wait for all
run_batch() {
  local -n _names=$1
  local -n _extras=$2
  local batch_size=${#_names[@]}
  local pids=()
  local fails=0

  echo ""
  echo "──────────────────────────────────────────────────"
  echo "Launching batch: ${batch_size} runs"
  echo "──────────────────────────────────────────────────"

  for i in $(seq 0 $((batch_size - 1))); do
    local gpu=${GPUS[$((i % per_batch))]}
    run_one "${gpu}" "${_names[$i]}" "${_extras[$i]}" &
    pids+=($!)
    total_runs=$((total_runs + 1))
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fails=$((fails + 1))
    fi
  done

  total_failures=$((total_failures + fails))
  echo "──────────────────────────────────────────────────"
  echo "Batch done: ${fails}/${batch_size} failures"
  echo "──────────────────────────────────────────────────"
}

# =============================================================================
# STAGE 1: β × ρ Sweep (9 runs → 2 batches: 8 + 1)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STAGE 1: β × ρ Sweep (9 runs)                 ║"
echo "╚══════════════════════════════════════════════════╝"

BETAS=(1 3 5)
RHOS=(0 0.1 0.5)

names=()
extras=()
for beta in "${BETAS[@]}"; do
  for rho in "${RHOS[@]}"; do
    names+=("sweep_b${beta}_r${rho}")
    extras+=("model.loss_fn.soft_beta=${beta} model.loss_fn.soft_rho=${rho} tags=[rebuttal_sweep,beta_${beta},rho_${rho}] logger.wandb.group=rebuttal_sweep_beta_rho")
  done
done

# Split into batches of $per_batch
batch_names=()
batch_extras=()
for i in "${!names[@]}"; do
  batch_names+=("${names[$i]}")
  batch_extras+=("${extras[$i]}")
  if (( ${#batch_names[@]} >= per_batch )); then
    run_batch batch_names batch_extras
    batch_names=()
    batch_extras=()
  fi
done
if (( ${#batch_names[@]} > 0 )); then
  run_batch batch_names batch_extras
fi

# =============================================================================
# STAGE 2: η Sweep (3 runs → 1 batch)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STAGE 2: η Sweep (3 runs)                     ║"
echo "╚══════════════════════════════════════════════════╝"

BEST_BETA=3
BEST_RHO=0.5

names=()
extras=()
for eta in 0.1 0.25 0.5; do
  names+=("sweep_eta${eta}")
  extras+=("model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} model.loss_fn.aux_weight=${eta} tags=[rebuttal_sweep,eta_${eta}] logger.wandb.group=rebuttal_sweep_eta")
done
run_batch names extras

# =============================================================================
# STAGE 3: k_min Ablation (3 runs → 1 batch)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STAGE 3: k_min Ablation (3 runs)              ║"
echo "╚══════════════════════════════════════════════════╝"

BEST_ETA=0.25

names=(
  "sweep_kmin_fixed3"
  "sweep_kmin_7to3"
  "sweep_kmin_fixed7"
)
extras=(
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} model.loss_fn.aux_weight=${BEST_ETA} model.factor_schedulers.k_min.start=3 model.factor_schedulers.k_min.end=3 model.factor_schedulers.k_min.horizon=5000 tags=[rebuttal_sweep,kmin_fixed3] logger.wandb.group=rebuttal_sweep_kmin"
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} model.loss_fn.aux_weight=${BEST_ETA} model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=3 model.factor_schedulers.k_min.horizon=5000 tags=[rebuttal_sweep,kmin_7to3] logger.wandb.group=rebuttal_sweep_kmin"
  "model.loss_fn.soft_beta=${BEST_BETA} model.loss_fn.soft_rho=${BEST_RHO} model.loss_fn.aux_weight=${BEST_ETA} model.factor_schedulers.k_min.start=7 model.factor_schedulers.k_min.end=7 model.factor_schedulers.k_min.horizon=5000 tags=[rebuttal_sweep,kmin_fixed7] logger.wandb.group=rebuttal_sweep_kmin"
)
run_batch names extras

# =============================================================================
# STAGE 4: Merge all eval CSVs into one summary table
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STAGE 4: Merge Results                         ║"
echo "╚══════════════════════════════════════════════════╝"

${PYTHON} - <<'PYEOF'
import pandas as pd
from pathlib import Path

results_dir = Path("results/rebuttal_sweep")
dfs = []
for f in sorted(results_dir.glob("sweep_*.csv")):
    df = pd.read_csv(f)
    df.insert(0, "run", f.stem)
    # Parse beta, rho, eta, kmin from run name
    name = f.stem
    if name.startswith("sweep_b"):
        parts = name.split("_")
        df["beta"] = float(parts[1][1:])
        df["rho"] = float(parts[2][1:])
        df["sweep_type"] = "beta_rho"
    elif name.startswith("sweep_eta"):
        df["eta"] = float(name.split("eta")[1])
        df["sweep_type"] = "eta"
    elif name.startswith("sweep_kmin"):
        df["kmin_variant"] = name.split("kmin_")[1]
        df["sweep_type"] = "kmin"
    dfs.append(df)

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(results_dir / "all_sweep.csv", index=False)

    cols = ["run", "Acc", "Unique_valid", "NormCov", "KL(pi->p*)", "KL(p*->pi)", "JS_tok", "log_pterm"]
    avail = [c for c in cols if c in merged.columns]
    print("\n" + "=" * 90)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 90)
    print(merged[avail].to_string(index=False))
    print(f"\nSaved: {results_dir / 'all_sweep.csv'}")
else:
    print("WARNING: No sweep results found!")
PYEOF

# =============================================================================
# STAGE 5: Generate figures
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  STAGE 5: Generate Figures                      ║"
echo "╚══════════════════════════════════════════════════╝"

${PYTHON} - <<'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("results/rebuttal_sweep")
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

csv_path = results_dir / "all_sweep.csv"
if not csv_path.exists():
    print("No all_sweep.csv found, skipping figures")
    exit(0)

df = pd.read_csv(csv_path)

# --- Figure 1: β × ρ Heatmap ---
beta_rho = df[df.get("sweep_type", pd.Series()) == "beta_rho"].copy()
if len(beta_rho) > 0:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        for ax, metric, title, fmt, cmap in [
            (axes[0], "NormCov", "NormCov ↑", ".3f", "YlGnBu"),
            (axes[1], "log_pterm", r"log $p_{\mathrm{term}}(\tau)$", ".2f", "RdYlGn"),
        ]:
            if metric not in beta_rho.columns:
                continue
            pivot = beta_rho.pivot_table(index="rho", columns="beta", values=metric)
            pivot = pivot.sort_index(ascending=True)
            pivot.columns = [f"β={c}" for c in pivot.columns]
            pivot.index = [f"ρ={i}" for i in pivot.index]

            sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                        linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax.set_title(title, fontsize=13)
            ax.set_ylabel("")
            ax.set_xlabel("")

        fig.suptitle("Expr24 (RP): β × ρ sensitivity", fontsize=14, y=1.02)
        fig.tight_layout()
        out = fig_dir / "sweep_beta_rho_heatmap.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)
    except Exception as e:
        print(f"Heatmap failed: {e}")

# --- Figure 2: k_min Bar Chart ---
kmin = df[df.get("sweep_type", pd.Series()) == "kmin"].copy()
if len(kmin) > 0:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["NormCov", "Acc", "log_pterm"]
        labels_m = ["NormCov ↑", "Acc ↑", r"log $p_{\mathrm{term}}(\tau)$"]
        variant_order = ["fixed3", "7to3", "fixed7"]
        variant_labels = {"fixed3": "Fixed\nk=3", "7to3": "Schedule\n7→3", "fixed7": "Fixed\nk=7"}

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        colors = ["#4c72b0", "#55a868", "#c44e52"]

        for ax, metric, label in zip(axes, metrics, labels_m):
            if metric not in kmin.columns:
                continue
            vals, names_v = [], []
            for v in variant_order:
                row = kmin[kmin["kmin_variant"] == v]
                val = row[metric].values[0] if len(row) > 0 else 0
                vals.append(val)
                names_v.append(variant_labels.get(v, v))

            bars = ax.bar(names_v, vals, color=colors, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            ax.set_title(label, fontsize=12)
            ax.set_ylim(bottom=min(0, min(vals) * 1.2) if min(vals) < 0 else 0)

        fig.suptitle(r"Expr24: $k_{\min}$ schedule ablation", fontsize=14, y=1.02)
        fig.tight_layout()
        out = fig_dir / "sweep_kmin_barplot.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)
    except Exception as e:
        print(f"k_min barplot failed: {e}")

# --- Figure 3: η Bar Chart ---
eta_df = df[df.get("sweep_type", pd.Series()) == "eta"].copy()
if len(eta_df) > 0:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        eta_vals = sorted(eta_df["eta"].unique())

        for ax, metric, label in [(axes[0], "NormCov", "NormCov ↑"), (axes[1], "Acc", "Acc ↑")]:
            vals = [eta_df[eta_df["eta"] == e][metric].values[0] for e in eta_vals]
            ax.bar([f"η={e}" for e in eta_vals], vals,
                   color=["#4c72b0", "#55a868", "#c44e52"], edgecolor="black", linewidth=0.5)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
            ax.set_title(label, fontsize=12)

        fig.suptitle("Expr24: η (aux weight) sweep", fontsize=14, y=1.02)
        fig.tight_layout()
        out = fig_dir / "sweep_eta_barplot.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)
    except Exception as e:
        print(f"eta barplot failed: {e}")

print("\nAll figures done.")
PYEOF

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Total runs:     ${total_runs}"
echo "  Failures:       ${total_failures}"
echo "  Results:        ${RESULTS_DIR}/all_sweep.csv"
echo "  Figures:        figures/sweep_beta_rho_heatmap.pdf"
echo "                  figures/sweep_kmin_barplot.pdf"
echo "                  figures/sweep_eta_barplot.pdf"
echo ""
echo "  Finished at:    $(timestamp)"

if (( total_failures > 0 )); then
  exit 1
fi
