#!/usr/bin/env python3
"""
Analyze RapTB hyperparameter sweep results and produce rebuttal figures.

Usage:
    # Option 1: from wandb (recommended)
    python scripts/sweep/analyze_sweep.py --source wandb \
        --wandb-project ChemGFN --wandb-group sweep1_beta_rho

    # Option 2: from CSV (export from wandb UI or manual)
    python scripts/sweep/analyze_sweep.py --source csv --csv-path results.csv

Outputs:
    figures/sweep1_beta_rho_heatmap.pdf   - NormCov + log_pterm heatmap
    figures/sweep3_kmin_barplot.pdf        - k_min ablation bars
    stdout: lexicographic ranking of configs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# Metric extraction
# ============================================================================

EXPR24_METRICS = [
    "test/acc",           # Accuracy
    "test/norm_coverage", # NormCov
    "test/log_pterm",     # log p_term(tau) — termination calibration
    "test/kl_div",        # KL divergence
    "test/js_div",        # JS divergence
]

SMILES_METRICS = [
    "test/validity",
    "test/acc",
    "test/fp_diversity",  # FPDiv (fingerprint diversity)
    "test/macro_fp",      # MacroFP
    "test/score",
    "test/avg_len",
]


def load_from_wandb(project: str, group: str, entity: str = None) -> pd.DataFrame:
    """Pull sweep runs from wandb API."""
    try:
        import wandb
    except ImportError:
        print("ERROR: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group})

    records = []
    for run in runs:
        row = {"run_name": run.name, "state": run.state}
        # Extract config
        cfg = run.config
        row["beta"] = cfg.get("model/loss_fn/soft_beta", cfg.get("model.loss_fn.soft_beta"))
        row["rho"] = cfg.get("model/loss_fn/soft_rho", cfg.get("model.loss_fn.soft_rho"))
        row["eta"] = cfg.get("model/loss_fn/aux_weight", cfg.get("model.loss_fn.aux_weight"))
        row["kmin_start"] = cfg.get(
            "model/factor_schedulers/k_min/start",
            cfg.get("model.factor_schedulers.k_min.start"),
        )
        row["kmin_end"] = cfg.get(
            "model/factor_schedulers/k_min/end",
            cfg.get("model.factor_schedulers.k_min.end"),
        )
        # Extract summary metrics
        summary = run.summary
        for m in EXPR24_METRICS + SMILES_METRICS:
            key = m.replace("/", "_")
            row[key] = summary.get(m)
        records.append(row)

    return pd.DataFrame(records)


def load_from_csv(path: str) -> pd.DataFrame:
    """Load sweep results from a CSV file.

    Expected columns: run_name, beta, rho, eta, kmin_start, kmin_end,
                      test_acc, test_norm_coverage, test_log_pterm, ...
    """
    return pd.read_csv(path)


# ============================================================================
# Lexicographic selection
# ============================================================================


def rank_expr24(df: pd.DataFrame) -> pd.DataFrame:
    """Lexicographic ranking for Expr24:
    1. Filter: Acc >= 0.99
    2. Sort by NormCov descending
    3. Tiebreak: |log_pterm| ascending (closer to 0 is better)
    4. Then KL ascending
    """
    eligible = df[df["test_acc"] >= 0.99].copy()
    if eligible.empty:
        print("WARNING: no configs with Acc >= 0.99, relaxing to >= 0.95")
        eligible = df[df["test_acc"] >= 0.95].copy()
    if eligible.empty:
        print("WARNING: no configs with Acc >= 0.95, showing all")
        eligible = df.copy()

    eligible["abs_log_pterm"] = eligible["test_log_pterm"].abs()
    ranked = eligible.sort_values(
        by=["test_norm_coverage", "abs_log_pterm"],
        ascending=[False, True],
    )
    return ranked.drop(columns=["abs_log_pterm"])


def rank_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Lexicographic ranking for SMILES:
    1. Filter: validity not collapsed
    2. Sort by FPDiv descending
    3. Tiebreak: Score descending
    4. Then filter extreme lengths
    """
    eligible = df.copy()
    if "test_validity" in eligible.columns:
        eligible = eligible[eligible["test_validity"] >= 0.90]

    ranked = eligible.sort_values(
        by=["test_fp_diversity", "test_score"],
        ascending=[False, False],
    )
    return ranked


# ============================================================================
# Figures
# ============================================================================


def plot_beta_rho_heatmap(df: pd.DataFrame, out_dir: Path):
    """Fig 1: beta x rho heatmap with NormCov (color) + log_pterm (annotation)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric, title, fmt, cmap in [
        (axes[0], "test_norm_coverage", "NormCov", ".3f", "YlGnBu"),
        (axes[1], "test_log_pterm", r"log $p_{\mathrm{term}}(\tau)$", ".2f", "RdYlGn_r"),
    ]:
        pivot = df.pivot_table(index="rho", columns="beta", values=metric)
        # Sort index for display
        pivot = pivot.sort_index(ascending=True)
        pivot.columns = [f"$\\beta$={c}" for c in pivot.columns]
        pivot.index = [f"$\\rho$={i}" for i in pivot.index]

        sns.heatmap(
            pivot,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.suptitle(
        r"Expr24 (RP replay): $\beta \times \rho$ sensitivity",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / "sweep1_beta_rho_heatmap.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_kmin_barplot(df: pd.DataFrame, out_dir: Path):
    """Fig 2: k_min 3 variants bar comparison."""
    metrics = ["test_norm_coverage", "test_acc", "test_log_pterm"]
    labels = ["NormCov", "Acc", r"log $p_{\mathrm{term}}(\tau)$"]
    variant_labels = {
        "fixed_low": "Fixed low\n(k=3)",
        "schedule_default": "Schedule\n(7 -> 3)",
        "fixed_high": "Fixed high\n(k=7)",
    }

    # Try to infer variant from run_name
    df = df.copy()
    df["variant"] = df["run_name"].apply(
        lambda x: "fixed_low" if "fixed_low" in x
        else "fixed_high" if "fixed_high" in x
        else "schedule_default"
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, metric, label in zip(axes, metrics, labels):
        vals = []
        names = []
        for v in ["fixed_low", "schedule_default", "fixed_high"]:
            row = df[df["variant"] == v]
            val = row[metric].values[0] if len(row) > 0 else 0
            vals.append(val)
            names.append(variant_labels[v])

        colors = ["#4c72b0", "#55a868", "#c44e52"]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(label, fontsize=12)
        ax.set_ylim(bottom=min(0, min(vals) * 1.1))

    fig.suptitle(r"Expr24: $k_{\min}$ schedule ablation", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "sweep3_kmin_barplot.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze RapTB sweep results")
    parser.add_argument("--source", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="ChemGFN")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="sweep1_beta_rho",
        help="wandb group to filter runs",
    )
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument(
        "--sweep",
        choices=["beta_rho", "kmin", "all"],
        default="all",
        help="Which sweep to analyze",
    )
    parser.add_argument(
        "--task",
        choices=["expr24", "smiles"],
        default="expr24",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.source == "wandb":
        df = load_from_wandb(args.wandb_project, args.wandb_group, args.wandb_entity)
    else:
        if not args.csv_path:
            print("ERROR: --csv-path required for csv source")
            sys.exit(1)
        df = load_from_csv(args.csv_path)

    print(f"Loaded {len(df)} runs")
    print(df[["run_name", "beta", "rho", "eta", "kmin_start", "kmin_end"]].to_string())
    print()

    # Ranking
    if args.task == "expr24":
        ranked = rank_expr24(df)
    else:
        ranked = rank_smiles(df)

    print("=" * 60)
    print(f"Lexicographic ranking ({args.task}):")
    print("=" * 60)
    display_cols = ["run_name", "beta", "rho", "eta"]
    if args.task == "expr24":
        display_cols += ["test_acc", "test_norm_coverage", "test_log_pterm"]
    else:
        display_cols += ["test_validity", "test_fp_diversity", "test_score", "test_avg_len"]
    available = [c for c in display_cols if c in ranked.columns]
    print(ranked[available].to_string(index=False))
    print()

    # Figures
    if args.sweep in ("beta_rho", "all"):
        if "beta" in df.columns and "rho" in df.columns:
            plot_beta_rho_heatmap(df, out_dir)

    if args.sweep in ("kmin", "all"):
        if any("fixed_low" in str(n) or "fixed_high" in str(n) for n in df["run_name"]):
            plot_kmin_barplot(df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
