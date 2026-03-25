#!/usr/bin/env python3
"""
Collect full validation results: mean +/- std across seeds, per (config, task).

Usage:
    # From wandb
    python scripts/sweep/collect_full_validation.py --source wandb

    # From CSV
    python scripts/sweep/collect_full_validation.py --source csv --csv-path results.csv

Outputs:
    results/full_validation_summary.csv   — per-config-task mean±std
    results/full_validation_all.csv       — all individual runs
    stdout: LaTeX-ready table
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPR24_METRICS = [
    "test/acc",
    "test/norm_coverage",
    "test/log_pterm",
    "test/kl_div",
    "test/js_div",
]

SMILES_METRICS = [
    "test/validity",
    "test/acc",
    "test/fp_diversity",
    "test/macro_fp",
    "test/score",
    "test/avg_len",
]

ALL_METRICS = list(dict.fromkeys(EXPR24_METRICS + SMILES_METRICS))


def load_from_wandb(project: str, group: str, entity: str = None) -> pd.DataFrame:
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
        cfg = run.config
        row["seed"] = cfg.get("seed", None)

        # Parse tag and task from run name: full_{tag}_{task}_s{seed}
        name = run.name
        row["tag"] = "unknown"
        row["task"] = "unknown"
        if "expr24_rp" in name:
            row["task"] = "expr24_rp"
        elif "expr24_oracle" in name:
            row["task"] = "expr24_oracle"
        elif "smiles" in name:
            row["task"] = "smiles"
        if "topA" in name:
            row["tag"] = "topA"
        elif "topB" in name:
            row["tag"] = "topB"

        # Extract hyperparams
        row["beta"] = cfg.get("model/loss_fn/soft_beta", cfg.get("model.loss_fn.soft_beta"))
        row["rho"] = cfg.get("model/loss_fn/soft_rho", cfg.get("model.loss_fn.soft_rho"))
        row["eta"] = cfg.get("model/loss_fn/aux_weight", cfg.get("model.loss_fn.aux_weight"))

        # Summary metrics
        summary = run.summary
        for m in ALL_METRICS:
            key = m.replace("/", "_")
            row[key] = summary.get(m)
        records.append(row)

    return pd.DataFrame(records)


def load_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fmt_mean_std(mean, std, fmt=".3f"):
    return f"{mean:{fmt}} ± {std:{fmt}}"


def fmt_latex(mean, std, fmt=".3f"):
    return f"${mean:{fmt}} \\pm {std:{fmt}}$"


def summarize(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all individual runs
    all_path = out_dir / "full_validation_all.csv"
    df.to_csv(all_path, index=False)
    print(f"Saved all runs: {all_path}")

    # Group by (tag, task), compute mean/std
    metric_cols = [c for c in df.columns if c.startswith("test_")]
    groups = df.groupby(["tag", "task"])

    summary_rows = []
    for (tag, task), grp in groups:
        row = {"config": tag, "task": task, "n_seeds": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std()
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    sum_path = out_dir / "full_validation_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Saved summary: {sum_path}")

    # Print human-readable table
    print("\n" + "=" * 80)
    print("Full Validation Results (mean ± std across seeds)")
    print("=" * 80)

    for task in ["expr24_rp", "expr24_oracle", "smiles"]:
        task_df = summary[summary["task"] == task]
        if task_df.empty:
            continue

        metrics = EXPR24_METRICS if "expr24" in task else SMILES_METRICS
        metric_keys = [m.replace("/", "_") for m in metrics]

        print(f"\n--- {task.upper()} ---")
        header = f"{'Config':<10}"
        for m in metrics:
            header += f"  {m.split('/')[-1]:>16}"
        print(header)
        print("-" * len(header))

        for _, row in task_df.iterrows():
            line = f"{row['config']:<10}"
            for mk in metric_keys:
                mean = row.get(f"{mk}_mean")
                std = row.get(f"{mk}_std")
                if mean is not None and std is not None:
                    line += f"  {fmt_mean_std(mean, std):>16}"
                else:
                    line += f"  {'N/A':>16}"
            print(line)

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table (copy-paste ready)")
    print("=" * 80)

    for task in ["expr24_rp", "expr24_oracle", "smiles"]:
        task_df = summary[summary["task"] == task]
        if task_df.empty:
            continue

        metrics = EXPR24_METRICS if "expr24" in task else SMILES_METRICS
        metric_keys = [m.replace("/", "_") for m in metrics]
        short_names = [m.split("/")[-1] for m in metrics]

        print(f"\n% {task}")
        print(f"% Columns: Config & {' & '.join(short_names)} \\\\")
        for _, row in task_df.iterrows():
            cells = [row["config"]]
            for mk in metric_keys:
                mean = row.get(f"{mk}_mean")
                std = row.get(f"{mk}_std")
                if mean is not None and std is not None:
                    cells.append(fmt_latex(mean, std))
                else:
                    cells.append("--")
            print(" & ".join(cells) + " \\\\")


def main():
    parser = argparse.ArgumentParser(description="Collect full validation results")
    parser.add_argument("--source", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="ChemGFN")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="full_validation")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    if args.source == "wandb":
        df = load_from_wandb(args.wandb_project, args.wandb_group, args.wandb_entity)
    else:
        if not args.csv_path:
            print("ERROR: --csv-path required for csv source")
            sys.exit(1)
        df = load_from_csv(args.csv_path)

    print(f"Loaded {len(df)} runs")
    if len(df) == 0:
        print("No runs found. Check wandb group name.")
        sys.exit(1)

    finished = df[df["state"] == "finished"] if "state" in df.columns else df
    print(f"Finished: {len(finished)}/{len(df)}")

    if len(finished) < len(df):
        not_done = df[df["state"] != "finished"]
        print(f"WARNING: {len(not_done)} runs not finished:")
        for _, r in not_done.iterrows():
            print(f"  {r['run_name']}: {r['state']}")

    summarize(finished, Path(args.out_dir))


if __name__ == "__main__":
    main()
