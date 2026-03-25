#!/usr/bin/env python3
"""
Detailed k_min analysis: extract actual metrics from ALL historical + current runs.
Focus on comparable finished runs with real metric values.
"""

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb


def get_nested(cfg, dotpath):
    """Get value from nested dict using dot or slash path."""
    for sep in ["/", "."]:
        parts = dotpath.split(sep)
        node = cfg
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                node = None
                break
        if node is not None:
            return node
    return None


def main():
    api = wandb.Api()
    project = "ChemGFN"

    print("Fetching ALL runs from wandb...")
    all_runs = list(api.runs(project))
    print(f"Total runs in project: {len(all_runs)}")

    # Filter to k_min related runs (by name pattern)
    kmin_runs = [
        r
        for r in all_runs
        if "kmin" in r.name.lower()
        or "k_min" in r.name.lower()
        or r.group == "full_validation"
        or r.group == "sweep3_kmin"
    ]

    print(f"k_min related: {len(kmin_runs)}")
    finished = [r for r in kmin_runs if r.state == "finished"]
    running = [r for r in kmin_runs if r.state == "running"]
    print(f"Finished: {len(finished)}, Running: {len(running)}")

    # Build DataFrame
    records = []
    for r in kmin_runs:
        cfg = r.config
        summary = r.summary

        # Determine task
        name = r.name.lower()
        if "smiles" in name:
            task = "smiles"
        elif "oracle" in name:
            task = "expr24_oracle"
        elif "expr24" in name or "varexpr" in name:
            task = "expr24_rp"
        else:
            task = "unknown"

        # k_min config
        ks = get_nested(cfg, "model/factor_schedulers/k_min/start")
        ke = get_nested(cfg, "model/factor_schedulers/k_min/end")
        k_static = get_nested(cfg, "model/loss_fn/k_min")

        if ks is not None and ke is not None:
            if ks == ke:
                kmin_label = f"fixed_{int(ks)}"
            else:
                kmin_label = f"{int(ks)}→{int(ke)}"
        else:
            # Parse from name
            if "kmin_0" in name and "to" not in name.split("kmin_0")[0][-5:]:
                kmin_label = "fixed_0"
            elif "kmin_2" in name and "to" not in name:
                kmin_label = "fixed_2"
            elif "kmin_3" in name and "to" not in name:
                kmin_label = "fixed_3"
            elif "kmin_5" in name and "to" not in name:
                kmin_label = "fixed_5"
            elif "kmin_7" in name and "to" not in name:
                kmin_label = "fixed_7"
            elif "kmin_10" in name and "to" not in name:
                kmin_label = "fixed_10"
            else:
                kmin_label = "from_name"

        # Hyperparams
        beta = get_nested(cfg, "model/loss_fn/soft_beta")
        rho = get_nested(cfg, "model/loss_fn/soft_rho")
        eta = get_nested(cfg, "model/loss_fn/aux_weight")
        max_steps = get_nested(cfg, "trainer/max_steps")

        row = {
            "name": r.name,
            "state": r.state,
            "group": r.group or "",
            "task": task,
            "kmin_label": kmin_label,
            "kmin_start": ks,
            "kmin_end": ke,
            "beta": beta,
            "rho": rho,
            "eta": eta,
            "max_steps": max_steps,
            "global_step": summary.get("_step"),
            "seed": cfg.get("seed"),
        }

        # All metrics - prefer test/ over val/
        for prefix in ["test", "val", "train"]:
            for m in [
                "acc",
                "norm_coverage",
                "log_pterm",
                "kl_div",
                "js_div",
                "validity",
                "fp_diversity",
                "macro_fp",
                "score",
                "avg_len",
                "loss",
                "reward_mean",
            ]:
                key = f"{prefix}/{m}"
                row[f"{prefix}_{m}"] = summary.get(key)

        records.append(row)

    df = pd.DataFrame(records)

    # =========================================================================
    # SECTION 1: Expr24 RP - k_min comparison (finished runs only)
    # =========================================================================
    print("\n" + "=" * 100)
    print(" EXPR24 RP: k_min Comparison (finished runs)")
    print("=" * 100)

    expr_rp = df[(df["task"] == "expr24_rp") & (df["state"] == "finished")].copy()
    if not expr_rp.empty:
        # Use test metrics if available, fallback to val
        for m in ["acc", "norm_coverage", "log_pterm", "kl_div", "js_div"]:
            expr_rp[m] = expr_rp[f"test_{m}"].combine_first(expr_rp[f"val_{m}"])

        cols = [
            "name",
            "kmin_label",
            "kmin_start",
            "kmin_end",
            "beta",
            "rho",
            "global_step",
            "acc",
            "norm_coverage",
            "log_pterm",
            "kl_div",
            "js_div",
        ]
        avail = [c for c in cols if c in expr_rp.columns]
        expr_rp_display = expr_rp[avail].sort_values("kmin_label")

        pd.set_option("display.max_colwidth", 55)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        print(expr_rp_display.to_string(index=False))

        # Aggregate by kmin_label
        print("\n--- Aggregated by k_min variant ---")
        agg_metrics = ["acc", "norm_coverage", "log_pterm", "kl_div", "js_div"]
        agg_results = []
        for label, grp in expr_rp.groupby("kmin_label"):
            row = {"kmin_variant": label, "n_runs": len(grp)}
            for m in agg_metrics:
                vals = grp[m].dropna()
                if len(vals) > 0:
                    row[f"{m}_mean"] = vals.mean()
                    row[f"{m}_std"] = vals.std() if len(vals) > 1 else 0
                    row[f"{m}_best"] = vals.max() if m in ["acc", "norm_coverage"] else vals.min()
            agg_results.append(row)

        agg_df = pd.DataFrame(agg_results)
        print(agg_df.to_string(index=False))

    # =========================================================================
    # SECTION 2: Expr24 Oracle - k_min comparison
    # =========================================================================
    print("\n" + "=" * 100)
    print(" EXPR24 ORACLE: k_min Comparison (finished runs)")
    print("=" * 100)

    expr_oracle = df[(df["task"] == "expr24_oracle") & (df["state"] == "finished")].copy()
    if not expr_oracle.empty:
        for m in ["acc", "norm_coverage", "log_pterm", "kl_div", "js_div"]:
            expr_oracle[m] = expr_oracle[f"test_{m}"].combine_first(expr_oracle[f"val_{m}"])

        cols = [
            "name",
            "kmin_label",
            "kmin_start",
            "kmin_end",
            "beta",
            "rho",
            "global_step",
            "acc",
            "norm_coverage",
            "log_pterm",
            "kl_div",
        ]
        avail = [c for c in cols if c in expr_oracle.columns]
        print(expr_oracle[avail].sort_values("kmin_label").to_string(index=False))

        print("\n--- Aggregated by k_min variant ---")
        agg_results = []
        for label, grp in expr_oracle.groupby("kmin_label"):
            row = {"kmin_variant": label, "n_runs": len(grp)}
            for m in ["acc", "norm_coverage", "log_pterm"]:
                vals = grp[m].dropna()
                if len(vals) > 0:
                    row[f"{m}_mean"] = vals.mean()
                    row[f"{m}_best"] = vals.max() if m in ["acc", "norm_coverage"] else vals.min()
            agg_results.append(row)
        print(pd.DataFrame(agg_results).to_string(index=False))

    # =========================================================================
    # SECTION 3: SMILES - k_min comparison
    # =========================================================================
    print("\n" + "=" * 100)
    print(" SMILES: k_min Comparison (finished runs)")
    print("=" * 100)

    smiles = df[(df["task"] == "smiles") & (df["state"] == "finished")].copy()
    if not smiles.empty:
        for m in ["validity", "acc", "fp_diversity", "macro_fp", "score", "avg_len"]:
            smiles[m] = smiles[f"test_{m}"].combine_first(smiles[f"val_{m}"])

        cols = [
            "name",
            "kmin_label",
            "kmin_start",
            "kmin_end",
            "beta",
            "rho",
            "global_step",
            "validity",
            "acc",
            "fp_diversity",
            "score",
            "avg_len",
        ]
        avail = [c for c in cols if c in smiles.columns]
        print(smiles[avail].sort_values("kmin_label").to_string(index=False))

        print("\n--- Aggregated by k_min variant ---")
        agg_results = []
        for label, grp in smiles.groupby("kmin_label"):
            row = {"kmin_variant": label, "n_runs": len(grp)}
            for m in ["validity", "acc", "fp_diversity", "score", "avg_len"]:
                vals = grp[m].dropna()
                if len(vals) > 0:
                    row[f"{m}_mean"] = vals.mean()
                    row[f"{m}_best"] = (
                        vals.max()
                        if m in ["validity", "acc", "fp_diversity", "score"]
                        else vals.min()
                    )
            agg_results.append(row)
        print(pd.DataFrame(agg_results).to_string(index=False))

    # =========================================================================
    # SECTION 4: Currently running experiments
    # =========================================================================
    print("\n" + "=" * 100)
    print(" CURRENTLY RUNNING")
    print("=" * 100)

    running_df = df[df["state"] == "running"]
    if not running_df.empty:
        cols = ["name", "task", "kmin_label", "seed", "global_step", "group"]
        avail = [c for c in cols if c in running_df.columns]
        print(running_df[avail].to_string(index=False))
    else:
        print("  No running experiments.")

    # =========================================================================
    # SECTION 5: Key findings summary
    # =========================================================================
    print("\n" + "=" * 100)
    print(" KEY FINDINGS")
    print("=" * 100)

    # Expr24 RP best
    if not expr_rp.empty:
        best_cov = (
            expr_rp.loc[expr_rp["norm_coverage"].idxmax()]
            if expr_rp["norm_coverage"].notna().any()
            else None
        )
        best_acc = expr_rp.loc[expr_rp["acc"].idxmax()] if expr_rp["acc"].notna().any() else None
        if best_cov is not None:
            print(f"\n  Expr24 RP - Best NormCov: {best_cov['name']}")
            print(
                f"    k_min={best_cov['kmin_label']}, NormCov={best_cov['norm_coverage']:.4f}, Acc={best_cov['acc']:.4f}"
            )
        if best_acc is not None:
            print(f"  Expr24 RP - Best Acc: {best_acc['name']}")
            print(
                f"    k_min={best_acc['kmin_label']}, Acc={best_acc['acc']:.4f}, NormCov={best_acc.get('norm_coverage', 'N/A')}"
            )

    # SMILES best
    if not smiles.empty:
        best_div = (
            smiles.loc[smiles["fp_diversity"].idxmax()]
            if smiles["fp_diversity"].notna().any()
            else None
        )
        if best_div is not None:
            print(f"\n  SMILES - Best FP Diversity: {best_div['name']}")
            print(
                f"    k_min={best_div['kmin_label']}, FPDiv={best_div['fp_diversity']:.4f}, Score={best_div.get('score', 'N/A')}"
            )

    # Coverage of k_min variants
    print("\n  k_min variant coverage:")
    for task in ["expr24_rp", "expr24_oracle", "smiles"]:
        task_df = df[(df["task"] == task) & (df["state"] == "finished")]
        variants = sorted(task_df["kmin_label"].unique())
        print(f"    {task}: {variants}")

    print("\n  WARNING: Historical runs may use DIFFERENT code versions, reward functions,")
    print("  and other hyperparams. Direct cross-run comparison requires caution.")
    print("  The sweep3_kmin screening (controlled ablation) has NOT been run yet.")


if __name__ == "__main__":
    main()
