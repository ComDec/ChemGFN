#!/usr/bin/env python3
"""
Analyze k_min sweep experiments from wandb.

Pulls all runs with tags/groups related to k_min ablation from wandb,
computes summary statistics, and generates a comprehensive analysis.

Usage:
    python scripts/sweep/analyze_kmin_wandb.py
    python scripts/sweep/analyze_kmin_wandb.py --entity comdec --project ChemGFN
"""

import argparse
import json
import sys
from collections import defaultdict

import wandb


def fetch_kmin_runs(api, project, entity=None):
    """Fetch all runs related to k_min experiments."""
    path = f"{entity}/{project}" if entity else project

    # Strategy 1: search by group
    kmin_runs = []
    for group_name in ["sweep3_kmin", "full_validation"]:
        try:
            runs = api.runs(path, filters={"group": group_name})
            for r in runs:
                kmin_runs.append(r)
        except Exception as e:
            print(f"  Warning fetching group '{group_name}': {e}")

    # Strategy 2: search by tags
    try:
        runs = api.runs(path, filters={"tags": {"$in": ["sweep3_kmin"]}})
        existing_ids = {r.id for r in kmin_runs}
        for r in runs:
            if r.id not in existing_ids:
                kmin_runs.append(r)
    except Exception as e:
        print(f"  Warning fetching by tag: {e}")

    # Strategy 3: search by name pattern
    try:
        runs = api.runs(path, filters={"display_name": {"$regex": "kmin|k_min"}})
        existing_ids = {r.id for r in kmin_runs}
        for r in runs:
            if r.id not in existing_ids:
                kmin_runs.append(r)
    except Exception as e:
        print(f"  Warning fetching by name: {e}")

    return kmin_runs


def extract_run_info(run):
    """Extract structured info from a run."""
    info = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "group": run.group,
        "tags": list(run.tags),
        "created_at": run.created_at,
    }

    # Extract config
    cfg = run.config
    info["seed"] = cfg.get("seed")

    # Try nested and flat config key formats
    key_map = {
        "beta": ["model/loss_fn/soft_beta", "model.loss_fn.soft_beta"],
        "rho": ["model/loss_fn/soft_rho", "model.loss_fn.soft_rho"],
        "eta": ["model/loss_fn/aux_weight", "model.loss_fn.aux_weight"],
        "k_min_static": ["model/loss_fn/k_min", "model.loss_fn.k_min"],
        "kmin_start": [
            "model/factor_schedulers/k_min/start",
            "model.factor_schedulers.k_min.start",
        ],
        "kmin_end": [
            "model/factor_schedulers/k_min/end",
            "model.factor_schedulers.k_min.end",
        ],
        "kmin_horizon": [
            "model/factor_schedulers/k_min/horizon",
            "model.factor_schedulers.k_min.horizon",
        ],
        "max_steps": ["trainer/max_steps", "trainer.max_steps"],
    }

    for field, keys in key_map.items():
        info[field] = None
        for k in keys:
            # Try flat
            if k in cfg:
                info[field] = cfg[k]
                break
            # Try nested
            parts = k.split("/")
            node = cfg
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    node = None
                    break
            if node is not None:
                info[field] = node
                break

    # Derive k_min variant label
    ks = info.get("kmin_start")
    ke = info.get("kmin_end")
    if ks is not None and ke is not None:
        if ks == ke:
            info["kmin_variant"] = f"fixed_{int(ks)}"
        else:
            info["kmin_variant"] = f"schedule_{int(ks)}_to_{int(ke)}"
    else:
        # Try from run name
        name = run.name.lower()
        if "fixed_low" in name:
            info["kmin_variant"] = "fixed_3"
        elif "fixed_high" in name:
            info["kmin_variant"] = "fixed_7"
        elif "schedule_default" in name or "7_to_3" in name:
            info["kmin_variant"] = "schedule_7_to_3"
        elif "5_to_2" in name:
            info["kmin_variant"] = "schedule_5_to_2"
        else:
            info["kmin_variant"] = "unknown"

    # Derive task
    name = run.name.lower()
    if "smiles" in name:
        info["task"] = "smiles"
    elif "oracle" in name:
        info["task"] = "expr24_oracle"
    elif "expr24" in name or "varexpr" in name:
        info["task"] = "expr24_rp"
    else:
        info["task"] = "unknown"

    # Extract summary metrics
    summary = run.summary
    metric_keys = [
        "test/acc",
        "test/norm_coverage",
        "test/log_pterm",
        "test/kl_div",
        "test/js_div",
        "test/validity",
        "test/fp_diversity",
        "test/macro_fp",
        "test/score",
        "test/avg_len",
        "val/acc",
        "val/norm_coverage",
        "val/log_pterm",
        "val/kl_div",
        "val/js_div",
        "val/validity",
        "val/fp_diversity",
        "val/macro_fp",
        "val/score",
        "val/avg_len",
        "train/loss",
        "train/reward_mean",
    ]
    for m in metric_keys:
        safe_key = m.replace("/", "_")
        info[safe_key] = summary.get(m)

    # Also grab _step for progress info
    info["global_step"] = summary.get("_step", summary.get("trainer/global_step"))

    return info


def fetch_history_metrics(run, metrics, step_sample=50):
    """Fetch training curve data for specific metrics."""
    try:
        history = run.scan_history(keys=metrics + ["_step"], page_size=500)
        rows = list(history)
        if len(rows) > step_sample:
            step = max(1, len(rows) // step_sample)
            rows = rows[::step]
        return rows
    except Exception:
        return []


def print_separator(title, char="=", width=80):
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def main():
    parser = argparse.ArgumentParser(description="Analyze k_min sweep experiments from wandb")
    parser.add_argument("--project", type=str, default="ChemGFN")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    print(f"Fetching k_min related runs from {args.entity or ''}/{args.project}...")
    runs = fetch_kmin_runs(api, args.project, args.entity)
    print(f"Found {len(runs)} runs total")

    if not runs:
        print("No runs found. Check project/entity.")
        sys.exit(1)

    # Extract info from all runs
    all_info = []
    for r in runs:
        info = extract_run_info(r)
        all_info.append(info)

    # =========================================================================
    # 1. Overview: all runs
    # =========================================================================
    print_separator("1. ALL RUNS OVERVIEW")
    print(
        f"{'Name':<50} {'State':<10} {'Task':<15} {'k_min variant':<20} {'Seed':<6} {'Group':<20}"
    )
    print("-" * 130)
    for info in sorted(all_info, key=lambda x: (x["task"], x["kmin_variant"], str(x["seed"]))):
        print(
            f"{info['name']:<50} {info['state']:<10} {info['task']:<15} "
            f"{info['kmin_variant']:<20} {str(info.get('seed','?')):<6} {str(info.get('group','')):<20}"
        )

    # =========================================================================
    # 2. Screening sweep3_kmin results (1750 steps)
    # =========================================================================
    screening = [i for i in all_info if i.get("group") == "sweep3_kmin"]
    if screening:
        print_separator("2. SCREENING SWEEP (sweep3_kmin group, 1750 steps)")

        # Expr24 metrics
        expr24_metrics = [
            "val_acc",
            "val_norm_coverage",
            "val_log_pterm",
            "val_kl_div",
            "val_js_div",
        ]
        test_metrics = [
            "test_acc",
            "test_norm_coverage",
            "test_log_pterm",
            "test_kl_div",
            "test_js_div",
        ]

        # Use val or test depending on availability
        use_metrics = test_metrics  # prefer test
        metric_labels = ["Acc", "NormCov", "log_pterm", "KL", "JS"]

        header = f"{'Variant':<25} {'State':<10}"
        for label in metric_labels:
            header += f" {label:>12}"
        print(header)
        print("-" * len(header))

        for info in sorted(screening, key=lambda x: x["kmin_variant"]):
            line = f"{info['kmin_variant']:<25} {info['state']:<10}"
            for m in use_metrics:
                val = info.get(m)
                if val is not None:
                    line += f" {val:>12.4f}"
                else:
                    # fallback to val_ version
                    val_key = m.replace("test_", "val_")
                    val2 = info.get(val_key)
                    if val2 is not None:
                        line += f" {val2:>12.4f}"
                    else:
                        line += f" {'N/A':>12}"
            print(line)

        # Analysis
        print("\n--- Analysis ---")
        finished = [i for i in screening if i["state"] == "finished"]
        if finished:
            best_by_coverage = max(
                finished,
                key=lambda x: x.get("test_norm_coverage") or x.get("val_norm_coverage") or 0,
            )
            best_by_acc = max(finished, key=lambda x: x.get("test_acc") or x.get("val_acc") or 0)
            print(
                f"  Best NormCov: {best_by_coverage['kmin_variant']} = {best_by_coverage.get('test_norm_coverage') or best_by_coverage.get('val_norm_coverage')}"
            )
            print(
                f"  Best Acc:     {best_by_acc['kmin_variant']} = {best_by_acc.get('test_acc') or best_by_acc.get('val_acc')}"
            )
        else:
            print("  No finished runs yet.")
    else:
        print_separator("2. SCREENING SWEEP (sweep3_kmin)")
        print("  No sweep3_kmin runs found.")

    # =========================================================================
    # 3. Full validation runs with k_min variation
    # =========================================================================
    full_val = [i for i in all_info if i.get("group") == "full_validation"]
    if full_val:
        print_separator("3. FULL VALIDATION RUNS (full_validation group, 5000 steps)")

        # Group by (task, kmin_variant, tag)
        by_task = defaultdict(list)
        for info in full_val:
            by_task[info["task"]].append(info)

        for task in sorted(by_task.keys()):
            print(f"\n--- Task: {task} ---")
            task_runs = by_task[task]

            if "expr24" in task:
                use_metrics = ["test_acc", "test_norm_coverage", "test_log_pterm", "test_kl_div"]
                metric_labels = ["Acc", "NormCov", "log_pterm", "KL"]
            else:
                use_metrics = [
                    "test_validity",
                    "test_acc",
                    "test_fp_diversity",
                    "test_score",
                    "test_avg_len",
                ]
                metric_labels = ["Valid", "Acc", "FPDiv", "Score", "AvgLen"]

            header = f"{'Name':<45} {'State':<10} {'kmin':<18} {'Seed':<6}"
            for label in metric_labels:
                header += f" {label:>10}"
            print(header)
            print("-" * len(header))

            for info in sorted(
                task_runs, key=lambda x: (x["kmin_variant"], str(x.get("seed", "")))
            ):
                line = f"{info['name']:<45} {info['state']:<10} {info['kmin_variant']:<18} {str(info.get('seed','?')):<6}"
                for m in use_metrics:
                    val = info.get(m)
                    if val is not None:
                        line += f" {val:>10.4f}"
                    else:
                        line += f" {'N/A':>10}"
                print(line)

            # Compute mean±std per kmin_variant (across seeds)
            finished = [i for i in task_runs if i["state"] == "finished"]
            if finished:
                variants = defaultdict(list)
                for i in finished:
                    variants[i["kmin_variant"]].append(i)

                print(f"\n  Aggregated (finished runs, mean ± std):")
                header2 = f"  {'kmin_variant':<22} {'n':>3}"
                for label in metric_labels:
                    header2 += f" {label:>16}"
                print(header2)
                print("  " + "-" * (len(header2) - 2))

                for variant in sorted(variants.keys()):
                    vr = variants[variant]
                    line = f"  {variant:<22} {len(vr):>3}"
                    for m in use_metrics:
                        vals = [i[m] for i in vr if i.get(m) is not None]
                        if vals:
                            import numpy as np

                            mean = np.mean(vals)
                            std = np.std(vals)
                            line += f" {mean:>7.4f}±{std:.4f}"
                        else:
                            line += f" {'N/A':>16}"
                    print(line)
    else:
        print_separator("3. FULL VALIDATION RUNS")
        print("  No full_validation runs found.")

    # =========================================================================
    # 4. Historical runs with k_min in name (broader search)
    # =========================================================================
    other_runs = [i for i in all_info if i.get("group") not in ("sweep3_kmin", "full_validation")]
    if other_runs:
        print_separator("4. OTHER k_min-RELATED RUNS (historical / misc)")
        print(f"{'Name':<55} {'State':<10} {'k_min variant':<20} {'Steps':<8}")
        print("-" * 100)
        for info in sorted(other_runs, key=lambda x: x["name"]):
            steps = info.get("global_step", "?")
            print(
                f"{info['name']:<55} {info['state']:<10} {info['kmin_variant']:<20} {str(steps):<8}"
            )

    # =========================================================================
    # 5. Training curves for screening sweep (if available)
    # =========================================================================
    if screening:
        finished_screening = [i for i in screening if i["state"] == "finished"]
        if finished_screening:
            print_separator("5. TRAINING CURVE SNAPSHOTS (screening sweep)")
            curve_metrics = ["val/acc", "val/norm_coverage", "val/log_pterm", "train/loss"]
            for info in finished_screening:
                run_obj = api.run(
                    f"{args.entity + '/' if args.entity else ''}{args.project}/{info['id']}"
                )
                rows = fetch_history_metrics(run_obj, curve_metrics, step_sample=10)
                if rows:
                    print(f"\n  {info['kmin_variant']}:")
                    print(f"  {'Step':>8}", end="")
                    for m in curve_metrics:
                        print(f"  {m.split('/')[-1]:>14}", end="")
                    print()
                    for row in rows:
                        step = row.get("_step", "?")
                        print(f"  {str(step):>8}", end="")
                        for m in curve_metrics:
                            v = row.get(m)
                            if v is not None:
                                print(f"  {v:>14.4f}", end="")
                            else:
                                print(f"  {'':>14}", end="")
                        print()

    # =========================================================================
    # 6. Summary & Recommendations
    # =========================================================================
    print_separator("6. SUMMARY & RECOMMENDATIONS")

    total = len(all_info)
    finished = sum(1 for i in all_info if i["state"] == "finished")
    running = sum(1 for i in all_info if i["state"] == "running")
    failed = sum(1 for i in all_info if i["state"] not in ("finished", "running"))

    print(f"  Total runs found: {total}")
    print(f"  Finished: {finished}, Running: {running}, Other: {failed}")

    groups = defaultdict(int)
    for i in all_info:
        groups[i.get("group", "none")] += 1
    print(f"  By group: {dict(groups)}")

    variants = defaultdict(int)
    for i in all_info:
        variants[i["kmin_variant"]] += 1
    print(f"  By k_min variant: {dict(variants)}")

    if running > 0:
        print(
            f"\n  NOTE: {running} runs still in progress. Re-run this analysis after completion."
        )


if __name__ == "__main__":
    main()
