import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from chemgfn.models.validators import RDKitValidator


def extract_smiles(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    s = text.split("<|end_of_text|>")[0]
    s = re.sub(r"<\|.*?\|>", "", s)
    return "".join(s.split())


def parse_token_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if not isinstance(value, str):
        return []
    value = value.strip()
    if not value:
        return []
    try:
        out = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if isinstance(out, list):
        return [int(v) for v in out]
    return []


def find_sample_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Sampled sentence",
        "sample",
        "Sample",
        "sampled_sentence",
        "sentence",
    ]
    for name in candidates:
        if name in df.columns:
            return name
    return None


def find_run_dir(path: Path) -> Path | None:
    for parent in [path.parent, *path.parents]:
        if (parent / ".hydra" / "config.yaml").exists():
            return parent
    return None


def load_hydra_config(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        from omegaconf import OmegaConf
    except Exception:
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None


def parse_meta_from_path(path: Path) -> dict[str, Any]:
    parts = path.parts
    meta: dict[str, Any] = {}
    if "logs" in parts:
        idx = parts.index("logs")
        if idx + 2 < len(parts):
            meta["exp_name"] = parts[idx + 2]
            meta["task_name"] = parts[idx + 1]
    if "runs" in parts:
        ridx = parts.index("runs")
        if ridx + 1 < len(parts):
            meta["run_id"] = parts[ridx + 1]
    m = re.search(r"samples_test_(\d+)\.csv", path.name)
    if m:
        meta["global_step"] = int(m.group(1))
    return meta


def parse_method_from_exp(exp_name: str | None) -> str:
    if not exp_name:
        return "unknown"
    lower = exp_name.lower()
    if "raptb_v2" in lower:
        return "RapTB_v2"
    if "raptb_v1" in lower:
        return "RapTB_v1"
    if "raptb" in lower:
        return "RapTB"
    if "tb" in lower:
        return "TB"
    return exp_name


def parse_weight_from_exp(exp_name: str | None) -> float | None:
    if not exp_name:
        return None
    m = re.search(r"weight[_-]?([0-9.]+)", exp_name)
    if not m:
        m = re.search(r"rap_weight[_-]?([0-9.]+)", exp_name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def compute_metrics(
    smiles_list: list[str],
    token_ids_list: list[list[int]],
    validator: RDKitValidator,
) -> dict[str, Any]:
    valid_flags: list[bool] = []
    scores: list[float] = []
    valid_scores: list[float] = []
    valid_mols: list[Chem.Mol] = []
    valid_smiles: list[str] = []
    char_lens: list[int] = []
    tok_lens: list[int] = []
    valid_tok_lens: list[int] = []
    valid_char_lens: list[int] = []

    for idx, smi in enumerate(smiles_list):
        smi = smi or ""
        char_lens.append(len(smi))
        tok_len = len(token_ids_list[idx]) if idx < len(token_ids_list) else len(smi)
        tok_lens.append(tok_len)

        mol = None
        is_valid = False
        if smi:
            if validator._is_valid_smiles(smi):
                mol = Chem.MolFromSmiles(smi)
                is_valid = bool(mol)
        valid_flags.append(is_valid)

        if is_valid and mol is not None:
            score = float(validator.score_function(mol))
            scores.append(score)
            valid_scores.append(score)
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))
            valid_tok_lens.append(tok_len)
            valid_char_lens.append(len(smi))
        else:
            scores.append(0.0)

    n_samples = len(smiles_list)
    n_valid = int(sum(valid_flags))
    acc = float(n_valid / n_samples) if n_samples else 0.0

    avg_score = float(sum(scores) / n_samples) if n_samples else 0.0
    avg_score_valid = float(sum(valid_scores) / n_valid) if n_valid else 0.0

    if n_valid >= 2:
        fps = [validator._morgan_fp(m) for m in valid_mols]
        mean_sim = validator._mean_pairwise_tanimoto(fps)
        fp_div_internal_valid = 1.0 - float(mean_sim)

        k = min(int(validator.topk_diversity), n_valid)
        if k >= 2:
            top_idx = sorted(range(n_valid), key=lambda j: valid_scores[j], reverse=True)[:k]
            top_fps = [fps[j] for j in top_idx]
            top_mean_sim = validator._mean_pairwise_tanimoto(top_fps)
            fp_div_topk_valid = 1.0 - float(top_mean_sim)
        else:
            fp_div_topk_valid = 0.0
    else:
        fp_div_internal_valid = 0.0
        fp_div_topk_valid = 0.0

    unique_valid = len(set(valid_smiles))
    unique_rate_valid = float(unique_valid / n_valid) if n_valid else 0.0

    tok_mean = float(np.mean(tok_lens)) if tok_lens else 0.0
    tok_valid_mean = float(np.mean(valid_tok_lens)) if valid_tok_lens else 0.0
    char_mean = float(np.mean(char_lens)) if char_lens else 0.0
    char_valid_mean = float(np.mean(valid_char_lens)) if valid_char_lens else 0.0

    return {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "acc": acc,
        f"{validator.scorer_name}_mean_all": avg_score,
        f"{validator.scorer_name}_mean_valid": avg_score_valid,
        "fp_div_internal_valid": float(fp_div_internal_valid),
        "fp_div_topk_valid": float(fp_div_topk_valid),
        "n_unique_valid": int(unique_valid),
        "unique_rate_valid": float(unique_rate_valid),
        "len_tok_mean": tok_mean,
        "len_tok_valid_mean": tok_valid_mean,
        "len_char_mean": char_mean,
        "len_char_valid_mean": char_valid_mean,
    }


def pareto_mask(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    arr = df[keys].to_numpy(dtype=float)
    n = arr.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                mask[i] = False
                break
    return mask


def plot_scatter(
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    color_key: str,
    out_path: Path,
    title: str,
    method_key: str = "method",
    pareto_points: pd.DataFrame | None = None,
) -> None:
    if df.empty:
        return
    methods = df[method_key].fillna("unknown").unique().tolist()
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods)}

    norm = plt.Normalize(df[color_key].min(), df[color_key].max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in df.groupby(method_key):
        colors = cmap(norm(group[color_key].to_numpy()))
        ax.scatter(
            group[x_key],
            group[y_key],
            c=colors,
            marker=marker_map.get(method, "o"),
            label=str(method),
            alpha=0.85,
            edgecolors="none",
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=color_key)

    if pareto_points is not None and not pareto_points.empty:
        ax.scatter(
            pareto_points[x_key],
            pareto_points[y_key],
            facecolors="none",
            edgecolors="black",
            s=90,
            linewidths=1.5,
            label="pareto",
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectories(df: pd.DataFrame, x_key: str, y_key: str, out_path: Path) -> None:
    if df.empty or "run_id" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    any_line = False
    for run_id, group in df.groupby("run_id"):
        if "global_step" not in group.columns:
            continue
        if group["global_step"].nunique() < 2:
            continue
        g = group.sort_values("global_step")
        ax.plot(g[x_key], g[y_key], marker="o", linewidth=1.2, alpha=0.7)
        any_line = True
    if not any_line:
        plt.close(fig)
        return
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("trajectory by run_id")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pareto from samples_test_*.csv.")
    parser.add_argument(
        "--samples-root",
        type=str,
        default="logs",
        help="Root directory to search for samples_test_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/pareto",
        help="Output directory for tables and figures.",
    )
    parser.add_argument("--scorer", type=str, default="qed")
    parser.add_argument("--backend", type=str, default="pa")
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-nbits", type=int, default=2048)
    parser.add_argument("--topk-diversity", type=int, default=20)
    parser.add_argument("--acc-threshold", type=float, default=0.99)
    args = parser.parse_args()

    samples_root = Path(args.samples_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(samples_root.rglob("samples_test_*.csv"))
    if not csv_paths:
        print(f"No samples_test_*.csv found under {samples_root}")
        return

    rows = []
    for path in csv_paths:
        df = pd.read_csv(path)
        col = find_sample_column(df)
        if col is None:
            continue

        smiles_list = [extract_smiles(x) for x in df[col].tolist()]
        token_ids_list = []
        if "token_ids" in df.columns:
            token_ids_list = [parse_token_ids(x) for x in df["token_ids"].tolist()]
        else:
            token_ids_list = [[] for _ in smiles_list]

        meta = parse_meta_from_path(path)
        run_dir = find_run_dir(path)
        cfg = load_hydra_config(run_dir) if run_dir else None

        scorer = args.scorer
        backend = args.backend
        fp_radius = args.fp_radius
        fp_nbits = args.fp_nbits
        topk = args.topk_diversity
        seed = None
        if cfg:
            exp_name = cfg.get("exp_name")
            if exp_name:
                meta["exp_name"] = exp_name
            seed = cfg.get("seed", None)
            validator_cfg = cfg.get("model", {}).get("reward", {}).get("sentence_validator", {})
            scorer = validator_cfg.get("scorer", scorer)
            backend = validator_cfg.get("backend", backend)
            fp_radius = int(validator_cfg.get("fp_radius", fp_radius))
            fp_nbits = int(validator_cfg.get("fp_nbits", fp_nbits))
            topk = int(validator_cfg.get("topk_diversity", topk))

        if seed is not None:
            meta["seed"] = seed

        validator = RDKitValidator(
            scorer=scorer,
            backend=backend,
            fp_radius=fp_radius,
            fp_nbits=fp_nbits,
            topk_diversity=topk,
        )

        metrics = compute_metrics(smiles_list, token_ids_list, validator)
        exp_name = meta.get("exp_name")
        meta["method"] = parse_method_from_exp(exp_name)
        meta["weight"] = parse_weight_from_exp(exp_name)
        meta["scorer"] = scorer
        meta["backend"] = backend
        meta["fp_radius"] = fp_radius
        meta["fp_nbits"] = fp_nbits
        meta["topk_diversity"] = topk

        rows.append({**meta, **metrics, "csv_path": str(path)})

    if not rows:
        print("No usable CSVs found.")
        return

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out_dir / "runs_table.csv", index=False)

    score_key = f"{runs_df.iloc[0]['scorer']}_mean_valid"
    pareto_keys = ["acc", score_key, "fp_div_topk_valid"]
    df_valid = runs_df.dropna(subset=pareto_keys)
    if not df_valid.empty:
        mask = pareto_mask(df_valid, pareto_keys)
        pareto_df = df_valid[mask].copy()
        pareto_df.to_csv(out_dir / "pareto_points_3d.csv", index=False)
    else:
        pareto_df = pd.DataFrame()

    thr = float(args.acc_threshold)
    df_thr = df_valid[df_valid["acc"] >= thr].copy()
    if not df_thr.empty:
        mask_thr = pareto_mask(df_thr, [score_key, "fp_div_topk_valid"])
        df_thr[mask_thr].to_csv(out_dir / f"pareto_points_acc_ge_{thr:.3f}.csv", index=False)

    plot_scatter(
        df_valid,
        score_key,
        "fp_div_topk_valid",
        "acc",
        out_dir / "fig_qed_vs_divTopk_all.png",
        "qed vs div (all)",
        pareto_points=pareto_df,
    )

    for thr_val in [0.98, 0.99, 0.995]:
        df_sub = df_valid[df_valid["acc"] >= thr_val]
        if df_sub.empty:
            continue
        plot_scatter(
            df_sub,
            score_key,
            "fp_div_topk_valid",
            "acc",
            out_dir / f"fig_qed_vs_divTopk_acc_ge_{thr_val:.3f}.png",
            f"qed vs div (acc >= {thr_val:.3f})",
        )

    plot_scatter(
        df_valid,
        "acc",
        "fp_div_topk_valid",
        score_key,
        out_dir / "fig_acc_vs_divTopk.png",
        "acc vs div",
    )

    plot_trajectories(
        df_valid,
        score_key,
        "fp_div_topk_valid",
        out_dir / "fig_trajectories.png",
    )

    summary = {
        "num_runs": int(len(runs_df)),
        "num_pareto_3d": int(len(pareto_df)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
