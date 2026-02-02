from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# ============================
# Configuration
# ============================

PAD_EOS_IDS: tuple[int, ...] = (128001,)  # tokenizer EOS/pad used in reference buffer

# ============================================================
# Style helpers (kept close to draw_smiles for a consistent look)
# ============================================================


@dataclass
class LineParams:
    marker: str = "o"
    markersize: float = 4.2
    linewidth: float = 3.0
    markeredgewidth: float = 0.0
    markeredgecolor: str | None = None
    alpha: float = 0.95

    def kwargs(
        self,
        color: Any | None = None,
        marker_override: str | None = None,
        linestyle: str | None = None,
    ) -> dict[str, Any]:
        kw = {
            "marker": marker_override if marker_override is not None else self.marker,
            "markersize": self.markersize,
            "linewidth": self.linewidth,
            "markeredgewidth": self.markeredgewidth,
            "markeredgecolor": self.markeredgecolor,
            "alpha": self.alpha,
        }
        if linestyle is not None:
            kw["linestyle"] = linestyle
        if color is not None:
            kw["color"] = color
        return kw


@dataclass
class BarParams:
    edgecolor: str = "black"
    linewidth: float = 0.5
    alpha: float = 0.9

    def kwargs(self, color: Any | None = None) -> dict[str, Any]:
        kw = {"edgecolor": self.edgecolor, "linewidth": self.linewidth, "alpha": self.alpha}
        if color is not None:
            kw["color"] = color
        return kw


@dataclass
class MethodStyle:
    color: Any | None = None
    linestyle: str | None = None
    marker: str | None = None


@dataclass
class PlotStyle:
    dpi: int = 300
    aspect_ratio: float = 2.0
    width_standard: float = 8.0
    width_divergence: float = 10.0
    width_prefix: float = 12.0
    width_nk: float = 7.0
    prefix_triplet_side: float = 3.0
    stacked_bylen_width: float = 7.0
    line: LineParams = field(default_factory=LineParams)
    bar: BarParams = field(default_factory=BarParams)
    # Pre-seed common Expr24 methods so color/linestyle are stable out of the box.
    method_styles: dict[str, MethodStyle] = field(
        default_factory=lambda: {
            "TB": MethodStyle(color="C0", linestyle="--", marker="o"),
            "SubTB": MethodStyle(color="C1", linestyle="-.", marker="s"),
            "RapTB": MethodStyle(color="C2", linestyle="-", marker="^"),
        }
    )
    grid_alpha: float = 0.25
    legend_ncol: int = 3
    legend_frameon: bool = True
    full_frame: bool = True
    shade_regions: list[tuple[float, float]] | None = None
    shade_alpha: float = 0.08
    suptitle_fontsize: int = 14
    title_fontsize: int = 15
    label_fontsize: int = 14
    sns_style: str = "whitegrid"
    sns_context: str = "paper"
    palette: str = "colorblind"
    font_scale: float = 1.0

    # --- sizing helpers (compatible with draw_smiles expectations) ---
    def size(self, width: float) -> tuple[float, float]:
        height = max(width / self.aspect_ratio, 1.0)
        return (width, height)

    @property
    def figsize_prefix(self) -> tuple[float, float]:
        return self.size(self.width_prefix)

    @property
    def figsize_prefix_triplet(self) -> tuple[float, float]:
        side = self.prefix_triplet_side
        return (side * 3.0, side)

    @property
    def figsize_nk(self) -> tuple[float, float]:
        return self.size(self.width_nk)

    @property
    def figsize_len_hist_bins(self) -> tuple[float, float]:
        return self.size(self.width_standard)

    @property
    def figsize_len_hist_fine(self) -> tuple[float, float]:
        return self.size(self.width_standard)

    @property
    def figsize_by_len(self) -> tuple[float, float]:
        return self.size(self.width_standard)

    @property
    def figsize_by_len_stacked(self) -> tuple[float, float]:
        w = self.stacked_bylen_width
        h = w * 1.1
        return (w, h)

    @property
    def figsize_standard(self) -> tuple[float, float]:
        h = max(self.width_standard / self.aspect_ratio, 1.0)
        return (self.width_standard, h)

    @property
    def figsize_divergence(self) -> tuple[float, float]:
        h = max(self.width_divergence / self.aspect_ratio, 1.0)
        return (self.width_divergence, h)


def apply_plot_style(style: PlotStyle) -> None:
    sns.set_theme(
        context=style.sns_context,
        style=style.sns_style,
        palette=style.palette,
        font_scale=style.font_scale,
    )
    plt.rcParams.update(
        {
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "legend.frameon": style.legend_frameon,
            "grid.alpha": style.grid_alpha,
        }
    )


def apply_full_border(ax, style: PlotStyle) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)


def add_shading(ax, regions: list[tuple[float, float]] | None, alpha: float) -> None:
    if not regions:
        return
    for lo, hi in regions:
        ax.axvspan(lo, hi, alpha=alpha)


def resolve_method_style(
    name: str, style: PlotStyle, cmap: dict[str, Any]
) -> tuple[Any | None, str | None, str | None]:
    ms = style.method_styles.get(name) if style and style.method_styles else None
    color = ms.color if ms and ms.color is not None else cmap.get(name)
    marker_override = ms.marker if ms and ms.marker is not None else None
    linestyle = ms.linestyle if ms and ms.linestyle is not None else None
    return color, marker_override, linestyle


def build_color_map(exps: dict[str, dict[str, Any]], palette: str) -> dict[str, tuple]:
    names = list(exps.keys())
    colors = sns.color_palette(palette, n_colors=max(len(names), 3))
    return {name: colors[i % len(colors)] for i, name in enumerate(names)}


def _compute_err(values: np.ndarray, mode: str = "sem") -> np.ndarray:
    if mode == "none":
        return np.zeros_like(values, dtype=float)
    if mode not in {"std", "sem", "ci95"}:
        raise ValueError(f"Unknown error mode: {mode}")
    std = np.nanstd(values, axis=0, ddof=1 if values.shape[0] > 1 else 0)
    if mode == "std":
        return std
    n = np.maximum(np.sum(~np.isnan(values), axis=0), 1)
    sem = std / np.sqrt(n)
    if mode == "ci95":
        return sem * 1.96
    return sem


# ============================================================
# Experiment path normalization (copy of draw_smiles helpers)
# ============================================================


def _ensure_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _append_if_exists(lst: list[str], path: Path) -> None:
    if path.exists():
        lst.append(str(path))


def _collect_single_run(dir_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Collect json/prefix/samples under a single directory."""
    json_paths: list[str] = []
    prefix_paths: list[str] = []
    samples_paths: list[str] = []

    prefix_candidates = itertools.chain(
        dir_path.glob("prefix_tables*/prefix_pos_test_k_correct_0.csv"),
        dir_path.glob("prefix_tables*/prefix_pos_test_k_correct_0*.csv"),
    )
    for pc in prefix_candidates:
        _append_if_exists(prefix_paths, pc)

    sample_candidates = itertools.chain(
        dir_path.glob("test_samples*/samples_test_0*.csv"),
        dir_path.glob("test_samples*/samples_test*.csv"),
    )
    for sc in sample_candidates:
        _append_if_exists(samples_paths, sc)

    json_candidates = list(dir_path.glob("json/test_metrics*.json"))
    if not json_candidates:
        json_candidates = list(dir_path.glob("json/*.json"))
    if not json_candidates:
        json_candidates = list(dir_path.glob("test_metrics*.json"))
    if not json_candidates:
        json_candidates = list(dir_path.glob("*.json"))
    if json_candidates:
        json_paths.append(str(sorted(json_candidates)[0]))

    return json_paths, prefix_paths, samples_paths


def _discover_repeat_runs(repeat_root: str) -> tuple[list[str], list[str], list[str]]:
    base = Path(repeat_root)
    json_paths: list[str] = []
    prefix_paths: list[str] = []
    samples_paths: list[str] = []

    if not base.exists():
        print(f"[warn] repeat_root not found: {repeat_root}")
        return json_paths, prefix_paths, samples_paths

    repeat_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("repeat_")]
    )

    if not repeat_dirs:
        jp, pp, sp = _collect_single_run(base)
        json_paths.extend(jp)
        prefix_paths.extend(pp)
        samples_paths.extend(sp)
        return json_paths, prefix_paths, samples_paths

    for rdir in repeat_dirs:
        jp, pp, sp = _collect_single_run(rdir)
        json_paths.extend(jp)
        prefix_paths.extend(pp)
        samples_paths.extend(sp)

    return json_paths, prefix_paths, samples_paths


def normalize_exps(exps_raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    norm: dict[str, dict[str, Any]] = {}
    for name, payload in exps_raw.items():
        json_paths = _ensure_list(payload.get("json"))
        roots = _ensure_list(payload.get("roots"))
        prefix_paths = _ensure_list(payload.get("prefix"))
        samples_paths = _ensure_list(payload.get("samples"))
        repeat_roots = _ensure_list(payload.get("repeat_root") or payload.get("repeat_roots"))

        for r in roots:
            prefix_paths.append(str(Path(r) / "prefix_tables" / "prefix_pos_test_k_correct_0.csv"))
            samples_paths.append(str(Path(r) / "test_samples" / "samples_test_0.csv"))

        for rr in repeat_roots:
            jp, pp, sp = _discover_repeat_runs(rr)
            json_paths.extend(jp)
            prefix_paths.extend(pp)
            samples_paths.extend(sp)

        norm[name] = {
            "json_paths": json_paths,
            "prefix_paths": prefix_paths,
            "samples_paths": samples_paths,
            "style": payload.get("style", {}),
        }
    return norm


def apply_method_styles_from_exps(exps_norm: dict[str, dict[str, Any]], style: PlotStyle) -> None:
    if style.method_styles is None:
        style.method_styles = {}

    default_linestyles = ["-", "--", "-.", ":"]
    default_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for idx, (name, payload) in enumerate(exps_norm.items()):
        user_style = payload.get("style", {}) or {}
        if name not in style.method_styles:
            ms = MethodStyle(
                color=user_style.get("color"),
                linestyle=user_style.get(
                    "linestyle", default_linestyles[idx % len(default_linestyles)]
                ),
                marker=user_style.get("marker", default_markers[idx % len(default_markers)]),
            )
            style.method_styles[name] = ms
        else:
            ms = style.method_styles[name]
            if user_style.get("color") is not None:
                ms.color = user_style["color"]
            if user_style.get("linestyle") is not None:
                ms.linestyle = user_style["linestyle"]
            if user_style.get("marker") is not None:
                ms.marker = user_style["marker"]


# ============================================================
# IO helpers
# ============================================================


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_prefix_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "k" not in df.columns or "n" not in df.columns:
        raise ValueError(f"Prefix CSV missing required columns k,n. Found: {list(df.columns)}")
    df["k"] = df["k"].astype(int)
    df = df.sort_values("k").reset_index(drop=True)
    base = float(df.iloc[0]["n"]) if len(df) else np.nan
    df["survival"] = df["n"] / base if (base and base > 0) else np.nan
    if "unique" in df.columns:
        df["unique_rate"] = df["unique"] / df["n"].replace(0, np.nan)
    return df


def load_samples_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _infer_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def _parse_token_list(x) -> list[int] | None:
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return None
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return v
        return None
    except Exception:
        return None


# ============================================================
# Coverage / uniqueness / positional divergence for Expr24
# ============================================================


@dataclass
class ExprSamplesConfig:
    token_ids_cols: tuple[str, ...] = ("token_ids", "tokens", "token_id_list")
    valid_cols: tuple[str, ...] = ("is_valid", "correct", "valid", "success")


def load_reference_sequences(path: str) -> list[list[int]]:
    ref = torch.load(path)
    seqs = ref.tolist() if hasattr(ref, "tolist") else list(ref)
    return [trim_eos_padding(s, PAD_EOS_IDS) for s in seqs]


def load_correct_token_lists(samples_path: str, cfg: ExprSamplesConfig) -> list[list[int]]:
    df = load_samples_csv(samples_path)
    token_col = _infer_col(df, cfg.token_ids_cols)
    if token_col is None:
        raise ValueError(f"Cannot find token column in {samples_path}; tried {cfg.token_ids_cols}")

    valid_col = _infer_col(df, cfg.valid_cols)
    if valid_col is not None:
        df = df[_to_bool_series(df[valid_col])].copy()

    tokens: list[list[int]] = []
    for raw in df[token_col].tolist():
        t = _parse_token_list(raw)
        if isinstance(t, list):
            t = trim_eos_padding(t, PAD_EOS_IDS)
            if len(t) > 0:
                tokens.append(t)
    return tokens


def load_token_lists(
    samples_path: str, cfg: ExprSamplesConfig, valid_only: bool = False
) -> list[list[int]]:
    df = load_samples_csv(samples_path)
    token_col = _infer_col(df, cfg.token_ids_cols)
    if token_col is None:
        raise ValueError(f"Cannot find token column in {samples_path}; tried {cfg.token_ids_cols}")

    if valid_only:
        valid_col = _infer_col(df, cfg.valid_cols)
        if valid_col is not None:
            df = df[_to_bool_series(df[valid_col])].copy()

    tokens: list[list[int]] = []
    for raw in df[token_col].tolist():
        t = _parse_token_list(raw)
        if isinstance(t, list):
            t = trim_eos_padding(t, PAD_EOS_IDS)
            if len(t) > 0:
                tokens.append(t)
    return tokens


def compute_coverage(
    reference: list[list[int]], sampled: list[list[int]]
) -> tuple[int, int, float]:
    ref_set = {tuple(s) for s in reference}
    samp_set = {tuple(s) for s in sampled}
    covered = len(ref_set.intersection(samp_set))
    total = len(ref_set)
    rate = covered / total if total > 0 else np.nan
    return covered, total, rate


def compute_uniqueness(sampled: list[list[int]]) -> tuple[int, int, float]:
    n = len(sampled)
    uniq = len({tuple(s) for s in sampled})
    rate = uniq / n if n > 0 else np.nan
    return uniq, n, rate


def _length_counts(seqs: list[list[int]]) -> Counter:
    c = Counter()
    for s in seqs:
        L = len(s)
        if L > 0:
            c[L] += 1
    return c


def _js_from_counters(c1: Counter, c2: Counter, eps: float = 1e-9) -> float:
    keys = sorted(set(c1.keys()) | set(c2.keys()))
    if not keys:
        return np.nan
    v1 = np.array([c1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([c2.get(k, 0) for k in keys], dtype=float)
    if v1.sum() == 0 or v2.sum() == 0:
        return np.nan
    p = v1 / v1.sum()
    q = v2 / v2.sum()
    return _js_div(p, q, eps=eps)


def check_sample_counts(
    exps: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Check sample counts across different methods and repeats.

    Reads samples_test CSV files for each experiment and each repeat,
    counts the number of samples, and returns a summary table.
    Also prints warnings if counts are inconsistent across methods.

    Args:
        exps: Normalized experiment dictionary (output of normalize_exps).
        verbose: If True, print summary and warnings.

    Returns:
        DataFrame with columns: experiment, repeat_idx, samples_path, n_samples
    """
    rows = []
    for exp_name, payload in exps.items():
        samples_paths = payload.get("samples_paths", [])
        for repeat_idx, spath in enumerate(samples_paths):
            if not spath:
                continue
            spath_path = Path(spath)
            if not spath_path.exists():
                if verbose:
                    print(f"[warn] samples csv not found: {spath}")
                rows.append(
                    {
                        "experiment": exp_name,
                        "repeat_idx": repeat_idx,
                        "samples_path": spath,
                        "n_samples": np.nan,
                        "exists": False,
                    }
                )
                continue
            try:
                df = pd.read_csv(spath)
                n_samples = len(df)
            except Exception as e:
                if verbose:
                    print(f"[warn] failed to read {spath}: {e}")
                n_samples = np.nan
            rows.append(
                {
                    "experiment": exp_name,
                    "repeat_idx": repeat_idx,
                    "samples_path": spath,
                    "n_samples": n_samples,
                    "exists": True,
                }
            )

    if not rows:
        if verbose:
            print("No samples files found.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    if verbose:
        # Print summary table
        print("\n" + "=" * 80)
        print("Sample Counts Summary")
        print("=" * 80)

        # Group by experiment and show counts
        for exp_name in result_df["experiment"].unique():
            exp_df = result_df[result_df["experiment"] == exp_name]
            counts = exp_df["n_samples"].dropna().tolist()
            n_repeats = len(exp_df)
            n_valid = len(counts)

            if counts:
                min_c, max_c = int(min(counts)), int(max(counts))
                mean_c = np.mean(counts)
                consistent = "YES" if min_c == max_c else "NO"
                print(f"\n{exp_name}:")
                print(f"  Repeats: {n_repeats} (valid: {n_valid})")
                print(f"  Sample counts: min={min_c}, max={max_c}, mean={mean_c:.1f}")
                print(f"  Consistent: {consistent}")
                if min_c != max_c:
                    print(f"  [WARNING] Inconsistent sample counts across repeats!")
                    for _, row in exp_df.iterrows():
                        print(
                            f"    repeat_{row['repeat_idx']}: {int(row['n_samples']) if not np.isnan(row['n_samples']) else 'N/A'}"
                        )
            else:
                print(f"\n{exp_name}: No valid samples files found")

        # Cross-method comparison
        print("\n" + "-" * 80)
        print("Cross-Method Comparison (first repeat of each method):")
        print("-" * 80)
        first_repeat_counts = {}
        for exp_name in result_df["experiment"].unique():
            exp_df = result_df[result_df["experiment"] == exp_name]
            if len(exp_df) > 0 and not np.isnan(exp_df.iloc[0]["n_samples"]):
                first_repeat_counts[exp_name] = int(exp_df.iloc[0]["n_samples"])

        if first_repeat_counts:
            counts_set = set(first_repeat_counts.values())
            all_same = len(counts_set) == 1
            print(f"All methods have same sample count: {'YES' if all_same else 'NO'}")
            for exp_name, count in first_repeat_counts.items():
                print(f"  {exp_name}: {count}")
            if not all_same:
                print("[WARNING] Different methods have different sample counts!")

        print("=" * 80 + "\n")

    return result_df


def make_valid_ratio_table(
    exps: dict[str, dict[str, Any]],
    cfg: ExprSamplesConfig,
    error_mode: str = "ci95",
) -> pd.DataFrame:
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, spath in enumerate(payload.get("samples_paths", [])):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            df = load_samples_csv(spath)
            total = len(df)
            valid_col = _infer_col(df, cfg.valid_cols)
            if total == 0 or valid_col is None:
                continue
            m = _to_bool_series(df[valid_col])
            n_valid = int(m.sum())
            ratio = n_valid / total if total > 0 else np.nan
            rows.append(
                {
                    "experiment": exp_name,
                    "run": run_idx,
                    "n_total": total,
                    "n_valid": n_valid,
                    "valid_ratio": ratio,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _aggregate_numeric(df, group_key="experiment", error_mode=error_mode)


def make_json_metrics_table(
    exps: dict[str, dict[str, Any]],
    error_mode: str = "ci95",
) -> pd.DataFrame:
    """
    Extract specific metrics from JSON files:
    - log_pterm_by_len_terminal: value at max length key
    - log_pterm_by_len[9] (kept for backwards compatibility)
    - test/Mean(log_pterm - log_pterm_ref)
    - test/logP(s) (avg)
    - test/logZ (supports test/logZ, test/log_z_b, test/log_z)
    - test/acc (if present)
    """
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, jpath in enumerate(payload.get("json_paths", [])):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue

            try:
                with open(jpath) as f:
                    j = json.load(f)
            except Exception as e:
                print(f"[warn] failed to load json {jpath}: {e}")
                continue

            row = {
                "experiment": exp_name,
                "run": run_idx,
            }

            # Extract log_pterm_by_len at terminal (max length)
            log_pterm_by_len = j.get("log_pterm_by_len", {})
            if isinstance(log_pterm_by_len, dict):
                try:
                    len_keys = [int(k) for k in log_pterm_by_len.keys()]
                except Exception:
                    len_keys = []
                if len_keys:
                    max_len = max(len_keys)
                    val_term = log_pterm_by_len.get(str(max_len))
                    if val_term is None:
                        val_term = log_pterm_by_len.get(max_len)
                    row["log_pterm_by_len_terminal"] = (
                        float(val_term) if val_term is not None else np.nan
                    )
                else:
                    row["log_pterm_by_len_terminal"] = np.nan
                # Keep length-9 for compatibility
                val_9 = log_pterm_by_len.get("9") or log_pterm_by_len.get(9)
                row["log_pterm_by_len_9"] = float(val_9) if val_9 is not None else np.nan
            else:
                row["log_pterm_by_len_terminal"] = np.nan
                row["log_pterm_by_len_9"] = np.nan

            # Extract test/Mean(log_pterm - log_pterm_ref)
            row["test_mean_log_pterm_diff"] = float(
                j.get("test/Mean(log_pterm - log_pterm_ref)", np.nan)
            )

            # Extract test/logP(s) (avg)
            row["test_logP_avg"] = float(j.get("test/logP(s) (avg)", np.nan))

            # Extract test/logZ (supports multiple key variants)
            logz_val = j.get("test/logZ")
            if logz_val is None:
                logz_val = j.get("test/log_z_b")
            if logz_val is None:
                logz_val = j.get("test/log_z")
            row["test_logZ"] = float(logz_val) if logz_val is not None else np.nan

            # Extract test/acc (if present)
            acc_val = j.get("test/acc")
            if acc_val is None:
                acc_val = j.get("test/validator/acc")
            row["test_acc"] = float(acc_val) if acc_val is not None else np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return _aggregate_numeric(df, group_key="experiment", error_mode=error_mode)


def make_pterm_by_length_table(
    exps: dict[str, dict[str, Any]],
    error_mode: str = "ci95",
) -> pd.DataFrame:
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, jpath in enumerate(payload.get("json_paths", [])):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue
            try:
                with open(jpath) as f:
                    j = json.load(f)
            except Exception as e:
                print(f"[warn] failed to load json {jpath}: {e}")
                continue
            log_pterm_by_len = j.get("log_pterm_by_len", {})
            if not isinstance(log_pterm_by_len, dict):
                continue
            for k, v in log_pterm_by_len.items():
                try:
                    ell = int(k)
                except Exception:
                    continue
                rows.append(
                    {
                        "experiment": exp_name,
                        "run": run_idx,
                        "length": ell,
                        "log_pterm_by_len": float(v),
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _aggregate_numeric(df, group_key=["experiment", "length"], error_mode=error_mode)


def position_distributions(seqs: list[list[int]]) -> list[Counter]:
    max_len = max((len(s) for s in seqs), default=0)
    pos_counters = []
    for i in range(max_len):
        c = Counter(s[i] for s in seqs if len(s) > i)
        pos_counters.append(c)
    return pos_counters


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p_safe, q_safe = p + eps, q + eps
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m, eps) + 0.5 * _kl_div(q, m, eps)


def compute_position_divergence(
    ref_pos: list[Counter], seqs: list[list[int]], eps: float = 1e-9
) -> pd.DataFrame:
    rows = []
    pos_list = position_distributions(seqs)
    n_pos = max(len(ref_pos), len(pos_list))
    for i in range(n_pos):
        ref_c = ref_pos[i] if i < len(ref_pos) else Counter()
        samp_c = pos_list[i] if i < len(pos_list) else Counter()
        tokens = sorted(set(ref_c.keys()) | set(samp_c.keys()), key=str)
        if not tokens:
            continue
        ref_total = sum(ref_c.values()) or 1
        samp_total = sum(samp_c.values()) or 1
        ref_probs = np.array([ref_c[t] / ref_total for t in tokens], dtype=float)
        samp_probs = np.array([samp_c[t] / samp_total for t in tokens], dtype=float)
        kl_sr = _kl_div(samp_probs, ref_probs, eps)
        kl_rs = _kl_div(ref_probs, samp_probs, eps)
        js = _js_div(samp_probs, ref_probs, eps)
        l1 = float(np.sum(np.abs(samp_probs - ref_probs)))
        rows.append(
            {
                "position": i,
                "kl_sample_to_ref": kl_sr,
                "kl_ref_to_sample": kl_rs,
                "js": js,
                "l1": l1,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Padding / EOS trimming helper
# ============================================================


def trim_eos_padding(seq: list[int], pad_ids: tuple[int, ...] = PAD_EOS_IDS) -> list[int]:
    if not isinstance(seq, list):
        return seq
    while seq and seq[-1] in pad_ids:
        seq.pop()
    return seq


# ============================================================
# Length-wise coverage & uniqueness (correct-only)
# ============================================================


def _ref_len_sets(reference: list[list[int]]) -> dict[int, set[tuple[int, ...]]]:
    ref_map: dict[int, set[tuple[int, ...]]] = {}
    for seq in reference:
        key = len(seq)
        ref_map.setdefault(key, set()).add(tuple(seq))
    return ref_map


def _sample_len_sets(seqs: list[list[int]]) -> dict[int, list[list[int]]]:
    m: dict[int, list[list[int]]] = {}
    for s in seqs:
        s = trim_eos_padding(s, PAD_EOS_IDS)
        if len(s) == 0:
            continue
        m.setdefault(len(s), []).append(s)
    return m


def make_length_coverage_table(
    exps: dict[str, dict[str, Any]],
    reference_data: list[list[int]],
    cfg: ExprSamplesConfig,
    error_mode: str = "ci95",
) -> pd.DataFrame:
    ref_map = _ref_len_sets(reference_data)
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, spath in enumerate(payload.get("samples_paths", [])):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            # hit/correct-only
            hit_seqs = load_correct_token_lists(spath, cfg)
            hit_map = _sample_len_sets(hit_seqs)

            # all generated (no valid filter)
            gen_seqs = load_token_lists(spath, cfg, valid_only=False)
            gen_map = _sample_len_sets(gen_seqs)

            lengths = sorted(set(ref_map.keys()) | set(gen_map.keys()) | set(hit_map.keys()))
            for L in lengths:
                hit_list = hit_map.get(L, [])
                gen_list = gen_map.get(L, [])
                uniq = len({tuple(s) for s in hit_list})
                n_hit = len(hit_list)
                n_gen = len(gen_list)
                ref_set = ref_map.get(L, set())
                covered = len({tuple(s) for s in hit_list} & ref_set)
                len_ref = len(ref_set)
                rate = covered / len_ref if len_ref > 0 else np.nan
                rows.append(
                    {
                        "experiment": exp_name,
                        "run": run_idx,
                        "length": int(L),
                        "n_correct_by_len": n_hit,
                        "unique_correct_by_len": uniq,
                        "coverage_count_by_len": covered,
                        "coverage_rate_by_len": rate,
                        "len_ref": len_ref,
                        "gen_count_by_len": n_gen,
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = _aggregate_numeric(df, group_key=["experiment", "length"], error_mode=error_mode)
    return agg


def make_js_length_table(
    exps: dict[str, dict[str, Any]],
    reference_data: list[list[int]],
    cfg: ExprSamplesConfig,
    error_mode: str = "ci95",
) -> pd.DataFrame:
    ref_counts = _length_counts(reference_data)
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, spath in enumerate(payload.get("samples_paths", [])):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            # all generated (no valid filter)
            gen_seqs = load_token_lists(spath, cfg, valid_only=False)
            # covered/hit (valid & correct)
            hit_seqs = load_correct_token_lists(spath, cfg)

            gen_counts = _length_counts(gen_seqs)
            hit_counts = _length_counts(hit_seqs)

            js_len_gen = _js_from_counters(gen_counts, ref_counts)
            js_len_hit = _js_from_counters(hit_counts, ref_counts)

            rows.append(
                {
                    "experiment": exp_name,
                    "run": run_idx,
                    "js_len_gen": js_len_gen,
                    "js_len_hit": js_len_hit,
                    "n_gen": sum(gen_counts.values()),
                    "n_hit": sum(hit_counts.values()),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _aggregate_numeric(df, group_key="experiment", error_mode=error_mode)


def plot_length_coverage_uniqueness(
    table: pd.DataFrame,
    style: PlotStyle,
    color_map: dict[str, Any],
    save_dir: Path | None = None,
    pos_offset: int = 3,
) -> None:
    if table is None or len(table) == 0:
        print("No length-wise coverage data to plot.")
        return
    df = table.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_standard, dpi=style.dpi)
    ax_cov, ax_uniq = axes
    # coverage (rate)
    for name in df["experiment"].unique():
        sub = df[df["experiment"] == name].sort_values("length")
        x = sub["length"].to_numpy(dtype=int) + pos_offset  # shift since sequences start at 3
        y = sub["coverage_rate_by_len_mean"].to_numpy(dtype=float)
        err = sub["coverage_rate_by_len_err"].to_numpy(dtype=float)
        color, marker_override, linestyle = resolve_method_style(name, style, color_map)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax_cov.errorbar(x, y, yerr=err, label=name, **line_kw)
    ax_cov.set_xlabel("length", fontsize=style.label_fontsize)
    ax_cov.set_ylabel("coverage rate", fontsize=style.label_fontsize)
    ymax = np.nanmax(df["coverage_rate_by_len_mean"]) if len(df) else 0.0
    ax_cov.set_ylim(0, max(0.05, (ymax if np.isfinite(ymax) else 0.0) * 1.25))
    ax_cov.set_title("Coverage by length", fontsize=style.title_fontsize)
    ax_cov.grid(True, alpha=style.grid_alpha)
    apply_full_border(ax_cov, style)

    # uniqueness (count)
    for name in df["experiment"].unique():
        sub = df[df["experiment"] == name].sort_values("length")
        x = sub["length"].to_numpy(dtype=int) + pos_offset
        y = sub["unique_correct_by_len_mean"].to_numpy(dtype=float)
        err = sub["unique_correct_by_len_err"].to_numpy(dtype=float)
        color, marker_override, linestyle = resolve_method_style(name, style, color_map)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax_uniq.errorbar(x, y, yerr=err, label=name, **line_kw)
    ax_uniq.set_xlabel("length", fontsize=style.label_fontsize)
    ax_uniq.set_ylabel("unique correct count", fontsize=style.label_fontsize)
    ymax_u = np.nanmax(df["unique_correct_by_len_mean"]) if len(df) else 0.0
    ax_uniq.set_ylim(0, max(1.0, (ymax_u if np.isfinite(ymax_u) else 0.0) * 1.25))
    ax_uniq.set_title("Uniqueness by length", fontsize=style.title_fontsize)
    ax_uniq.grid(True, alpha=style.grid_alpha)
    apply_full_border(ax_uniq, style)

    handles, labels = ax_cov.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(3, min(style.legend_ncol, len(labels))),
            frameon=style.legend_frameon,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "length_coverage_uniqueness.pdf", bbox_inches="tight")
    plt.show()


def plot_pterm_by_length(
    table: pd.DataFrame,
    style: PlotStyle,
    color_map: dict[str, Any],
    save_dir: Path | None = None,
) -> None:
    if table is None or len(table) == 0:
        print("No pterm-by-length data to plot.")
        return
    df = table.reset_index()
    fig, ax = plt.subplots(1, 1, figsize=style.figsize_standard, dpi=style.dpi)
    for name in df["experiment"].unique():
        sub = df[df["experiment"] == name].sort_values("length")
        x = sub["length"].to_numpy(dtype=int)
        y = sub["log_pterm_by_len_mean"].to_numpy(dtype=float)
        err = sub["log_pterm_by_len_err"].to_numpy(dtype=float)
        color, marker_override, linestyle = resolve_method_style(name, style, color_map)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax.errorbar(x, y, yerr=err, label=name, **line_kw)
    ax.set_xlabel("length", fontsize=style.label_fontsize)
    ax.set_ylabel("log pterm", fontsize=style.label_fontsize)
    ax.set_title("log pterm by length", fontsize=style.title_fontsize)
    ax.grid(True, alpha=style.grid_alpha)
    apply_full_border(ax, style)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(3, min(style.legend_ncol, len(labels))),
            frameon=style.legend_frameon,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "pterm_by_length.pdf", bbox_inches="tight")
    plt.show()


def make_coverage_table(
    exps: dict[str, dict[str, Any]],
    reference_data: list[list[int]],
    cfg: ExprSamplesConfig,
    error_mode: str = "ci95",
) -> pd.DataFrame:
    ref_set = {tuple(s) for s in reference_data}
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, spath in enumerate(payload.get("samples_paths", [])):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            seqs = load_correct_token_lists(spath, cfg)
            uniq, n, uniq_rate = compute_uniqueness(seqs)
            covered, ref_total, cov_rate = compute_coverage(reference_data, seqs)
            rows.append(
                {
                    "experiment": exp_name,
                    "run": run_idx,
                    "n_correct": n,
                    "unique_correct": uniq,
                    "unique_rate_correct": uniq_rate,
                    "coverage_count": covered,
                    "coverage_rate": cov_rate,
                    "ref_total": ref_total,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _aggregate_numeric(df, group_key="experiment", error_mode=error_mode)


def collect_divergence(
    exps: dict[str, dict[str, Any]],
    reference_data: list[list[int]],
    cfg: ExprSamplesConfig,
) -> dict[str, list[pd.DataFrame]]:
    ref_pos = position_distributions(reference_data)
    out: dict[str, list[pd.DataFrame]] = {}
    for exp_name, payload in exps.items():
        divs: list[pd.DataFrame] = []
        for spath in payload.get("samples_paths", []):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            seqs = load_correct_token_lists(spath, cfg)
            if not seqs:
                continue
            divs.append(compute_position_divergence(ref_pos, seqs))
        if divs:
            out[exp_name] = divs
    return out


def _aggregate_numeric(df: pd.DataFrame, group_key: Any, error_mode: str = "sem") -> pd.DataFrame:
    if df.empty:
        return df
    keys = [group_key] if isinstance(group_key, str) else list(group_key)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for key_vals, g in df.groupby(keys):
        key_vals = (key_vals,) if len(keys) == 1 else tuple(key_vals)
        row = {"n_runs": len(g)}
        for k_name, k_val in zip(keys, key_vals):
            row[k_name] = k_val
        for c in numeric_cols:
            vals = g[c].astype(float).to_numpy()
            mean = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
            err = (
                float(_compute_err(vals.reshape(-1, 1), mode=error_mode).item())
                if len(vals) > 0
                else np.nan
            )
            row[f"{c}_mean"] = mean
            row[f"{c}_err"] = err
        rows.append(row)
    return pd.DataFrame(rows).set_index(keys).sort_index()


def _aggregate_divergence_series(
    div_list: list[pd.DataFrame], metric: str, error_mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not div_list:
        return None
    all_pos = sorted(set().union(*[set(df["position"].tolist()) for df in div_list]))
    if len(all_pos) == 0:
        return None
    arr = np.full((len(div_list), len(all_pos)), np.nan, dtype=float)
    pos_to_idx = {v: i for i, v in enumerate(all_pos)}
    for r, df in enumerate(div_list):
        sub = df.set_index("position")
        for p, v in sub[metric].items():
            arr[r, pos_to_idx[int(p)]] = float(v)
    mean = np.nanmean(arr, axis=0)
    err = _compute_err(arr, mode=error_mode)
    return np.array(all_pos, dtype=int), mean, err


# ============================================================
# Plotting
# ============================================================


def plot_coverage_bar(
    coverage_df: pd.DataFrame,
    style: PlotStyle,
    save_dir: Path | None = None,
) -> None:
    if coverage_df.empty:
        print("No coverage data to plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_standard, dpi=style.dpi)
    metrics = [
        ("coverage_rate_mean", "coverage_rate_err", "Coverage (correct-only, rate)"),
        ("unique_correct_mean", "unique_correct_err", "Uniqueness count (correct-only)"),
    ]
    names = coverage_df.index.tolist()
    x = np.arange(len(names))
    width = 0.6
    cmap = build_color_map({n: {} for n in names}, style.palette)
    for ax, (mean_col, err_col, title) in zip(axes, metrics):
        means = coverage_df[mean_col].to_numpy(dtype=float)
        errs = coverage_df[err_col].to_numpy(dtype=float)
        colors = [cmap[n] for n in names]
        ax.bar(x, means, yerr=errs, color=colors, **style.bar.kwargs())
        ax.set_xticks(x, names, rotation=20, ha="right")
        ylabel = "fraction" if "coverage" in mean_col else "count"
        ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
        ax.set_title(title, fontsize=style.title_fontsize)
        # Adaptive y-range: keep a small headroom even for tiny coverage values.
        max_val = np.nanmax(means) if len(means) else 0.0
        upper = max(0.05, (max_val if np.isfinite(max_val) else 0.0) * 1.25)
        ax.set_ylim(0, upper)
        ax.grid(True, axis="y", alpha=style.grid_alpha)
        apply_full_border(ax, style)
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "coverage_uniqueness.pdf", bbox_inches="tight")
    plt.show()


def plot_position_divergence(
    div_map: dict[str, list[pd.DataFrame]],
    style: PlotStyle,
    color_map: dict[str, Any],
    save_dir: Path | None = None,
    error_mode: str = "sem",
    pos_offset: int = 3,
) -> None:
    if not div_map:
        print("No positional divergence data to plot.")
        return
    metrics = ["kl_sample_to_ref", "kl_ref_to_sample", "js", "l1"]
    fig, axes = plt.subplots(2, 2, figsize=style.figsize_divergence, dpi=style.dpi)
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        for name, div_list in div_map.items():
            agg = _aggregate_divergence_series(div_list, metric, error_mode=error_mode)
            if agg is None:
                continue
            x, mean, err = agg
            x = x + pos_offset  # sequences start from position 3
            color, marker_override, linestyle = resolve_method_style(name, style, color_map)
            line_kw = style.line.kwargs(
                color=color, marker_override=marker_override, linestyle=linestyle
            )
            ax.errorbar(x, mean, yerr=err, label=name, **line_kw)
        ax.set_xlabel("position", fontsize=style.label_fontsize)
        ax.set_ylabel(metric, fontsize=style.label_fontsize)
        ax.set_title(metric, fontsize=style.title_fontsize)
        ax.grid(True, alpha=style.grid_alpha)
        apply_full_border(ax, style)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(4, min(style.legend_ncol, len(labels))),
            frameon=style.legend_frameon,
            bbox_to_anchor=(0.5, 1.04),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "pos_divergence.pdf", bbox_inches="tight")
    plt.show()


# ============================================================
# Runner
# ============================================================


def run_analysis(
    exps: dict[str, dict[str, Any]],
    reference_path: str = "/data1/xw3763/project/gflow/ChemGFN/data/24_points/buffer_24_non_zero.pt",
    style: PlotStyle = PlotStyle(),
    samples_cfg: ExprSamplesConfig = ExprSamplesConfig(),
    error_mode: str = "ci95",
    plot_prefix: bool = True,
    plot_len_hist: bool = True,
    plot_by_len_json: bool = True,
    plot_by_len_samples: bool = False,
    plot_divergence: bool = True,
    plot_coverage: bool = True,
    plot_pterm_by_len: bool = True,
    check_samples: bool = True,
    save_fig_dir: Path | None = Path("figures_expr24"),
    len_hist_bins_y: tuple[float, float] | None = None,
    len_hist_fine_y: tuple[float, float] | None = None,
    json_bylen_y: dict[str, tuple[float, float]] | None = None,
    prefix_ylims: dict[str, tuple[float, float]] | None = None,
    prefix_nk_ylim: tuple[float, float] | None = None,
    output_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Analysis entrypoint (Expr24). Interface mirrors draw_smiles with extra coverage/divergence.
    """
    apply_plot_style(style)
    exps_norm = normalize_exps(exps)
    apply_method_styles_from_exps(exps_norm, style)
    color_map = build_color_map(exps_norm, style.palette)

    out_tables: dict[str, pd.DataFrame] = {}
    reference_data = load_reference_sequences(reference_path)

    # change cwd for outputs if requested (do this before saving any files)
    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        cwd_prev = Path(".").resolve()
        os.chdir(output_root)
    else:
        cwd_prev = None

    # Check sample counts across methods and repeats
    if check_samples:
        sample_counts_df = check_sample_counts(exps_norm, verbose=True)
        out_tables["sample_counts"] = sample_counts_df
        if len(sample_counts_df) > 0:
            sample_counts_df.to_csv("sample_counts.csv", index=False)

    # Coverage / uniqueness
    cov_table = make_coverage_table(
        exps_norm, reference_data, cfg=samples_cfg, error_mode=error_mode
    )
    out_tables["coverage"] = cov_table
    if len(cov_table) > 0:
        cov_table.to_csv("coverage_table.csv")

    # JS over length histograms (gen vs ref; hit vs ref)
    js_len_table = make_js_length_table(
        exps_norm, reference_data, cfg=samples_cfg, error_mode=error_mode
    )
    out_tables["js_length"] = js_len_table
    if len(js_len_table) > 0:
        js_len_table.to_csv("js_length.csv")

    # Valid ratio (per experiment)
    valid_table = make_valid_ratio_table(exps_norm, cfg=samples_cfg, error_mode=error_mode)
    out_tables["valid_ratio"] = valid_table
    if len(valid_table) > 0:
        valid_table.to_csv("valid_ratio.csv")

    # JSON metrics (log_pterm_by_len[9], test/Mean(log_pterm - log_pterm_ref), test/logP(s) (avg), test/logZ)
    json_metrics_table = make_json_metrics_table(exps_norm, error_mode=error_mode)
    out_tables["json_metrics"] = json_metrics_table
    if len(json_metrics_table) > 0:
        json_metrics_table.to_csv("json_metrics_table.csv")

    # Pterm by length (from json)
    pterm_by_len = make_pterm_by_length_table(exps_norm, error_mode=error_mode)
    out_tables["pterm_by_length"] = pterm_by_len
    if len(pterm_by_len) > 0:
        pterm_by_len.to_csv("pterm_by_length.csv")

    # Positional divergence
    div_map = collect_divergence(exps_norm, reference_data, cfg=samples_cfg)
    if div_map:
        # summary table: average over positions per run, then aggregate
        rows = []
        for exp_name, div_list in div_map.items():
            for run_idx, df in enumerate(div_list):
                r = {"experiment": exp_name, "run": run_idx}
                for metric in ["kl_sample_to_ref", "kl_ref_to_sample", "js", "l1"]:
                    r[f"{metric}_mean_over_pos"] = (
                        float(df[metric].mean()) if not df.empty else np.nan
                    )
                rows.append(r)
        if rows:
            div_df = pd.DataFrame(rows)
            div_summary = _aggregate_numeric(div_df, group_key="experiment", error_mode=error_mode)
            out_tables["divergence_summary"] = div_summary
            div_summary.to_csv("pos_divergence_summary.csv")

    # Optional: reuse draw_smiles plotting for prefix/length JSON
    try:
        from draw_smiles import Buckets, JsonKeys
        from draw_smiles import SamplesConfig as SmilesSamplesConfig
        from draw_smiles import (
            plot_fp_score_stacked,
            plot_length_histogram_binned,
            plot_length_histogram_fine,
            plot_metric_by_length_from_json,
            plot_prefix_triplet,
            plot_samples_metric_by_length,
        )
    except Exception:
        Buckets = JsonKeys = SmilesSamplesConfig = None
        print("[warn] draw_smiles not importable; skipping prefix/len plots.")

    save_dir = Path(save_fig_dir) if save_fig_dir is not None else None
    buckets = Buckets() if "Buckets" in locals() and Buckets is not None else None
    keys = JsonKeys() if "JsonKeys" in locals() and JsonKeys is not None else None
    smiles_cfg = (
        SmilesSamplesConfig()
        if "SmilesSamplesConfig" in locals() and SmilesSamplesConfig is not None
        else None
    )

    if buckets is not None and keys is not None:
        if plot_len_hist:
            plot_length_histogram_binned(
                exps_norm,
                buckets=buckets,
                style=style,
                valid_only=True,
                title="Length histogram",
                color_map=color_map,
                save_dir=save_dir,
                y_range=len_hist_bins_y,
                error_mode=error_mode,
            )
            plot_length_histogram_fine(
                exps_norm,
                style=style,
                normalize=True,
                valid_only=True,
                title="Length Distribution",
                color_map=color_map,
                save_dir=save_dir,
                y_range=len_hist_fine_y,
                error_mode=error_mode,
            )
        if plot_by_len_json:
            plot_metric_by_length_from_json(
                exps_norm,
                style=style,
                key=keys.score_mean_by_len_valid,
                title="Score by length",
                ylabel="score mean",
                color_map=color_map,
                save_dir=save_dir,
                y_range=(json_bylen_y or {}).get(keys.score_mean_by_len_valid),
                error_mode=error_mode,
            )
            plot_metric_by_length_from_json(
                exps_norm,
                style=style,
                key=keys.diversity_by_len_valid,
                title="Token diversity by length",
                ylabel="token diversity",
                color_map=color_map,
                save_dir=save_dir,
                y_range=(json_bylen_y or {}).get(keys.diversity_by_len_valid),
                error_mode=error_mode,
            )
        if plot_prefix and buckets is not None:
            plot_prefix_triplet(
                exps_norm,
                buckets=buckets,
                style=style,
                color_map=color_map,
                save_dir=save_dir,
                ylims=prefix_ylims,
                error_mode=error_mode,
            )
        if plot_by_len_samples and smiles_cfg is not None:
            plot_samples_metric_by_length(
                exps_norm,
                style=style,
                cfg=smiles_cfg,
                metric="unique_rate_str",
                title="Unique (string) rate by length",
                ylabel="unique / n",
                color_map=color_map,
                save_dir=save_dir,
                y_range=None,
                error_mode=error_mode,
            )
            plot_samples_metric_by_length(
                exps_norm,
                style=style,
                cfg=smiles_cfg,
                metric="n",
                title="Valid samples count by length",
                ylabel="count",
                color_map=color_map,
                save_dir=save_dir,
                y_range=None,
                error_mode=error_mode,
            )
        if plot_by_len_samples and plot_len_hist and smiles_cfg is not None:
            plot_fp_score_stacked(
                exps_norm,
                style=style,
                keys=keys,
                cfg=smiles_cfg,
                color_map=color_map,
                save_dir=save_dir,
                y_range_fp=None,
                y_range_score=None,
                error_mode=error_mode,
            )

    # Length-wise coverage & uniqueness table/plot
    len_cov_table = make_length_coverage_table(
        exps_norm, reference_data, cfg=samples_cfg, error_mode=error_mode
    )
    out_tables["coverage_by_length"] = len_cov_table
    if len(len_cov_table) > 0:
        len_cov_table.to_csv("length_coverage_by_length.csv")

    if plot_coverage and cov_table is not None:
        plot_coverage_bar(cov_table, style=style, save_dir=save_dir)
    if plot_coverage and len_cov_table is not None:
        plot_length_coverage_uniqueness(
            len_cov_table, style=style, color_map=color_map, save_dir=save_dir, pos_offset=3
        )

    if plot_pterm_by_len and pterm_by_len is not None:
        plot_pterm_by_length(pterm_by_len, style=style, color_map=color_map, save_dir=save_dir)

    if plot_divergence:
        plot_position_divergence(
            div_map,
            style=style,
            color_map=color_map,
            save_dir=save_dir,
            error_mode=error_mode,
            pos_offset=3,
        )

    if cwd_prev is not None:
        os.chdir(cwd_prev)

    return out_tables


# ============================================================
# Example usage
# ============================================================

# Main Papers

exps = {
    # baselines
    "TB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB/eval/runs/2026-01-17_10-12-52",
        "style": {"color": "C0", "linestyle": "--", "marker": "o"},
    },
    "SubTB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB/eval/runs/2026-01-16_07-51-36",
        "style": {"color": "C1", "linestyle": "-.", "marker": "s"},
    },
    "RapTB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB/eval/runs/2026-01-16_07-51-36",
        "style": {"color": "C2", "linestyle": "-", "marker": "^"},
    },
    "RapTB-SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_extend_v0/eval/runs/2026-01-17_05-45-41",
        "style": {"color": "C8", "linestyle": "-", "marker": "^"},
    },
    # "TB-SubM":
    #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_SubM/eval/runs/2026-01-24_06-01-46",
    #      "style": {"color": "C3", "linestyle": "--", "marker": "o"}
    #     },
    # "SubTB-SubM":
    #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB_SubM/eval/runs/2026-01-24_06-01-46",
    #      "style": {"color": "C4", "linestyle": "-.", "marker": "s"}
    #     },
    "TB-SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_SubM_ext_size/eval/runs/2026-01-25_05-27-46",
        "style": {"color": "C3", "linestyle": "--", "marker": "o"},
    },
    "SubTB-SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB_SubM_ext_size/eval/runs/2026-01-25_05-27-46",
        "style": {"color": "C4", "linestyle": "-.", "marker": "s"},
    },
    # oracle buffer
    "TB-Oracle": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_Oracle/eval/runs/2026-01-17_10-12-52",
        "style": {"color": "C6", "linestyle": "--", "marker": "o"},
    },
    "SubTB-Oracle": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB_Oracle/eval/runs/2026-01-16_07-51-36",
        "style": {"color": "C7", "linestyle": "-.", "marker": "s"},
    },
    "RapTB-Oracle": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v2/eval/runs/2026-01-16_06-14-49",
        "style": {"color": "C8", "linestyle": "-", "marker": "^"},
    },
    # PRT variants
    "TB-PRT": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_PRT/eval/runs/2026-01-24_05-28-24",
        "style": {"color": "C3", "linestyle": "--", "marker": "o"},
    },
    "SubTB-PRT": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB_PRT/eval/runs/2026-01-24_05-28-24",
        "style": {"color": "C4", "linestyle": "-.", "marker": "s"},
    },
    "RapTB-PRT": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_PRT/eval/runs/2026-01-24_05-28-24",
        "style": {"color": "C5", "linestyle": "-", "marker": "^"},
    },
    # SubTB variants
    "RootSubTBLogZ-Oracle": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RootSubTBLogZ_Oracle/eval/runs/2026-01-16_09-17-52",
        "style": {"color": "C9", "linestyle": "-", "marker": "^"},
    },
    "RootSubTBLogZ": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RootSubTBLogZ/eval/runs/2026-01-16_09-17-52",
        "style": {"color": "C9", "linestyle": "-", "marker": "^"},
    },
}


# exps = {

#     # Oracle buffer
#     "TB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense_oracle/eval/runs/2026-01-14_12-25-03",
#          "style": {"color": "C6", "linestyle": "--", "marker": "o"}
#     },
#     "RapTB_oracle_0":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/RapTB_oracel_v0/eval/runs/2026-01-16_05-43-20",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_1":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/RapTB_oracel_v1/eval/runs/2026-01-16_05-43-20",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_2":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/RapTB_oracel_v2/eval/runs/2026-01-16_05-43-20",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_3":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/RapTB_oracel_v3/eval/runs/2026-01-16_05-43-20",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_4":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/RapTB_oracel_v4/eval/runs/2026-01-16_05-43-20",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_5":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/RapTB_oracel_v5/eval/runs/2026-01-16_05-43-20",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
# }

# SubM hyperparameters
# exps = {

#     # Oracle buffer

#     "TB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C0", "linestyle": "--", "marker": "o"}
#         },

#     # "TB_SubM":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_SubM/eval/runs/2026-01-16_07-51-36",
#     #      "style": {"color": "C3", "linestyle": "--", "marker": "o"}
#     #     },

#     "RapTB":
#             {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB/eval/runs/2026-01-16_07-51-36",
#             "style": {"color": "C2", "linestyle": "-",  "marker": "^"}
#         },

#     "RapTB_SubM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM/eval/runs/2026-01-16_09-08-10",
#          "style": {"color": "C6", "linestyle": "--", "marker": "o"}
#     },

#     # "RapTB_SubM_wo_len_v0":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_v0/eval/runs/2026-01-17_05-45-41",
#     #      "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     #     },
#     # "RapTB_SubM_wo_len_v1":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_v1/eval/runs/2026-01-17_05-45-41",
#     #      "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     #     },
#     "RapTB_SubM_wo_len_v2":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_v2/eval/runs/2026-01-17_05-45-41",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#         },
#     # "RapTB_SubM_wo_len_v2_small_sample":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_v2_small_sample/eval/runs/2026-01-17_05-45-41",
#     #      "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     #     },
#     # "RapTB_SubM_wo_len_extend_v0":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_extend_v0/eval/runs/2026-01-17_05-45-41",
#     #      "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     #     },
#     # "RapTB_SubM_wo_len_extend_v1":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_extend_v1/eval/runs/2026-01-17_05-45-41",
#     #      "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     #     },
#     # "RapTB_SubM_wo_len_extend_v2":
#     #     {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM_wo_len_extend_v2/eval/runs/2026-01-17_05-45-41",
#     #      "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     #     },
# }

# fill this dict before running (same structure as draw_smiles)
# exps = {

#     # Baseline
#     "TB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C0", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C1", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C2", "linestyle": "-",  "marker": "^"}
#     },

#     # Submodular buffer
#     "TB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_SubM/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C3", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB_SubM/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C4", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_SubM/eval/runs/2026-01-16_09-08-10",
#          "style": {"color": "C5", "linestyle": "-",  "marker": "^"}
#     },

#     # Oracle buffer
#     "TB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_TB_Oracle/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C6", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_SubTB_Oracle/eval/runs/2026-01-16_07-51-36",
#          "style": {"color": "C7", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RapTB_Oracle/eval/runs/2026-01-16_09-08-10",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },

#     # RapTB_oracel_hyperparams

#     "RapTB_oracle_0":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v0/eval/runs/2026-01-16_06-14-49",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_1":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v1/eval/runs/2026-01-16_06-14-49",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_2":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v2/eval/runs/2026-01-16_06-14-49",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_3":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v3/eval/runs/2026-01-16_06-14-49",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_4":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v4/eval/runs/2026-01-16_06-14-49",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
#     "RapTB_oracle_5":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/Expr24_RapTB_Oracle_v5/eval/runs/2026-01-16_06-14-49",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },

#     # SubTB variants
#     "RootSubTBLogZ_Oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RootSubTBLogZ_Oracle/eval/runs/2026-01-16_09-17-52",
#          "style": {"color": "C9", "linestyle": "-",  "marker": "^"}
#     },
#     "RootSubTBLogZ":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_RootSubTBLogZ/eval/runs/2026-01-16_09-17-52",
#          "style": {"color": "C9", "linestyle": "-",  "marker": "^"}
#     },
# }

# 200K samples
# exps = {

#     # Baseline
#     "TB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense/eval/runs/2026-01-15_04-54-26",
#          "style": {"color": "C0", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense/eval/runs/2026-01-15_04-54-28",
#          "style": {"color": "C1", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_v0/eval/runs/2026-01-15_04-54-26",
#          "style": {"color": "C2", "linestyle": "-",  "marker": "^"}
#     },


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_root",
        "--output-name",
        "-o",
        dest="output_root",
        type=str,
        default="expr24_outputs",
        help="Directory to write CSVs/LaTeX outputs (default: expr24_outputs).",
    )
    ap.add_argument(
        "--reference_path",
        "--reference-path",
        "-r",
        dest="reference_path",
        type=str,
        default="/data1/xw3763/project/gflow/ChemGFN/data/24_points/buffer_24_len1to9_non_zero.pt",
        help="Path to reference buffer (.pt).",
    )
    ap.add_argument(
        "--check_samples_only",
        "--check-samples-only",
        action="store_true",
        help="Only check sample counts across methods/repeats (write sample_counts.csv) and exit.",
    )
    ap.add_argument(
        "--save_fig_dir",
        type=str,
        default="figures_expr24",
        help="Subdirectory for figures (under output_root).",
    )
    ap.add_argument(
        "--error_mode",
        type=str,
        default="ci95",
        choices=["none", "std", "sem", "ci95"],
        help="Error bar mode for plots/tables.",
    )
    ap.add_argument("--no_plots", action="store_true", help="Skip all plots.")
    ap.add_argument("--no_tables", action="store_true", help="Skip LaTeX table generation.")
    ap.add_argument(
        "--lengths",
        type=str,
        default="3,5,7,9",
        help="Comma-separated lengths for per-length tables.",
    )
    ap.add_argument(
        "--tb_logz_exclude_regex",
        type=str,
        default="",
        help="Regex to exclude methods in per-length NormCov table.",
    )
    args = ap.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    save_fig_dir = Path(args.save_fig_dir) if not args.no_plots else None
    length_list = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    if args.check_samples_only:
        exps_norm = normalize_exps(exps)
        df = check_sample_counts(exps_norm, verbose=True)
        out_dir = output_root or Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(df) > 0:
            df.to_csv(out_dir / "sample_counts.csv", index=False)
        return

    run_analysis(
        exps,
        reference_path=args.reference_path,
        style=PlotStyle(),
        error_mode=args.error_mode,
        plot_prefix=not args.no_plots,
        plot_len_hist=not args.no_plots,
        plot_by_len_json=not args.no_plots,
        plot_by_len_samples=False,
        plot_divergence=not args.no_plots,
        plot_coverage=not args.no_plots,
        plot_pterm_by_len=not args.no_plots,
        save_fig_dir=save_fig_dir,
        output_root=output_root,
    )

    if args.no_tables:
        return

    if output_root is None:
        output_root = Path(".")

    try:
        from gen_expr24_table import (
            build_appendix_len_table,
            build_main_table,
            build_main_table_ci,
            build_pterm_by_len_table,
            build_pterm_diagnostic_table,
        )
    except Exception as e:
        print(f"[warn] failed to import gen_expr24_table helpers: {e}")
        return

    coverage_csv = output_root / "coverage_table.csv"
    pos_div_csv = output_root / "pos_divergence_summary.csv"
    valid_ratio_csv = output_root / "valid_ratio.csv"
    length_by_len_csv = output_root / "length_coverage_by_length.csv"
    json_metrics_csv = output_root / "json_metrics_table.csv"

    if (
        coverage_csv.exists()
        and pos_div_csv.exists()
        and valid_ratio_csv.exists()
        and length_by_len_csv.exists()
    ):
        _, main_tex = build_main_table(
            coverage_csv=str(coverage_csv),
            pos_div_csv=str(pos_div_csv),
            valid_ratio_csv=str(valid_ratio_csv),
        )
        (output_root / "expr24_main.tex").write_text(main_tex, encoding="utf-8")

        _, main_ci_tex = build_main_table_ci(
            coverage_csv=str(coverage_csv),
            pos_div_csv=str(pos_div_csv),
            valid_ratio_csv=str(valid_ratio_csv),
        )
        (output_root / "expr24_main_ci.tex").write_text(main_ci_tex, encoding="utf-8")

        # Single per-length NormCov table (method-dependent N_ell)
        _, normcov_by_len_tex = build_appendix_len_table(
            length_by_len_csv=str(length_by_len_csv),
            exclude_regex="",
            lengths=length_list,
        )
        (output_root / "expr24_normcov_by_len_all.tex").write_text(
            normcov_by_len_tex, encoding="utf-8"
        )
    else:
        print("[warn] missing core CSVs; skip main/appendix LaTeX.")

    if coverage_csv.exists() and valid_ratio_csv.exists() and json_metrics_csv.exists():
        _, pterm_tex = build_pterm_diagnostic_table(
            coverage_csv=str(coverage_csv),
            valid_ratio_csv=str(valid_ratio_csv),
            json_metrics_csv=str(json_metrics_csv),
        )
        (output_root / "expr24_pterm_diag.tex").write_text(pterm_tex, encoding="utf-8")
    else:
        print("[warn] missing CSVs for pterm diagnostics; skip pterm LaTeX.")

    # (Removed) legacy duplicate per-length NormCov table.

    pterm_by_len_csv = output_root / "pterm_by_length.csv"
    if pterm_by_len_csv.exists():
        _, pterm_by_len_tex = build_pterm_by_len_table(
            pterm_by_len_csv=str(pterm_by_len_csv),
            lengths=length_list,
        )
        (output_root / "expr24_pterm_by_len.tex").write_text(pterm_by_len_tex, encoding="utf-8")
    else:
        print("[warn] missing pterm_by_length.csv; skip per-length pterm table.")


if __name__ == "__main__":
    main()

#     # Submodular buffer
#     "TB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_no_data_buffer_hit24_dense_subM_div_on_valid/eval/runs/2026-01-15_04-54-24",
#          "style": {"color": "C3", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_subM_div_on_valid/eval/runs/2026-01-15_04-54-25",
#          "style": {"color": "C4", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_subM_div_on_valid_tune_v0/eval/runs/2026-01-15_04-54-25",
#          "style": {"color": "C5", "linestyle": "-",  "marker": "^"}
#     },

#     # Oracle buffer
#     "TB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense_oracle/eval/runs/2026-01-15_04-54-26",
#          "style": {"color": "C6", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_oracle/eval/runs/2026-01-15_04-54-26",
#          "style": {"color": "C7", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_oracle/eval/runs/2026-01-15_06-22-55",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
# }

# 50

# exps = {

#     # Baseline
#     "TB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C0", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C1", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_v0/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C2", "linestyle": "-",  "marker": "^"}
#     },

#     # Submodular buffer
#     "TB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_no_data_buffer_hit24_dense_subM_div_on_valid/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C3", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_subM_div_on_valid/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C4", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_subM_div_on_valid_tune_v0/eval/runs/2026-01-15_04-54-25",
#          "style": {"color": "C5", "linestyle": "-",  "marker": "^"}
#     },

#     # Oracle buffer
#     "TB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense_oracle/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C6", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_oracle/eval/runs/2026-01-15_07-09-20",
#          "style": {"color": "C7", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_oracle/eval/runs/2026-01-15_07-47-31",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
# }

# Extensive sampling results

# exps = {

#     # Baseline
#     "TB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense/eval/runs/2026-01-15_02-03-45",
#          "style": {"color": "C0", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense/eval/runs/2026-01-15_02-03-44",
#          "style": {"color": "C1", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_v0/eval/runs/2026-01-15_02-03-44",
#          "style": {"color": "C2", "linestyle": "-",  "marker": "^"}
#     },

#     # Submodular buffer
#     "TB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_no_data_buffer_hit24_dense_subM_div_on_valid/eval/runs/2026-01-15_02-03-45",
#          "style": {"color": "C3", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_subM_div_on_valid/eval/runs/2026-01-15_02-03-45",
#          "style": {"color": "C4", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_subM":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_subM_div_on_valid_tune_v0/eval/runs/2026-01-15_02-03-45",
#          "style": {"color": "C5", "linestyle": "-",  "marker": "^"}
#     },

#     # Oracle buffer
#     "TB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_TB_hit24_dense_oracle/eval/runs/2026-01-15_02-03-45",
#          "style": {"color": "C6", "linestyle": "--", "marker": "o"}
#     },
#     "SubTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_oracle/eval/runs/2026-01-15_02-03-45",
#          "style": {"color": "C7", "linestyle": "-.", "marker": "s"}
#     },
#     "RapTB_oracle":
#         {"repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_oracle/eval/runs/2026-01-15_05-39-32",
#          "style": {"color": "C8", "linestyle": "-",  "marker": "^"}
#     },
# }


def check_sample_counts_standalone(
    exps_raw: dict[str, dict[str, Any]],
    save_csv: str | None = None,
) -> pd.DataFrame:
    """Standalone helper (library use)."""
    exps_norm = normalize_exps(exps_raw)
    result_df = check_sample_counts(exps_norm, verbose=True)
    if save_csv and len(result_df) > 0:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_csv, index=False)
        print(f"Sample counts saved to: {save_csv}")
    return result_df
