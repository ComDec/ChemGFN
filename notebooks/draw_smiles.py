from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

from dataclasses import dataclass, field

try:
    from IPython.display import display
except Exception:
    display = print

# =========================
# Global configs (edit here)
# =========================
DEFAULT_SHADE_REGIONS_L10 = [(1, 3), (4, 7), (8, 10)]
DEFAULT_PREFIX_BINS_L10 = {"short": (1, 3), "mid": (4, 7), "long": (8, 10)}
DEFAULT_LEN_BIN_LABELS_L10 = ["0-2", "3-5", "6-8", "9-10"]

DEFAULT_SHADE_REGIONS_L15 = [(1, 3), (4, 7), (8, 10), (11, 15)]
DEFAULT_PREFIX_BINS_L15 = {"short": (1, 3), "mid": (4, 7), "long": (8, 10), "longer": (11, 15)}
DEFAULT_LEN_BIN_LABELS_L15 = ["0-2", "3-5", "6-8", "9-10", "11-15"]

# Backwards-compatible defaults (L10)
DEFAULT_SHADE_REGIONS = DEFAULT_SHADE_REGIONS_L10
DEFAULT_PREFIX_BINS = DEFAULT_PREFIX_BINS_L10
DEFAULT_LEN_BIN_LABELS = DEFAULT_LEN_BIN_LABELS_L10


EXPS_L15 = {
    "TB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_TB_len_15/eval/runs/2026-01-13_09-28-54",
        "style": {"marker": "o", "linestyle": "--", "color": "C0"},
    },
    "SubTB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_subTB_len_15/eval/runs/2026-01-13_10-49-10",
        "style": {"marker": "o", "linestyle": "-.", "color": "C1"},
    },
    "RapTB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_RapTB_v2_kmin_12_to_8_mix_fix_len15/eval/runs/2026-01-13_10-49-10",
        "style": {"marker": "o", "linestyle": "-", "color": "C2"},
    },
    "RapTB_SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_RapTB_v2_kmin_12_to_8_mix_fix_len15_subM/eval/runs/2026-01-14_09-01-29",
        "style": {"marker": "o", "linestyle": "-", "color": "C3"},
    },
}

EXPS_L10 = {
    "TB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_TB/eval/runs/2026-01-12_12-26-14",
        "style": {"marker": "o", "linestyle": "--", "color": "C0"},
    },
    "TB-wo-ref": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_TB_wo_ref/eval/runs/2026-01-26_12-11-01",
        "style": {"marker": "o", "linestyle": "--", "color": "C0"},
    },
    "SubTB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_subTB/eval/runs/2026-01-12_12-26-14",
        "style": {"marker": "o", "linestyle": "-.", "color": "C1"},
    },
    "RapTB": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_RapTB_v2_kmin_5_to_2_mix_fix_softmax_overflow/eval/runs/2026-01-27_05-11-32",
        "style": {"marker": "o", "linestyle": "-", "color": "C2"},
    },
    "RapTB-SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_RapTB_v2_kmin_5_to_2_mix_fix_softmax_overflow_subM/eval/runs/2026-01-12_12-26-14",
        "style": {"marker": "o", "linestyle": "-", "color": "C3"},
    },
    "TB-SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_TB_subM_replay_add_len_func/eval/runs/2026-01-24_07-00-09",
        "style": {"marker": "o", "linestyle": "-", "color": "C4"},
    },
    "SubTB-SubM": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_CFG_SubTB_subM_full/eval/runs/2026-01-24_07-00-09",
        "style": {"marker": "o", "linestyle": "-", "color": "C5"},
    },
    "RapTB-MaxOnly": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_RapTB_v2_kmin_5_to_2_max_only_v2/eval/runs/2026-01-26_23-04-55",
        "style": {"marker": "o", "linestyle": "-", "color": "C6"},
    },
    "RapTB-SoftOnly": {
        "repeat_root": "/data1/xw3763/project/gflow/ChemGFN/logs/eval/smiles_RapTB_v2_kmin_5_to_2_soft_only_v2/eval/runs/2026-01-26_23-04-55",
        "style": {"marker": "o", "linestyle": "-", "color": "C7"},
    },
}

EXPS_PRESETS = {
    "L10": EXPS_L10,
    "L15": EXPS_L15,
}

SHADE_REGIONS_PRESETS = {
    "L10": DEFAULT_SHADE_REGIONS_L10,
    "L15": DEFAULT_SHADE_REGIONS_L15,
}


def get_exps(preset: str) -> dict[str, dict[str, Any]]:
    if preset not in EXPS_PRESETS:
        raise KeyError(f"Unknown preset: {preset}")
    return EXPS_PRESETS[preset]


def get_buckets(preset: str) -> Buckets:
    if preset == "L15":
        return Buckets(
            prefix_bins=DEFAULT_PREFIX_BINS_L15, len_bin_labels=DEFAULT_LEN_BIN_LABELS_L15
        )
    return Buckets(prefix_bins=DEFAULT_PREFIX_BINS_L10, len_bin_labels=DEFAULT_LEN_BIN_LABELS_L10)


exps = EXPS_L10


@dataclass
class Buckets:
    # prefix-length buckets (k)
    prefix_bins: dict[str, tuple[int, int]] | None = None
    # length histogram buckets in JSON keys
    len_bin_labels: list[str] | None = None

    def __post_init__(self):
        if self.prefix_bins is None:
            self.prefix_bins = DEFAULT_PREFIX_BINS.copy()
        if self.len_bin_labels is None:
            self.len_bin_labels = list(DEFAULT_LEN_BIN_LABELS)


def plot_prefix_triplet(
    exps: dict[str, dict[str, str]],
    buckets: Buckets,
    style: PlotStyle,
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    metrics: list[tuple[str, str, str]] | None = None,
    ylims: dict[str, tuple[float, float]] | None = None,
    error_mode: str = "sem",
    name_prefix: str = "",
) -> None:
    """Plot three prefix metrics in a single horizontal row of square subplots."""
    pref_runs: dict[str, list[pd.DataFrame]] = {}
    for exp_name, payload in exps.items():
        dfs = []
        for ppath in payload.get("prefix_paths", []):
            if not ppath:
                continue
            if not Path(ppath).exists():
                print(f"[warn] prefix csv not found, skip: {ppath}")
                continue
            dfs.append(load_prefix_csv(ppath))
        if dfs:
            pref_runs[exp_name] = dfs
    if not pref_runs:
        print("No prefix csv provided; skipping prefix triplet plots.")
        return

    if metrics is None:
        metrics = [
            ("survival", "Prefix survival", "n(k) / n(1)"),
            ("entropy", "Prefix entropy", "entropy"),
            ("top1", "Top-1 mass (collapse ↑)", "top1"),
        ]

    fig, axes = plt.subplots(1, 3, figsize=style.figsize_prefix_triplet, dpi=style.dpi)
    cmap = color_map or {}

    for ax, (ycol, title, ylabel) in zip(axes, metrics):
        for name, dfs in pref_runs.items():
            all_k = sorted(set().union(*[set(df["k"].tolist()) for df in dfs]))
            if not all_k:
                continue
            arr = np.full((len(dfs), len(all_k)), np.nan, dtype=float)
            k_to_idx = {v: i for i, v in enumerate(all_k)}
            for r, df in enumerate(dfs):
                if ycol not in df.columns:
                    continue
                for k, v in zip(df["k"], df[ycol]):
                    arr[r, k_to_idx[int(k)]] = float(v)
            mean = np.nanmean(arr, axis=0)
            err = _compute_err(arr, mode=error_mode)
            color, marker_override, linestyle = resolve_method_style(name, style, cmap)
            line_kw = style.line.kwargs(
                color=color, marker_override=marker_override, linestyle=linestyle
            )
            ax.errorbar(all_k, mean, yerr=err, label=name, **line_kw)
        add_shading(
            ax, style.shade_regions or list(buckets.prefix_bins.values()), style.shade_alpha
        )
        apply_full_border(ax, style)
        ax.set_title(title, fontsize=style.title_fontsize)
        ax.set_xlabel("k (prefix length)", fontsize=style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
        if ylims and ycol in ylims:
            ax.set_ylim(*ylims[ycol])
        ax.grid(True, alpha=style.grid_alpha)
        ax.set_box_aspect(1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(4, min(style.legend_ncol, len(labels))),
            frameon=style.legend_frameon,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=0.2)
    if save_dir is not None:
        prefix = f"{name_prefix}_" if name_prefix else ""
        _save_fig(fig, save_dir, f"{prefix}prefix_triplet")
    plt.show()


# ============================================================
# ICML-style analysis toolkit (JSON + prefix CSV + samples CSV)
# Directly compatible with your file formats:
#   - samples CSV: columns like "Sampled sentence", "token_ids", "is_valid"
#   - prefix CSV: columns like epoch,k,top1,top5,entropy,eff,n,unique
#   - eval JSON: keys like score_mean_by_len_valid, diversity_by_len_valid, score_count_by_len_valid
# ============================================================


# =========================
# Style / Config (edit here)
# =========================
@dataclass
class LineParams:
    marker: str = "o"
    markersize: float = 4.2
    linewidth: float = 3.8
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

    # size helpers (height = width / aspect_ratio)
    aspect_ratio: float = 2.0
    width_prefix: float = 12.0
    width_standard: float = 8.0
    width_nk: float = 7.0

    line: LineParams = field(default_factory=LineParams)
    bar: BarParams = field(default_factory=BarParams)

    # global shading for all line plots
    shade_regions: list[tuple[float, float]] | None = None
    shade_alpha: float = 0.08

    # method-specific overrides
    method_styles: dict[str, MethodStyle] = field(default_factory=dict)

    # frame and sizing
    full_frame: bool = True
    prefix_triplet_side: float = 3.0
    stacked_bylen_width: float = 7.0

    grid_alpha: float = 0.25

    legend_ncol: int = 3
    legend_frameon: bool = True

    suptitle_fontsize: int = 14
    title_fontsize: int = 15
    label_fontsize: int = 14

    sns_style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] | dict[
        str, Any
    ] = "whitegrid"
    sns_context: Literal["paper", "notebook", "talk", "poster"] | dict[str, Any] = "paper"
    palette: str = "colorblind"
    font_scale: float = 1.0

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


def apply_plot_style(style: PlotStyle) -> None:
    """Apply a consistent seaborn/matplotlib style globally."""
    if sns is not None:
        sns.set_theme(
            context=style.sns_context,
            style=style.sns_style,
            palette=style.palette,
            font_scale=style.font_scale,
        )
    plt.rcParams.update(
        {
            "axes.spines.top": style.full_frame,
            "axes.spines.right": style.full_frame,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "legend.frameon": style.legend_frameon,
            "grid.alpha": style.grid_alpha,
        }
    )


def apply_full_border(ax, style: PlotStyle) -> None:
    if style.full_frame:
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


def build_color_map(exps: dict[str, dict[str, str]], palette: str) -> dict[str, tuple]:
    names = list(exps.keys())
    if sns is not None:
        colors = sns.color_palette(palette, n_colors=max(len(names), 3))
        return {name: colors[i % len(colors)] for i, name in enumerate(names)}

    # Fallback: use matplotlib categorical palette.
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(max(len(names), 3))]
    return {name: colors[i % len(colors)] for i, name in enumerate(names)}


def _make_slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_.]+", "_", name).strip("_")
    return s or "figure"


def _save_fig(fig, save_dir: Path | None, name: str) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    slug = _make_slug(name)
    path = save_dir / f"{slug}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")


@dataclass
class JsonKeys:
    # by-length keys in your eval JSON
    score_mean_by_len: str = "score_mean_by_len"  # overall score/QED by length
    score_mean_by_len_valid: str = (
        "score_mean_by_len_valid"  # (SMILES: QED-by-length) (Expr24: score-by-length)
    )
    diversity_by_len: str = "diversity_by_len"  # token diversity by length (all samples)
    diversity_by_len_valid: str = (
        "diversity_by_len_valid"  # token-diversity-by-length (valid-only)
    )
    score_count_by_len_valid: str = (
        "score_count_by_len_valid"  # valid-only count-by-length (fine length histogram)
    )
    score_count_by_len: str = (
        "score_count_by_len"  # overall count-by-length (fallback if len_counts missing)
    )

    # termination statistics by length
    log_pterm_by_len: str = "log_pterm_by_len"

    # termination probability by length (if present)
    pterm_by_len: str = "pterm_by_len"

    # binned fractions (these are typical in your SMILES JSONs; may be absent in Expr24)
    len_valid_frac_prefix: str = "len_valid_frac"  # expects keys like len_valid_frac[0-2]


@dataclass
class SamplesConfig:
    # Your samples CSV (direct compatibility)
    text_cols: tuple[str, ...] = (
        "Sampled sentence",  # your current file
        "smiles",
        "SMILES",
        "sample",
        "text",
        "generated",
        "gen_smiles",
        "sentence",
        "output",
        "completion",
    )
    token_ids_cols: tuple[str, ...] = ("token_ids", "tokens", "token_id_list")
    length_cols: tuple[str, ...] = (
        "len_tok",
        "len",
        "length",
        "tok_len",
        "n_tokens",
        "num_tokens",
        "token_len",
    )
    valid_cols: tuple[str, ...] = (
        "is_valid",
        "correct",
        "valid",
        "rdkit_valid",
        "can_parse",
        "success",
    )

    # cleaning: strip everything after "<|" (handles <|end_of_text|> spam)
    strip_after_special_token: bool = True

    # FPdiv-by-length settings (if RDKit available and strings look like SMILES)
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    max_per_len: int = 512  # cap per length to control O(n^2) pairwise

    # if your CSV uses different names, override here:
    text_col_override: str | None = None
    token_ids_col_override: str | None = None
    length_col_override: str | None = None
    valid_col_override: str | None = None


# =========================
# Experiment normalization / aggregation helpers
# =========================
def _ensure_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _append_if_exists(lst: list[str], path: Path) -> None:
    if path.exists():
        lst.append(str(path))


def _discover_repeat_runs(repeat_root: str) -> tuple[list[str], list[str], list[str]]:
    """
    Given a root that contains repeat_x subdirectories, collect json/prefix/samples paths.
    Each repeat_x is expected to contain:
      - prefix_tables/prefix_pos_test_k_correct_0.csv
      - test_samples/samples_test_0.csv
      - a json file (prefers test_metrics*.json, fallback to first *.json)
    """
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
    for rdir in repeat_dirs:
        # prefix tables: handle both prefix_tables and prefix_tables_runX
        prefix_candidates = itertools.chain(
            rdir.glob("prefix_tables*/prefix_pos_test_k_correct_0.csv"),
            rdir.glob("prefix_tables*/prefix_pos_test_k_correct_0*.csv"),
        )
        for pc in prefix_candidates:
            _append_if_exists(prefix_paths, pc)

        # samples: handle test_samples and test_samples_runX
        sample_candidates = itertools.chain(
            rdir.glob("test_samples*/samples_test_0*.csv"),
            rdir.glob("test_samples*/samples_test*.csv"),
        )
        for sc in sample_candidates:
            _append_if_exists(samples_paths, sc)

        # json: prefer json/test_metrics*.json, fallback any *.json inside repeat dir
        json_candidates = list(rdir.glob("json/test_metrics*.json"))
        if not json_candidates:
            json_candidates = list(rdir.glob("json/*.json"))
        if not json_candidates:
            json_candidates = list(rdir.glob("test_metrics*.json"))
        if not json_candidates:
            json_candidates = list(rdir.glob("*.json"))
        if json_candidates:
            json_paths.append(str(sorted(json_candidates)[0]))

    return json_paths, prefix_paths, samples_paths


def normalize_exps(exps_raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Normalize experiment payload:
      - Ensure json/prefix/samples become lists (json_paths/prefix_paths/samples_paths).
      - If roots provided, auto-derive prefix/samples paths:
            root/prefix_tables/prefix_pos_test_k_correct_0.csv
            root/test_samples/samples_test_0.csv
      - If repeat_root(s) provided, scan repeat_* subdirs inside each and collect json/prefix/samples.
      - Keep user-specified prefix/samples and extend with derived ones.
      - Preserve per-experiment style hint (marker/linestyle/color).
    """
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
    """
    Copy per-method style hints from exps into PlotStyle.method_styles.
    If not provided, assign default alternating linestyles/markers for readability.
    """
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


def _compute_err(values: np.ndarray, mode: str = "sem") -> np.ndarray:
    """
    mode in {"std","sem","none"}; nan-aware.
    If mode == "ci95", returns 1.96 * SEM (approx 95% CI for normal assumption).
    """
    if mode == "none":
        return np.zeros_like(values, dtype=float)
    if mode not in {"std", "sem", "ci95"}:
        raise ValueError(f"Unknown error mode: {mode}")
    std = np.nanstd(values, axis=0, ddof=1 if values.shape[0] > 1 else 0)
    if mode == "std":
        return std
    # sem or ci95
    n = np.maximum(np.sum(~np.isnan(values), axis=0), 1)
    sem = std / np.sqrt(n)
    if mode == "ci95":
        return sem * 1.96
    return sem


# =========================
# IO helpers
# =========================
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


def _infer_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def _clean_text(s: str, strip_after_special_token: bool = True) -> str:
    s = "" if s is None else str(s)
    if strip_after_special_token and "<|" in s:
        s = s.split("<|", 1)[0]
    return s.strip()


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


# =========================
# JSON extraction & tables
# =========================
def extract_by_len_series(j: dict[str, Any], key: str) -> pd.Series | None:
    if key not in j or j[key] is None or not isinstance(j[key], dict):
        return None
    items = []
    for k, v in j[key].items():
        try:
            items.append((int(k), float(v)))
        except Exception:
            continue
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    return pd.Series({k: v for k, v in items}).sort_index()


def extract_main_metrics(j: dict[str, Any], buckets: Buckets) -> dict[str, Any]:
    out = {}

    # robust: your JSON sometimes uses nested names; keep core ones here
    out["acc"] = j.get("acc", j.get("test/validator/acc"))
    out["diversity_valid"] = j.get("diversity_valid", j.get("test/diversity_valid"))
    out["diversity"] = j.get("test/diversity", j.get("diversity"))
    out["score_mean_valid"] = j.get("test/score_mean_valid", j.get("score_mean_valid", None))
    out["score_mean"] = j.get("test/score_mean", j.get("score_mean", None))

    out["len_mean"] = j.get("len_mean", None)
    out["len_std"] = j.get("len_std", None)
    out["len_p50"] = j.get("len_p50", None)
    out["len_p90"] = j.get("len_p90", None)
    out["len_p95"] = j.get("len_p95", None)

    # binned fractions if present (SMILES json usually has these; Expr24 may not)
    for b in buckets.len_bin_labels:
        out[f"len_valid_frac[{b}]"] = j.get(f"len_valid_frac[{b}]", None)
        out[f"len_frac[{b}]"] = j.get(f"len_frac[{b}]", None)

    return out


from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _get_first(j: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        if k in j and j[k] is not None:
            return j[k]
    return default


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


def make_main_table(
    exps: dict[str, dict[str, Any]],
    buckets: Buckets,
    error_mode: str = "ci95",
    samples_cfg: SamplesConfig | None = None,
) -> pd.DataFrame:
    rows = []

    for exp_name, payload in exps.items():
        json_paths = payload.get("json_paths", [])
        samples_paths = payload.get("samples_paths", [])

        # Determine the number of runs based on the longer list
        n_runs = max(len(json_paths), len(samples_paths))
        if n_runs == 0:
            continue

        for run_idx in range(n_runs):
            row: dict[str, Any] = {"experiment": exp_name, "run": run_idx}

            # Initialize all metrics with nan
            row["acc"] = np.nan
            row["diversity_valid"] = np.nan
            row["diversity"] = np.nan
            row["score_mean_valid"] = np.nan
            row["score_mean"] = np.nan
            row["len_mean"] = np.nan
            row["len_std"] = np.nan
            row["len_p50"] = np.nan
            row["len_p90"] = np.nan
            row["len_p95"] = np.nan
            row["fp_div"] = np.nan
            for b in buckets.len_bin_labels:
                row[f"len_valid_frac[{b}]"] = np.nan
                row[f"len_frac[{b}]"] = np.nan

            # ---- Extract from JSON if available ----
            jpath = json_paths[run_idx] if run_idx < len(json_paths) else None
            if jpath and Path(jpath).exists():
                j = load_json(jpath)

                # ---- core metrics ----
                row["acc"] = float(_get_first(j, ["test/validator/acc", "acc"], default=np.nan))

                # diversity: your json often contains both test/* and plain keys
                row["diversity_valid"] = float(
                    _get_first(j, ["test/diversity_valid", "diversity_valid"], default=np.nan)
                )
                row["diversity"] = float(
                    _get_first(j, ["test/diversity", "diversity"], default=np.nan)
                )

                # score/QED: map to your validator keys (qed_filter is usually valid-only)
                row["score_mean_valid"] = float(
                    _get_first(
                        j,
                        [
                            "test/validator/qed_filter",
                            "test/qed_filter",
                            "test/validator/qed",
                            "test/qed",
                        ],
                        default=np.nan,
                    )
                )
                row["score_mean"] = float(
                    _get_first(j, ["test/validator/qed", "test/qed"], default=np.nan)
                )

                # ---- length summary (valid-only) ----
                row["len_mean"] = float(
                    _get_first(
                        j,
                        ["test/validator/len_tok_valid_mean", "test/validator/len_tok_mean"],
                        default=np.nan,
                    )
                )
                row["len_std"] = float(
                    _get_first(
                        j,
                        ["test/validator/len_tok_valid_std", "test/validator/len_tok_std"],
                        default=np.nan,
                    )
                )
                row["len_p50"] = float(
                    _get_first(
                        j,
                        ["test/validator/len_tok_valid_p50", "test/validator/len_tok_p50"],
                        default=np.nan,
                    )
                )
                row["len_p90"] = float(
                    _get_first(
                        j,
                        ["test/validator/len_tok_valid_p90", "test/validator/len_tok_p90"],
                        default=np.nan,
                    )
                )
                row["len_p95"] = float(
                    _get_first(
                        j,
                        ["test/validator/len_tok_valid_p95", "test/validator/len_tok_p95"],
                        default=np.nan,
                    )
                )

                # ---- length bin fractions (compute robustly via extract_len_bin_fracs you fixed) ----
                # valid-only binned
                v_fracs = extract_len_bin_fracs(j, buckets.len_bin_labels, valid_only=True)
                a_fracs = extract_len_bin_fracs(j, buckets.len_bin_labels, valid_only=False)

                if v_fracs is not None:
                    for b, val in zip(buckets.len_bin_labels, v_fracs):
                        row[f"len_valid_frac[{b}]"] = float(val) if val is not None else np.nan

                if a_fracs is not None:
                    for b, val in zip(buckets.len_bin_labels, a_fracs):
                        row[f"len_frac[{b}]"] = float(val) if val is not None else np.nan
            elif jpath:
                print(f"[warn] json not found, skip: {jpath}")

            # ---- Compute FP Diversity from samples if available ----
            spath = samples_paths[run_idx] if run_idx < len(samples_paths) else None
            if spath and Path(spath).exists() and samples_cfg is not None:
                try:
                    sdf = load_samples_csv(spath)
                    byL = compute_samples_by_length(sdf, samples_cfg)
                    # Get overall weighted FP diversity
                    fp_div_all = byL.attrs.get("fp_div_mean_all", np.nan)
                    if fp_div_all is None or (
                        isinstance(fp_div_all, float) and np.isnan(fp_div_all)
                    ):
                        # Fallback: compute from the dataframe if attrs not set
                        if "fp_div" in byL.columns and "n" in byL.columns:
                            fp_vals = byL["fp_div"].astype(float)
                            weights = byL["n"].astype(float)
                            mask = fp_vals.notnull() & weights.notnull()
                            if mask.any():
                                w = weights[mask]
                                f = fp_vals[mask]
                                denom = float(w.sum())
                                if denom > 0:
                                    fp_div_all = float(np.average(f, weights=w))
                    row["fp_div"] = float(fp_div_all) if fp_div_all is not None else np.nan
                except Exception as e:
                    print(f"[warn] failed to compute FP diversity from {spath}: {e}")
            elif spath and samples_cfg is not None:
                print(f"[warn] samples csv not found, skip: {spath}")

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = _aggregate_numeric(df, group_key="experiment", error_mode=error_mode)
    return agg


# =========================
# Plots: JSON length histograms (FIXED for your JSON structure)
# =========================
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_len_bin_label(label: str) -> tuple[int, int]:
    """
    Supports:
      "0-2"  -> (0,2)
      "3-5"  -> (3,5)
      "11+"  -> (11,-1)  # -1 means open-ended
      "11- -1" (rare) -> (11,-1)
    """
    s = label.strip()
    if s.endswith("+"):
        lo = int(s[:-1])
        return lo, -1
    m = re.match(r"^\s*(\d+)\s*-\s*(-?\d+)\s*$", s)
    if not m:
        raise ValueError(f"Unrecognized bin label: {label}")
    lo = int(m.group(1))
    hi = int(m.group(2))
    return lo, hi


def _get_counts_dict(j: dict[str, Any], valid_only: bool) -> dict[int, float] | None:
    """
    Prefer len_counts(_valid). Fallback to score_count_by_len(_valid).
    Returns dict[int] -> count.
    """
    candidates = []
    if valid_only:
        candidates = ["len_counts_valid", "score_count_by_len_valid"]
    else:
        candidates = ["len_counts", "score_count_by_len"]

    for k in candidates:
        v = j.get(k, None)
        if isinstance(v, dict) and len(v) > 0:
            out = {}
            for kk, vv in v.items():
                try:
                    out[int(kk)] = float(vv)
                except Exception:
                    pass
            if out:
                return out
    return None


def extract_len_bin_fracs(
    j: dict[str, Any],
    bins: list[str],
    valid_only: bool = True,
) -> list[float] | None:
    """
    Returns a list of bin fractions aligned with `bins`.

    Priority:
      1) Compute from len_counts(_valid) (most robust).
      2) Fallback to reading test/validator/len_tok(_valid)_{lo}_{hi}_frac.
      3) Legacy fallback: len_valid_frac[...] / len_frac[...] (if you ever have them).
    """
    # (1) Compute from counts dict
    counts = _get_counts_dict(j, valid_only=valid_only)
    if counts is not None:
        total = float(sum(counts.values()))
        if total <= 0:
            return None

        out = []
        for b in bins:
            lo, hi = _parse_len_bin_label(b)
            if hi == -1:
                c = sum(v for L, v in counts.items() if L >= lo)
            else:
                c = sum(v for L, v in counts.items() if lo <= L <= hi)
            out.append(float(c / total))
        return out

    # (2) Fallback to validator frac keys (your SMILES JSON has these)
    # Example keys:
    #   test/validator/len_tok_valid_0_2_frac
    #   test/validator/len_tok_0_2_frac
    prefix = "test/validator/len_tok_valid" if valid_only else "test/validator/len_tok"
    vals, found_any = [], False
    for b in bins:
        lo, hi = _parse_len_bin_label(b)
        key_hi = hi
        if hi == -1:
            # your json uses 11_-1 for open ended
            key_hi = -1
        k = f"{prefix}_{lo}_{key_hi}_frac"
        if k in j and j[k] is not None:
            vals.append(float(j[k]))
            found_any = True
        else:
            vals.append(np.nan)
    if found_any:
        return vals

    # (3) Legacy fallback (your previous assumption)
    legacy_prefix = "len_valid_frac" if valid_only else "len_frac"
    vals, found_any = [], False
    for b in bins:
        k = f"{legacy_prefix}[{b}]"
        if k in j and j[k] is not None:
            vals.append(float(j[k]))
            found_any = True
        else:
            vals.append(np.nan)
    return vals if found_any else None


def plot_length_histogram_binned(
    exps: dict[str, dict[str, str]],
    buckets,
    style,
    valid_only: bool = True,
    title: str = "Length histogram (valid-only, binned)",
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    y_range: tuple[float, float] | None = None,
    error_mode: str = "sem",
) -> None:
    names, means, errs = [], [], []
    for exp_name, payload in exps.items():
        runs = []
        for jpath in payload.get("json_paths", []):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue
            j = load_json(jpath)
            v = extract_len_bin_fracs(j, buckets.len_bin_labels, valid_only=valid_only)
            if v is None:
                continue
            runs.append(np.asarray(v, dtype=float))
        if not runs:
            continue
        arr = np.stack(runs, axis=0)
        names.append(exp_name)
        means.append(np.nanmean(arr, axis=0))
        errs.append(_compute_err(arr, mode=error_mode))

    if not means:
        print("No usable length bins found (counts/validator fracs/legacy fracs).")
        return

    fracs_mean = np.asarray(means, dtype=float)
    fracs_err = np.asarray(errs, dtype=float)
    E, B = fracs_mean.shape
    x = np.arange(B)
    width = 0.8 / max(E, 1)

    cmap = color_map or {name: None for name in names}
    fig, ax = plt.subplots(figsize=style.figsize_len_hist_bins, dpi=style.dpi)
    for i, exp_name in enumerate(names):
        color = cmap.get(exp_name)
        ax.bar(
            x + i * width - (E - 1) * width / 2,
            fracs_mean[i],
            width=width,
            label=exp_name,
            yerr=fracs_err[i],
            capsize=3.5,
            **style.bar.kwargs(color=color),
        )

    ax.set_xticks(x, buckets.len_bin_labels)
    ax.set_ylabel("Fraction", fontsize=style.label_fontsize)
    ax.set_title(title, fontsize=style.title_fontsize)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.grid(True, axis="y", alpha=style.grid_alpha)
    ax.legend(
        ncol=max(4, min(style.legend_ncol, len(names))),
        frameon=style.legend_frameon,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
    )
    apply_full_border(ax, style)
    plt.tight_layout()
    _save_fig(fig, save_dir, title)
    plt.show()


def plot_length_histogram_fine(
    exps: dict[str, dict[str, str]],
    style,
    normalize: bool = True,
    valid_only: bool = True,
    title: str = "Length histogram (valid-only, per-length)",
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    y_range: tuple[float, float] | None = None,
    error_mode: str = "sem",
) -> None:
    """
    FIX: use len_counts(_valid) if present; fallback to score_count_by_len(_valid).
    """
    series_map: dict[str, list[pd.Series]] = {}
    for exp_name, payload in exps.items():
        run_series = []
        for jpath in payload.get("json_paths", []):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue
            j = load_json(jpath)

            counts = _get_counts_dict(j, valid_only=valid_only)
            if counts is None:
                continue

            s = pd.Series(counts).sort_index()
            if normalize:
                denom = float(s.sum())
                s = s / denom if denom > 0 else s * np.nan
            run_series.append(s)
        if run_series:
            series_map[exp_name] = run_series

    if not series_map:
        print("No usable per-length counts found (len_counts/score_count_by_len).")
        return

    fig, ax = plt.subplots(figsize=style.figsize_len_hist_fine, dpi=style.dpi)
    cmap = color_map or {}
    for name, series_list in series_map.items():
        agg = _aggregate_series(series_list, error_mode=error_mode)
        if agg is None:
            continue
        x, mean, err = agg
        color, marker_override, linestyle = resolve_method_style(name, style, cmap)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax.errorbar(x, mean, yerr=err, label=name, **line_kw)

    add_shading(ax, style.shade_regions, style.shade_alpha)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.set_title(title, fontsize=style.title_fontsize)
    ax.set_xlabel("Length (tokens)", fontsize=style.label_fontsize)
    ax.set_ylabel("Fraction" if normalize else "Count", fontsize=style.label_fontsize)
    ax.grid(True, alpha=style.grid_alpha)
    ax.legend(ncol=min(style.legend_ncol, len(series_map)), frameon=style.legend_frameon)
    apply_full_border(ax, style)
    plt.tight_layout()
    _save_fig(fig, save_dir, title)
    plt.show()


def plot_metric_by_length_from_json(
    exps: dict[str, dict[str, str]],
    style,
    key: str,
    title: str,
    ylabel: str,
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    y_range: tuple[float, float] | None = None,
    error_mode: str = "sem",
) -> None:
    """
    This part is OK for your JSON: score_mean_by_len_valid and diversity_by_len_valid
    are dict[str length]->value, so extract_by_len_series works.
    """
    series_map: dict[str, list[pd.Series]] = {}
    for exp_name, payload in exps.items():
        run_series = []
        for jpath in payload.get("json_paths", []):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue
            j = load_json(jpath)
            s = extract_by_len_series(j, key)
            if s is None:
                continue
            run_series.append(s)
        if run_series:
            series_map[exp_name] = run_series

    if not series_map:
        print(f"No key '{key}' found as dict in any json; skipping '{title}'.")
        return

    fig, ax = plt.subplots(figsize=style.figsize_by_len, dpi=style.dpi)
    cmap = color_map or {}
    for name, series_list in series_map.items():
        agg = _aggregate_series(series_list, error_mode=error_mode)
        if agg is None:
            continue
        x, mean, err = agg
        color, marker_override, linestyle = resolve_method_style(name, style, cmap)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax.errorbar(x, mean, yerr=err, label=name, **line_kw)

    add_shading(ax, style.shade_regions, style.shade_alpha)
    apply_full_border(ax, style)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.set_title(title, fontsize=style.title_fontsize)
    ax.set_xlabel("Length (tokens)", fontsize=style.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
    ax.grid(True, alpha=style.grid_alpha)
    ax.legend(ncol=min(style.legend_ncol, len(series_map)), frameon=style.legend_frameon)
    plt.tight_layout()
    _save_fig(fig, save_dir, title)
    plt.show()


def _collect_score_series(exps: dict[str, dict[str, str]], key: str) -> dict[str, pd.Series]:
    out = {}
    for exp_name, payload in exps.items():
        jpath = payload.get("json")
        if not jpath:
            continue
        j = load_json(jpath)
        s = extract_by_len_series(j, key)
        if s is None:
            continue
        out[exp_name] = s
    return out


def _aggregate_series(
    series_list: list[pd.Series], error_mode: str = "sem"
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not series_list:
        return None
    all_idx = sorted(set().union(*[set(s.index.tolist()) for s in series_list]))
    if len(all_idx) == 0:
        return None
    arr = np.full((len(series_list), len(all_idx)), np.nan, dtype=float)
    idx_map = {v: i for i, v in enumerate(all_idx)}
    for r, s in enumerate(series_list):
        for k, v in s.items():
            arr[r, idx_map[int(k)]] = float(v)
    mean = np.nanmean(arr, axis=0)
    err = _compute_err(arr, mode=error_mode)
    # if all runs missing at a position, show as 0 with 0 error to avoid broken lines
    all_nan = np.all(np.isnan(arr), axis=0)
    mean[all_nan] = 0.0
    err[all_nan] = 0.0
    return np.array(all_idx, dtype=int), mean, err


def extract_json_metrics_by_length(j: dict[str, Any], keys: JsonKeys) -> pd.DataFrame:
    """
    Pull per-length metrics (all + valid-only) from a single eval JSON.
    Returns DataFrame indexed by length with columns:
      score_mean_all/valid, diversity_all/valid, count_all/valid, frac_all/valid, acc
    """
    score_all = extract_by_len_series(j, keys.score_mean_by_len)
    score_valid = extract_by_len_series(j, keys.score_mean_by_len_valid)
    div_all = extract_by_len_series(j, keys.diversity_by_len)
    div_valid = extract_by_len_series(j, keys.diversity_by_len_valid)

    counts_all = _get_counts_dict(j, valid_only=False)
    counts_valid = _get_counts_dict(j, valid_only=True)
    total_all = float(sum(counts_all.values())) if counts_all else 0.0
    total_valid = float(sum(counts_valid.values())) if counts_valid else 0.0

    lengths = set()
    for s in (score_all, score_valid, div_all, div_valid):
        if s is not None:
            lengths.update(int(k) for k in s.index.tolist())
    for d in (counts_all, counts_valid):
        if d:
            lengths.update(int(k) for k in d.keys())

    if not lengths:
        return pd.DataFrame()

    rows = []
    for L in sorted(lengths):
        ca = counts_all.get(L) if counts_all else None
        cv = counts_valid.get(L) if counts_valid else None
        row = {
            "length": int(L),
            "score_mean_all": float(score_all.get(L))
            if (score_all is not None and L in score_all.index)
            else np.nan,
            "score_mean_valid": float(score_valid.get(L))
            if (score_valid is not None and L in score_valid.index)
            else np.nan,
            "diversity_all": float(div_all.get(L))
            if (div_all is not None and L in div_all.index)
            else np.nan,
            "diversity_valid": float(div_valid.get(L))
            if (div_valid is not None and L in div_valid.index)
            else np.nan,
            "count_all": float(ca) if ca is not None else np.nan,
            "count_valid": float(cv) if cv is not None else np.nan,
            "frac_all": (float(ca) / total_all) if (ca is not None and total_all > 0) else np.nan,
            "frac_valid": (float(cv) / total_valid)
            if (cv is not None and total_valid > 0)
            else np.nan,
            "acc": (float(cv) / float(ca))
            if (ca is not None and ca > 0 and cv is not None)
            else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("length").sort_index()


def _collect_fp_series_from_samples(
    exps: dict[str, dict[str, str]], cfg: SamplesConfig
) -> dict[str, pd.Series]:
    out = {}
    for exp_name, payload in exps.items():
        spath = payload.get("samples")
        if not spath:
            continue
        df = load_samples_csv(spath)
        byL = compute_samples_by_length(df, cfg)
        if "fp_div" not in byL.columns:
            continue
        out[exp_name] = byL["fp_div"]
    return out


def plot_fp_score_stacked(
    exps: dict[str, dict[str, str]],
    style: PlotStyle,
    keys: JsonKeys,
    cfg: SamplesConfig,
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    y_range_fp: tuple[float, float] | None = None,
    y_range_score: tuple[float, float] | None = None,
    title_fp: str = "FP diversity by length",
    title_score: str = "Score by length",
    error_mode: str = "sem",
    name_prefix: str = "",
) -> None:
    score_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fp_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # score
    for exp_name, payload in exps.items():
        series_list = []
        for jpath in payload.get("json_paths", []):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue
            j = load_json(jpath)
            s = extract_by_len_series(j, keys.score_mean_by_len_valid)
            if s is not None:
                series_list.append(s)
        agg = _aggregate_series(series_list, error_mode=error_mode) if series_list else None
        if agg is not None:
            score_map[exp_name] = agg

    # fp from samples
    for exp_name, payload in exps.items():
        series_list = []
        for spath in payload.get("samples_paths", []):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            sdf = load_samples_csv(spath)
            byL = compute_samples_by_length(sdf, cfg)
            if "fp_div" in byL.columns:
                series_list.append(byL["fp_div"])
        agg = _aggregate_series(series_list, error_mode=error_mode) if series_list else None
        if agg is not None:
            fp_map[exp_name] = agg

    if not score_map and not fp_map:
        print("No score or FP data available; skipping stacked plot.")
        return

    cmap = color_map or {}
    per_panel_w, per_panel_h = style.figsize_len_hist_bins
    # Keep the combined stacked figure about as tall as a single length-hist plot.
    # Each subplot will be roughly half the height.
    stacked_h = per_panel_h
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(per_panel_w, stacked_h),
        dpi=style.dpi,
        sharex=False,
    )
    fig.subplots_adjust(hspace=0.25)

    # FP (top)
    ax_fp = axes[0]
    if fp_map:
        for name, (x_fp, mean_fp, err_fp) in fp_map.items():
            color, marker_override, linestyle = resolve_method_style(name, style, cmap)
            line_kw = style.line.kwargs(
                color=color, marker_override=marker_override, linestyle=linestyle
            )
            ax_fp.errorbar(x_fp, mean_fp, yerr=err_fp, label=name, **line_kw)
        add_shading(ax_fp, style.shade_regions, style.shade_alpha)
        apply_full_border(ax_fp, style)
        if y_range_fp is not None:
            ax_fp.set_ylim(*y_range_fp)
        ax_fp.set_title(title_fp, fontsize=style.title_fontsize)
        ax_fp.set_ylabel("1 - mean Tanimoto", fontsize=style.label_fontsize)
        ax_fp.grid(True, alpha=style.grid_alpha)

    # Score (bottom)
    ax_sc = axes[1]
    if score_map:
        for name, (x_sc, mean_sc, err_sc) in score_map.items():
            color, marker_override, linestyle = resolve_method_style(name, style, cmap)
            line_kw = style.line.kwargs(
                color=color, marker_override=marker_override, linestyle=linestyle
            )
            ax_sc.errorbar(x_sc, mean_sc, yerr=err_sc, label=name, **line_kw)
        add_shading(ax_sc, style.shade_regions, style.shade_alpha)
        apply_full_border(ax_sc, style)
        if y_range_score is not None:
            ax_sc.set_ylim(*y_range_score)
        ax_sc.set_title(title_score, fontsize=style.title_fontsize)
        ax_sc.set_xlabel("Length (tokens)", fontsize=style.label_fontsize)
        ax_sc.set_ylabel("score mean", fontsize=style.label_fontsize)
        ax_sc.grid(True, alpha=style.grid_alpha)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(4, min(style.legend_ncol, len(labels))),
            frameon=style.legend_frameon,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=0.35)
    if save_dir is not None:
        prefix = f"{name_prefix}_" if name_prefix else ""
        _save_fig(fig, save_dir, f"{prefix}fp_score_stacked")
    plt.show()


# =========================
# Prefix plots + bucket table
# =========================
def add_bin_shading(ax, prefix_bins: dict[str, tuple[int, int]]):
    for _, (lo, hi) in prefix_bins.items():
        ax.axvspan(lo, hi, alpha=0.08)


def plot_prefix_panels(
    exps: dict[str, dict[str, str]],
    buckets: Buckets,
    style: PlotStyle,
    suptitle: str = "Prefix metrics (correct-only)",
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    ylims: dict[str, tuple[float, float]] | None = None,
    y_range_nk: tuple[float, float] | None = None,
    error_mode: str = "sem",
) -> None:
    """Legacy 2x3 prefix grid + nk panel with aggregation + error bars."""
    pref_runs: dict[str, list[pd.DataFrame]] = {}
    for exp_name, payload in exps.items():
        dfs = []
        for ppath in payload.get("prefix_paths", []):
            if not ppath:
                continue
            dfs.append(load_prefix_csv(ppath))
        if dfs:
            pref_runs[exp_name] = dfs

    if not pref_runs:
        print("No prefix csv provided; skipping prefix plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=style.figsize_prefix, dpi=style.dpi)
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    cmap = color_map or {}

    def plot_curve(ax, ycol, title, ylabel, ylim=None):
        for name, dfs in pref_runs.items():
            all_k = sorted(set().union(*[set(df["k"].tolist()) for df in dfs]))
            if not all_k:
                continue
            arr = np.full((len(dfs), len(all_k)), np.nan, dtype=float)
            k_to_idx = {v: i for i, v in enumerate(all_k)}
            for r, df in enumerate(dfs):
                if ycol not in df.columns:
                    continue
                for k, v in zip(df["k"], df[ycol]):
                    arr[r, k_to_idx[int(k)]] = float(v)
            mean = np.nanmean(arr, axis=0)
            err = _compute_err(arr, mode=error_mode)
            color, marker_override, linestyle = resolve_method_style(name, style, cmap)
            line_kw = style.line.kwargs(
                color=color, marker_override=marker_override, linestyle=linestyle
            )
            ax.errorbar(
                all_k,
                mean,
                yerr=err,
                label=name,
                **line_kw,
            )
        add_shading(
            ax, style.shade_regions or list(buckets.prefix_bins.values()), style.shade_alpha
        )
        apply_full_border(ax, style)
        ax.set_title(title, fontsize=style.title_fontsize)
        ax.set_xlabel("k (prefix length)", fontsize=style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
        target_ylim = (ylims or {}).get(ycol, ylim)
        if target_ylim is not None:
            ax.set_ylim(*target_ylim)
        ax.grid(True, alpha=style.grid_alpha)

    # metrics configuration for reuse and individual saving
    panels = [
        (axes[0, 0], "survival", "Prefix survival rate", "n(k) / n(1)", (0, 1.05)),
        (axes[0, 1], "entropy", "Prefix entropy vs k", "entropy", None),
        (axes[0, 2], "eff", "Effective support size vs k", "eff", None),
        (axes[1, 0], "top1", "Top-1 mass vs k (collapse ↑)", "top1", (0, 1.0)),
        (axes[1, 1], "unique", "Unique tokens vs k", "unique", None),
        (axes[1, 2], "unique_rate", "Unique rate vs k", "unique / n", None),
    ]

    for ax, ycol, title, ylabel, ylim in panels:
        plot_curve(ax, ycol, title, ylabel, ylim=ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(4, min(style.legend_ncol, len(labels))),
        frameon=style.legend_frameon,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(suptitle, y=0.98, fontsize=style.suptitle_fontsize)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()

    # save each panel individually if requested
    if save_dir is not None:
        for _, ycol, title, ylabel, ylim in panels:
            fig_single, ax_single = plt.subplots(figsize=style.figsize_by_len, dpi=style.dpi)
            for name, dfs in pref_runs.items():
                all_k = sorted(set().union(*[set(df["k"].tolist()) for df in dfs]))
                if not all_k:
                    continue
                arr = np.full((len(dfs), len(all_k)), np.nan, dtype=float)
                k_to_idx = {v: i for i, v in enumerate(all_k)}
                for r, df in enumerate(dfs):
                    if ycol not in df.columns:
                        continue
                    for k, v in zip(df["k"], df[ycol]):
                        arr[r, k_to_idx[int(k)]] = float(v)
                mean = np.nanmean(arr, axis=0)
                err = _compute_err(arr, mode=error_mode)
                color, marker_override, linestyle = resolve_method_style(name, style, cmap)
                line_kw = style.line.kwargs(
                    color=color, marker_override=marker_override, linestyle=linestyle
                )
                ax_single.errorbar(
                    all_k,
                    mean,
                    yerr=err,
                    label=name,
                    **line_kw,
                )
            add_shading(
                ax_single,
                style.shade_regions or list(buckets.prefix_bins.values()),
                style.shade_alpha,
            )
            apply_full_border(ax_single, style)
            ax_single.set_title(title, fontsize=style.title_fontsize)
            ax_single.set_xlabel("k (prefix length)", fontsize=style.label_fontsize)
            ax_single.set_ylabel(ylabel, fontsize=style.label_fontsize)
            if ylim is not None:
                ax_single.set_ylim(*ylim)
            ax_single.grid(True, alpha=style.grid_alpha)
            ax_single.legend(
                ncol=min(style.legend_ncol, len(pref_runs)), frameon=style.legend_frameon
            )
            plt.tight_layout()
            _save_fig(fig_single, save_dir, title)
            plt.close(fig_single)

    # n(k)
    fig2, ax2 = plt.subplots(figsize=style.figsize_nk, dpi=style.dpi)
    for name, dfs in pref_runs.items():
        all_k = sorted(set().union(*[set(df["k"].tolist()) for df in dfs]))
        if not all_k:
            continue
        arr = np.full((len(dfs), len(all_k)), np.nan, dtype=float)
        k_to_idx = {v: i for i, v in enumerate(all_k)}
        for r, df in enumerate(dfs):
            for k, v in zip(df["k"], df["n"]):
                arr[r, k_to_idx[int(k)]] = float(v)
        mean = np.nanmean(arr, axis=0)
        err = _compute_err(arr, mode=error_mode)
        color, marker_override, linestyle = resolve_method_style(name, style, cmap)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax2.errorbar(all_k, mean, yerr=err, label=name, **line_kw)
    add_shading(ax2, style.shade_regions or list(buckets.prefix_bins.values()), style.shade_alpha)
    apply_full_border(ax2, style)
    if y_range_nk is not None:
        ax2.set_ylim(*y_range_nk)
    ax2.set_title(
        "Reachable correct samples count n(k) vs k (correct-only)", fontsize=style.title_fontsize
    )
    ax2.set_xlabel("k", fontsize=style.label_fontsize)
    ax2.set_ylabel("n(k)", fontsize=style.label_fontsize)
    ax2.grid(True, alpha=style.grid_alpha)
    ax2.legend(ncol=min(style.legend_ncol, len(pref_runs)), frameon=style.legend_frameon)
    apply_full_border(ax2, style)
    plt.tight_layout()
    _save_fig(fig2, save_dir, "prefix_nk")
    plt.show()


def make_prefix_bucket_table(
    exps: dict[str, dict[str, Any]], buckets: Buckets, error_mode: str = "ci95"
) -> pd.DataFrame:
    AGG_COLS = ["survival", "entropy", "eff", "top1", "unique", "unique_rate", "n"]

    def agg_over_bin(df: pd.DataFrame, lo: int, hi: int) -> dict[str, float]:
        g = df[(df["k"] >= lo) & (df["k"] <= hi)]
        if len(g) == 0:
            out = {c: np.nan for c in AGG_COLS}
            out["survival_end"] = np.nan
            out["n_end"] = np.nan
            return out
        out = {}
        for c in AGG_COLS:
            out[c] = float(g[c].mean()) if c in g.columns else np.nan
        out["survival_end"] = (
            float(df[df["k"] == hi]["survival"].iloc[0]) if (df["k"] == hi).any() else np.nan
        )
        out["n_end"] = float(df[df["k"] == hi]["n"].iloc[0]) if (df["k"] == hi).any() else np.nan
        return out

    rows = []
    for exp_name, payload in exps.items():
        for run_idx, ppath in enumerate(payload.get("prefix_paths", [])):
            if not ppath:
                continue
            if not Path(ppath).exists():
                print(f"[warn] prefix csv not found, skip: {ppath}")
                continue
            df = load_prefix_csv(ppath)
            for bucket_name, (lo, hi) in buckets.prefix_bins.items():
                r = {
                    "experiment": exp_name,
                    "bucket": bucket_name,
                    "run": run_idx,
                    "k_lo": lo,
                    "k_hi": hi,
                }
                r.update(agg_over_bin(df, lo, hi))
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = _aggregate_numeric(df, group_key=["experiment", "bucket"], error_mode=error_mode)
    return agg


def make_prefix_length_table(
    exps: dict[str, dict[str, Any]], error_mode: str = "ci95"
) -> pd.DataFrame:
    """
    Per-length prefix metrics table (survival/entropy/eff/top1/unique/unique_rate/n) aggregated across runs.
    Index: experiment, k
    """
    COLS = ["survival", "entropy", "eff", "top1", "unique", "unique_rate", "n"]
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, ppath in enumerate(payload.get("prefix_paths", [])):
            if not ppath:
                continue
            if not Path(ppath).exists():
                print(f"[warn] prefix csv not found, skip: {ppath}")
                continue
            df = load_prefix_csv(ppath)
            for _, row in df.iterrows():
                r = {"experiment": exp_name, "k": int(row["k"]), "run": run_idx}
                for c in COLS:
                    r[c] = float(row[c]) if c in row and pd.notnull(row[c]) else np.nan
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = _aggregate_numeric(df, group_key=["experiment", "k"], error_mode=error_mode)
    return agg


# =========================
# Samples: FPdiv-by-length + unique-by-length (directly compatible)
# =========================
def compute_samples_by_length(samples_df: pd.DataFrame, cfg: SamplesConfig) -> pd.DataFrame:
    """
    Builds a per-length table from your samples CSV.
    Always computes:
      - n
      - unique_str / unique_rate_str (string-level, cleaned)
    Additionally (if RDKit available + enough parsable SMILES):
      - unique_mol / unique_rate_mol (canonical smiles)
      - fp_div (1 - mean pairwise Tanimoto)
    """
    df = samples_df.copy()

    text_col = cfg.text_col_override or _infer_col(df, cfg.text_cols)
    if text_col is None:
        raise ValueError(f"Cannot infer text/SMILES column. Tried: {cfg.text_cols}")

    valid_col = cfg.valid_col_override or _infer_col(df, cfg.valid_cols)
    length_col = cfg.length_col_override or _infer_col(df, cfg.length_cols)
    token_ids_col = cfg.token_ids_col_override or _infer_col(df, cfg.token_ids_cols)

    # clean string
    df["_text"] = df[text_col].map(lambda s: _clean_text(s, cfg.strip_after_special_token))

    # length
    if length_col is not None:
        df["_len"] = pd.to_numeric(df[length_col], errors="coerce")
    elif token_ids_col is not None:
        df["_len"] = (
            df[token_ids_col]
            .map(_parse_token_list)
            .map(lambda v: len(v) if isinstance(v, list) else np.nan)
        )
    else:
        # last-resort fallback (avoid if you care about true token lengths)
        df["_len"] = df["_text"].astype(str).str.len()

    df = df[df["_len"].notnull()].copy()
    df["_len"] = df["_len"].astype(int)

    # Try RDKit (optional)
    rdkit_ok = False
    Chem = DataStructs = AllChem = None
    try:
        from rdkit import Chem as _Chem
        from rdkit import DataStructs as _DataStructs
        from rdkit.Chem import AllChem as _AllChem

        try:
            # Silence verbose RDKit SMILES parse errors during batch processing.
            from rdkit import RDLogger as _RDLogger

            _RDLogger.DisableLog("rdApp.error")
            _RDLogger.DisableLog("rdApp.warning")
        except Exception:
            pass
        Chem, DataStructs, AllChem = _Chem, _DataStructs, _AllChem
        rdkit_ok = True
    except Exception:
        rdkit_ok = False

    if valid_col is None and not rdkit_ok:
        raise ImportError(
            "RDKit is required to determine validity when no valid column is present. "
            "Install RDKit or provide a valid indicator column (e.g., is_valid)."
        )

    # determine validity: prefer provided valid column, but always require RDKit parseable when available
    base_valid = (
        _to_bool_series(df[valid_col])
        if valid_col is not None
        else pd.Series(True, index=df.index)
    )

    parse_mask = None
    if rdkit_ok:
        parse_mask = df["_text"].map(lambda s: Chem.MolFromSmiles(s) is not None)
        valid_mask = base_valid & parse_mask
        parse_ratio = float(parse_mask.mean()) if len(parse_mask) > 0 else 0.0
    else:
        valid_mask = base_valid
        parse_ratio = 0.0

    df = df[valid_mask].copy()

    # string-level unique
    rows = []
    rng = np.random.default_rng(0)

    # Pre-check if this looks like SMILES at all: reuse parse ratio when available
    use_rdkit = rdkit_ok and (parse_ratio >= 0.2)  # heuristic: at least some SMILES parse

    for L, g in df.groupby("_len"):
        texts = g["_text"].tolist()
        n = len(texts)
        uniq_str = len(set(texts))
        uniq_rate_str = uniq_str / n if n > 0 else np.nan

        out = {
            "length": int(L),
            "n": int(n),
            "unique_str": int(uniq_str),
            "unique_rate_str": float(uniq_rate_str),
            "unique_mol": np.nan,
            "unique_rate_mol": np.nan,
            "fp_div": np.nan,
        }

        if use_rdkit:
            # subsample per length for fp-div computation
            idx = np.arange(n)
            if n > cfg.max_per_len:
                idx = rng.choice(idx, size=cfg.max_per_len, replace=False)
                g2 = g.iloc[idx]
            else:
                g2 = g

            mols, canon = [], []
            for s in g2["_text"].tolist():
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                mols.append(mol)
                canon.append(Chem.MolToSmiles(mol, canonical=True))

            if len(canon) > 0:
                um = len(set(canon))
                out["unique_mol"] = int(um)
                out["unique_rate_mol"] = float(um / len(canon))

            if len(mols) > 1:
                from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

                gen = GetMorganGenerator(radius=cfg.morgan_radius, fpSize=cfg.morgan_nbits)
                fps = [gen.GetFingerprint(m) for m in mols]

                sum_sims = 0.0
                pairs = 0
                for i in range(1, len(fps)):
                    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
                    sum_sims += float(np.sum(sims))
                    pairs += i
                mean_sim = sum_sims / pairs if pairs > 0 else np.nan
                out["fp_div"] = float(1.0 - mean_sim) if mean_sim == mean_sim else np.nan

        rows.append(out)

    result = pd.DataFrame(rows).sort_values("length").set_index("length")

    fp_div_mean_all = np.nan
    if "fp_div" in result.columns:
        fp_vals = result["fp_div"].astype(float)
        weights = result["n"].astype(float)
        mask = fp_vals.notnull() & weights.notnull()
        if mask.any():
            w = weights[mask]
            f = fp_vals[mask]
            denom = float(w.sum())
            if denom > 0:
                fp_div_mean_all = float(np.average(f, weights=w))
            else:
                fp_div_mean_all = float(f.mean())
    result["fp_div_mean_all"] = fp_div_mean_all
    result.attrs["fp_div_mean_all"] = fp_div_mean_all

    return result


def aggregate_samples_by_length(
    exps: dict[str, dict[str, Any]], cfg: SamplesConfig, error_mode: str = "ci95"
) -> pd.DataFrame:
    """
    Build aggregated samples-by-length table across runs for each experiment.
    Output index: (experiment, length); columns: metric_mean/metric_err.
    """
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, spath in enumerate(payload.get("samples_paths", [])):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            sdf = load_samples_csv(spath)
            byL = compute_samples_by_length(sdf, cfg)
            for L, r in byL.iterrows():
                entry = {"experiment": exp_name, "length": int(L), "run": run_idx}
                for col, val in r.items():
                    entry[col] = float(val) if pd.notnull(val) else np.nan
                rows.append(entry)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = _aggregate_numeric(df, group_key=["experiment", "length"], error_mode=error_mode)
    return agg


def aggregate_json_by_length(
    exps: dict[str, dict[str, Any]], keys: JsonKeys, error_mode: str = "ci95"
) -> pd.DataFrame:
    """
    Aggregate JSON-derived per-length metrics across runs.
    Output index: (experiment, length); columns include score/diversity/count/frac/acc mean+err.
    """
    rows = []
    for exp_name, payload in exps.items():
        for run_idx, jpath in enumerate(payload.get("json_paths", [])):
            if not jpath:
                continue
            if not Path(jpath).exists():
                print(f"[warn] json not found, skip: {jpath}")
                continue
            j = load_json(jpath)
            by_len = extract_json_metrics_by_length(j, keys)
            if by_len is None or len(by_len) == 0:
                continue
            for L, r in by_len.iterrows():
                entry = {"experiment": exp_name, "length": int(L), "run": run_idx}
                for col, val in r.items():
                    entry[col] = float(val) if pd.notnull(val) else np.nan
                rows.append(entry)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = _aggregate_numeric(df, group_key=["experiment", "length"], error_mode=error_mode)
    return agg


def make_metrics_by_length_table(
    main_table: pd.DataFrame, samples_by_length: pd.DataFrame, json_by_length: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine main-table metrics with per-length samples metrics.
    main_table: index=experiment with *_mean/*_err columns.
    samples_by_length/json_by_length: index=(experiment, length) with per-length metrics.
    """
    base = None

    if json_by_length is not None and len(json_by_length) > 0:
        json_reset = json_by_length.rename(columns={"n_runs": "json_n_runs"})
        base = json_reset

    if samples_by_length is not None and len(samples_by_length) > 0:
        samples_reset = samples_by_length.rename(columns={"n_runs": "samples_n_runs"})
        base = (
            samples_reset
            if base is None
            else base.join(samples_reset, how="outer", lsuffix="_json", rsuffix="_samples")
        )

    if base is None or len(base) == 0:
        return pd.DataFrame()

    combined = base.reset_index()

    if main_table is not None and len(main_table) > 0:
        main_reset = main_table.reset_index().rename(columns={"n_runs": "main_n_runs"})
        combined = combined.merge(main_reset, on="experiment", how="left")

    combined = combined.sort_values(["experiment", "length"]).set_index(["experiment", "length"])
    return combined


def plot_samples_metric_by_length(
    exps: dict[str, dict[str, str]],
    style: PlotStyle,
    cfg: SamplesConfig,
    metric: str,
    title: str,
    ylabel: str,
    color_map: dict[str, Any] | None = None,
    save_dir: Path | None = None,
    y_range: tuple[float, float] | None = None,
    error_mode: str = "ci95",
) -> None:
    series_map: dict[str, list[pd.Series]] = {}
    for exp_name, payload in exps.items():
        run_series = []
        for spath in payload.get("samples_paths", []):
            if not spath:
                continue
            if not Path(spath).exists():
                print(f"[warn] samples csv not found, skip: {spath}")
                continue
            df = load_samples_csv(spath)
            byL = compute_samples_by_length(df, cfg)
            if metric not in byL.columns:
                continue
            run_series.append(byL[metric])
        if run_series:
            series_map[exp_name] = run_series

    if not series_map:
        print(f"No samples csv / metric '{metric}' available; skipping '{title}'.")
        return

    fig, ax = plt.subplots(figsize=style.figsize_by_len, dpi=style.dpi)
    cmap = color_map or {}
    for name, series_list in series_map.items():
        agg = _aggregate_series(series_list, error_mode=error_mode)
        if agg is None:
            continue
        x, mean, err = agg
        color, marker_override, linestyle = resolve_method_style(name, style, cmap)
        line_kw = style.line.kwargs(
            color=color, marker_override=marker_override, linestyle=linestyle
        )
        ax.errorbar(x, mean, yerr=err, label=name, **line_kw)

    add_shading(ax, style.shade_regions, style.shade_alpha)
    apply_full_border(ax, style)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.set_title(title, fontsize=style.title_fontsize)
    ax.set_xlabel("Length", fontsize=style.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
    ax.grid(True, alpha=style.grid_alpha)
    ax.legend(ncol=min(style.legend_ncol, len(series_map)), frameon=style.legend_frameon)
    plt.tight_layout()
    _save_fig(fig, save_dir, title)
    plt.show()


# =========================
# Tables + plotting runners (tables and plots can be called separately)
# =========================
def run_tables(
    exps: dict[str, dict[str, str]],
    buckets: Buckets = Buckets(),
    keys: JsonKeys = JsonKeys(),
    samples_cfg: SamplesConfig = SamplesConfig(),
    error_mode: str = "ci95",
    output_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    exps_norm = normalize_exps(exps)

    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        cwd_prev = Path(".").resolve()
        os.chdir(output_root)
    else:
        cwd_prev = None

    main_df = make_main_table(
        exps_norm, buckets=buckets, error_mode=error_mode, samples_cfg=samples_cfg
    )
    tables["main_table"] = main_df
    if len(main_df) > 0:
        display(main_df)
        main_df.to_csv("main_table.csv")

    pref_bucket = make_prefix_bucket_table(exps_norm, buckets=buckets, error_mode=error_mode)
    tables["prefix_bucket_table"] = pref_bucket
    if len(pref_bucket) > 0:
        display(pref_bucket)
        pref_bucket.to_csv("prefix_bucket_table.csv")

    pref_bylen = make_prefix_length_table(exps_norm, error_mode=error_mode)
    tables["prefix_by_length_table"] = pref_bylen
    if len(pref_bylen) > 0:
        display(pref_bylen)
        pref_bylen.to_csv("prefix_by_length_table.csv")

    samples_agg = aggregate_samples_by_length(exps_norm, cfg=samples_cfg, error_mode=error_mode)
    tables["samples_by_length"] = samples_agg
    if len(samples_agg) > 0:
        display(samples_agg)

    json_by_length = aggregate_json_by_length(exps_norm, keys=keys, error_mode=error_mode)
    tables["json_by_length"] = json_by_length
    if len(json_by_length) > 0:
        display(json_by_length)
        json_by_length.to_csv("json_by_length.csv")

    metrics_by_length = make_metrics_by_length_table(main_df, samples_agg, json_by_length)
    tables["metrics_by_length"] = metrics_by_length
    if len(metrics_by_length) > 0:
        display(metrics_by_length)
        metrics_by_length.to_csv("metrics_by_length.csv")

    if cwd_prev is not None:
        os.chdir(cwd_prev)

    return tables


def run_plots(
    exps: dict[str, dict[str, str]],
    style: PlotStyle = PlotStyle(),
    buckets: Buckets = Buckets(),
    keys: JsonKeys = JsonKeys(),
    samples_cfg: SamplesConfig = SamplesConfig(),
    plot_prefix: bool = True,
    plot_len_hist: bool = True,
    plot_by_len_json: bool = True,
    plot_by_len_samples: bool = True,
    save_fig_dir: Path | None = Path("figures_pdf"),
    len_hist_bins_y: tuple[float, float] | None = None,
    len_hist_fine_y: tuple[float, float] | None = None,
    json_bylen_y: dict[str, tuple[float, float]] | None = None,
    prefix_ylims: dict[str, tuple[float, float]] | None = None,
    prefix_nk_ylim: tuple[float, float] | None = None,
    samples_y: dict[str, tuple[float, float]] | None = None,
    plot_score_fp_stacked: bool = True,
    stacked_fp_y: tuple[float, float] | None = None,
    stacked_score_y: tuple[float, float] | None = None,
    error_mode: str = "ci95",
    output_root: Path | None = None,
    name_prefix: str = "",
) -> None:
    apply_plot_style(style)
    exps_norm = normalize_exps(exps)
    apply_method_styles_from_exps(exps_norm, style)
    color_map = build_color_map(exps_norm, style.palette)
    base_save_dir = Path(save_fig_dir) if save_fig_dir is not None else None
    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        save_dir = output_root
    else:
        save_dir = base_save_dir

    def _with_prefix(title: str) -> str:
        return f"{name_prefix} {title}".strip() if name_prefix else title

    if plot_len_hist:
        plot_length_histogram_binned(
            exps_norm,
            buckets=buckets,
            style=style,
            valid_only=True,
            title=_with_prefix("Length histogram"),
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
            title=_with_prefix("Length Distribution"),
            color_map=color_map,
            save_dir=save_dir,
            y_range=len_hist_fine_y,
            error_mode=error_mode,
        )

    if plot_by_len_json:
        if not plot_score_fp_stacked:
            plot_metric_by_length_from_json(
                exps_norm,
                style=style,
                key=keys.score_mean_by_len_valid,
                title=_with_prefix("Score by length"),
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
            title=_with_prefix("Token diversity by length"),
            ylabel="token diversity",
            color_map=color_map,
            save_dir=save_dir,
            y_range=(json_bylen_y or {}).get(keys.diversity_by_len_valid),
            error_mode=error_mode,
        )
        plot_metric_by_length_from_json(
            exps_norm,
            style=style,
            key=keys.log_pterm_by_len,
            title=_with_prefix("Log pterm by length"),
            ylabel="log pterm",
            color_map=color_map,
            save_dir=save_dir,
            y_range=(json_bylen_y or {}).get(keys.log_pterm_by_len),
            error_mode=error_mode,
        )

        # Added: per-length p(term) curve.
        plot_metric_by_length_from_json(
            exps_norm,
            style=style,
            key=keys.pterm_by_len,
            title=_with_prefix("p(term) by length"),
            ylabel="p(term)",
            color_map=color_map,
            save_dir=save_dir,
            y_range=(json_bylen_y or {}).get(keys.pterm_by_len),
            error_mode=error_mode,
        )

    if plot_prefix:
        plot_prefix_triplet(
            exps_norm,
            buckets=buckets,
            style=style,
            color_map=color_map,
            save_dir=save_dir,
            ylims=prefix_ylims,
            error_mode=error_mode,
            name_prefix=name_prefix,
        )

    if plot_by_len_samples:
        if plot_score_fp_stacked:
            plot_fp_score_stacked(
                exps_norm,
                style=style,
                keys=keys,
                cfg=samples_cfg,
                color_map=color_map,
                save_dir=save_dir,
                y_range_fp=stacked_fp_y,
                y_range_score=stacked_score_y,
                error_mode=error_mode,
                name_prefix=name_prefix,
            )
        else:
            plot_samples_metric_by_length(
                exps_norm,
                style=style,
                cfg=samples_cfg,
                metric="fp_div",
                title=_with_prefix("FP diversity by length"),
                ylabel="1 - mean Tanimoto",
                color_map=color_map,
                save_dir=save_dir,
                y_range=(samples_y or {}).get("fp_div"),
                error_mode=error_mode,
            )
        plot_samples_metric_by_length(
            exps_norm,
            style=style,
            cfg=samples_cfg,
            metric="unique_rate_str",
            title=_with_prefix("Unique (string) rate by length"),
            ylabel="unique / n",
            color_map=color_map,
            save_dir=save_dir,
            y_range=(samples_y or {}).get("unique_rate_str"),
            error_mode=error_mode,
        )
        plot_samples_metric_by_length(
            exps_norm,
            style=style,
            cfg=samples_cfg,
            metric="n",
            title=_with_prefix("Valid samples count by length"),
            ylabel="count",
            color_map=color_map,
            save_dir=save_dir,
            y_range=(samples_y or {}).get("n"),
            error_mode=error_mode,
        )


# compatibility wrapper
# returns tables and also runs plots unless disabled


def run_analysis(
    exps: dict[str, dict[str, Any]],
    style: PlotStyle = PlotStyle(),
    buckets: Buckets = Buckets(),
    keys: JsonKeys = JsonKeys(),  # kept for by-length metrics keys
    samples_cfg: SamplesConfig = SamplesConfig(),
    plot_prefix: bool = True,
    plot_len_hist: bool = True,
    plot_by_len_json: bool = True,
    plot_by_len_samples: bool = True,
    save_fig_dir: Path | None = Path("figures_pdf"),
    len_hist_bins_y: tuple[float, float] | None = None,
    len_hist_fine_y: tuple[float, float] | None = None,
    json_bylen_y: dict[str, tuple[float, float]] | None = None,
    prefix_ylims: dict[str, tuple[float, float]] | None = None,
    prefix_nk_ylim: tuple[float, float] | None = None,
    samples_y: dict[str, tuple[float, float]] | None = None,
    make_tables: bool = True,
    make_plots: bool = True,
    plot_score_fp_stacked: bool = True,
    stacked_fp_y: tuple[float, float] | None = None,
    stacked_score_y: tuple[float, float] | None = None,
    error_mode: str = "ci95",
    output_root: Path | None = None,
    name_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    output_root_path = Path(output_root) if output_root is not None else None
    if make_tables:
        tables = run_tables(
            exps,
            buckets=buckets,
            keys=keys,
            samples_cfg=samples_cfg,
            error_mode=error_mode,
            output_root=output_root_path,
        )

    if make_plots:
        run_plots(
            exps,
            style=style,
            buckets=buckets,
            keys=keys,
            samples_cfg=samples_cfg,
            plot_prefix=plot_prefix,
            plot_len_hist=plot_len_hist,
            plot_by_len_json=plot_by_len_json,
            plot_by_len_samples=plot_by_len_samples,
            save_fig_dir=save_fig_dir,
            len_hist_bins_y=len_hist_bins_y,
            len_hist_fine_y=len_hist_fine_y,
            json_bylen_y=json_bylen_y,
            prefix_ylims=prefix_ylims,
            prefix_nk_ylim=prefix_nk_ylim,
            samples_y=samples_y,
            plot_score_fp_stacked=plot_score_fp_stacked,
            stacked_fp_y=stacked_fp_y,
            stacked_score_y=stacked_score_y,
            error_mode=error_mode,
            output_root=output_root_path,
            name_prefix=name_prefix,
        )

    return tables


# ============================================================
# Example usage (edit style here globally)
# ============================================================
style = PlotStyle(
    dpi=300,
    aspect_ratio=1.8,
    line=LineParams(linewidth=2.0, markersize=6.5, markeredgewidth=0.0, alpha=0.9),
    shade_regions=DEFAULT_SHADE_REGIONS,
    shade_alpha=0.04,
    method_styles={
        # Example per-method overrides
        # "TB": MethodStyle(color="C0", linestyle="-", marker="o"),
        # "RapTB": MethodStyle(color="C1", linestyle="--", marker="s"),
    },
)
buckets = Buckets()

samples_cfg = SamplesConfig(max_per_len=512)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMILES eval plotting/tables.")
    parser.add_argument(
        "--output-name",
        "-o",
        dest="output_root",
        type=str,
        default="smiles_outputs",
        help="Directory to store all outputs (tables + figures).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_root) if args.output_root else None

    tables = run_analysis(
        exps,
        style=style,
        buckets=buckets,
        keys=JsonKeys(),
        samples_cfg=samples_cfg,
        plot_prefix=True,
        plot_len_hist=True,
        plot_by_len_json=True,
        plot_by_len_samples=True,
        output_root=out_dir,
    )
