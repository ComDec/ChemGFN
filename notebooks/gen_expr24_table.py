#!/usr/bin/env python3
"""
Expr24 table generator (main + appendix + diagnostics) from a directory of CSVs.

Main table (paper):
  Replay x Objective -> {Unique_checkmark, NormCov, Acc, KL(pi->p*), KL(p*->pi), JS_tok}

Appendix table:
  Replay x Objective -> per-length NormCov_ell columns.

Diagnostics table:
  Termination/length calibration (Acc, Cov, log p_term(tau), log Z).

TBLogZ table:
  Per-length Cov for RootSubTBLogZ (RP vs Oracle).

Per-length normalized coverage:
  NormCov_ell = CovCount_ell / min(N, |Y*_ell|)
where N is the sampling cap, estimated as sum(gen_count_by_len_mean) across lengths for that experiment.

Defaults:
- Prefer SubM ext-size variant if both SubM and SubM-ext-size exist.
- Exclude RootSubTBLogZ (regex "RootSubTBLogZ") unless --exclude_regex "".

Usage:
  python gen_expr24_table_v3.py --input_dir /path/to/csvs \
      --main_tex expr24_main.tex --appendix_tex expr24_len_normcov.tex \
      --pterm_tex expr24_pterm_diag.tex --tb_logz_len_tex expr24_normcov_by_len_all.tex \
      --pterm_by_len_tex expr24_pterm_by_len.tex

You may also override individual file paths:
  --coverage_csv ... --pos_div_csv ... --valid_ratio_csv ... --length_by_len_csv ... --json_metrics_csv ... --pterm_by_len_csv ...
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Helpers: parsing + formatting
# -----------------------------


def _canon_exp_name(s: str) -> str:
    """Handle tuple-like experiment names stored as strings."""
    if not isinstance(s, str):
        return str(s)
    s = s.strip()
    m = re.match(r"\(\s*'([^']+)'\s*,?\s*\)", s)
    return m.group(1) if m else s


def _pick_first_existing(cols: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(cols)
    for c in candidates:
        if c in cols:
            return c
    return None


@dataclass(frozen=True)
class ExpKey:
    method: str
    replay: str
    exp: str
    subm_variant: str  # "base" or "ext"


def _infer_replay(exp: str) -> tuple[str, str]:
    low = exp.lower()
    if "oracle" in low:
        return "oracle", "base"
    if re.search(r"\bprt\b", low):
        return "prt", "base"
    if "subm" in low:
        return "subm", ("ext" if "ext" in low else "base")
    return "hu_rp", "base"


def _strip_tokens_keep_method(exp: str) -> str:
    """Remove replay tokens from experiment name to get the base objective name."""
    s = exp.replace(" ", "")
    # Remove replay descriptors (case-insensitive)
    patterns = [
        r"(?i)[_-]oracle\b",
        r"(?i)\boracle\b",
        r"(?i)[_-]prt\b",
        r"(?i)\bprt\b",
        r"(?i)[_-]subm(?:[_-]ext(?:[_-]?size)?)?\b",
        r"(?i)\bsubm(?:[_-]ext(?:[_-]?size)?)?\b",
        r"(?i)[_-]rp\b",
        r"(?i)\brp\b",
    ]
    for p in patterns:
        s = re.sub(p, "", s)
    s = re.sub(r"[_-]+$", "", s)
    s = re.sub(r"^[_-]+", "", s)
    s = re.sub(r"[_-]{2,}", "_", s)
    return s


def _expkey(exp: str) -> ExpKey:
    replay, subm_variant = _infer_replay(exp)
    method = _strip_tokens_keep_method(exp)
    return ExpKey(method=method, replay=replay, exp=exp, subm_variant=subm_variant)


def _prefer_subm_ext(
    df: pd.DataFrame, exp_col="exp", replay_col="replay", method_col="method"
) -> pd.DataFrame:
    """If both SubM and SubM-ext variants exist for the same (replay=subm, method), keep only ext."""
    if replay_col not in df.columns:
        return df
    kept = []
    for (replay, method), g in df.groupby([replay_col, method_col], sort=False):
        if replay != "subm":
            kept.append(g)
            continue
        g_ext = g[g[exp_col].str.lower().str.contains("ext", na=False)]
        kept.append(g_ext if len(g_ext) else g)
    return pd.concat(kept, axis=0, ignore_index=True)


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def _fmt_sci(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e3:
        return f"{x:.{digits}f}"
    return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")


def _fmt_pm(mean: float, err: float, digits: int = 2) -> str:
    """mean +/- err (err is 95% CI)."""
    if pd.isna(mean):
        return ""
    # Use $\pm$ so this works in text-mode table cells.
    if pd.isna(err):
        err = 0.0
    return f"{_fmt_sci(mean, digits)}$\\pm${_fmt_sci(err, digits)}"


def _infer_err_col(cols: Iterable[str], mean_col: str | None) -> str | None:
    """Best-effort: map '<name>_mean' -> '<name>_err'."""
    if not mean_col:
        return None
    if mean_col.endswith("_mean"):
        cand = mean_col[:-5] + "_err"
        if cand in set(cols):
            return cand
    return None


def _fmt_pm_fixed(mean: float, err: float, digits: int = 3) -> str:
    """Fixed-decimal mean +/- err (err is 95% CI)."""
    if pd.isna(mean):
        return ""
    if pd.isna(err):
        err = 0.0
    return f"{float(mean):.{digits}f}$\\pm${float(err):.{digits}f}"


def _ci95_ratio_err(
    num_mean: pd.Series,
    num_err95: pd.Series | None,
    den_mean: pd.Series,
    den_err95: pd.Series | None,
    eps: float = 1e-12,
) -> pd.Series:
    """Propagate 95% CI for ratio num/den using SEM approximation.

    Assumes errors are 95% CI, converts to SEM via /1.96, then uses
    first-order delta method:
      var(num/den) ~= (sem_num/den)^2 + (num*sem_den/den^2)^2
    """
    den = den_mean.astype(float).replace(0, np.nan)
    num = num_mean.astype(float)
    sem_num = (num_err95.astype(float) / 1.96) if num_err95 is not None else 0.0
    sem_den = (den_err95.astype(float) / 1.96) if den_err95 is not None else 0.0
    den_safe = den.abs().clip(lower=eps)
    var = (sem_num / den_safe) ** 2 + ((num * sem_den) / (den_safe**2)) ** 2
    return 1.96 * np.sqrt(var)


def _fmt_num_or_dash(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return _fmt_num(float(x), digits)


# -----------------------------
# File discovery
# -----------------------------


def _find_first_csv(
    input_dir: Path,
    include_any: Iterable[str],
    include_all: Iterable[str] = (),
    exclude_any: Iterable[str] = (),
) -> Path | None:
    cands = sorted(input_dir.glob("*.csv"))

    def ok(p: Path) -> bool:
        n = p.name.lower()
        if any(tok.lower() not in n for tok in include_all):
            return False
        if any(tok.lower() in n for tok in exclude_any):
            return False
        if include_any:
            return any(tok.lower() in n for tok in include_any)
        return True

    for p in cands:
        if ok(p):
            return p
    return None


def discover_csvs(input_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Return (coverage_csv, pos_div_csv, valid_ratio_csv, length_by_len_csv, json_metrics_csv, pterm_by_len_csv).
    """
    coverage = _find_first_csv(
        input_dir, include_any=("coverage",), exclude_any=("by_length", "bylen", "length_coverage")
    )
    if coverage is None:
        raise FileNotFoundError(
            "Missing coverage summary CSV (name should include 'coverage' but not 'by_length')."
        )

    pos = _find_first_csv(input_dir, include_any=("pos",), include_all=())
    if pos is None:
        raise FileNotFoundError("Missing pos-divergence CSV (name should include 'pos').")

    valid = _find_first_csv(
        input_dir, include_any=("valid", "acc", "ratio"), include_all=("valid",)
    )
    if valid is None:
        valid = _find_first_csv(input_dir, include_any=("valid",), include_all=())
    if valid is None:
        raise FileNotFoundError("Missing valid-ratio CSV (name should include 'valid').")

    length_by_len = _find_first_csv(
        input_dir,
        include_any=("length_coverage_by_length", "by_length", "bylen"),
        include_all=("length",),
    )
    if length_by_len is None:
        alt = input_dir / "length_coverage_by_length.csv"
        if alt.exists():
            length_by_len = alt
    if length_by_len is None:
        raise FileNotFoundError(
            "Missing length-by-length CSV (name should include 'length' and 'by_length')."
        )

    json_metrics = _find_first_csv(
        input_dir, include_any=("json_metrics", "pterm", "log_pterm"), include_all=()
    )
    if json_metrics is None:
        alt = input_dir / "json_metrics_table.csv"
        if alt.exists():
            json_metrics = alt
    if json_metrics is None:
        raise FileNotFoundError("Missing json-metrics CSV (name should include 'json_metrics').")

    pterm_by_len = _find_first_csv(
        input_dir, include_any=("pterm_by_length", "pterm_by_len"), include_all=()
    )
    if pterm_by_len is None:
        alt = input_dir / "pterm_by_length.csv"
        if alt.exists():
            pterm_by_len = alt
    if pterm_by_len is None:
        raise FileNotFoundError(
            "Missing pterm-by-length CSV (name should include 'pterm_by_length')."
        )

    return coverage, pos, valid, length_by_len, json_metrics, pterm_by_len


# -----------------------------
# Main table
# -----------------------------


def build_main_table(
    coverage_csv: str,
    pos_div_csv: str,
    valid_ratio_csv: str,
    exclude_regex: str = r"RootSubTBLogZ",
    prefer_subm_ext: bool = True,
) -> tuple[pd.DataFrame, str]:
    cov = pd.read_csv(coverage_csv)
    pos = pd.read_csv(pos_div_csv)
    vr = pd.read_csv(valid_ratio_csv)

    for df in (cov, pos, vr):
        df["exp"] = df["experiment"].apply(_canon_exp_name)

    df = cov.merge(pos.drop(columns=["experiment"]), on="exp", how="left")
    df = df.merge(vr.drop(columns=["experiment"]), on="exp", how="left")

    # Column picking (robust)
    col_unique = _pick_first_existing(df.columns, ["unique_correct_mean"])
    col_ref = _pick_first_existing(df.columns, ["ref_total_mean"])
    col_n = _pick_first_existing(df.columns, ["n_total_mean", "n_total"])
    col_acc = _pick_first_existing(df.columns, ["valid_ratio_mean", "acc_mean", "accuracy_mean"])

    col_kl_s2r = _pick_first_existing(
        df.columns, ["kl_sample_to_ref_mean_over_pos_mean", "KL_s2r_mean", "kl_s2r_mean"]
    )
    col_kl_r2s = _pick_first_existing(
        df.columns, ["kl_ref_to_sample_mean_over_pos_mean", "KL_r2s_mean", "kl_r2s_mean"]
    )
    col_js_tok = _pick_first_existing(
        df.columns, ["js_mean_over_pos_mean", "JS_tok_mean", "js_tok_mean"]
    )

    needed = [
        ("unique", col_unique),
        ("ref_total", col_ref),
        ("n_total", col_n),
        ("acc", col_acc),
        ("KL_s2r", col_kl_s2r),
        ("KL_r2s", col_kl_r2s),
        ("JS_tok", col_js_tok),
    ]
    miss = [name for name, col in needed if col is None]
    if miss:
        raise ValueError(
            f"Missing required columns for main table: {miss}. Available: {list(df.columns)}"
        )

    n_total = pd.to_numeric(df[col_n], errors="coerce")
    _ = pd.to_numeric(df[col_ref], errors="coerce") if col_ref else None
    denom = pd.Series(n_total, index=df.index).replace(0, np.nan)
    df["normcov"] = pd.to_numeric(df[col_unique], errors="coerce") / denom

    keys = df["exp"].apply(_expkey)
    df["method"] = keys.apply(lambda k: k.method)
    df["replay"] = keys.apply(lambda k: k.replay)

    if exclude_regex:
        df = df[~df["method"].astype(str).str.contains(exclude_regex, regex=True, na=False)].copy()

    if prefer_subm_ext:
        df = _prefer_subm_ext(df, exp_col="exp", replay_col="replay", method_col="method")

    out = df[
        ["replay", "method", col_unique, "normcov", col_acc, col_kl_s2r, col_kl_r2s, col_js_tok]
    ].rename(
        columns={
            col_unique: "unique_correct",
            col_acc: "acc",
            col_kl_s2r: "KL_s2r",
            col_kl_r2s: "KL_r2s",
            col_js_tok: "JS_tok",
        }
    )

    # Sorting
    replay_order = {"prt": 0, "hu_rp": 1, "subm": 2, "oracle": 3}
    method_order = {"TB": 0, "SubTB": 1, "RapTB": 2}
    out["_ro"] = out["replay"].map(lambda x: replay_order.get(str(x), 99))
    out["_mo"] = out["method"].map(lambda x: method_order.get(str(x), 99))
    out = out.sort_values(["_ro", "_mo"]).drop(columns=["_ro", "_mo"]).reset_index(drop=True)

    replay_name = {"prt": "PRT", "hu_rp": "Hu RP", "subm": "SubM", "oracle": "Oracle"}

    header = [
        "Replay",
        "Objective",
        "Unique$_\\checkmark$",
        "NormCov",
        "Acc",
        "KL($\\pi\\!\\to\\!p^*$)",
        "KL($p^*\\!\\to\\!\\pi$)",
        "JS$_{\\text{tok}}$",
    ]
    col_spec = "ll" + "c" * (len(header) - 2)

    lines: list[str] = []
    for replay, g in out.groupby("replay", sort=False):
        g = g.copy()
        n = len(g)
        for i, (_, r) in enumerate(g.iterrows()):
            left = ""
            if i == 0:
                left = f"\\multirow{{{n}}}{{*}}{{{replay_name.get(replay, replay)}}}"
            uc = f"{float(r['unique_correct']):.1f}" if pd.notna(r["unique_correct"]) else ""
            row = [
                left,
                str(r["method"]),
                uc,
                _fmt_num(r["normcov"], 3),
                _fmt_num(r["acc"], 3),
                _fmt_num(r["KL_s2r"], 3),
                _fmt_num(r["KL_r2s"], 3),
                _fmt_num(r["JS_tok"], 3),
            ]
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]

    latex = (
        "% Requires \\usepackage{booktabs,multirow}\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.08}\n"
        "\\caption{\\textbf{Expr24 results under four replay schemes.} "
        "For \\textbf{SubM}, we report the \\emph{ext-size} buffer variant for TB and SubTB for a fairer replay-strength comparison. "
        "NormCov $=\\frac{\\text{unique}_{\\checkmark}}{\\min(N,|\\mathcal{Y}^*|)}$ accounts for the sampling cap. "
        "JS$_{\\text{tok}}$ is the token/position-level JS divergence (original JS$_{pos}$).}\n"
        "\\label{tab:expr24_main}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + " & ".join(header) + " \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    return out, latex


def build_main_table_ci(
    coverage_csv: str,
    pos_div_csv: str,
    valid_ratio_csv: str,
    exclude_regex: str = r"RootSubTBLogZ",
    prefer_subm_ext: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Main table, but every numeric cell is mean +/- 95% CI."""
    cov = pd.read_csv(coverage_csv)
    pos = pd.read_csv(pos_div_csv)
    vr = pd.read_csv(valid_ratio_csv)

    for df in (cov, pos, vr):
        df["exp"] = df["experiment"].apply(_canon_exp_name)

    df = cov.merge(pos.drop(columns=["experiment"]), on="exp", how="left")
    df = df.merge(vr.drop(columns=["experiment"]), on="exp", how="left")

    # Means
    col_unique_m = _pick_first_existing(df.columns, ["unique_correct_mean"])
    col_n_m = _pick_first_existing(df.columns, ["n_total_mean", "n_total"])
    col_acc_m = _pick_first_existing(df.columns, ["valid_ratio_mean", "acc_mean", "accuracy_mean"])
    col_kl_s2r_m = _pick_first_existing(
        df.columns, ["kl_sample_to_ref_mean_over_pos_mean", "KL_s2r_mean", "kl_s2r_mean"]
    )
    col_kl_r2s_m = _pick_first_existing(
        df.columns, ["kl_ref_to_sample_mean_over_pos_mean", "KL_r2s_mean", "kl_r2s_mean"]
    )
    col_js_tok_m = _pick_first_existing(
        df.columns, ["js_mean_over_pos_mean", "JS_tok_mean", "js_tok_mean"]
    )

    needed = [
        ("unique_correct_mean", col_unique_m),
        ("n_total_mean", col_n_m),
        ("acc_mean", col_acc_m),
        ("KL_s2r_mean", col_kl_s2r_m),
        ("KL_r2s_mean", col_kl_r2s_m),
        ("JS_tok_mean", col_js_tok_m),
    ]
    miss = [name for name, col in needed if col is None]
    if miss:
        raise ValueError(
            f"Missing required columns for CI main table: {miss}. Available: {list(df.columns)}"
        )

    # Errors (95% CI)
    col_unique_e = _infer_err_col(df.columns, col_unique_m)
    col_n_e = _infer_err_col(df.columns, col_n_m)
    col_acc_e = _infer_err_col(df.columns, col_acc_m)
    col_kl_s2r_e = _infer_err_col(df.columns, col_kl_s2r_m)
    col_kl_r2s_e = _infer_err_col(df.columns, col_kl_r2s_m)
    col_js_tok_e = _infer_err_col(df.columns, col_js_tok_m)

    df["unique_correct_mean"] = pd.to_numeric(df[col_unique_m], errors="coerce")
    df["unique_correct_err"] = (
        pd.to_numeric(df[col_unique_e], errors="coerce")
        if col_unique_e
        else pd.Series(0.0, index=df.index)
    )
    n_total_m = pd.to_numeric(df[col_n_m], errors="coerce")
    n_total_e = (
        pd.to_numeric(df[col_n_e], errors="coerce") if col_n_e else pd.Series(0.0, index=df.index)
    )

    df["acc_mean"] = pd.to_numeric(df[col_acc_m], errors="coerce")
    df["acc_err"] = (
        pd.to_numeric(df[col_acc_e], errors="coerce")
        if col_acc_e
        else pd.Series(0.0, index=df.index)
    )
    df["KL_s2r_mean"] = pd.to_numeric(df[col_kl_s2r_m], errors="coerce")
    df["KL_s2r_err"] = (
        pd.to_numeric(df[col_kl_s2r_e], errors="coerce")
        if col_kl_s2r_e
        else pd.Series(0.0, index=df.index)
    )
    df["KL_r2s_mean"] = pd.to_numeric(df[col_kl_r2s_m], errors="coerce")
    df["KL_r2s_err"] = (
        pd.to_numeric(df[col_kl_r2s_e], errors="coerce")
        if col_kl_r2s_e
        else pd.Series(0.0, index=df.index)
    )
    df["JS_tok_mean"] = pd.to_numeric(df[col_js_tok_m], errors="coerce")
    df["JS_tok_err"] = (
        pd.to_numeric(df[col_js_tok_e], errors="coerce")
        if col_js_tok_e
        else pd.Series(0.0, index=df.index)
    )

    denom = n_total_m.replace(0, np.nan)
    df["normcov"] = df["unique_correct_mean"] / denom
    df["normcov_err"] = _ci95_ratio_err(
        df["unique_correct_mean"], df["unique_correct_err"], n_total_m, n_total_e
    )

    keys = df["exp"].apply(_expkey)
    df["method"] = keys.apply(lambda k: k.method)
    df["replay"] = keys.apply(lambda k: k.replay)

    if exclude_regex:
        df = df[~df["method"].astype(str).str.contains(exclude_regex, regex=True, na=False)].copy()

    if prefer_subm_ext:
        df = _prefer_subm_ext(df, exp_col="exp", replay_col="replay", method_col="method")

    out = df[
        [
            "replay",
            "method",
            "unique_correct_mean",
            "unique_correct_err",
            "normcov",
            "normcov_err",
            "acc_mean",
            "acc_err",
            "KL_s2r_mean",
            "KL_s2r_err",
            "KL_r2s_mean",
            "KL_r2s_err",
            "JS_tok_mean",
            "JS_tok_err",
        ]
    ]

    # Sorting
    replay_order = {"prt": 0, "hu_rp": 1, "subm": 2, "oracle": 3}
    method_order = {"TB": 0, "SubTB": 1, "RapTB": 2}
    out["_ro"] = out["replay"].map(lambda x: replay_order.get(str(x), 99))
    out["_mo"] = out["method"].map(lambda x: method_order.get(str(x), 99))
    out = out.sort_values(["_ro", "_mo"]).drop(columns=["_ro", "_mo"]).reset_index(drop=True)

    replay_name = {"prt": "PRT", "hu_rp": "Hu RP", "subm": "SubM", "oracle": "Oracle"}

    header = [
        "Replay",
        "Objective",
        "Unique$_\\checkmark$",
        "NormCov",
        "Acc",
        "KL($\\pi\\!\\to\\!p^*$)",
        "KL($p^*\\!\\to\\!\\pi$)",
        "JS$_{\\text{tok}}$",
    ]
    col_spec = "ll" + "c" * (len(header) - 2)

    lines: list[str] = []
    for replay, g in out.groupby("replay", sort=False):
        g = g.copy()
        n = len(g)
        for i, (_, r) in enumerate(g.iterrows()):
            left = ""
            if i == 0:
                left = f"\\multirow{{{n}}}{{*}}{{{replay_name.get(replay, replay)}}}"
            row = [
                left,
                str(r["method"]),
                _fmt_pm_fixed(r["unique_correct_mean"], r["unique_correct_err"], digits=1),
                _fmt_pm_fixed(r["normcov"], r["normcov_err"], digits=3),
                _fmt_pm_fixed(r["acc_mean"], r["acc_err"], digits=3),
                _fmt_pm_fixed(r["KL_s2r_mean"], r["KL_s2r_err"], digits=3),
                _fmt_pm_fixed(r["KL_r2s_mean"], r["KL_r2s_err"], digits=3),
                _fmt_pm_fixed(r["JS_tok_mean"], r["JS_tok_err"], digits=3),
            ]
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]

    latex = (
        "% Requires \\usepackage{booktabs,multirow}\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.08}\n"
        "\\caption{\\textbf{Expr24 results under four replay schemes.} "
        "For \\textbf{SubM}, we report the \\emph{ext-size} buffer variant for TB and SubTB for a fairer replay-strength comparison. "
        "NormCov $=\\frac{\\text{unique}_{\\checkmark}}{\\min(N,|\\mathcal{Y}^*|)}$ accounts for the sampling cap. "
        "JS$_{\\text{tok}}$ is the token/position-level JS divergence (original JS$_{pos}$).}\n"
        "\\label{tab:expr24_main_ci}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + " & ".join(header) + " \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    return out, latex


# -----------------------------
# Appendix table: per-length NormCov
# -----------------------------


def build_appendix_len_table(
    length_by_len_csv: str,
    exclude_regex: str = r"RootSubTBLogZ",
    prefer_subm_ext: bool = True,
    lengths: list[int] | None = None,
) -> tuple[pd.DataFrame, str]:
    lcov = pd.read_csv(length_by_len_csv)
    lcov["exp"] = lcov["experiment"].apply(_canon_exp_name)

    required = [
        "length",
        "unique_correct_by_len_mean",
        "unique_correct_by_len_err",
        "len_ref_mean",
        "gen_count_by_len_mean",
    ]
    for c in required:
        if c not in lcov.columns:
            raise ValueError(
                f"length-by-length CSV must contain '{c}'. Available: {list(lcov.columns)}"
            )

    keys = lcov["exp"].apply(_expkey)
    lcov["method"] = keys.apply(lambda k: k.method)
    lcov["replay"] = keys.apply(lambda k: k.replay)

    if exclude_regex:
        lcov = lcov[
            ~lcov["method"].astype(str).str.contains(exclude_regex, regex=True, na=False)
        ].copy()

    if prefer_subm_ext:
        tmp = lcov[["exp", "replay", "method"]].drop_duplicates()
        tmp = _prefer_subm_ext(tmp, exp_col="exp", replay_col="replay", method_col="method")
        keep = set(tmp["exp"].tolist())
        lcov = lcov[lcov["exp"].isin(keep)].copy()

    # Note: sampling cap is length-dependent (N_ell)

    all_lengths = sorted(lcov["length"].unique().tolist())
    if lengths is None:
        lengths = all_lengths
    else:
        lengths = [int(x) for x in lengths]

    rows = []
    for (replay, method, exp), g in lcov.groupby(["replay", "method", "exp"], sort=False):
        for _, r in g.iterrows():
            ell = int(r["length"])
            if ell not in lengths:
                continue
            ref = float(r["len_ref_mean"])
            n_len = float(r["gen_count_by_len_mean"])
            denom = min(n_len, ref) if (not np.isnan(n_len) and not np.isnan(ref)) else np.nan
            denom = denom if denom and denom > 0 else np.nan
            mean = float(r["unique_correct_by_len_mean"]) / denom if denom else np.nan
            err = float(r["unique_correct_by_len_err"]) / denom if denom else np.nan
            rows.append(
                {"replay": replay, "method": method, "length": ell, "mean": mean, "err": err}
            )

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No rows left for appendix table (check parsing/exclusion).")

    wide_mean = long.pivot_table(
        index=["replay", "method"], columns="length", values="mean", aggfunc="first"
    ).reindex(columns=lengths)
    wide_err = long.pivot_table(
        index=["replay", "method"], columns="length", values="err", aggfunc="first"
    ).reindex(columns=lengths)

    replay_order = {"prt": 0, "hu_rp": 1, "subm": 2, "oracle": 3}
    method_order = {"TB": 0, "SubTB": 1, "RapTB": 2}
    idx = wide_mean.index.to_frame(index=False)
    idx["_ro"] = idx["replay"].map(replay_order).fillna(99)
    idx["_mo"] = idx["method"].map(method_order).fillna(99)
    idx = idx.sort_values(["_ro", "_mo"]).drop(columns=["_ro", "_mo"])
    wide_mean = wide_mean.loc[pd.MultiIndex.from_frame(idx)]
    wide_err = wide_err.loc[wide_mean.index]

    replay_name = {"prt": "PRT", "hu_rp": "Hu RP", "subm": "SubM (ext-size)", "oracle": "Oracle"}
    header = ["Replay", "Objective"] + [f"$\\ell={ell}$" for ell in lengths]
    col_spec = "ll" + "c" * len(lengths)

    # Reference sizes (assumed constant)
    ref_sizes = (
        lcov.groupby("length", as_index=True)["len_ref_mean"].mean().reindex(lengths).to_dict()
    )
    ref_note = ", ".join(
        [
            f"$|\\mathcal{{Y}}^*_{{{ell}}}|={int(ref_sizes[ell])}$"
            for ell in lengths
            if ell in ref_sizes
        ]
    )

    lines: list[str] = []
    for replay, g in wide_mean.groupby(level=0, sort=False):
        methods = g.index.get_level_values(1).tolist()
        n = len(methods)
        for i, method in enumerate(methods):
            left = ""
            if i == 0:
                left = f"\\multirow{{{n}}}{{*}}{{{replay_name.get(replay, replay)}}}"
            vals = []
            for ell in lengths:
                m = wide_mean.loc[(replay, method), ell]
                e = wide_err.loc[(replay, method), ell]
                vals.append(_fmt_pm(m, e, digits=2))
            lines.append(" & ".join([left, str(method)] + vals) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]

    latex = (
        "% Requires \\usepackage{booktabs,multirow}\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.12}\n"
        "\\caption{Expr24 per-length normalized coverage. "
        "We report $\\mathrm{NormCov}_\\ell=\\frac{\\mathrm{unique}_{\\checkmark,\\ell}}{\\min(N_\\ell,|\\mathcal{Y}^*_\\ell|)}$ (mean$\\pm$95\\% CI), where $N_\\ell$ is the number of generated samples with length $\\ell$ (method-dependent). "
        + "Reference sizes: "
        + ref_note
        + ".}\n"
        "\\label{tab:expr24_normcov_by_len}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + " & ".join(header) + r" \\" + "\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    out = wide_mean.copy()
    out.columns = [f"ell_{c}" for c in out.columns]
    out = out.reset_index()
    return out, latex


# -----------------------------
# Diagnostics table: pterm/length
# -----------------------------


def build_pterm_diagnostic_table(
    coverage_csv: str,
    valid_ratio_csv: str,
    json_metrics_csv: str,
    exclude_regex: str = "",
    include_subm_methods: tuple[str, ...] = ("RapTB",),
    include_prt: bool = False,
) -> tuple[pd.DataFrame, str]:
    cov = pd.read_csv(coverage_csv)
    vr = pd.read_csv(valid_ratio_csv)
    jm = pd.read_csv(json_metrics_csv)

    for df in (cov, vr, jm):
        df["exp"] = df["experiment"].apply(_canon_exp_name)

    df = cov.merge(vr.drop(columns=["experiment"]), on="exp", how="left")
    df = df.merge(jm.drop(columns=["experiment"]), on="exp", how="left")

    col_acc = _pick_first_existing(df.columns, ["valid_ratio_mean", "acc_mean", "test_acc_mean"])
    col_cov = _pick_first_existing(df.columns, ["coverage_rate_mean", "coverage_rate"])
    col_logpterm = _pick_first_existing(
        df.columns, ["log_pterm_by_len_terminal_mean", "log_pterm_by_len_9_mean"]
    )
    col_logz = _pick_first_existing(df.columns, ["test_logZ_mean", "test_logZ"])

    needed = [("acc", col_acc), ("cov", col_cov), ("logpterm", col_logpterm), ("logz", col_logz)]
    miss = [name for name, col in needed if col is None]
    if miss:
        raise ValueError(
            f"Missing required columns for pterm table: {miss}. Available: {list(df.columns)}"
        )

    keys = df["exp"].apply(_expkey)
    df["method"] = keys.apply(lambda k: k.method)
    df["replay"] = keys.apply(lambda k: k.replay)

    if exclude_regex:
        df = df[~df["method"].astype(str).str.contains(exclude_regex, regex=True, na=False)].copy()

    if not include_prt:
        df = df[df["replay"] != "prt"].copy()

    if include_subm_methods:
        df = df[~((df["replay"] == "subm") & (~df["method"].isin(include_subm_methods)))].copy()
    else:
        df = df[df["replay"] != "subm"].copy()

    df["section"] = np.where(df["replay"] == "oracle", "oracle", "rp")
    df["method_label"] = df.apply(
        lambda r: f"{r['method']}+SubM" if r["replay"] == "subm" else str(r["method"]),
        axis=1,
    )

    out = df[["section", "method_label", col_acc, col_cov, col_logpterm, col_logz]].rename(
        columns={
            col_acc: "acc",
            col_cov: "cov",
            col_logpterm: "log_pterm",
            col_logz: "logZ",
        }
    )

    section_order = {"rp": 0, "oracle": 1}
    method_order = {"TB": 0, "SubTB": 1, "RapTB": 2, "RapTB+SubM": 3, "RootSubTBLogZ": 4}
    out["_so"] = out["section"].map(section_order).fillna(99)
    out["_mo"] = out["method_label"].map(method_order).fillna(99)
    out = out.sort_values(["_so", "_mo"]).drop(columns=["_so", "_mo"]).reset_index(drop=True)

    header = ["Method", "Acc", "Cov", "$\\log p_{\\text{term}}(\\tau)$", "$\\log Z$"]
    col_spec = "l" + "c" * (len(header) - 1)
    highlight = {"SubTB", "RootSubTBLogZ"}

    lines: list[str] = []
    for section, g in out.groupby("section", sort=False):
        section_title = "RP Replay" if section == "rp" else "Oracle Replay"
        lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\textbf{{{section_title}}}}}\\\\")
        for _, r in g.iterrows():
            prefix = "\\rowcolor{rowgray} " if r["method_label"] in highlight else ""
            row = [
                str(r["method_label"]),
                _fmt_num_or_dash(r["acc"], 3),
                _fmt_num_or_dash(r["cov"], 3),
                _fmt_num_or_dash(r["log_pterm"], 3),
                _fmt_num_or_dash(r["logZ"], 3),
            ]
            lines.append(prefix + " & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]

    latex = (
        "% Requires \\usepackage{booktabs,xcolor}\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{\\textbf{Termination/length calibration diagnostic on Expr24.} "
        "More negative values indicate overly suppressed termination, which directly reduces hit rate in variable-length generation. "
        "$\\log p_{\\text{term}}(\\tau)$ is termination log probability at terminal.}\n"
        "\\label{tab:expr24_json_metrics}\n\n"
        "\\colorlet{rowgray}{black!10}\n\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + " & ".join(header) + " \\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return out, latex


# -----------------------------
# Per-length pterm table (all methods)
# -----------------------------


def build_pterm_by_len_table(
    pterm_by_len_csv: str,
    exclude_regex: str = "",
    lengths: list[int] | None = None,
    include_methods: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    pt = pd.read_csv(pterm_by_len_csv)
    pt["exp"] = pt["experiment"].apply(_canon_exp_name)

    required = ["length", "log_pterm_by_len_mean"]
    for c in required:
        if c not in pt.columns:
            raise ValueError(
                f"pterm-by-length CSV must contain '{c}'. Available: {list(pt.columns)}"
            )

    keys = pt["exp"].apply(_expkey)
    pt["method"] = keys.apply(lambda k: k.method)
    pt["replay"] = keys.apply(lambda k: k.replay)

    if exclude_regex:
        pt = pt[~pt["method"].astype(str).str.contains(exclude_regex, regex=True, na=False)].copy()

    if include_methods is None:
        include_methods = ["TB", "SubTB", "RapTB", "RootSubTBLogZ"]
    pt = pt[pt["method"].isin(include_methods)].copy()

    if lengths is None:
        lengths = sorted(pt["length"].unique().tolist())
    else:
        lengths = [int(x) for x in lengths]

    if pt.empty:
        raise ValueError("No rows left for per-length pterm table (check exclude_regex).")

    pt = pt[pt["replay"].isin(["hu_rp", "oracle"])].copy()
    pt["section"] = np.where(pt["replay"] == "oracle", "Oracle", "RP")
    wide = pt.pivot_table(
        index=["section", "method"],
        columns="length",
        values="log_pterm_by_len_mean",
        aggfunc="first",
    ).reindex(columns=lengths)

    section_order = {"RP": 0, "Oracle": 1}
    method_order = {"TB": 0, "SubTB": 1, "RapTB": 2, "RootSubTBLogZ": 3}
    idx = wide.index.to_frame(index=False)
    idx["_so"] = idx["section"].map(section_order).fillna(99)
    idx["_mo"] = idx["method"].map(method_order).fillna(99)
    idx = idx.sort_values(["_so", "_mo"]).drop(columns=["_so", "_mo"])
    wide = wide.loc[pd.MultiIndex.from_frame(idx)]

    header = ["Replay", "Objective"] + [f"$\\ell={ell}$" for ell in lengths]
    col_spec = "ll" + "c" * len(lengths)

    lines: list[str] = []
    for section, g in wide.groupby(level=0, sort=False):
        methods = g.index.get_level_values(1).tolist()
        n = len(methods)
        for i, method in enumerate(methods):
            left = ""
            if i == 0:
                left = f"\\multirow{{{n}}}{{*}}{{{section}}}"
            vals = []
            for ell in lengths:
                v = wide.loc[(section, method), ell]
                vals.append(_fmt_num_or_dash(v, 3))
            lines.append(" & ".join([left, str(method)] + vals) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]

    latex = (
        "% Requires \\usepackage{booktabs,multirow}\n"
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\label{tab:expr24_pterm_by_len}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + " & ".join(header) + " \\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Per-length $\\log p_{\\text{term}}$ on Expr24.}\n"
        "\\end{table}\n"
    )

    out = wide.copy()
    out.columns = [f"ell_{c}" for c in out.columns]
    out = out.reset_index()
    return out, latex


# -----------------------------
# Per-length NormCov table (all methods, exclude regex)
# -----------------------------


def build_tb_logz_len_table(
    length_by_len_csv: str,
    valid_ratio_csv: str | None = None,
    exclude_regex: str = "",
    lengths: list[int] | None = None,
) -> tuple[pd.DataFrame, str]:
    lcov = pd.read_csv(length_by_len_csv)
    lcov["exp"] = lcov["experiment"].apply(_canon_exp_name)

    n_total_map = {}
    if valid_ratio_csv:
        vr = pd.read_csv(valid_ratio_csv)
        vr["exp"] = vr["experiment"].apply(_canon_exp_name)
        col_n = _pick_first_existing(vr.columns, ["n_total_mean", "n_total"])
        if col_n:
            n_total_map = dict(zip(vr["exp"].tolist(), vr[col_n].astype(float).tolist()))

    required = ["length", "unique_correct_by_len_mean", "len_ref_mean", "gen_count_by_len_mean"]
    for c in required:
        if c not in lcov.columns:
            raise ValueError(
                f"length-by-length CSV must contain '{c}'. Available: {list(lcov.columns)}"
            )

    keys = lcov["exp"].apply(_expkey)
    lcov["method"] = keys.apply(lambda k: k.method)
    lcov["replay"] = keys.apply(lambda k: k.replay)

    if exclude_regex:
        lcov = lcov[
            ~lcov["method"].astype(str).str.contains(exclude_regex, regex=True, na=False)
        ].copy()

    if lengths is None:
        lengths = sorted(lcov["length"].unique().tolist())
    else:
        lengths = [int(x) for x in lengths]

    if lcov.empty:
        raise ValueError("No rows left for TBLogZ length table (check include_regex).")

    if n_total_map:
        n_total = pd.Series(n_total_map)
    else:
        n_total = lcov.groupby("exp", as_index=True)["gen_count_by_len_mean"].sum().astype(float)
    rows = []
    for (replay, method, exp), g in lcov.groupby(["replay", "method", "exp"], sort=False):
        Nt = float(n_total.loc[exp]) if exp in n_total.index else np.nan
        for _, r in g.iterrows():
            ell = int(r["length"])
            if ell not in lengths:
                continue
            ref = float(r["len_ref_mean"])
            denom = min(Nt, ref) if (not np.isnan(Nt) and not np.isnan(ref)) else np.nan
            denom = denom if denom and denom > 0 else np.nan
            mean = float(r["unique_correct_by_len_mean"]) / denom if denom else np.nan
            rows.append({"replay": replay, "method": method, "length": ell, "mean": mean})

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No rows left for per-length NormCov table (check exclude_regex).")

    wide = long.pivot_table(
        index=["replay", "method"],
        columns="length",
        values="mean",
        aggfunc="first",
    ).reindex(columns=lengths)

    replay_order = {"hu_rp": 0, "subm": 1, "oracle": 2, "prt": 3}
    idx = wide.index.to_frame(index=False)
    idx["_ro"] = idx["replay"].map(replay_order).fillna(99)
    idx = idx.sort_values(["_ro", "method"]).drop(columns=["_ro"])
    wide = wide.loc[pd.MultiIndex.from_frame(idx)]

    header = ["Replay", "Method"] + [f"$\\ell={ell}$" for ell in lengths]
    col_spec = "ll" + "c" * len(lengths)

    replay_label = {
        "hu_rp": "RP Replay",
        "subm": "RP Replay",
        "oracle": "Oracle Replay",
        "prt": "RP Replay",
    }
    lines: list[str] = []
    for section, g in wide.groupby(level=0, sort=False):
        section_title = replay_label.get(section, section)
        lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\textbf{{{section_title}}}}}\\\\")
        for (_, method), row in g.iterrows():
            vals = [_fmt_num_or_dash(row.get(ell), 3) for ell in lengths]
            lines.append(" & ".join([section_title, str(method)] + vals) + " \\\\")
        lines.append("\\midrule")
    if lines:
        lines = lines[:-1]

    latex = (
        "% Requires \\usepackage{booktabs}\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        "\\caption{\\textbf{Per-length NormCov on Expr24.} "
        "$\\mathrm{NormCov}_\\ell=\\frac{\\mathrm{CovCount}_\\ell}{\\min(N,|\\mathcal{Y}^*_\\ell|)}$.}\n"
        "\\label{tab:expr24_normcov_by_len_all}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n" + " & ".join(header) + " \\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    out = wide.copy()
    out.columns = [f"ell_{c}" for c in out.columns]
    out = out.reset_index()
    return out, latex


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing Expr24 CSV files."
    )
    ap.add_argument(
        "--coverage_csv", type=str, default="", help="Override: coverage summary CSV path."
    )
    ap.add_argument(
        "--pos_div_csv", type=str, default="", help="Override: pos-divergence CSV path."
    )
    ap.add_argument(
        "--valid_ratio_csv", type=str, default="", help="Override: valid-ratio CSV path."
    )
    ap.add_argument(
        "--length_by_len_csv", type=str, default="", help="Override: length-by-length CSV path."
    )
    ap.add_argument(
        "--json_metrics_csv", type=str, default="", help="Override: json-metrics CSV path."
    )
    ap.add_argument(
        "--pterm_by_len_csv", type=str, default="", help="Override: pterm-by-length CSV path."
    )
    ap.add_argument(
        "--main_tex", type=str, default="", help="Optional path to write main table LaTeX."
    )
    ap.add_argument(
        "--main_ci_tex",
        type=str,
        default="",
        help="Optional path to write main table LaTeX with 95% CI.",
    )
    ap.add_argument(
        "--appendix_tex", type=str, default="", help="Optional path to write appendix table LaTeX."
    )
    ap.add_argument(
        "--pterm_tex",
        type=str,
        default="",
        help="Optional path to write pterm diagnostic table LaTeX.",
    )
    ap.add_argument(
        "--tb_logz_len_tex",
        type=str,
        default="",
        help="Optional path to write TBLogZ per-length Cov LaTeX.",
    )
    ap.add_argument(
        "--pterm_by_len_tex",
        type=str,
        default="",
        help="Optional path to write per-length pterm LaTeX.",
    )
    ap.add_argument(
        "--exclude_regex",
        type=str,
        default=r"RootSubTBLogZ",
        help="Regex to exclude methods. Use empty string to disable exclusion.",
    )
    ap.add_argument(
        "--diag_exclude_regex",
        type=str,
        default="",
        help="Regex to exclude methods in diagnostics tables.",
    )
    ap.add_argument(
        "--tb_logz_exclude_regex",
        type=str,
        default="",
        help="Regex to exclude methods in per-length NormCov table.",
    )
    ap.add_argument(
        "--no_prefer_subm_ext",
        action="store_true",
        help="Do not prefer SubM ext-size variant when multiple SubM variants exist.",
    )
    ap.add_argument(
        "--lengths",
        type=str,
        default="3,5,7,9",
        help="Comma-separated lengths for appendix table (default: 3,5,7,9).",
    )
    ap.add_argument(
        "--tb_logz_lengths",
        type=str,
        default="",
        help="Comma-separated lengths for TBLogZ table (defaults to --lengths).",
    )
    args = ap.parse_args()

    d = Path(args.input_dir)
    prefer_ext = not args.no_prefer_subm_ext

    if (
        args.coverage_csv
        and args.pos_div_csv
        and args.valid_ratio_csv
        and args.length_by_len_csv
        and args.json_metrics_csv
        and args.pterm_by_len_csv
    ):
        coverage = Path(args.coverage_csv)
        pos = Path(args.pos_div_csv)
        valid = Path(args.valid_ratio_csv)
        length_by_len = Path(args.length_by_len_csv)
        json_metrics = Path(args.json_metrics_csv)
        pterm_by_len = Path(args.pterm_by_len_csv)
    else:
        coverage, pos, valid, length_by_len, json_metrics, pterm_by_len = discover_csvs(d)

    lens = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    tb_logz_lens = lens
    if args.tb_logz_lengths:
        tb_logz_lens = [int(x.strip()) for x in args.tb_logz_lengths.split(",") if x.strip()]

    _, main_tex = build_main_table(
        coverage_csv=str(coverage),
        pos_div_csv=str(pos),
        valid_ratio_csv=str(valid),
        exclude_regex=args.exclude_regex,
        prefer_subm_ext=prefer_ext,
    )
    _, app_tex = build_appendix_len_table(
        length_by_len_csv=str(length_by_len),
        exclude_regex=args.exclude_regex,
        prefer_subm_ext=prefer_ext,
        lengths=lens,
    )

    _, pterm_tex = build_pterm_diagnostic_table(
        coverage_csv=str(coverage),
        valid_ratio_csv=str(valid),
        json_metrics_csv=str(json_metrics),
        exclude_regex=args.diag_exclude_regex,
    )

    _, tb_logz_tex = build_tb_logz_len_table(
        length_by_len_csv=str(length_by_len),
        valid_ratio_csv=str(valid),
        exclude_regex=args.tb_logz_exclude_regex,
        lengths=tb_logz_lens,
    )

    _, pterm_by_len_tex = build_pterm_by_len_table(
        pterm_by_len_csv=str(pterm_by_len),
        exclude_regex=args.diag_exclude_regex,
        lengths=tb_logz_lens,
    )

    _, main_ci_tex = build_main_table_ci(
        coverage_csv=str(coverage),
        pos_div_csv=str(pos),
        valid_ratio_csv=str(valid),
        exclude_regex=args.exclude_regex,
        prefer_subm_ext=prefer_ext,
    )

    if args.main_tex:
        Path(args.main_tex).write_text(main_tex, encoding="utf-8")
    if args.main_ci_tex:
        Path(args.main_ci_tex).write_text(main_ci_tex, encoding="utf-8")
    if args.appendix_tex:
        Path(args.appendix_tex).write_text(app_tex, encoding="utf-8")
    if args.pterm_tex:
        Path(args.pterm_tex).write_text(pterm_tex, encoding="utf-8")
    if args.tb_logz_len_tex:
        Path(args.tb_logz_len_tex).write_text(tb_logz_tex, encoding="utf-8")
    if args.pterm_by_len_tex:
        Path(args.pterm_by_len_tex).write_text(pterm_by_len_tex, encoding="utf-8")

    print("% ===== Expr24 MAIN TABLE =====")
    print(main_tex)
    print("% ===== Expr24 MAIN TABLE (MEAN +- 95% CI) =====")
    print(main_ci_tex)
    print("% ===== Expr24 APPENDIX (PER-LENGTH NORMCOV) =====")
    print(app_tex)
    print("% ===== Expr24 PTERM DIAGNOSTICS =====")
    print(pterm_tex)
    print("% ===== Expr24 PER-LENGTH NORMCOV (ALL METHODS) =====")
    print(tb_logz_tex)
    print("% ===== Expr24 PTERM PER-LENGTH =====")
    print(pterm_by_len_tex)


if __name__ == "__main__":
    main()
