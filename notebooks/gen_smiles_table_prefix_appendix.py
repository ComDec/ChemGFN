#!/usr/bin/env python3
"""
csv2latex_prefix_bylen.py

Generate ICML-style LaTeX tables for "prefix by length" CSV like:
  prefix_by_length_table.csv

CSV expected columns (at least):
  experiment, k,
  survival_mean, survival_err,
  entropy_mean, entropy_err,
  eff_mean, eff_err,
  top1_mean, top1_err,
  unique_rate_mean, unique_rate_err,
  unique_mean, unique_err,
  n_mean, n_err

Usage:
  python csv2latex_prefix_bylen.py -i prefix_by_length_table.csv -o prefix_bylen.tex
  python csv2latex_prefix_bylen.py -i prefix_by_length_table.csv --rename-subm > prefix_bylen.tex
"""

from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ---------------------------
# Formatting helpers
# ---------------------------


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def canonicalize_xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse col_x/col_y duplicates by choosing the one with more non-NaN entries.
    """
    cols = list(df.columns)
    base_to_variants: dict[str, list[str]] = {}

    for c in cols:
        m = re.match(r"^(.*)_(x|y)$", c)
        if m:
            base_to_variants.setdefault(m.group(1), []).append(c)

    for base, variants in base_to_variants.items():
        candidates = []
        if base in df.columns:
            candidates.append(base)
        candidates.extend(variants)

        best = max(candidates, key=lambda k: int(df[k].notna().sum()))
        df[base] = df[best]

        for k in candidates:
            if k != base and k in df.columns:
                df.drop(columns=[k], inplace=True)

    return df


def pick_experiment_order(exps: list[str]) -> list[str]:
    """
    Heuristic ordering for common baselines.
    """

    def key(e: str) -> tuple[int, str]:
        if e == "TB":
            return (0, e)
        if e == "SubTB":
            return (1, e)
        if e == "RapTB":
            return (2, e)
        if e in ("RapTB_SubM", "RapTB-SubM", "RapTB+SubM"):
            return (3, e)
        return (99, e)

    return sorted(exps, key=key)


def _isnan(x) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def format_number(v: float, digits: int) -> str:
    s = f"{v:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _coerce_scalar(value: float | pd.Series | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        series = value.dropna()
        if len(series) == 0:
            return None
        return float(series.mean())
    try:
        return float(value)
    except Exception:
        return None


def format_pm(
    mean: float | pd.Series | None,
    err: float | pd.Series | None,
    metric_key: str,
) -> str:
    mean_val = _coerce_scalar(mean)
    err_val = _coerce_scalar(err)
    if mean_val is None or (isinstance(mean_val, float) and math.isnan(mean_val)):
        return "--"
    if err_val is None or (isinstance(err_val, float) and math.isnan(err_val)):
        return format_number(float(mean_val), 3 if abs(mean_val) < 1 else 2)

    mean = float(mean_val)
    err = float(err_val)
    if mean is None or _isnan(mean):
        return "--"
    if err is None or _isnan(err):
        # mean only
        return format_number(float(mean), 3)

    mean = float(mean)
    err = float(err)

    # counts
    if metric_key in ("unique", "n"):
        mean_str = (
            str(int(round(mean))) if abs(mean - round(mean)) < 1e-6 else format_number(mean, 1)
        )
        err_str = format_number(err, 1)
        return f"{mean_str}$\\pm${err_str}"

    # probability-like
    if metric_key in ("survival", "top1", "unique_rate"):
        return f"{format_number(mean, 3)}$\\pm${format_number(err, 3)}"

    # entropy: usually a few decimals
    if metric_key == "entropy":
        return f"{format_number(mean, 3)}$\\pm${format_number(err, 3)}"

    # eff: scale can be large; keep 2 decimals
    if metric_key == "eff":
        return f"{format_number(mean, 2)}$\\pm${format_number(err, 2)}"

    # fallback
    return f"{format_number(mean, 3)}$\\pm${format_number(err, 3)}"


# ---------------------------
# Metrics specs for this CSV
# ---------------------------

# key -> (pretty_name, mean_col, err_col)
METRICS: OrderedDict[str, tuple[str, str, str]] = OrderedDict(
    [
        ("survival", ("Survival", "survival_mean", "survival_err")),
        ("entropy", ("Entropy", "entropy_mean", "entropy_err")),
        ("eff", ("Eff", "eff_mean", "eff_err")),
        ("top1", ("Top1", "top1_mean", "top1_err")),
        ("unique_rate", ("UniqueRate", "unique_rate_mean", "unique_rate_err")),
        ("unique", ("Unique", "unique_mean", "unique_err")),
        ("n", ("N", "n_mean", "n_err")),
    ]
)

DEFAULT_GROUPS: list[tuple[str, list[str], str]] = [
    (
        "prefix_bylen_main",
        ["survival", "entropy", "eff", "top1", "unique_rate"],
        r"\textbf{Prefix statistics by length.} Mean$\pm$95\% CI.",
    ),
    (
        "prefix_bylen_counts",
        ["unique", "n"],
        r"\textbf{Prefix counts by length.} Mean$\pm$95\% CI.",
    ),
]


# ---------------------------
# LaTeX table generation
# ---------------------------


def make_table(
    df: pd.DataFrame,
    exp_col: str,
    len_col: str,
    exps: list[str],
    metric_keys: list[str],
    label: str,
    caption: str,
    display_name_map: dict[str, str],
    clearpage_after: bool = True,
) -> str:
    # resolve available metrics
    active: list[tuple[str, str, str, str]] = []  # (key, pretty, mean_col, err_col)
    for k in metric_keys:
        if k not in METRICS:
            continue
        pretty, mc, ec = METRICS[k]
        if mc in df.columns and df[mc].notna().any():
            active.append((k, pretty, mc, ec if ec in df.columns else ""))

    if not active:
        return ""

    idx = df.set_index([exp_col, len_col])
    lengths = sorted(df[len_col].dropna().unique().tolist(), key=lambda x: int(x))
    n_metrics = len(active)

    colspec = "c" + ("c" * (len(exps) * n_metrics))

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.06}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # header row 1
    h1 = [r"$k$"]
    for e in exps:
        disp = latex_escape(display_name_map.get(e, e))
        h1.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{{disp}}}")
    lines.append(" & ".join(h1) + r" \\")
    lines.append(r"\midrule")

    # header row 2
    h2 = [""]
    for _ in exps:
        for _, pretty, _, _ in active:
            h2.append(latex_escape(pretty))
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # body
    for k_val in lengths:
        row = [str(int(k_val)) if float(k_val).is_integer() else str(k_val)]
        for e in exps:
            for key, _, mc, ec in active:
                mean = float("nan")
                err = None
                try:
                    mean = idx.loc[(e, k_val), mc]
                except Exception:
                    pass
                if ec:
                    try:
                        err = idx.loc[(e, k_val), ec]
                    except Exception:
                        err = float("nan")
                row.append(format_pm(mean, err, key))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}%")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\end{table*}")

    if clearpage_after:
        lines.append("")
        lines.append(r"\clearpage")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", "-i", required=True, help="Input CSV path (prefix_by_length_table.csv)."
    )
    ap.add_argument(
        "--output", "-o", default="", help="Output .tex path. If omitted, print to stdout."
    )
    ap.add_argument("--exp-col", default="experiment", help="Experiment/model column name.")
    ap.add_argument("--len-col", default="k", help="Prefix length column name (default: k).")
    ap.add_argument(
        "--no-clearpage", action="store_true", help="Do not insert \\clearpage between tables."
    )
    ap.add_argument(
        "--rename-subm", action="store_true", help="Display RapTB_SubM as RapTB+SubM (cosmetic)."
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = canonicalize_xy_columns(df)

    if args.exp_col not in df.columns or args.len_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.exp_col}' and '{args.len_col}'.")

    exps = [str(x) for x in df[args.exp_col].dropna().unique().tolist()]
    exps = pick_experiment_order(exps)

    display_name_map: dict[str, str] = {}
    for e in exps:
        disp = e
        if args.rename_subm:
            disp = disp.replace("RapTB_SubM", "RapTB+SubM").replace("RapTB-SubM", "RapTB+SubM")
        display_name_map[e] = disp

    tex_parts: list[str] = []
    tex_parts.append("% Auto-generated by csv2latex_prefix_bylen.py")
    tex_parts.append("% Requires: \\usepackage{booktabs}")
    tex_parts.append("% Optional: \\usepackage{graphicx}  (for \\resizebox)")
    tex_parts.append("")

    clearpage_after = not args.no_clearpage

    for label, keys, caption in DEFAULT_GROUPS:
        part = make_table(
            df=df,
            exp_col=args.exp_col,
            len_col=args.len_col,
            exps=exps,
            metric_keys=keys,
            label=label,
            caption=caption,
            display_name_map=display_name_map,
            clearpage_after=clearpage_after,
        )
        if part.strip():
            tex_parts.append(part)

    out = "\n".join(tex_parts).rstrip() + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
