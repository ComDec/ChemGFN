#!/usr/bin/env python3
"""
csv2latex_bylen_valid.py
Generate ICML-friendly appendix LaTeX tables from a "metrics_by_length.csv"-style file.

Default behavior:
- rows: length
- columns: experiments/models
- cells: mean ± 95% CI (from *_mean / *_err)
- valid-only metrics (Score_valid, Frac_valid, Count_valid, Diversity_valid, FPDiv, Uniqueness, etc.)
- skip metrics that are missing entirely; show '--' for missing cells
- no bolding

Usage:
  python csv2latex_bylen_valid.py --input metrics_by_length.csv --output appendix_bylen.tex
  python csv2latex_bylen_valid.py --input metrics_by_length.csv > appendix_bylen.tex
"""

from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ---------------------------
# Pretty names / presets
# ---------------------------

# Metric key -> (pretty header, mean_col_regex, err_col_regex)
# The regexes are matched against canonicalized column names (see canonicalize_xy_columns).
METRIC_SPECS: OrderedDict[str, tuple[str, str, str]] = OrderedDict(
    [
        ("acc", ("Acc", r"^acc_mean$", r"^acc_err$")),
        ("score_valid", ("Score", r"^score_mean_valid_mean$", r"^score_mean_valid_err$")),
        ("frac_valid", ("Frac", r"^frac_valid_mean$", r"^frac_valid_err$")),
        ("count_valid", ("Count", r"^count_valid_mean$", r"^count_valid_err$")),
        ("div_valid", ("Div", r"^diversity_valid_mean$", r"^diversity_valid_err$")),
        ("fpdiv", ("FPDiv", r"^fp_div_mean$", r"^fp_div_err$")),
        ("uniq_str", ("UniqStr", r"^unique_str_mean$", r"^unique_str_err$")),
        ("uniq_mol", ("UniqMol", r"^unique_mol_mean$", r"^unique_mol_err$")),
        ("uniqrate_str", ("UniqRateStr", r"^unique_rate_str_mean$", r"^unique_rate_str_err$")),
        ("uniqrate_mol", ("UniqRateMol", r"^unique_rate_mol_mean$", r"^unique_rate_mol_err$")),
    ]
)

DEFAULT_GROUPS: list[tuple[str, list[str], str]] = [
    (
        "bylen_valid_core",
        ["acc", "score_valid", "frac_valid", "count_valid"],
        r"Per-length valid-only core metrics (mean$\pm$95\% CI).",
    ),
    (
        "bylen_valid_div",
        ["div_valid", "fpdiv"],
        r"Per-length valid-only diversity metrics (mean$\pm$95\% CI).",
    ),
    (
        "bylen_valid_uniq",
        ["uniq_str", "uniq_mol", "uniqrate_str", "uniqrate_mol"],
        r"Per-length valid-only uniqueness metrics (mean$\pm$95\% CI).",
    ),
]


# ---------------------------
# Helpers
# ---------------------------


def latex_escape(s: str) -> str:
    # Minimal escaping for table headers.
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
    If a CSV was produced by merges, you often get col_x/col_y duplicates.
    This collapses them into a single base column by choosing the variant
    with the most non-NaN entries.
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

        # pick the best filled candidate
        best = max(candidates, key=lambda k: int(df[k].notna().sum()))
        df[base] = df[best]

        # drop all other candidates except base
        for k in candidates:
            if k != base and k in df.columns:
                df.drop(columns=[k], inplace=True)

    return df


def pick_experiment_order(exps: list[str]) -> list[str]:
    """
    Heuristic ordering: TB, SubTB, RapTB first; then their +/-SubM variants; else alphabetical.
    """

    def key(e: str) -> tuple[int, str]:
        # Display mapping handled later; here we sort raw ids
        base_rank = 99
        if e == "TB":
            base_rank = 0
        elif e == "SubTB":
            base_rank = 1
        elif e == "RapTB":
            base_rank = 2
        elif e.startswith("TB"):
            base_rank = 10
        elif e.startswith("SubTB"):
            base_rank = 11
        elif e.startswith("RapTB"):
            base_rank = 12
        return (base_rank, e)

    return sorted(exps, key=key)


def format_number(v: float, digits: int) -> str:
    fmt = f"{{:.{digits}f}}"
    s = fmt.format(v)
    # trim trailing zeros for cleaner LaTeX (but keep at least one digit after dot if dot exists)
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
        # mean only
        return format_number(float(mean_val), 3 if abs(mean_val) < 1 else 2)

    mean = float(mean_val)
    err = float(err_val)

    # Heuristics by metric type
    if any(k in metric_key for k in ["count", "uniq"]):
        # counts: integer-ish mean, 1 decimal err
        mean_str = (
            str(int(round(mean))) if abs(mean - round(mean)) < 1e-6 else format_number(mean, 1)
        )
        err_str = format_number(err, 1)
        return f"{mean_str}$\\pm${err_str}"

    if (
        "fpdiv" in metric_key
        or "acc" in metric_key
        or "frac" in metric_key
        or "rate" in metric_key
        or "score" in metric_key
    ):
        # probabilities/scores: 3 decimals typically
        d = 3 if max(abs(mean), abs(err)) < 1 else 3
        return f"{format_number(mean, d)}$\\pm${format_number(err, d)}"

    if "div" in metric_key:
        # diversity: 2 decimals
        return f"{format_number(mean, 2)}$\\pm${format_number(err, 2)}"

    # fallback
    d = 3 if abs(mean) < 1 else 2
    return f"{format_number(mean, d)}$\\pm${format_number(err, d)}"


def find_column(df: pd.DataFrame, pattern: str) -> str | None:
    r = re.compile(pattern)
    for c in df.columns:
        if r.match(c):
            return c
    return None


def metric_available(df: pd.DataFrame, mean_col: str) -> bool:
    if mean_col not in df.columns:
        return False
    col = df[mean_col]
    if isinstance(col, pd.DataFrame):
        return bool(col.notna().any().any())
    return bool(col.notna().any())


# ---------------------------
# LaTeX generation
# ---------------------------


def make_table_latex(
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
    # Resolve columns for each metric
    metrics: list[tuple[str, str, str]] = []  # (key, mean_col, err_col)
    headers: list[str] = []

    for k in metric_keys:
        if k not in METRIC_SPECS:
            continue
        pretty, mean_pat, err_pat = METRIC_SPECS[k]
        mean_col = find_column(df, mean_pat)
        err_col = find_column(df, err_pat)
        if mean_col is None:
            continue
        if not metric_available(df, mean_col):
            continue
        # err_col can be None; we still allow mean-only (but most should have err)
        metrics.append((k, mean_col, err_col if err_col is not None else ""))
        headers.append(pretty)

    if not metrics:
        return ""  # nothing to print

    # Lookup for quick access
    idx = df.set_index([exp_col, len_col])

    lengths = sorted(df[len_col].dropna().unique().tolist(), key=lambda x: int(x))
    n_metrics = len(metrics)

    # Column spec: 1 + (n_exps * n_metrics)
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

    # Header row 1: experiments
    h1 = [r"$L$"]
    for e in exps:
        disp = latex_escape(display_name_map.get(e, e))
        h1.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{{disp}}}")
    lines.append(" & ".join(h1) + r" \\")
    lines.append(r"\midrule")

    # Header row 2: metric names repeated per experiment
    h2 = [""]
    for _ in exps:
        for name in headers:
            h2.append(latex_escape(name))
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # Body
    for L in lengths:
        row = [str(int(L)) if float(L).is_integer() else str(L)]
        for e in exps:
            for k, mc, ec in metrics:
                try:
                    mean = idx.loc[(e, L), mc]
                except KeyError:
                    mean = float("nan")
                err = None
                if ec:
                    try:
                        err = idx.loc[(e, L), ec]
                    except KeyError:
                        err = float("nan")
                row.append(format_pm(mean, err, k))
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
    ap.add_argument("--input", "-i", required=True, help="Input CSV file path.")
    ap.add_argument(
        "--output", "-o", default="", help="Output .tex file path. If omitted, print to stdout."
    )
    ap.add_argument(
        "--exp-col",
        default="experiment",
        help="Experiment/model column name (default: experiment).",
    )
    ap.add_argument("--len-col", default="length", help="Length column name (default: length).")
    ap.add_argument(
        "--no-clearpage", action="store_true", help="Do not insert \\clearpage between tables."
    )
    ap.add_argument(
        "--rename-subm",
        action="store_true",
        help="Display '-SubM' as '+SubM' in headers (purely cosmetic).",
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
            disp = disp.replace("-SubM", "+SubM")
        display_name_map[e] = disp

    tex_parts: list[str] = []
    tex_parts.append("% Auto-generated by csv2latex_bylen_valid.py")
    tex_parts.append("% Requires: \\usepackage{booktabs}")
    tex_parts.append("% Optional: \\usepackage{graphicx}  (for \\resizebox)")
    tex_parts.append("")

    clearpage_after = not args.no_clearpage

    for label, keys, caption in DEFAULT_GROUPS:
        part = make_table_latex(
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
