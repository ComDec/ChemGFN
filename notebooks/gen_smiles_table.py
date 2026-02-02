#!/usr/bin/env python3

"""
Unified SMILES table generator.

Workflow:
  1) Uses draw_smiles.py presets (L10/L15) to build CSV tables via run_tables.
  2) Renders paper stress tables (with optional subset of methods).
  3) Renders appendix tables (all-length average + by-length + prefix-by-length).

This avoids manual toggling in draw_smiles.py when switching L10/L15.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import draw_smiles as ds
import gen_smiles_L15_table as l15
import gen_smiles_table_appendix as bylen
import gen_smiles_table_prefix_appendix as pref
import numpy as np
import pandas as pd

DEFAULT_PAPER_METHODS = ["TB", "SubTB", "RapTB", "RapTB+SubM"]


def _norm_method_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _display_method(name: str) -> str:
    s = str(name)
    s = s.replace("RapTB_SubM", "RapTB+SubM").replace("RapTB-SubM", "RapTB+SubM")
    s = s.replace("RapTB-MaxOnly", "RapTB-Max")
    s = s.replace("RapTB-SoftOnly", "RapTB-Soft")
    s = s.replace("SubTB_SubM", "SubTB+SubM").replace("TB_SubM", "TB+SubM")
    s = s.replace("_SubM", "+SubM").replace("-SubM", "+SubM")
    return s


def _pick_experiment_order(exps: Sequence[str]) -> list[str]:
    uniq = list(pd.unique(list(exps)))
    try:
        return bylen.pick_experiment_order(uniq)
    except Exception:
        return sorted(uniq)


def _format_num(v: float, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{float(v):.{digits}f}"


def _format_pm(mean: float, err: float | None, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and (math.isnan(mean) or math.isinf(mean))):
        return "--"
    if err is None or (isinstance(err, float) and (math.isnan(err) or math.isinf(err))):
        return _format_num(mean, digits=digits)
    return f"{_format_num(mean, digits=digits)}$\\pm${_format_num(err, digits=digits)}"


def _best_mask(values: dict[str, float], higher_is_better: bool) -> dict[str, bool]:
    vals = [
        v
        for v in values.values()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return {k: False for k in values}
    best = max(vals) if higher_is_better else min(vals)
    return {
        k: (v == best) if v is not None and not (isinstance(v, float) and math.isnan(v)) else False
        for k, v in values.items()
    }


def _bf(s: str) -> str:
    return f"\\textbf{{{s}}}"


def _select_rows(rows: list[l15.Row], methods: Sequence[str] | None) -> list[l15.Row]:
    if not methods:
        return rows
    want = {_norm_method_label(m) for m in methods}
    out = []
    for r in rows:
        if _norm_method_label(r.disp) in want or _norm_method_label(r.method_key) in want:
            out.append(r)
    return out


def render_stress_table(
    rows: list[l15.Row],
    table_label: str,
    caption: str,
    include_ci: bool = False,
    include_macro_fpdiv: bool = True,
    include_fpdiv: bool = True,
    include_tokdiv: bool = False,
    digits: int = 3,
) -> str:
    acc_vals = {r.method_key: r.acc[0] for r in rows}
    score_vals = {r.method_key: r.score[0] for r in rows}
    frac0_vals = {r.method_key: r.frac_bins.get("0-5", (0.0, 0.0))[0] for r in rows}
    frac11_vals = {r.method_key: r.frac_bins.get("11+", (0.0, 0.0))[0] for r in rows}
    surv_vals = {r.method_key: r.prefix_all.get("SurvEnd", (0.0, 0.0))[0] for r in rows}
    ent_vals = {r.method_key: r.prefix_all.get("Ent", (0.0, 0.0))[0] for r in rows}
    top1_vals = {r.method_key: r.prefix_all.get("Top1", (0.0, 0.0))[0] for r in rows}
    macro_fp_vals = {r.method_key: r.macro_fpdiv[0] for r in rows}
    fpdiv_vals = {r.method_key: r.fpdiv_all[0] for r in rows}

    best_acc = _best_mask(acc_vals, higher_is_better=True)
    best_score = _best_mask(score_vals, higher_is_better=True)
    best_frac0 = _best_mask(frac0_vals, higher_is_better=False)
    best_frac11 = _best_mask(frac11_vals, higher_is_better=True)
    best_surv = _best_mask(surv_vals, higher_is_better=True)
    best_ent = _best_mask(ent_vals, higher_is_better=True)
    best_top1 = _best_mask(top1_vals, higher_is_better=False)
    best_macro_fp = _best_mask(macro_fp_vals, higher_is_better=True)
    best_fpdiv = _best_mask(fpdiv_vals, higher_is_better=True)

    cols = [
        "Method",
        "Acc$\\uparrow$",
        "Score$\\uparrow$",
        "Len",
        "Frac(0--5)$\\downarrow$",
        "Frac(6--10)",
        "Frac(11+)$\\uparrow$",
        "SurvEnd$\\uparrow$",
        "Ent$\\uparrow$",
        "Top1$\\downarrow$",
    ]

    if include_macro_fpdiv:
        cols.append("MacroFPDiv$\\uparrow$")
    if include_tokdiv:
        cols.append("TokDiv$\\uparrow$")
    if include_fpdiv:
        cols.append("FPDiv$\\uparrow$")

    col_spec = "@{}l" + "c" * (len(cols) - 1) + "@{}"
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.08}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\vspace{-0.35em}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\")
    lines.append("\\midrule")

    def cell_val(v: tuple[float, float], digs: int = digits) -> str:
        return (
            _format_pm(v[0], v[1], digits=digs) if include_ci else _format_num(v[0], digits=digs)
        )

    for r in rows:
        cells = [_display_method(r.disp)]

        acc = cell_val(r.acc)
        if best_acc.get(r.method_key, False):
            acc = _bf(acc)
        cells.append(acc)

        score = cell_val(r.score)
        if best_score.get(r.method_key, False):
            score = _bf(score)
        cells.append(score)

        cells.append(cell_val(r.length_mean))

        f0 = cell_val(r.frac_bins.get("0-5", (0.0, 0.0)))
        if best_frac0.get(r.method_key, False):
            f0 = _bf(f0)
        cells.append(f0)

        cells.append(cell_val(r.frac_bins.get("6-10", (0.0, 0.0))))

        f11 = cell_val(r.frac_bins.get("11+", (0.0, 0.0)))
        if best_frac11.get(r.method_key, False):
            f11 = _bf(f11)
        cells.append(f11)

        surv = cell_val(r.prefix_all.get("SurvEnd", (0.0, 0.0)))
        if best_surv.get(r.method_key, False):
            surv = _bf(surv)
        cells.append(surv)

        ent = cell_val(r.prefix_all.get("Ent", (0.0, 0.0)))
        if best_ent.get(r.method_key, False):
            ent = _bf(ent)
        cells.append(ent)

        top1 = cell_val(r.prefix_all.get("Top1", (0.0, 0.0)))
        if best_top1.get(r.method_key, False):
            top1 = _bf(top1)
        cells.append(top1)

        if include_macro_fpdiv:
            mfp = cell_val(r.macro_fpdiv)
            if best_macro_fp.get(r.method_key, False):
                mfp = _bf(mfp)
            cells.append(mfp)
        if include_tokdiv:
            cells.append(cell_val(r.tokdiv))
        if include_fpdiv:
            fpd = cell_val(r.fpdiv_all)
            if best_fpdiv.get(r.method_key, False):
                fpd = _bf(fpd)
            cells.append(fpd)

        lines.append(" & ".join(cells) + " \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.25em}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_all_length_table(
    main_df: pd.DataFrame,
    buckets: ds.Buckets,
    table_label: str,
    caption: str,
    digits_prob: int = 3,
    digits_len: int = 2,
) -> str:
    if main_df is None or len(main_df) == 0:
        return ""

    df = main_df.copy().reset_index()
    if "experiment" not in df.columns:
        return ""
    df["experiment"] = df["experiment"].apply(l15.clean_experiment_name)

    exps = _pick_experiment_order(df["experiment"].dropna().astype(str).tolist())

    bin_labels = list(buckets.len_bin_labels or [])
    frac_cols = []
    for b in bin_labels:
        if f"len_frac[{b}]_mean" in df.columns:
            frac_cols.append((b, "len_frac"))
        elif f"len_valid_frac[{b}]_mean" in df.columns:
            frac_cols.append((b, "len_valid_frac"))

    headers = [
        "Method",
        "Acc",
        "Score",
        "Div",
        "FPDiv",
        "Len$_{\\mu}$",
        "Len$_{50}$",
        "Len$_{90}$",
    ]
    headers += [f"Frac[{b}]" for b, _ in frac_cols]

    colspec = "l" + "c" * (len(headers) - 1)
    lines = []
    lines.append("\\begin{table*}[!htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.06}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\")
    lines.append("\\midrule")

    idx = df.set_index("experiment")
    for e in exps:
        if e not in idx.index:
            continue
        r = idx.loc[e]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        row = [_display_method(e)]
        row.append(_format_pm(r.get("acc_mean"), r.get("acc_err"), digits_prob))
        row.append(
            _format_pm(r.get("score_mean_valid_mean"), r.get("score_mean_valid_err"), digits_prob)
        )
        row.append(
            _format_pm(r.get("diversity_valid_mean"), r.get("diversity_valid_err"), digits_prob)
        )
        row.append(_format_pm(r.get("fp_div_mean"), r.get("fp_div_err"), digits_prob))
        row.append(_format_pm(r.get("len_mean_mean"), r.get("len_mean_err"), digits_len))
        row.append(_format_pm(r.get("len_p50_mean"), r.get("len_p50_err"), digits_len))
        row.append(_format_pm(r.get("len_p90_mean"), r.get("len_p90_err"), digits_len))
        for b, prefix in frac_cols:
            row.append(
                _format_pm(r.get(f"{prefix}[{b}]_mean"), r.get(f"{prefix}[{b}]_err"), digits_prob)
            )
        lines.append(" & ".join(row) + " \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}%")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_main_summary_table(
    main_df: pd.DataFrame,
    methods: Sequence[str],
    table_label: str,
    lmax: str,
) -> str:
    if main_df is None or len(main_df) == 0:
        return ""

    df = main_df.copy().reset_index()
    if "experiment" not in df.columns:
        return ""
    df["experiment"] = df["experiment"].apply(l15.clean_experiment_name)

    idx = df.set_index("experiment")
    rows = []
    for m in methods:
        if m not in idx.index:
            continue
        r = idx.loc[m]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append((m, r))

    if not rows:
        return ""

    acc_vals = {m: float(r.get("acc_mean", np.nan)) for m, r in rows}
    score_vals = {m: float(r.get("score_mean_valid_mean", np.nan)) for m, r in rows}
    ent_vals = {m: float(r.get("diversity_valid_mean", np.nan)) for m, r in rows}
    fp_vals = {m: float(r.get("fp_div_mean", np.nan)) for m, r in rows}

    best_acc = _best_mask(acc_vals, higher_is_better=True)
    best_score = _best_mask(score_vals, higher_is_better=True)
    best_ent = _best_mask(ent_vals, higher_is_better=True)
    best_fp = _best_mask(fp_vals, higher_is_better=True)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.08}")
    lines.append(
        "\\caption{"
        "\\textbf{SMILES generation performance.} "
        "Unless specified, all metrics are computed on valid samples. "
        f"\\texttt{{Len}} denotes the mean token length of valid samples ($L_{{\\max}}={lmax}$)."
        "}"
    )
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\vspace{-0.35em}")
    lines.append("\\begin{tabular}{@{}lccccc@{}}")
    lines.append("\\toprule")
    lines.append(
        "Method & Acc $\\uparrow$ & Score $\\uparrow$ & Entropy $\\uparrow$ & FPDiv $\\uparrow$ & Len \\"
    )
    lines.append("\\midrule")

    for m, r in rows:
        acc = _format_num(r.get("acc_mean"), 3)
        score = _format_num(r.get("score_mean_valid_mean"), 3)
        ent = _format_num(r.get("diversity_valid_mean"), 3)
        fp = _format_num(r.get("fp_div_mean"), 3)
        ln = _format_num(r.get("len_mean_mean"), 3)

        if best_acc.get(m, False):
            acc = _bf(acc)
        if best_score.get(m, False):
            score = _bf(score)
        if best_ent.get(m, False):
            ent = _bf(ent)
        if best_fp.get(m, False):
            fp = _bf(fp)

        disp = _display_method(m)
        if disp == "RapTB+SubM":
            disp = "RapTB + SubM"
        lines.append(f"{disp} & {acc} & {score} & {ent} & {fp} & {ln} \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_rtb_ablation_table(
    main_df: pd.DataFrame,
    methods: Sequence[str],
    table_label: str,
    lmax: str,
) -> str:
    if main_df is None or len(main_df) == 0:
        return ""

    df = main_df.copy().reset_index()
    if "experiment" not in df.columns:
        return ""

    def _extract_exp_name(x: str) -> str:
        s = str(x).strip()
        m = re.findall(r"'([^']+)'", s)
        if m:
            return m[0]
        return s

    df["experiment"] = df["experiment"].apply(_extract_exp_name)

    idx = df.set_index("experiment")
    rows = []
    for m in methods:
        if m not in idx.index:
            continue
        r = idx.loc[m]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append((m, r))

    if not rows:
        return ""

    acc_vals = {m: float(r.get("acc_mean", np.nan)) for m, r in rows}
    score_vals = {m: float(r.get("score_mean_valid_mean", np.nan)) for m, r in rows}
    ent_vals = {m: float(r.get("diversity_valid_mean", np.nan)) for m, r in rows}
    fp_vals = {m: float(r.get("fp_div_mean", np.nan)) for m, r in rows}

    best_acc = _best_mask(acc_vals, higher_is_better=True)
    best_score = _best_mask(score_vals, higher_is_better=True)
    best_ent = _best_mask(ent_vals, higher_is_better=True)
    best_fp = _best_mask(fp_vals, higher_is_better=True)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.08}")
    lines.append(
        "\\caption{"
        "\\textbf{RapTB mixing ablation.} "
        "Metrics are computed on valid samples unless noted. "
        f"\\texttt{{Len}} is the mean token length of valid samples ($L_{{\\max}}={lmax}$)."
        "}"
    )
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\vspace{-0.35em}")
    lines.append("\\begin{tabular}{@{}lccccc@{}}")
    lines.append("\\toprule")
    lines.append(
        "Method & Acc $\\uparrow$ & Score $\\uparrow$ & Entropy $\\uparrow$ & FPDiv $\\uparrow$ & Len \\"
    )
    lines.append("\\midrule")

    for m, r in rows:
        acc = _format_num(r.get("acc_mean"), 3)
        score = _format_num(r.get("score_mean_valid_mean"), 3)
        ent = _format_num(r.get("diversity_valid_mean"), 3)
        fp = _format_num(r.get("fp_div_mean"), 3)
        ln = _format_num(r.get("len_mean_mean"), 3)

        if best_acc.get(m, False):
            acc = _bf(acc)
        if best_score.get(m, False):
            score = _bf(score)
        if best_ent.get(m, False):
            ent = _bf(ent)
        if best_fp.get(m, False):
            fp = _bf(fp)

        disp = _display_method(m)
        lines.append(f"{disp} & {acc} & {score} & {ent} & {fp} & {ln} \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_appendix_by_length(
    metrics_df: pd.DataFrame,
    rename_subm: bool = True,
    label_prefix: str = "",
) -> str:
    if metrics_df is None or len(metrics_df) == 0:
        return ""
    df = metrics_df.reset_index()
    df = bylen.canonicalize_xy_columns(df)
    df["experiment"] = df["experiment"].apply(l15.clean_experiment_name)
    exps = _pick_experiment_order(df["experiment"].dropna().astype(str).tolist())
    display_map = {e: _display_method(e) if rename_subm else e for e in exps}

    parts = []
    for label, keys, caption in bylen.DEFAULT_GROUPS:
        full_label = f"{label_prefix}{label}"
        part = bylen.make_table_latex(
            df=df,
            exp_col="experiment",
            len_col="length",
            exps=exps,
            metric_keys=keys,
            label=full_label,
            caption=caption,
            display_name_map=display_map,
            clearpage_after=False,
        )
        if part.strip():
            parts.append(part)
    return "\n\n".join(parts)


def render_prefix_by_length_grouped(
    prefix_df: pd.DataFrame,
    groups: Sequence[tuple[str, Sequence[str], str]],
    rename_subm: bool = True,
) -> str:
    if prefix_df is None or len(prefix_df) == 0:
        return ""
    df = prefix_df.reset_index()
    df = pref.canonicalize_xy_columns(df)
    df["experiment"] = df["experiment"].apply(l15.clean_experiment_name)

    metric_keys = ["survival", "entropy", "eff", "top1", "unique_rate"]
    parts = []
    for label, methods, caption in groups:
        exps = [m for m in methods if m in set(df["experiment"].astype(str))]
        if not exps:
            continue
        display_map = {e: _display_method(e) if rename_subm else e for e in exps}
        part = pref.make_table(
            df=df,
            exp_col="experiment",
            len_col="k",
            exps=exps,
            metric_keys=metric_keys,
            label=label,
            caption=caption,
            display_name_map=display_map,
            clearpage_after=False,
        )
        if part.strip():
            parts.append(part)
    return "\n\n".join(parts)


def build_rows_for_preset(
    preset: str,
    output_dir: Path,
    error_mode: str,
    kmax_prefix_avg: int | None,
) -> tuple[list[l15.Row], dict[str, pd.DataFrame]]:
    exps = ds.get_exps(preset)
    buckets = ds.get_buckets(preset)
    tables = ds.run_tables(
        exps=exps,
        buckets=buckets,
        keys=ds.JsonKeys(),
        samples_cfg=ds.SamplesConfig(),
        error_mode=error_mode,
        output_root=output_dir,
    )

    main_df = tables.get("main_table")
    prefix_bucket_df = tables.get("prefix_bucket_table")
    prefix_by_k_df = tables.get("prefix_by_length_table")
    samples_by_length_df = tables.get("samples_by_length")

    if main_df is None:
        main_df = pd.DataFrame()
    else:
        if "experiment" not in main_df.columns:
            main_df = main_df.reset_index()

    if prefix_bucket_df is not None and "experiment" not in prefix_bucket_df.columns:
        prefix_bucket_df = prefix_bucket_df.reset_index()

    if prefix_by_k_df is not None and "experiment" not in prefix_by_k_df.columns:
        prefix_by_k_df = prefix_by_k_df.reset_index()

    if samples_by_length_df is not None and "experiment" not in samples_by_length_df.columns:
        samples_by_length_df = samples_by_length_df.reset_index()

    if kmax_prefix_avg is None:
        kmax_prefix_avg = 10 if preset == "L10" else 15

    rows = l15.build_rows(
        main_df=main_df,
        prefix_bucket_df=prefix_bucket_df,
        prefix_by_k_df=prefix_by_k_df,
        samples_by_length_df=samples_by_length_df,
        k_max_for_prefix_avg=kmax_prefix_avg,
    )
    return rows, tables


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SMILES LaTeX tables from draw_smiles presets."
    )
    ap.add_argument(
        "--preset", action="append", choices=sorted(ds.EXPS_PRESETS.keys()), default=None
    )
    ap.add_argument("--all-presets", action="store_true", help="Generate tables for all presets.")
    ap.add_argument("--out-root", default="smiles_tables", help="Output root directory.")
    ap.add_argument("--paper-methods", default=",".join(DEFAULT_PAPER_METHODS))
    ap.add_argument("--no-paper-subset", action="store_true", help="Skip subset paper table.")
    ap.add_argument("--no-all-table", action="store_true", help="Skip all-methods stress table.")
    ap.add_argument("--no-appendix", action="store_true", help="Skip appendix tables.")
    ap.add_argument("--kmax-prefix-avg", type=int, default=None)
    ap.add_argument("--error-mode", default="ci95", choices=["std", "sem", "ci95", "none"])
    args = ap.parse_args()

    presets = args.preset or []
    if args.all_presets or not presets:
        presets = sorted(ds.EXPS_PRESETS.keys())

    paper_methods = [m.strip() for m in args.paper_methods.split(",") if m.strip()]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for preset in presets:
        output_dir = out_root / preset
        output_dir.mkdir(parents=True, exist_ok=True)

        rows, tables = build_rows_for_preset(
            preset=preset,
            output_dir=output_dir,
            error_mode=args.error_mode,
            kmax_prefix_avg=args.kmax_prefix_avg,
        )

        buckets = ds.get_buckets(preset)
        main_df = tables.get("main_table")
        metrics_df = tables.get("metrics_by_length")
        prefix_df = tables.get("prefix_by_length_table")

        tex_parts: list[str] = []
        tex_parts.append("% Auto-generated by gen_smiles_table.py")
        tex_parts.append("% Requires: \\usepackage{booktabs}")
        tex_parts.append("% Optional: \\usepackage{graphicx}  (for \\resizebox)")
        tex_parts.append("")

        lmax = "10" if preset == "L10" else "15"
        label_case = preset

        if not args.no_paper_subset:
            subset_rows = _select_rows(rows, paper_methods)
            caption = (
                f"\\textbf{{SMILES generation with extended horizon ($L_{{\\max}}={lmax}$).}} "
                "Frac columns are valid-only length fractions. "
                "Prefix metrics (SurvEnd, Ent, Top1) are computed on correct-only samples and averaged over $k$. "
                "MacroFPDiv is a length-balanced fingerprint diversity that averages FPDiv across bins (0--5, 6--10, 11+)."
            )
            tex_parts.append(
                render_stress_table(
                    rows=subset_rows,
                    table_label=f"tab:smiles_{label_case}_stress",
                    caption=caption,
                    include_ci=False,
                    include_macro_fpdiv=True,
                    include_fpdiv=True,
                    include_tokdiv=False,
                )
            )
            tex_parts.append("")

        if main_df is not None and len(main_df) > 0:
            main_methods = ["TB", "SubTB", "RapTB", "RapTB_SubM"]
            main_label = "tab:smiles_main" if preset == "L10" else "tab:smiles_main_L15"
            tex_parts.append(
                render_main_summary_table(
                    main_df=main_df,
                    methods=main_methods,
                    table_label=main_label,
                    lmax=lmax,
                )
            )
            tex_parts.append("")

            if preset == "L10":
                ablation_methods = ["RapTB", "RapTB-MaxOnly", "RapTB-SoftOnly"]
                tex_parts.append(
                    render_rtb_ablation_table(
                        main_df=main_df,
                        methods=ablation_methods,
                        table_label="tab:smiles_rtb_ablation",
                        lmax=lmax,
                    )
                )
                tex_parts.append("")

        if not args.no_all_table:
            caption_all = (
                f"\\textbf{{SMILES generation with extended horizon ($L_{{\\max}}={lmax}$).}} "
                "All methods in this preset; same columns as the main stress table."
            )
            tex_parts.append(
                render_stress_table(
                    rows=rows,
                    table_label=f"tab:smiles_{label_case}_stress_all",
                    caption=caption_all,
                    include_ci=False,
                    include_macro_fpdiv=True,
                    include_fpdiv=True,
                    include_tokdiv=False,
                )
            )
            tex_parts.append("")

        if not args.no_appendix:
            if preset == "L10" and main_df is not None and len(main_df) > 0:
                caption = (
                    f"All-length averaged SMILES performance and induced length distribution ($L_{{\\max}}={lmax}$; "
                    "mean$\\pm$95\\% CI over runs). "
                    "Acc is computed over all samples; Score/Div/FPDiv are computed on valid samples; "
                    "Frac[$\\cdot$] is computed over all samples when available."
                )
                tex_parts.append(
                    render_all_length_table(
                        main_df=main_df,
                        buckets=buckets,
                        table_label="tab:smiles_all_length_avg",
                        caption=caption,
                    )
                )
                tex_parts.append("")

            if metrics_df is not None:
                bylen_prefix = "smiles_L15_" if preset == "L15" else ""
                appendix_bylen = render_appendix_by_length(
                    metrics_df,
                    rename_subm=True,
                    label_prefix=bylen_prefix,
                )
                if appendix_bylen.strip():
                    tex_parts.append(appendix_bylen)
                    tex_parts.append("")

            if prefix_df is not None:
                if preset == "L10":
                    prefix_groups = [
                        (
                            "prefix_bylen_base",
                            ["TB", "SubTB", "RapTB"],
                            "Prefix statistics by depth. Mean$\\pm$95\\% CI.",
                        ),
                        (
                            "prefix_bylen_subm",
                            ["TB_SubM", "SubTB_SubM", "RapTB_SubM"],
                            "Prefix statistics by depth (Continue). Mean$\\pm$95\\% CI.",
                        ),
                    ]
                else:
                    prefix_groups = [
                        (
                            "smiles_L15_prefix_base",
                            ["TB", "SubTB"],
                            "Prefix statistics by depth on SMILES generation (mean$\\pm$95\\% CI, $L_{\\max}=15$): TB vs. SubTB.",
                        ),
                        (
                            "smiles_L15_prefix_raptb",
                            ["RapTB", "RapTB_SubM"],
                            "Prefix statistics by depth on SMILES generation (mean$\\pm$95\\% CI, $L_{\\max}=15$): RapTB vs. RapTB+SubM.",
                        ),
                    ]

                appendix_prefix = render_prefix_by_length_grouped(
                    prefix_df, prefix_groups, rename_subm=True
                )
                if appendix_prefix.strip():
                    tex_parts.append(appendix_prefix)
                    tex_parts.append("")

        out_path = output_dir / "tables.tex"
        out_path.write_text("\n".join(tex_parts).rstrip() + "\n", encoding="utf-8")
        print(f"[ok] {preset}: wrote {out_path}")


if __name__ == "__main__":
    main()
