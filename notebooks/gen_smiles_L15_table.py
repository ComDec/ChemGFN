#!/usr/bin/env python3

"""
Generate LaTeX tables for SMILES L=15 results from a results folder.

Expected files (by name) under --dir:
  - main_table.csv
  - prefix_bucket_table.csv
  - prefix_by_length_table.csv
  - samples_by_length.csv
  - samples_by_length_merged.csv (optional; not required)

Outputs (ICML paper-friendly):
  - Paper "stress-test" table: collapse/coverage-first column order (means by default)
    * Acc / Score / Len
    * Length-mass bins Frac(0-5), Frac(6-10), Frac(11+)
    * Prefix-collapse summaries (SurvEnd, Ent, Top1)
    * TokDiv/FPDiv are OPTIONAL and (when included) marked as length-confounded (*)
    * Optional length-balanced FP diversity (MacroFPDiv) to avoid length confounding

  - Appendix tables (mean ± 95% CI):
    * Stress-test table with CI
    * Prefix buckets
    * Prefix-by-k (sidewaystable*)
    * Samples-by-length (sidewaystable*)
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Utilities
# -------------------------


def read_csv_maybe(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def clean_experiment_name(x) -> str:
    """
    Normalize experiment labels like:
      "('TB',)" -> "TB"
      "('RapTB_SubM',)" -> "RapTB_SubM"
      "('RapTB-SubM',)" -> "RapTB_SubM"
      "RapTB" -> "RapTB"
    """
    s = str(x).strip()
    if "SubTB" in s:
        if "SubM" in s:
            return "SubTB_SubM"
        return "SubTB"
    if "RapTB" in s:
        if "SubM" in s:
            return "RapTB_SubM"
        return "RapTB"
    if "TB" in s:
        if "SubM" in s:
            return "TB_SubM"
        return "TB"
    tokens = re.findall(r"[A-Za-z0-9_]+", s)
    return tokens[0] if tokens else s


def display_method(name: str) -> str:
    mapping = {
        "TB": "TB",
        "SubTB": "SubTB",
        "RapTB": "RapTB",
        "RapTB_12_8_mix": "RapTB",
        "RapTB_SubM": "RapTB+SubM",
        "RapTBSubM": "RapTB+SubM",
        "RapTB_SubM_": "RapTB+SubM",
    }
    return mapping.get(name, name)


def method_sort_key(name: str) -> int:
    order = ["TB", "SubTB", "RapTB", "RapTB_SubM", "RapTB+SubM"]
    dn = display_method(name)
    if dn == "RapTB+SubM":
        dn = "RapTB_SubM"
    try:
        return order.index(dn)
    except ValueError:
        return 999


def parse_frac_bins_from_main(main_df: pd.DataFrame) -> list[str]:
    """
    Find all bins in columns like len_valid_frac[0-2]_mean.
    Return sorted bin strings like ["0-2","3-5",...]
    """
    pat = re.compile(r"^len_valid_frac\[(.+?)\]_mean$")
    bins = []
    for c in main_df.columns:
        m = pat.match(c)
        if m:
            bins.append(m.group(1))

    def key(b: str) -> tuple[int, int]:
        mm = re.match(r"(\d+)\s*-\s*(\d+|\+)", b)
        if not mm:
            return (10**9, 10**9)
        lo = int(mm.group(1))
        hi = mm.group(2)
        hi_val = 10**9 if hi == "+" else int(hi)
        return (lo, hi_val)

    return sorted(set(bins), key=key)


def parse_bin_range(bin_str: str) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\s*-\s*(\d+|\+)", bin_str)
    if not m:
        return None
    lo = int(m.group(1))
    hi_raw = m.group(2)
    hi = 10**9 if hi_raw == "+" else int(hi_raw)
    return lo, hi


def fmt_num(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "0"
    return f"{float(x):.{digits}f}"


def fmt_pm(mean: float, err: float, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and (np.isnan(mean) or np.isinf(mean))):
        mean = 0.0
    if err is None or (isinstance(err, float) and (np.isnan(err) or np.isinf(err))):
        err = 0.0
    return f"{float(mean):.{digits}f} $\\pm$ {float(err):.{digits}f}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def _is_bad(x: float) -> bool:
    return x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))


# -------------------------
# Length bins for stress table
# -------------------------

# Paper-facing bins (coverage / early-termination)
TARGET_FRAC_BINS: list[tuple[str, int, int]] = [
    ("0-5", 0, 5),
    ("6-10", 6, 10),
    ("11+", 11, 10**9),
]


def aggregate_frac_bins_for_row(
    row: pd.Series,
    source_bins: list[str],
    target_bins: list[tuple[str, int, int]] = TARGET_FRAC_BINS,
) -> dict[str, tuple[float, float]]:
    """
    Merge fine-grained frac bins into coarser ones.
    Errors are combined via root-sum-square (conservative).
    """
    agg: dict[str, tuple[float, float]] = {name: (0.0, 0.0) for name, _, _ in target_bins}
    for b in source_bins:
        rng = parse_bin_range(b)
        if rng is None:
            continue
        lo, hi = rng
        mcol = f"len_valid_frac[{b}]_mean"
        ecol = f"len_valid_frac[{b}]_err"
        m = float(row.get(mcol, 0.0))
        e = float(row.get(ecol, 0.0))
        if _is_bad(m):
            m = 0.0
        if _is_bad(e):
            e = 0.0
        for name, t_lo, t_hi in target_bins:
            if lo >= t_lo and hi <= t_hi:
                cur_m, cur_e = agg[name]
                agg[name] = (cur_m + m, float(np.sqrt(cur_e**2 + e**2)))
                break
    return agg


def compute_prefix_overall_means(
    prefix_by_k_df: pd.DataFrame,
    k_max: int = 10,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Aggregate prefix metrics across all prefix positions (k=1..k_max).
    NOTE: Here we compute a simple mean across k, and RMS for err across k (conservative).
    """
    pbk = prefix_by_k_df.copy()
    pbk["method_key"] = pbk["experiment"].apply(clean_experiment_name)
    if "k" in pbk.columns:
        pbk = pbk[pbk["k"].between(1, k_max)]

    lookup: dict[str, dict[str, tuple[float, float]]] = {}
    metric_map = {
        "SurvEnd": "survival",
        "Ent": "entropy",
        "Eff": "eff",
        "Top1": "top1",
    }

    for mk, g in pbk.groupby("method_key"):
        res: dict[str, tuple[float, float]] = {}
        for out_key, base in metric_map.items():
            mcol = f"{base}_mean"
            ecol = f"{base}_err"
            means = g[mcol].astype(float).tolist() if mcol in g.columns else []
            errs = g[ecol].astype(float).tolist() if ecol in g.columns else []
            means = [0.0 if _is_bad(v) else float(v) for v in means]
            errs = [0.0 if _is_bad(v) else float(v) for v in errs]
            mean_val = float(np.mean(means)) if means else 0.0
            err_val = float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
            res[out_key] = (mean_val, err_val)
        lookup[mk] = res
    return lookup


def compute_macro_fpdiv(
    samples_by_length_df: pd.DataFrame,
    bins: list[tuple[str, int, int]] = TARGET_FRAC_BINS,
    fp_col_mean: str = "fp_div_mean",
    fp_col_err: str = "fp_div_err",
) -> dict[str, tuple[float, float]]:
    """
    Length-balanced fingerprint diversity:
      1) For each method and each bin, average FPDiv@Len over lengths inside the bin.
      2) Average across bins with equal weight (macro-average).
    This avoids TB looking "fine" just because it only generates short sequences.

    Error: conservative RMS over (bin-level errs) then /sqrt(#bins) is too optimistic,
    so we instead take RMS across bins (no /sqrt) as a robust uncertainty proxy.
    """
    df = samples_by_length_df.copy()
    df["method_key"] = df["experiment"].apply(clean_experiment_name)

    out: dict[str, tuple[float, float]] = {}
    for mk, g in df.groupby("method_key"):
        bin_means: list[float] = []
        bin_errs: list[float] = []
        for _name, lo, hi in bins:
            gg = g[(g["length"] >= lo) & (g["length"] <= hi if hi < 10**9 else True)]
            if len(gg) == 0:
                continue
            means = gg[fp_col_mean].astype(float).tolist() if fp_col_mean in gg.columns else []
            errs = gg[fp_col_err].astype(float).tolist() if fp_col_err in gg.columns else []
            means = [0.0 if _is_bad(v) else float(v) for v in means]
            errs = [0.0 if _is_bad(v) else float(v) for v in errs]
            if len(means) == 0:
                continue
            bm = float(np.mean(means))
            be = float(np.sqrt(np.mean(np.square(errs)))) if len(errs) else 0.0
            bin_means.append(bm)
            bin_errs.append(be)

        if len(bin_means) == 0:
            out[mk] = (0.0, 0.0)
        else:
            macro = float(np.mean(bin_means))
            macro_err = float(np.sqrt(np.mean(np.square(bin_errs)))) if len(bin_errs) else 0.0
            out[mk] = (macro, macro_err)
    return out


# -------------------------
# Data extraction
# -------------------------


@dataclass
class Row:
    method_key: str
    disp: str
    acc: tuple[float, float]
    score: tuple[float, float]
    tokdiv: tuple[float, float]
    fpdiv_all: tuple[float, float]
    macro_fpdiv: tuple[float, float]
    length_mean: tuple[float, float]
    frac_bins: dict[str, tuple[float, float]]  # "0-5"/"6-10"/"11+"
    prefix_all: dict[str, tuple[float, float]]  # SurvEnd/Ent/Eff/Top1


def build_rows(
    main_df: pd.DataFrame,
    prefix_bucket_df: pd.DataFrame | None,
    prefix_by_k_df: pd.DataFrame | None,
    samples_by_length_df: pd.DataFrame | None,
    wanted_methods: list[str] | None = None,
    prefix_bucket_name: str = "long",
    k_max_for_prefix_avg: int = 10,
) -> list[Row]:
    main_df = main_df.copy()
    main_df["method_key"] = main_df["experiment"].apply(clean_experiment_name)

    if wanted_methods is None:
        wanted_methods = sorted(main_df["method_key"].unique().tolist(), key=method_sort_key)

    source_frac_bins = parse_frac_bins_from_main(main_df)

    # Prefix fallback: bucket table (short/mid/long)
    pb_lookup = {}
    if prefix_bucket_df is not None:
        pb = prefix_bucket_df.copy()
        pb["method_key"] = pb["experiment"].apply(clean_experiment_name)
        for mk, g in pb.groupby("method_key"):
            sel = g[g["bucket"] == prefix_bucket_name]
            if len(sel) == 0:
                continue
            pb_lookup[mk] = sel.iloc[0].to_dict()

    # Preferred prefix summary: average across k
    prefix_overall_lookup: dict[str, dict[str, tuple[float, float]]] = {}
    if prefix_by_k_df is not None:
        prefix_overall_lookup = compute_prefix_overall_means(
            prefix_by_k_df, k_max=k_max_for_prefix_avg
        )

    # FPDiv overall (sample-weighted, potentially length-confounded): from samples_by_length.csv if present
    fp_all_lookup = {}
    macro_fp_lookup = {}
    if samples_by_length_df is not None:
        sb = samples_by_length_df.copy()
        sb["method_key"] = sb["experiment"].apply(clean_experiment_name)
        for mk, g in sb.groupby("method_key"):
            r0 = g.iloc[0].to_dict()
            fp_all_lookup[mk] = r0
        macro_fp_lookup = compute_macro_fpdiv(samples_by_length_df)

    rows: list[Row] = []
    for mk in wanted_methods:
        sub = main_df[main_df["method_key"] == mk]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]

        acc = (float(r.get("acc_mean", 0.0)), float(r.get("acc_err", 0.0)))
        score = (
            float(r.get("score_mean_valid_mean", 0.0)),
            float(r.get("score_mean_valid_err", 0.0)),
        )
        tokdiv = (
            float(r.get("diversity_valid_mean", 0.0)),
            float(r.get("diversity_valid_err", 0.0)),
        )
        length_mean = (float(r.get("len_mean_mean", 0.0)), float(r.get("len_mean_err", 0.0)))

        # FPDiv (all valid, sample-weighted; confounded by length distribution)
        fp_all_mean = (
            float(fp_all_lookup.get(mk, {}).get("fp_div_mean_all_mean", 0.0))
            if mk in fp_all_lookup
            else 0.0
        )
        fp_all_err = (
            float(fp_all_lookup.get(mk, {}).get("fp_div_mean_all_err", 0.0))
            if mk in fp_all_lookup
            else 0.0
        )
        fpdiv_all = (fp_all_mean, fp_all_err)

        # Macro FPDiv (length-balanced)
        macro_fpdiv = macro_fp_lookup.get(mk, (0.0, 0.0))

        # Frac bins (valid-only)
        frac_bins = aggregate_frac_bins_for_row(r, source_frac_bins, target_bins=TARGET_FRAC_BINS)

        # Prefix summary (correct-only)
        prefix_all: dict[str, tuple[float, float]] = {}
        if mk in prefix_overall_lookup:
            prefix_all = prefix_overall_lookup[mk]
        elif mk in pb_lookup:
            d = pb_lookup[mk]
            prefix_all["SurvEnd"] = (
                float(d.get("survival_end_mean", 0.0)),
                float(d.get("survival_end_err", 0.0)),
            )
            prefix_all["Ent"] = (
                float(d.get("entropy_mean", 0.0)),
                float(d.get("entropy_err", 0.0)),
            )
            prefix_all["Eff"] = (float(d.get("eff_mean", 0.0)), float(d.get("eff_err", 0.0)))
            prefix_all["Top1"] = (float(d.get("top1_mean", 0.0)), float(d.get("top1_err", 0.0)))
        else:
            prefix_all = {
                "SurvEnd": (0.0, 0.0),
                "Ent": (0.0, 0.0),
                "Eff": (0.0, 0.0),
                "Top1": (0.0, 0.0),
            }

        rows.append(
            Row(
                method_key=mk,
                disp=display_method(mk),
                acc=acc,
                score=score,
                tokdiv=tokdiv,
                fpdiv_all=fpdiv_all,
                macro_fpdiv=macro_fpdiv,
                length_mean=length_mean,
                frac_bins=frac_bins,
                prefix_all=prefix_all,
            )
        )

    rows = sorted(rows, key=lambda x: method_sort_key(x.method_key))
    return rows


# -------------------------
# LaTeX generation (ICML-style stress table)
# -------------------------


def _best_mask(
    values: dict[str, float],
    higher_is_better: bool = True,
    atol: float = 1e-12,
) -> dict[str, bool]:
    if not values:
        return {}
    vals = [v for v in values.values() if not _is_bad(v)]
    if not vals:
        return {k: False for k in values}
    best = max(vals) if higher_is_better else min(vals)
    out = {}
    for k, v in values.items():
        if _is_bad(v):
            out[k] = False
        else:
            out[k] = abs(v - best) <= atol
    return out


def _bf(s: str) -> str:
    return f"\\textbf{{{s}}}"


def latex_stress_table_icml(
    rows: list[Row],
    table_label: str = "tab:smiles_L15_stress",
    include_ci: bool = False,
    digits: int = 3,
    use_table_star: bool = True,
    include_tokfp: bool = True,
    include_macro_fpdiv: bool = True,
    caption: str | None = None,
) -> str:
    """
    ICML-friendly L=15 stress-test table:
      - Collapse/coverage-first column order
      - Optional TokDiv/FPDiv placed at the end and marked as length-confounded (*)
      - Optional MacroFPDiv (length-balanced)
      - Boldface best-in-column (Score, Frac(11+), SurvEnd, Ent, Top1 (min), Frac(0-5) (min))
    """
    env = "table*" if use_table_star else "table"

    if caption is None:
        extra = (
            "TokDiv/FPDiv are computed on valid samples but can be confounded by the length distribution; "
            "we therefore prioritize coverage and prefix-collapse diagnostics, and report length-conditioned plots in Fig.~\\ref{fig:smiles_length_breakdown_L15}."
        )
        caption = (
            "\\textbf{Long-horizon SMILES stress test (max len $=15$; mean over seeds).} "
            "Frac columns are valid-only length fractions. "
            "Prefix metrics (SurvEnd, Ent, Top1) are computed on correct-only samples and averaged over prefix lengths ($k=1$--$10$). "
            + extra
        )

    # Collect for bolding (means only)
    score_vals = {r.method_key: r.score[0] for r in rows}
    frac0_vals = {r.method_key: r.frac_bins.get("0-5", (0.0, 0.0))[0] for r in rows}
    frac11_vals = {r.method_key: r.frac_bins.get("11+", (0.0, 0.0))[0] for r in rows}
    surv_vals = {r.method_key: r.prefix_all.get("SurvEnd", (0.0, 0.0))[0] for r in rows}
    ent_vals = {r.method_key: r.prefix_all.get("Ent", (0.0, 0.0))[0] for r in rows}
    top1_vals = {r.method_key: r.prefix_all.get("Top1", (0.0, 0.0))[0] for r in rows}

    best_score = _best_mask(score_vals, higher_is_better=True)
    best_frac0 = _best_mask(
        frac0_vals, higher_is_better=False
    )  # lower early-termination fraction is better
    best_frac11 = _best_mask(frac11_vals, higher_is_better=True)
    best_surv = _best_mask(surv_vals, higher_is_better=True)
    best_ent = _best_mask(ent_vals, higher_is_better=True)
    best_top1 = _best_mask(top1_vals, higher_is_better=False)  # lower is better

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

    if include_tokfp:
        cols += ["TokDiv$^{*}\\uparrow$", "FPDiv$^{*}\\uparrow$"]

    # Column spec (ICML wide table)
    col_spec = "@{}l" + "c" * (len(cols) - 1) + "@{}"

    out = []
    out.append(f"\\begin{{{env}}}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{3.6pt}")
    out.append("\\renewcommand{\\arraystretch}{1.08}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{table_label}}}")
    out.append("\\vspace{-0.35em}")
    out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    out.append("\\toprule")
    out.append(" & ".join(cols) + " \\\\")
    out.append("\\midrule")

    for r in rows:
        cells = [latex_escape(r.disp)]

        def cell_val(mean_err: tuple[float, float], digs: int = digits) -> str:
            return (
                fmt_pm(mean_err[0], mean_err[1], digits=digs)
                if include_ci
                else fmt_num(mean_err[0], digits=digs)
            )

        # Acc
        cells.append(cell_val(r.acc))
        # Score (bold best)
        s = cell_val(r.score)
        if best_score.get(r.method_key, False):
            s = _bf(s)
        cells.append(s)
        # Len
        cells.append(cell_val(r.length_mean))

        # Frac bins
        f0 = cell_val(r.frac_bins.get("0-5", (0.0, 0.0)))
        if best_frac0.get(r.method_key, False):
            f0 = _bf(f0)
        cells.append(f0)

        cells.append(cell_val(r.frac_bins.get("6-10", (0.0, 0.0))))

        f11 = cell_val(r.frac_bins.get("11+", (0.0, 0.0)))
        if best_frac11.get(r.method_key, False):
            f11 = _bf(f11)
        cells.append(f11)

        # Prefix summaries
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

        # MacroFPDiv (length-balanced)
        if include_macro_fpdiv:
            cells.append(cell_val(r.macro_fpdiv))

        # TokDiv / FPDiv (length-confounded; keep but do not bold)
        if include_tokfp:
            cells.append(cell_val(r.tokdiv))
            cells.append(cell_val(r.fpdiv_all))

        out.append(" & ".join(cells) + " \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}")

    # footnote for * metrics
    if include_tokfp:
        out.append("\\vspace{0.25em}")
        out.append(
            "{\\footnotesize $^{*}$TokDiv/FPDiv are valid-only aggregate metrics and can be length-confounded; "
            "see length-conditioned breakdown in Fig.~\\ref{fig:smiles_length_breakdown_L15}.}"
        )

    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


# -------------------------
# Appendix tables (kept as before, but call the new stress-table generator)
# -------------------------


def latex_appendix_prefix_buckets(
    prefix_bucket_df: pd.DataFrame,
    table_label: str = "tab:smiles_L15_prefix_buckets",
    digits: int = 3,
) -> str:
    pb = prefix_bucket_df.copy()
    pb["method_key"] = pb["experiment"].apply(clean_experiment_name)
    bucket_order = ["short", "mid", "long"]
    pb["bucket_order"] = pb["bucket"].apply(
        lambda x: bucket_order.index(x) if x in bucket_order else 999
    )
    pb = pb.sort_values(["method_key", "bucket_order"])
    methods = sorted(pb["method_key"].unique().tolist(), key=method_sort_key)

    out = []
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{4.2pt}")
    out.append("\\renewcommand{\\arraystretch}{1.05}")
    out.append(
        "\\caption{\\textbf{Prefix-bucket diagnostics (correct-only; mean $\\pm$ 95\\% CI).} "
        "Buckets are short ($k{=}1$--$3$), mid ($k{=}4$--$7$), and long ($k{=}8$--$10$). "
        "Higher SurvEnd/Ent/Eff and lower Top1 indicate less prefix collapse.}"
    )
    out.append(f"\\label{{{table_label}}}")
    out.append("\\vspace{-0.35em}")
    out.append("\\begin{tabular}{@{}lcccccc@{}}")
    out.append("\\toprule")
    out.append(
        "Method & Bucket & SurvEnd$\\uparrow$ & Ent$\\uparrow$ & Eff$\\uparrow$ & Top1$\\downarrow$ & $n_{\\mathrm{end}}$ \\\\"
    )
    out.append("\\midrule")

    for mk in methods:
        g = pb[pb["method_key"] == mk].sort_values("bucket_order")
        first = True
        for _, rr in g.iterrows():
            method_cell = latex_escape(display_method(mk)) if first else ""
            first = False
            bucket = rr["bucket"]

            survend = fmt_pm(
                rr.get("survival_end_mean", 0.0), rr.get("survival_end_err", 0.0), digits=digits
            )
            ent = fmt_pm(rr.get("entropy_mean", 0.0), rr.get("entropy_err", 0.0), digits=digits)
            eff = fmt_pm(rr.get("eff_mean", 0.0), rr.get("eff_err", 0.0), digits=digits)
            top1 = fmt_pm(rr.get("top1_mean", 0.0), rr.get("top1_err", 0.0), digits=digits)

            n_end_m = rr.get("n_end_mean", 0.0)
            n_end_e = rr.get("n_end_err", 0.0)
            n_end = fmt_pm(n_end_m, n_end_e, digits=1)

            out.append(
                f"{method_cell} & {bucket} & {survend} & {ent} & {eff} & {top1} & {n_end} \\\\"
            )
        out.append("\\midrule")

    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out)


def latex_appendix_prefix_by_k(
    prefix_by_k_df: pd.DataFrame,
    methods: list[str] | None = None,
    k_max: int = 10,
    table_label: str = "tab:smiles_L15_prefix_by_k",
    digits: int = 3,
) -> str:
    df = prefix_by_k_df.copy()
    df["method_key"] = df["experiment"].apply(clean_experiment_name)
    if methods is None:
        methods = sorted(df["method_key"].unique().tolist(), key=method_sort_key)
    methods = [m for m in methods if m in df["method_key"].unique()]
    df = df[df["k"].between(1, k_max)].copy()

    metrics = [
        ("survival", "Surv$\\uparrow$"),
        ("entropy", "Ent$\\uparrow$"),
        ("eff", "Eff$\\uparrow$"),
        ("top1", "Top1$\\downarrow$"),
        ("unique", "Unique$\\uparrow$"),
        ("unique_rate", "UniqueRate$\\uparrow$"),
        ("n", "$n$"),
    ]

    out = []
    out.append("\\begin{sidewaystable*}[t]")
    out.append("\\centering")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{2.6pt}")
    out.append("\\renewcommand{\\arraystretch}{1.05}")
    out.append(
        "\\caption{\\textbf{Per-position prefix-collapse metrics (correct-only; mean $\\pm$ 95\\% CI).} "
        "Reported for each prefix position $k$ up to $k=10$.}"
    )
    out.append(f"\\label{{{table_label}}}")
    out.append("\\vspace{-0.35em}")

    subcols_per_method = len(metrics)
    col_spec = (
        "@{}r|" + ("".join(["c" * subcols_per_method + "|" for _ in methods])).rstrip("|") + "@{}"
    )
    out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    out.append("\\toprule")

    top = ["$k$"]
    for m in methods:
        top.append(
            f"\\multicolumn{{{subcols_per_method}}}{{c|}}{{{latex_escape(display_method(m))}}}"
        )
    top[-1] = top[-1].replace("{c|}", "{c}")
    out.append(" & ".join(top) + " \\\\")
    out.append("\\cmidrule(lr){2-%d}" % (1 + subcols_per_method * len(methods)))

    second = [""]
    for _m in methods:
        second += [lab for _, lab in metrics]
    out.append(" & ".join(second) + " \\\\")
    out.append("\\midrule")

    for k in range(1, k_max + 1):
        row = [str(k)]
        for m in methods:
            sub = df[(df["method_key"] == m) & (df["k"] == k)]
            if len(sub) == 0:
                row += ["0"] * len(metrics)
                continue
            rr = sub.iloc[0]
            for key, _lab in metrics:
                mean = rr.get(f"{key}_mean", 0.0)
                err = rr.get(f"{key}_err", 0.0)
                if key in ["unique", "n"]:
                    row.append(fmt_pm(mean, err, digits=1))
                else:
                    row.append(fmt_pm(mean, err, digits=digits))
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{sidewaystable*}")
    return "\n".join(out)


def latex_appendix_samples_by_length(
    samples_by_length_df: pd.DataFrame,
    methods: list[str] | None = None,
    length_max: int | None = None,
    table_label: str = "tab:smiles_L15_samples_by_length",
    digits: int = 3,
) -> str:
    df = samples_by_length_df.copy()
    df["method_key"] = df["experiment"].apply(clean_experiment_name)
    if methods is None:
        methods = sorted(df["method_key"].unique().tolist(), key=method_sort_key)
    methods = [m for m in methods if m in df["method_key"].unique()]

    if length_max is None:
        length_max = int(df["length"].max())

    out = []
    out.append("\\begin{sidewaystable*}[t]")
    out.append("\\centering")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{2.9pt}")
    out.append("\\renewcommand{\\arraystretch}{1.05}")
    out.append(
        "\\caption{\\textbf{Per-terminal-length statistics (valid-only; mean $\\pm$ 95\\% CI).} "
        "$n$ is the number of valid samples at each length; UniqueRate (mol) measures distinct molecules; "
        "FPDiv@Len is fingerprint diversity within the length.}"
    )
    out.append(f"\\label{{{table_label}}}")
    out.append("\\vspace{-0.35em}")

    subcols_per_method = 3
    col_spec = (
        "@{}r|" + ("".join(["c" * subcols_per_method + "|" for _ in methods])).rstrip("|") + "@{}"
    )
    out.append(f"\\begin{{tabular}}{{{col_spec}}}")
    out.append("\\toprule")

    top = ["Len"]
    for m in methods:
        top.append(
            f"\\multicolumn{{{subcols_per_method}}}{{c|}}{{{latex_escape(display_method(m))}}}"
        )
    top[-1] = top[-1].replace("{c|}", "{c}")
    out.append(" & ".join(top) + " \\\\")
    out.append("\\cmidrule(lr){2-%d}" % (1 + subcols_per_method * len(methods)))

    second = [""]
    for _m in methods:
        second += ["$n$", "UniqueRate", "FPDiv@Len"]
    out.append(" & ".join(second) + " \\\\")
    out.append("\\midrule")

    for L in range(1, length_max + 1):
        row = [str(L)]
        for m in methods:
            sub = df[(df["method_key"] == m) & (df["length"] == L)]
            if len(sub) == 0:
                row += ["0", "0", "0"]
                continue
            rr = sub.iloc[0]
            n_cell = fmt_pm(rr.get("n_mean", 0.0), rr.get("n_err", 0.0), digits=1)
            ur_cell = fmt_pm(
                rr.get("unique_rate_mol_mean", 0.0),
                rr.get("unique_rate_mol_err", 0.0),
                digits=digits,
            )
            fp_cell = fmt_pm(rr.get("fp_div_mean", 0.0), rr.get("fp_div_err", 0.0), digits=digits)
            row += [n_cell, ur_cell, fp_cell]
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{sidewaystable*}")
    return "\n".join(out)


# -------------------------
# Main
# -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Results directory containing the CSV files.")
    ap.add_argument(
        "--out", default=None, help="Optional output .tex file. If not set, print to stdout."
    )
    ap.add_argument("--digits", type=int, default=3, help="Decimal digits for floats.")
    ap.add_argument(
        "--no_table_star",
        action="store_true",
        help="Use single-column table instead of table* for the paper stress table.",
    )
    ap.add_argument(
        "--omit_tokfp", action="store_true", help="Omit TokDiv/FPDiv from the paper stress table."
    )
    ap.add_argument(
        "--omit_macro_fpdiv",
        action="store_true",
        help="Omit MacroFPDiv (length-balanced) from the paper stress table.",
    )
    ap.add_argument(
        "--kmax_prefix_avg",
        type=int,
        default=15,
        help="Max k for averaging prefix metrics (default: 10).",
    )
    args = ap.parse_args()

    d = args.dir
    main_path = os.path.join(d, "main_table.csv")
    pb_path = os.path.join(d, "prefix_bucket_table.csv")
    pbl_path = os.path.join(d, "prefix_by_length_table.csv")
    sbl_path = os.path.join(d, "samples_by_length.csv")

    main_df = read_csv_maybe(main_path)
    if main_df is None:
        raise FileNotFoundError(f"Missing {main_path}")

    prefix_bucket_df = read_csv_maybe(pb_path)
    prefix_by_k_df = read_csv_maybe(pbl_path)
    samples_by_length_df = read_csv_maybe(sbl_path)

    rows = build_rows(
        main_df=main_df,
        prefix_bucket_df=prefix_bucket_df,
        prefix_by_k_df=prefix_by_k_df,
        samples_by_length_df=samples_by_length_df,
        prefix_bucket_name="long",
        k_max_for_prefix_avg=args.kmax_prefix_avg,
    )

    methods = [r.method_key for r in rows]

    tex_parts: list[str] = []
    tex_parts.append("% =========================")
    tex_parts.append("% Paper: ICML-style stress-test table (means; collapse/coverage-first)")
    tex_parts.append("% =========================")
    tex_parts.append(
        latex_stress_table_icml(
            rows=rows,
            table_label="tab:smiles_L15_stress",
            include_ci=False,
            digits=args.digits,
            use_table_star=(not args.no_table_star),
            include_tokfp=(not args.omit_tokfp),
            include_macro_fpdiv=(not args.omit_macro_fpdiv and samples_by_length_df is not None),
        )
    )
    tex_parts.append("\n")

    tex_parts.append("% =========================")
    tex_parts.append("% Appendix: stress-test table with CI + diagnostic tables")
    tex_parts.append("% =========================")
    tex_parts.append(
        latex_stress_table_icml(
            rows=rows,
            table_label="tab:smiles_L15_stress_ci",
            include_ci=True,
            digits=args.digits,
            use_table_star=True,
            include_tokfp=True,
            include_macro_fpdiv=(samples_by_length_df is not None),
            caption="\\textbf{Long-horizon SMILES stress test (max len $=15$): mean $\\pm$ 95\\% CI over seeds.} "
            "Same metrics as Table~\\ref{tab:smiles_L15_stress}, with uncertainty.",
        )
    )
    tex_parts.append("\n")

    if prefix_bucket_df is not None:
        tex_parts.append(latex_appendix_prefix_buckets(prefix_bucket_df, digits=args.digits))
        tex_parts.append("\n")

    if prefix_by_k_df is not None:
        tex_parts.append(
            latex_appendix_prefix_by_k(
                prefix_by_k_df, methods=methods, k_max=args.kmax_prefix_avg, digits=args.digits
            )
        )
        tex_parts.append("\n")

    if samples_by_length_df is not None:
        tex_parts.append(
            latex_appendix_samples_by_length(
                samples_by_length_df, methods=methods, digits=args.digits
            )
        )
        tex_parts.append("\n")

    out_tex = "\n".join(tex_parts)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_tex)
    else:
        print(out_tex)


if __name__ == "__main__":
    main()
