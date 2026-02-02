"""
Unified SMILES report generator.

Outputs:
- Multiple LaTeX tables (A/B/C/D/E) as separate .tex files
- All plots (including per-length log pterm) into a single output directory

This script uses draw_smiles presets (L10/L15) and re-formats the tables to
match the paper-style layouts provided by the user.
"""

from __future__ import annotations

import argparse
import heapq
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import draw_smiles as ds
import gen_smiles_L15_table as l15
import gen_smiles_table_appendix as bylen
import gen_smiles_table_prefix_appendix as pref
import numpy as np
import pandas as pd

LATEX_ROW_END = r" \\"


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _clean_smiles_text(s: str) -> str:
    s = "" if s is None else str(s)
    # keep consistent with draw_smiles: strip any special-token suffix
    if "<|" in s:
        s = s.split("<|", 1)[0]
    return s.strip()


def _infer_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _wrap_smiles_for_legend(smiles: str, width: int = 28, max_lines: int = 4) -> str:
    s = str(smiles)
    if len(s) <= width:
        return s
    lines = [s[i : i + width] for i in range(0, len(s), width)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1][: max(0, width - 3)] + "..."
    return "\n".join(lines)


def _collect_top_qed(
    exp_name: str,
    payload: dict[str, Any],
    samples_cfg: ds.SamplesConfig,
    k: int,
) -> list[dict[str, Any]]:
    """Collect top-k QED molecules across all repeats for an experiment."""
    try:
        from rdkit import Chem
        from rdkit.Chem import QED

        try:
            from rdkit import RDLogger

            disable = getattr(RDLogger, "DisableLog", None)
            if callable(disable):
                disable("rdApp.error")
                disable("rdApp.warning")
        except Exception:
            pass
    except Exception:
        print(f"[warn] RDKit not available; skip top-QED for {exp_name}")
        return []

    # min-heap of (qed, canonical, uid) with lazy deletion
    heap: list[tuple[float, str, int]] = []
    best: dict[str, dict[str, Any]] = {}
    uid = 0

    def _clean_heap_top() -> None:
        while heap:
            q, canon, u = heap[0]
            cur = best.get(canon)
            if cur is None or cur.get("uid") != u or float(cur.get("qed", -1.0)) != float(q):
                heapq.heappop(heap)
                continue
            break

    def _trim_to_k() -> None:
        while len(best) > k:
            _clean_heap_top()
            if not heap:
                break
            q, canon, u = heapq.heappop(heap)
            cur = best.get(canon)
            if cur is not None and cur.get("uid") == u and float(cur.get("qed", -1.0)) == float(q):
                del best[canon]

    samples_paths = payload.get("samples_paths", []) or []
    if not samples_paths:
        print(f"[warn] no samples_paths for {exp_name}; skip top-QED")
        return []

    for spath in samples_paths:
        if not spath:
            continue
        p = Path(spath)
        if not p.exists():
            print(f"[warn] samples csv not found, skip: {spath}")
            continue

        try:
            sdf = ds.load_samples_csv(str(p))
        except Exception as e:
            print(f"[warn] failed to read samples csv, skip: {spath}: {e}")
            continue

        text_col = samples_cfg.text_col_override or _infer_col(sdf, samples_cfg.text_cols)
        if text_col is None:
            print(
                f"[warn] cannot infer SMILES column for {spath}; tried: {list(samples_cfg.text_cols)}"
            )
            continue

        for raw in sdf[text_col].tolist():
            smi = _clean_smiles_text(raw)
            if not smi:
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                qed = float(QED.qed(mol))
            except Exception:
                continue

            canon = Chem.MolToSmiles(mol, canonical=True)

            cur = best.get(canon)
            if cur is not None and float(cur["qed"]) >= qed:
                continue

            uid += 1
            best[canon] = {
                "uid": uid,
                "qed": qed,
                "smiles": canon,
                "smiles_raw": smi,
                "mol": mol,
            }
            heapq.heappush(heap, (qed, canon, uid))
            _trim_to_k()

    items = sorted(best.values(), key=lambda d: float(d["qed"]), reverse=True)
    out: list[dict[str, Any]] = []
    for i, d in enumerate(items, start=1):
        out.append(
            {
                "experiment": exp_name,
                "rank": i,
                "qed": float(d["qed"]),
                "smiles": str(d["smiles"]),
                "smiles_raw": str(d.get("smiles_raw") or d["smiles"]),
                "mol": d["mol"],
            }
        )
    return out


def _draw_top_qed_grid(
    exp_name: str,
    entries: list[dict[str, Any]],
    out_path: Path,
    ncols: int = 5,
) -> None:
    if not entries:
        return

    try:
        from rdkit.Chem import Draw
    except Exception:
        print(f"[warn] RDKit draw not available; skip image for {exp_name}")
        return

    mols = [e["mol"] for e in entries]
    legends: list[str] = []
    for e in entries:
        smi = str(e["smiles"])
        qed = float(e["qed"])
        legends.append(f"#{int(e['rank'])}  QED={qed:.3f}\n{_wrap_smiles_for_legend(smi)}")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=max(1, int(ncols)),
        subImgSize=(320, 280),
        legends=legends,
        useSVG=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img.save(str(out_path))
    except Exception as e:
        print(f"[warn] failed to save image {out_path}: {e}")


def _extract_exp_name(x: str) -> str:
    s = str(x).strip()
    m = re.findall(r"'([^']+)'", s)
    if m:
        return m[0]
    return s


def _normalize_exp_name(x: str) -> str:
    s = _extract_exp_name(x)
    s = s.replace("_SubM", "-SubM")
    return s


def _display_method(name: str, style: str = "plus") -> str:
    s = str(name)
    s = s.replace("_SubM", "-SubM")

    # Ablations / special variants
    if s.lower() in {"tb-wo-ref", "tb_wo_ref"}:
        return "TB w/o ref"
    if s.lower() in {"raptb-maxonly", "raptb_maxonly"}:
        return "RapTB (max-only)"
    if s.lower() in {"raptb-softonly", "raptb_softonly"}:
        return "RapTB (soft-only)"

    if style == "plus_space":
        s = s.replace("-SubM", " + SubM")
    elif style == "plus":
        s = s.replace("-SubM", "+SubM")
    elif style == "dash":
        s = s.replace("-SubM", "-SubM")
    return s


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


def _select_rows(rows: list[l15.Row], methods: Sequence[str]) -> list[l15.Row]:
    want = {re.sub(r"[^a-z0-9]+", "", m.lower()) for m in methods}
    out = []
    for r in rows:
        key = re.sub(r"[^a-z0-9]+", "", str(r.method_key).lower())
        disp = re.sub(r"[^a-z0-9]+", "", str(r.disp).lower())
        if key in want or disp in want:
            out.append(r)
    return out


def render_table_a(
    rows: list[l15.Row],
    table_label: str,
    caption: str,
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
        "MacroFPDiv$\\uparrow$",
        "FPDiv$\\uparrow$",
    ]

    col_spec = "@{}l" + "c" * (len(cols) - 1) + "@{}"
    lines: list[str] = []
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
    lines.append(" & ".join(cols) + LATEX_ROW_END)
    lines.append("\\midrule")

    def cell_val(v: tuple[float, float]) -> str:
        return _format_num(v[0], digits=digits)

    for r in rows:
        cells = [_display_method(r.disp, style="plus")]

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

        mfp = cell_val(r.macro_fpdiv)
        if best_macro_fp.get(r.method_key, False):
            mfp = _bf(mfp)
        cells.append(mfp)

        fp = cell_val(r.fpdiv_all)
        if best_fpdiv.get(r.method_key, False):
            fp = _bf(fp)
        cells.append(fp)

        lines.append(" & ".join(cells) + LATEX_ROW_END)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.25em}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_table_b(
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
    df["experiment"] = df["experiment"].apply(_normalize_exp_name)
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

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.08}")
    lines.append("\\caption{")
    lines.append("\\textbf{SMILES generation performance.}")
    lines.append("Unless specified, all metrics are computed on valid samples.")
    lines.append(
        f"\\texttt{{Len}} denotes the mean token length of valid samples ($L_{{\\max}}={lmax}$)."
    )
    lines.append("}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\vspace{-0.35em}")
    lines.append("\\begin{tabular}{@{}lccccc@{}}")
    lines.append("\\toprule")
    lines.append(
        "Method & Acc $\\uparrow$ & Score $\\uparrow$ & Entropy $\\uparrow$ & FPDiv $\\uparrow$ & Len"
        + LATEX_ROW_END
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

        disp = _display_method(m, style="plus_space")
        lines.append(f"{disp} & {acc} & {score} & {ent} & {fp} & {ln}" + LATEX_ROW_END)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_table_c(
    main_df: pd.DataFrame,
    buckets: ds.Buckets,
    methods: Sequence[str],
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
    df["experiment"] = df["experiment"].apply(_normalize_exp_name)
    idx = df.set_index("experiment")

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
    headers += [f"Frac[{b.replace('-', '--')}]" for b, _ in frac_cols]

    colspec = "l" + "c" * (len(headers) - 1)
    lines: list[str] = []
    lines.append("\\begin{table*}[!htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.06}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + LATEX_ROW_END)
    lines.append("\\midrule")

    for e in methods:
        if e not in idx.index:
            continue
        r = idx.loc[e]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        row = [_display_method(e, style="plus")]
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
        lines.append(" & ".join(row) + LATEX_ROW_END)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}%%")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_bylen_table(
    df: pd.DataFrame,
    exps: Sequence[str],
    metric_keys: list[str],
    label: str,
    caption: str,
    display_map: dict[str, str],
) -> str:
    if df is None or len(df) == 0:
        return ""
    return bylen.make_table_latex(
        df=df,
        exp_col="experiment",
        len_col="length",
        exps=list(exps),
        metric_keys=metric_keys,
        label=label,
        caption=caption,
        display_name_map=display_map,
        clearpage_after=False,
    )


def render_prefix_table(
    df: pd.DataFrame,
    exps: Sequence[str],
    label: str,
    caption: str,
    display_map: dict[str, str],
) -> str:
    if df is None or len(df) == 0:
        return ""
    return pref.make_table(
        df=df,
        exp_col="experiment",
        len_col="k",
        exps=list(exps),
        metric_keys=["survival", "entropy", "eff", "top1", "unique_rate"],
        label=label,
        caption=caption,
        display_name_map=display_map,
        clearpage_after=False,
    )


def _validate_latex(tex: str) -> list[str]:
    errors: list[str] = []
    stack: list[str] = []
    for m in re.finditer(r"\\begin\{([^}]+)\}|\\end\{([^}]+)\}", tex):
        env_begin = m.group(1)
        env_end = m.group(2)
        if env_begin:
            stack.append(env_begin)
        elif env_end:
            if not stack:
                errors.append(f"Unmatched \\end{{{env_end}}}")
                continue
            last = stack.pop()
            if last != env_end:
                errors.append(f"Mismatched env: \\begin{{{last}}} ... \\end{{{env_end}}}")
    for env in reversed(stack):
        errors.append(f"Unclosed \\begin{{{env}}}")

    # Common generator error: a single trailing backslash at row end (should be \\).
    for i, line in enumerate(tex.splitlines(), start=1):
        if re.search(r"[^\\]\\$", line.rstrip()):
            errors.append(
                f"Line {i}: suspicious single trailing \\\\ at end of row (expected \\\\\\)"
            )
    return errors


def _pdflatex_available() -> bool:
    return shutil.which("pdflatex") is not None


def _compile_tex_table(output_dir: Path, tex_filename: str) -> str | None:
    """Return error message if compilation fails, else None."""
    if not _pdflatex_available():
        return "pdflatex not found"

    check_dir = output_dir / "_texcheck"
    check_dir.mkdir(parents=True, exist_ok=True)

    wrapper = output_dir / "_texcheck_wrapper.tex"
    wrapper.write_text(
        "\\documentclass{article}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{lmodern}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{array}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        f"\\input{{{tex_filename}}}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={str(check_dir)}",
        str(wrapper.name),
    ]
    proc = subprocess.run(
        cmd, cwd=str(output_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        # Keep last ~80 lines for signal.
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        return f"pdflatex failed for {tex_filename}:\n{tail}"
    return None


def _write_tex(path: Path, content: str, problems: dict[str, list[str]]) -> None:
    path.write_text(content, encoding="utf-8")
    errs = _validate_latex(content)
    if errs:
        problems[str(path)] = errs


def _prepare_tables(
    preset: str,
    output_dir: Path,
    error_mode: str,
    kmax_prefix_avg: int | None,
) -> tuple[list[l15.Row], dict[str, pd.DataFrame], ds.Buckets]:
    exps = ds.get_exps(preset)
    buckets = ds.get_buckets(preset)
    cache_dir = output_dir / f"cache_{preset}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables = ds.run_tables(
        exps=exps,
        buckets=buckets,
        keys=ds.JsonKeys(),
        samples_cfg=ds.SamplesConfig(),
        error_mode=error_mode,
        output_root=cache_dir,
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
    return rows, tables, buckets


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SMILES report (tables + plots).")
    ap.add_argument("--output-dir", default="smiles_report", help="Directory for all outputs.")
    ap.add_argument(
        "--preset", action="append", choices=sorted(ds.EXPS_PRESETS.keys()), default=None
    )
    ap.add_argument("--error-mode", default="ci95", choices=["std", "sem", "ci95", "none"])
    ap.add_argument("--kmax-prefix-avg", type=int, default=None)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument(
        "--skip-topqed", action="store_true", help="Skip per-method Top-K QED molecule grids/CSVs."
    )
    ap.add_argument(
        "--topqed-k",
        type=int,
        default=10,
        help="Top-K molecules by QED per method (across repeats).",
    )
    ap.add_argument(
        "--topqed-ncols",
        type=int,
        default=5,
        help="Columns per grid image for Top-K QED molecules.",
    )
    ap.add_argument(
        "--check-tex",
        action="store_true",
        help="Compile generated .tex via pdflatex for syntax check.",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = args.preset or []
    if not presets:
        presets = sorted(ds.EXPS_PRESETS.keys())

    problems: dict[str, list[str]] = {}
    compile_problems: dict[str, str] = {}

    topqed_all_rows: list[dict[str, Any]] = []

    for preset in presets:
        rows, tables, buckets = _prepare_tables(
            preset=preset,
            output_dir=output_dir,
            error_mode=args.error_mode,
            kmax_prefix_avg=args.kmax_prefix_avg,
        )

        # Top-K QED molecules per method (across all repeat runs)
        if not args.skip_topqed and int(args.topqed_k) > 0:
            topqed_dir = output_dir / "topqed"
            topqed_dir.mkdir(parents=True, exist_ok=True)

            exps_norm = ds.normalize_exps(ds.get_exps(preset))
            samples_cfg = ds.SamplesConfig()
            preset_rows: list[dict[str, Any]] = []

            for exp_name, payload in exps_norm.items():
                entries = _collect_top_qed(
                    exp_name, payload, samples_cfg=samples_cfg, k=int(args.topqed_k)
                )
                if not entries:
                    continue

                exp_slug = _slugify(exp_name)
                img_path = topqed_dir / f"{preset}_{exp_slug}_top{int(args.topqed_k)}_qed.png"
                _draw_top_qed_grid(exp_name, entries, img_path, ncols=int(args.topqed_ncols))

                # Per-method CSV (SMILES + QED)
                per_method_rows = []
                for e in entries:
                    r = {
                        "preset": preset,
                        "method": exp_name,
                        "rank": int(e["rank"]),
                        "qed": float(e["qed"]),
                        "smiles": str(e["smiles"]),
                    }
                    per_method_rows.append(r)
                    preset_rows.append(r)
                    topqed_all_rows.append(r)

                pd.DataFrame(per_method_rows).to_csv(
                    topqed_dir / f"{preset}_{exp_slug}_top{int(args.topqed_k)}_qed.csv",
                    index=False,
                )

            if preset_rows:
                pd.DataFrame(preset_rows).to_csv(
                    topqed_dir / f"{preset}_top{int(args.topqed_k)}_qed.csv",
                    index=False,
                )

        main_df = tables.get("main_table")
        metrics_df = tables.get("metrics_by_length")
        prefix_df = tables.get("prefix_by_length_table")

        lmax = "10" if preset == "L10" else "15"

        # Table A: main stress table
        stress_methods = ["TB", "SubTB", "RapTB", "RapTB_SubM"]
        rows_a = _select_rows(rows, stress_methods)
        caption_a = (
            f"\\textbf{{SMILES generation with extended horizon ($L_{{\\max}}={lmax}$).}} "
            "Frac columns are valid-only length fractions. "
            "Prefix metrics (SurvEnd, Ent, Top1) are computed on correct-only samples and averaged over $k$. "
            "MacroFPDiv is a length-balanced fingerprint diversity that averages FPDiv across the three length bins (0--5, 6--10, 11+)."
        )
        table_a = render_table_a(
            rows=rows_a,
            table_label=f"tab:smiles_{preset}_stress",
            caption=caption_a,
        )
        _write_tex(output_dir / f"A_{preset}_stress.tex", table_a, problems)

        # Table B: main summary table
        if preset == "L10":
            table_b = render_table_b(
                main_df=main_df,
                methods=["TB", "SubTB", "RapTB", "RapTB-SubM"],
                table_label="tab:smiles_main",
                lmax=lmax,
            )
            _write_tex(output_dir / "B_L10_main.tex", table_b, problems)
        else:
            table_b = render_table_b(
                main_df=main_df,
                methods=["TB", "SubTB", "RapTB", "RapTB-SubM"],
                table_label="tab:smiles_main_L15",
                lmax=lmax,
            )
            _write_tex(output_dir / "B_L15_main.tex", table_b, problems)

        # Table C: all-length averaged performance (L10 focus)
        if preset in ("L10", "L15"):
            caption_c = (
                "All-length averaged SMILES performance and induced length distribution "
                f"($L_{{\\max}}={lmax}$; mean$\\pm$95\\% CI over 6 runs). "
                "Acc is computed over all samples; Score/Div/FPDiv are computed on valid samples only; "
                "Frac[$\\cdot$] is computed over all samples."
            )
            table_c = render_table_c(
                main_df=main_df,
                buckets=buckets,
                methods=(
                    ["TB", "TB-SubM", "RapTB", "RapTB-SubM", "SubTB", "SubTB-SubM"]
                    if preset == "L10"
                    else ["TB", "SubTB", "RapTB", "RapTB-SubM"]
                ),
                table_label=(
                    "tab:smiles_all_length_avg"
                    if preset == "L10"
                    else "tab:smiles_all_length_avg_L15"
                ),
                caption=caption_c,
            )
            _write_tex(output_dir / f"C_{preset}_all_length_avg.tex", table_c, problems)

        # L10 ablations (small tables)
        if preset == "L10":
            # RapTB max-only / soft-only
            table_ab1 = render_table_b(
                main_df=main_df,
                methods=["RapTB", "RapTB-MaxOnly", "RapTB-SoftOnly"],
                table_label="tab:smiles_ablation_raptb",
                lmax=lmax,
            )
            if table_ab1:
                table_ab1 = table_ab1.replace(
                    "SMILES generation performance.",
                    "SMILES generation ablations (RapTB variants).",
                )
                _write_tex(output_dir / "F_L10_ablation_raptb_max_soft.tex", table_ab1, problems)

            # TB reference ablation
            table_ab2 = render_table_b(
                main_df=main_df,
                methods=["TB", "TB-wo-ref"],
                table_label="tab:smiles_ablation_tb_ref",
                lmax=lmax,
            )
            if table_ab2:
                table_ab2 = table_ab2.replace(
                    "SMILES generation performance.", "SMILES generation ablations (TB reference)."
                )
                _write_tex(output_dir / "G_L10_ablation_tb_wo_ref.tex", table_ab2, problems)

        # Table D: by-length performance tables
        if metrics_df is not None and len(metrics_df) > 0:
            mdf = metrics_df.reset_index()
            mdf = bylen.canonicalize_xy_columns(mdf)
            mdf["experiment"] = mdf["experiment"].apply(_normalize_exp_name)

            if preset == "L10":
                exps_main = ["TB", "SubTB", "RapTB", "TB-SubM", "SubTB-SubM", "RapTB-SubM"]
                display_plus = {e: _display_method(e, style="plus") for e in exps_main}
                display_dash = {e: _display_method(e, style="dash") for e in exps_main}

                table_core = render_bylen_table(
                    df=mdf,
                    exps=[e for e in exps_main if e in set(mdf["experiment"])],
                    metric_keys=["acc", "score_valid", "frac_valid", "count_valid"],
                    label="bylen_valid_core",
                    caption=(
                        "Per-length valid-only core metrics of SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$)."
                    ),
                    display_map=display_plus,
                )
                _write_tex(output_dir / "D_L10_bylen_valid_core.tex", table_core, problems)

                table_div = render_bylen_table(
                    df=mdf,
                    exps=[e for e in exps_main if e in set(mdf["experiment"])],
                    metric_keys=["div_valid", "fpdiv"],
                    label="bylen_valid_div",
                    caption=(
                        "Per-length valid-only diversity metrics of SMILES generation "
                        f"(mean$\\pm$95\\% CI, $L_{{\\max}}={lmax}$)."
                    ),
                    display_map=display_plus,
                )
                _write_tex(output_dir / "D_L10_bylen_valid_div.tex", table_div, problems)

                table_uniq = render_bylen_table(
                    df=mdf,
                    exps=[e for e in exps_main if e in set(mdf["experiment"])],
                    metric_keys=["uniq_str", "uniq_mol", "uniqrate_str", "uniqrate_mol"],
                    label="smiles_L10_bylen_valid_uniq",
                    caption=(
                        "Per-length valid-only uniqueness metrics of SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$)."
                    ),
                    display_map=display_dash,
                )
                _write_tex(output_dir / "D_L10_bylen_valid_uniq.tex", table_uniq, problems)
            else:
                exps_l15 = ["TB", "SubTB", "RapTB", "RapTB-SubM"]
                display_plus = {e: _display_method(e, style="plus") for e in exps_l15}

                table_core = render_bylen_table(
                    df=mdf,
                    exps=[e for e in exps_l15 if e in set(mdf["experiment"])],
                    metric_keys=["acc", "score_valid", "frac_valid", "count_valid"],
                    label="smiles_L15_bylen_valid_core",
                    caption=(
                        "Per-length valid-only core metrics of SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$)."
                    ),
                    display_map=display_plus,
                )
                _write_tex(output_dir / "D_L15_bylen_valid_core.tex", table_core, problems)

                table_div = render_bylen_table(
                    df=mdf,
                    exps=[e for e in exps_l15 if e in set(mdf["experiment"])],
                    metric_keys=["div_valid", "fpdiv"],
                    label="smiles_L15_bylen_valid_div",
                    caption=(
                        "Per-length valid-only diversity metrics of SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$)."
                    ),
                    display_map=display_plus,
                )
                _write_tex(output_dir / "D_L15_bylen_valid_div.tex", table_div, problems)

                table_uniq = render_bylen_table(
                    df=mdf,
                    exps=[e for e in exps_l15 if e in set(mdf["experiment"])],
                    metric_keys=["uniq_str", "uniq_mol", "uniqrate_str", "uniqrate_mol"],
                    label="smiles_L15_bylen_valid_uniq",
                    caption=(
                        "Per-length valid-only uniqueness metrics of SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$)."
                    ),
                    display_map=display_plus,
                )
                _write_tex(output_dir / "D_L15_bylen_valid_uniq.tex", table_uniq, problems)

        # Table E: prefix by length (L10) / prefix by length (L15)
        if prefix_df is not None and len(prefix_df) > 0:
            pdf = prefix_df.reset_index()
            pdf = pref.canonicalize_xy_columns(pdf)
            pdf["experiment"] = pdf["experiment"].apply(_normalize_exp_name)

            if preset == "L10":
                base_exps = ["TB", "SubTB", "RapTB"]
                subm_exps = ["TB-SubM", "SubTB-SubM", "RapTB-SubM"]
                display_base = {e: _display_method(e, style="plus") for e in base_exps}
                display_subm = {e: _display_method(e, style="dash") for e in subm_exps}

                table_p1 = render_prefix_table(
                    df=pdf,
                    exps=[e for e in base_exps if e in set(pdf["experiment"])],
                    label="prefix_bylen_base",
                    caption="Prefix statistics by depth. Mean$\\pm$95\\% CI.",
                    display_map=display_base,
                )
                _write_tex(output_dir / "E_L10_prefix_base.tex", table_p1, problems)

                table_p2 = render_prefix_table(
                    df=pdf,
                    exps=[e for e in subm_exps if e in set(pdf["experiment"])],
                    label="prefix_bylen_subm",
                    caption="Prefix statistics by depth (Continue). Mean$\\pm$95\\% CI.",
                    display_map=display_subm,
                )
                _write_tex(output_dir / "E_L10_prefix_subm.tex", table_p2, problems)
            else:
                base_exps = ["TB", "SubTB"]
                raptb_exps = ["RapTB", "RapTB-SubM"]
                display_base = {e: _display_method(e, style="plus") for e in base_exps}
                display_rap = {e: _display_method(e, style="plus") for e in raptb_exps}

                table_p1 = render_prefix_table(
                    df=pdf,
                    exps=[e for e in base_exps if e in set(pdf["experiment"])],
                    label="smiles_L15_prefix_base",
                    caption=(
                        "Prefix statistics by depth on SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$): TB vs. SubTB."
                    ),
                    display_map=display_base,
                )
                _write_tex(output_dir / "E_L15_prefix_base.tex", table_p1, problems)

                table_p2 = render_prefix_table(
                    df=pdf,
                    exps=[e for e in raptb_exps if e in set(pdf["experiment"])],
                    label="smiles_L15_prefix_raptb",
                    caption=(
                        "Prefix statistics by depth on SMILES generation (mean$\\pm$95\\% CI, "
                        f"$L_{{\\max}}={lmax}$): RapTB vs. RapTB+SubM."
                    ),
                    display_map=display_rap,
                )
                _write_tex(output_dir / "E_L15_prefix_raptb.tex", table_p2, problems)

        if not args.skip_plots:
            ds.run_plots(
                exps=ds.get_exps(preset),
                style=ds.style,
                buckets=buckets,
                keys=ds.JsonKeys(),
                samples_cfg=ds.SamplesConfig(),
                plot_prefix=True,
                plot_len_hist=True,
                plot_by_len_json=True,
                plot_by_len_samples=True,
                save_fig_dir=output_dir,
                error_mode=args.error_mode,
                output_root=None,
                name_prefix=preset,
            )

    if topqed_all_rows:
        topqed_dir = output_dir / "topqed"
        topqed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(topqed_all_rows).to_csv(
            topqed_dir / f"top{int(args.topqed_k)}_qed_all_presets.csv",
            index=False,
        )

    if problems:
        print("[warn] LaTeX validation issues:")
        for path, errs in problems.items():
            print(f"  - {path}")
            for e in errs:
                print(f"    * {e}")
    else:
        print("[ok] LaTeX validation passed for all outputs.")

    if args.check_tex:
        # Compile-check every generated .tex in this output dir.
        for tex_path in sorted(output_dir.glob("*.tex")):
            err = _compile_tex_table(output_dir=output_dir, tex_filename=tex_path.name)
            if err and err != "pdflatex not found":
                compile_problems[str(tex_path)] = err
        if compile_problems:
            print("[warn] pdflatex compilation failed:")
            for p, msg in compile_problems.items():
                print(f"  - {p}")
                print(msg)
        elif _pdflatex_available():
            print("[ok] pdflatex compilation passed for all generated tables.")
        else:
            print("[warn] pdflatex not found; skipped compilation check.")


if __name__ == "__main__":
    main()
