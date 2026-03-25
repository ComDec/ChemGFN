#!/usr/bin/env python3
"""
Evaluate a trained Expr24 model and produce Table 3-compatible metrics.

Computes: Acc, Unique_valid, NormCov, KL(pi->p*), KL(p*->pi), JS_tok, log_pterm(tau)

Usage:
    python scripts/sweep/eval_expr24_table3.py \
        --run-dir logs/VarExpr24_CFG_AvgPrefixTB/runs/2026-03-25_12-00-00 \
        --n-samples 6400 --seeds 42 123 2024

    # Or from wandb run ID
    python scripts/sweep/eval_expr24_table3.py \
        --wandb-run comdec/ChemGFN/abc123 --n-samples 6400

Outputs:
    stdout: formatted table row
    results/expr24_table3_{exp_name}.csv
"""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def load_oracle_set(buffer_path: str) -> set[tuple]:
    """Load the enumerated oracle set Y* of all valid Expr24 solutions."""
    buf = torch.load(buffer_path, map_location="cpu", weights_only=False)
    if isinstance(buf, dict):
        buf = buf.get("samples", buf.get("data", list(buf.values())[0]))
    if isinstance(buf, torch.Tensor):
        oracle = set()
        eos_candidates = [128001, 128009, 2]  # common EOS token IDs
        for row in buf:
            tokens = tuple(
                t.item() for t in row if t.item() not in eos_candidates and t.item() != 0
            )
            if tokens:
                oracle.add(tokens)
        return oracle
    elif isinstance(buf, list):
        return {tuple(s) if isinstance(s, (list, tuple)) else (s,) for s in buf}
    return set()


def compute_table3_metrics(
    samples_csv_path: str,
    oracle_set: set[tuple],
    tokenizer,
    eos_id: int,
    max_seq_len: int = 9,
) -> dict:
    """Compute Table 3 metrics from a saved samples CSV."""
    import pandas as pd

    df = pd.read_csv(samples_csv_path)
    N = len(df)

    # Parse validity
    if "is_valid" in df.columns:
        valid_mask = df["is_valid"].astype(bool).values
    else:
        valid_mask = np.ones(N, dtype=bool)

    n_valid = int(valid_mask.sum())
    acc = n_valid / N if N > 0 else 0.0

    # Parse token IDs from the CSV
    if "token_ids" in df.columns:
        import ast

        token_ids_list = []
        for raw in df["token_ids"]:
            if isinstance(raw, str):
                try:
                    ids = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    ids = []
            elif isinstance(raw, (list, tuple)):
                ids = list(raw)
            else:
                ids = []
            # Strip EOS and padding
            ids = [t for t in ids if t != eos_id and t != 0]
            token_ids_list.append(tuple(ids))
    else:
        token_ids_list = [() for _ in range(N)]

    # Unique valid
    valid_tuples = [token_ids_list[i] for i in range(N) if valid_mask[i]]
    unique_valid_set = set(valid_tuples)
    unique_valid = len(unique_valid_set)

    # Coverage against oracle
    cov_count = len(unique_valid_set & oracle_set)
    oracle_size = len(oracle_set)
    norm_cov = cov_count / min(N, oracle_size) if min(N, oracle_size) > 0 else 0.0

    # Position-wise token marginals for KL/JS
    T_max = max_seq_len
    eps = 1e-9

    # Build vocabulary from oracle + samples
    all_tokens_at_pos: list[Counter] = [Counter() for _ in range(T_max)]
    oracle_tokens_at_pos: list[Counter] = [Counter() for _ in range(T_max)]
    sample_tokens_at_pos: list[Counter] = [Counter() for _ in range(T_max)]

    # Oracle marginals (uniform over oracle set)
    for seq in oracle_set:
        for t, tok in enumerate(seq):
            if t < T_max:
                oracle_tokens_at_pos[t][tok] += 1
                all_tokens_at_pos[t][tok] += 1

    # Sample marginals (from valid samples only, with duplicates)
    for seq in valid_tuples:
        for t, tok in enumerate(seq):
            if t < T_max:
                sample_tokens_at_pos[t][tok] += 1
                all_tokens_at_pos[t][tok] += 1

    # Compute per-position KL and JS
    kl_fwd_list = []  # KL(pi || p*)
    kl_rev_list = []  # KL(p* || pi)
    js_list = []

    for t in range(T_max):
        vocab = set(all_tokens_at_pos[t].keys())
        if not vocab:
            continue

        n_oracle = sum(oracle_tokens_at_pos[t].values())
        n_sample = sum(sample_tokens_at_pos[t].values())
        if n_oracle == 0 or n_sample == 0:
            continue

        # Build distributions
        p_star = {}  # oracle
        pi = {}  # sample
        for v in vocab:
            p_star[v] = (oracle_tokens_at_pos[t].get(v, 0) + eps) / (n_oracle + eps * len(vocab))
            pi[v] = (sample_tokens_at_pos[t].get(v, 0) + eps) / (n_sample + eps * len(vocab))

        # KL(pi || p*) = sum pi(v) log(pi(v) / p*(v))
        kl_fwd = sum(pi[v] * math.log((pi[v]) / (p_star[v])) for v in vocab)
        # KL(p* || pi) = sum p*(v) log(p*(v) / pi(v))
        kl_rev = sum(p_star[v] * math.log((p_star[v]) / (pi[v])) for v in vocab)
        # JS = 0.5 * KL(pi || m) + 0.5 * KL(p* || m) where m = 0.5*(pi + p*)
        m = {v: 0.5 * (pi[v] + p_star[v]) for v in vocab}
        js = 0.5 * sum(pi[v] * math.log(pi[v] / m[v]) for v in vocab) + 0.5 * sum(
            p_star[v] * math.log(p_star[v] / m[v]) for v in vocab
        )

        kl_fwd_list.append(kl_fwd)
        kl_rev_list.append(kl_rev)
        js_list.append(js)

    kl_fwd_avg = np.mean(kl_fwd_list) if kl_fwd_list else 0.0
    kl_rev_avg = np.mean(kl_rev_list) if kl_rev_list else 0.0
    js_avg = np.mean(js_list) if js_list else 0.0

    # Log pterm(tau) — extract from per-position list at the EOS position
    log_pterm_avg = None
    if "log_pterm" in df.columns and "token_ids" in df.columns:
        import ast as _ast

        log_pterm_vals = []
        for idx_row in range(N):
            raw_pterm = df["log_pterm"].iloc[idx_row]
            raw_tids = df["token_ids"].iloc[idx_row]
            try:
                pterm_list = (
                    _ast.literal_eval(raw_pterm) if isinstance(raw_pterm, str) else raw_pterm
                )
                tids = _ast.literal_eval(raw_tids) if isinstance(raw_tids, str) else raw_tids
            except (ValueError, SyntaxError):
                continue
            if not isinstance(pterm_list, (list, tuple)) or not isinstance(tids, (list, tuple)):
                continue
            # tau = length of generated tokens (before EOS)
            tau = len(tids)
            if tau < len(pterm_list):
                log_pterm_vals.append(float(pterm_list[tau]))
            elif len(pterm_list) > 0:
                # fallback: last non-zero entry
                for j in range(len(pterm_list) - 1, -1, -1):
                    if float(pterm_list[j]) != 0.0:
                        log_pterm_vals.append(float(pterm_list[j]))
                        break
        if log_pterm_vals:
            log_pterm_avg = float(np.mean(log_pterm_vals))

    return {
        "Acc": acc,
        "Unique_valid": unique_valid,
        "NormCov": norm_cov,
        "KL(pi->p*)": kl_fwd_avg,
        "KL(p*->pi)": kl_rev_avg,
        "JS_tok": js_avg,
        "log_pterm": log_pterm_avg,
        "N": N,
        "n_valid": n_valid,
        "CovCount": cov_count,
        "oracle_size": oracle_size,
    }


def find_latest_test_csv(run_dir: str) -> str | None:
    """Find the most recent test samples CSV in a run directory."""
    run_path = Path(run_dir)
    # Search common locations
    candidates = list(run_path.rglob("samples_test_*.csv"))
    if not candidates:
        candidates = list(run_path.rglob("test_samples/*.csv"))
    if not candidates:
        return None
    # Return the one with highest step number
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main():
    parser = argparse.ArgumentParser(
        description="Compute Table 3-compatible Expr24 metrics from test samples"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Direct path to samples CSV. If not given, searches --run-dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run output directory to search for test CSV.",
    )
    parser.add_argument(
        "--buffer-path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "24_points" / "buffer_24_non_zero.pt"),
        help="Path to oracle set (buffer_24_non_zero.pt).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=9,
        help="Maximum sequence length for position-wise metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save results CSV.",
    )
    args = parser.parse_args()

    # Find CSV
    csv_path = args.csv_path
    if csv_path is None and args.run_dir:
        csv_path = find_latest_test_csv(args.run_dir)
        if csv_path is None:
            print(f"ERROR: No test samples CSV found in {args.run_dir}")
            sys.exit(1)
    elif csv_path is None:
        print("ERROR: Provide --csv-path or --run-dir")
        sys.exit(1)

    print(f"Evaluating: {csv_path}")

    # Load tokenizer for EOS ID
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    eos_id = tokenizer.eos_token_id

    # Load oracle set
    oracle_set = load_oracle_set(args.buffer_path)
    print(f"Oracle set: {len(oracle_set)} unique solutions")

    # Compute metrics
    metrics = compute_table3_metrics(
        samples_csv_path=csv_path,
        oracle_set=oracle_set,
        tokenizer=tokenizer,
        eos_id=eos_id,
        max_seq_len=args.max_seq_len,
    )

    # Print formatted
    print("\n" + "=" * 70)
    print("Table 3-compatible Expr24 Metrics")
    print("=" * 70)
    print(f"  Acc:           {metrics['Acc']:.3f}")
    print(f"  Unique_valid:  {metrics['Unique_valid']}")
    print(
        f"  NormCov:       {metrics['NormCov']:.3f}  (CovCount={metrics['CovCount']}/{metrics['oracle_size']})"
    )
    print(f"  KL(pi->p*):    {metrics['KL(pi->p*)']:.3f}")
    print(f"  KL(p*->pi):    {metrics['KL(p*->pi)']:.3f}")
    print(f"  JS_tok:        {metrics['JS_tok']:.3f}")
    if metrics["log_pterm"] is not None:
        print(f"  log_pterm(τ):  {metrics['log_pterm']:.3f}")
    print(f"  N={metrics['N']}, n_valid={metrics['n_valid']}")

    # Save to CSV
    if args.output_csv:
        import pandas as pd

        out_dir = Path(args.output_csv).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
