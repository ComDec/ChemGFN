#!/usr/bin/env python3
"""
Extract rows of a merged CSV (with source_step column) back to a per-step CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rows for given step(s) from merged CSV.")
    parser.add_argument("--file", required=True, help="Path to merged csv.gz (with source_step).")
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        required=False,
        help="Step(s) to extract. If multiple provided, union of them.",
    )
    parser.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive step range to extract.",
    )
    parser.add_argument(
        "--step-col",
        type=str,
        default="source_step",
        help="Column name storing the original step. Default: source_step.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: alongside input, suffix with _<steps>.csv",
    )
    return parser.parse_args()


def build_steps(step: Iterable[int] | None, step_range: tuple[int, int] | None) -> set[int]:
    steps: set[int] = set()
    if step:
        steps.update(step)
    if step_range:
        start, end = step_range
        steps.update(range(start, end + 1))
    return steps


def main() -> None:
    args = parse_args()
    inp = Path(args.file).expanduser().resolve()
    if not inp.exists():
        raise FileNotFoundError(inp)

    steps = build_steps(args.step, tuple(args.step_range) if args.step_range else None)
    if not steps:
        raise ValueError("No steps provided; use --step or --step-range.")

    df = pd.read_csv(inp)
    if args.step_col not in df.columns:
        raise KeyError(f"Column '{args.step_col}' not found in {inp}")

    sub = df[df[args.step_col].isin(steps)]
    if sub.empty:
        raise ValueError(f"No rows found for steps {sorted(steps)} in {inp}")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        step_tag = (
            f"{min(steps)}-{max(steps)}"
            if len(steps) > 1 or (args.step_range and args.step_range[0] != args.step_range[1])
            else str(next(iter(steps)))
        )
        out_path = inp.with_suffix("").with_name(inp.stem.replace(".csv", f"_{step_tag}.csv"))

    sub.to_csv(out_path, index=False)
    print(f"[done] wrote {len(sub)} rows -> {out_path}")


if __name__ == "__main__":
    main()
