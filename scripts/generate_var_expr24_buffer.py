#!/usr/bin/env python
"""
Generate Expr24 solution buffer covering lengths 3, 5, 7, 9, and 11.

The script enumerates all digit/operator combinations (no parentheses) that
evaluate exactly to the target value (default 24) under standard precedence,
tokenizes them with the specified tokenizer (default: meta-llama/Llama-3.2-1B),
pads to the longest sequence with the EOS token, and saves a torch tensor.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

OPS: tuple[str, ...] = ("+", "-", "*", "/")


def eval_fraction(nums: Sequence[int], ops: Sequence[str]) -> Fraction | None:
    """Exact evaluation with precedence using Fraction."""
    total = Fraction(0, 1)
    cur = Fraction(nums[0], 1)
    for op, n in zip(ops, nums[1:]):
        if op == "+":
            total += cur
            cur = Fraction(n, 1)
        elif op == "-":
            total += cur
            cur = Fraction(-n, 1)
        elif op == "*":
            cur *= n
        elif op == "/":
            if n == 0:
                return None
            cur /= n
        else:
            raise ValueError(f"Unsupported op: {op}")
    return total + cur


def eval_vectorized(digits_float: np.ndarray, ops: Sequence[str]) -> np.ndarray:
    """Fast float evaluation for a batch of digit combinations."""
    total = np.zeros(digits_float.shape[0], dtype=np.float64)
    term = digits_float[:, 0].copy()
    for idx, op in enumerate(ops):
        nxt = digits_float[:, idx + 1]
        if op == "+":
            total += term
            term = nxt
        elif op == "-":
            total += term
            term = -nxt
        elif op == "*":
            term *= nxt
        elif op == "/":
            term /= nxt
        else:
            raise ValueError(f"Unsupported op: {op}")
    total += term
    return total


def build_expression(nums: Sequence[int], ops: Sequence[str]) -> str:
    parts: list[str] = []
    for n, op in zip(nums[:-1], ops):
        parts.append(str(int(n)))
        parts.append(op)
    parts.append(str(int(nums[-1])))
    return "".join(parts)


def generate_digits(num_digits: int, include_zero: bool) -> np.ndarray:
    symbols = np.arange(0 if include_zero else 1, 10, dtype=np.int8)
    mesh = np.stack(np.meshgrid(*([symbols] * num_digits), indexing="ij"), axis=-1)
    return mesh.reshape(-1, num_digits)


def collect_expressions(
    length: int,
    include_zero: bool,
    target: Fraction,
    float_tol: float,
) -> list[str]:
    """Return all expressions of the given length that equal the target."""
    num_digits = (length + 1) // 2
    op_count = num_digits - 1

    digit_int = generate_digits(num_digits, include_zero)
    digit_float = digit_int.astype(np.float64)

    expressions: list[str] = []
    for ops in product(OPS, repeat=op_count):
        values = eval_vectorized(digit_float, ops)
        candidates = np.nonzero(np.isclose(values, float(target), atol=float_tol))[0]
        if candidates.size == 0:
            continue
        for idx in candidates.tolist():
            nums = digit_int[idx].tolist()
            value = eval_fraction(nums, ops)
            if value is None or value != target:
                continue
            expressions.append(build_expression(nums, ops))
    return expressions


def pad_and_stack(seqs: Iterable[Sequence[int]], pad_id: int) -> torch.Tensor:
    seq_list = [list(seq) for seq in seqs]
    if not seq_list:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(len(seq) for seq in seq_list)
    out = torch.full((len(seq_list), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seq_list):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Expr24 buffer for var lengths.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/24_points/buffer_24_varlen_non_zero.pt"),
        help="Where to save the padded token tensor.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer name or path.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9, 11],
        help="Odd expression lengths to enumerate.",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Allow digit 0 in expressions (default: digits 1-9 only).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for the preliminary float equality check.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=24.0,
        help="Target value expressions must equal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define eos_token_id for padding.")

    target_fraction = Fraction(args.target).limit_denominator()

    all_token_seqs: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for length in args.lengths:
        if length % 2 == 0 or length < 1:
            raise ValueError(f"Length must be odd and positive, got {length}")
        exprs = collect_expressions(
            length=length,
            include_zero=args.include_zero,
            target=target_fraction,
            float_tol=args.tolerance,
        )
        for expr in exprs:
            tokens = tokenizer.encode(expr, add_special_tokens=False)
            key = tuple(tokens)
            if key not in seen:
                seen.add(key)
                all_token_seqs.append(key)
        print(f"[length {length}] found {len(exprs)} expressions, unique tokens {len(seen)}")

    tensor = pad_and_stack(all_token_seqs, pad_id=pad_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, args.output)
    print(f"Saved {tensor.shape[0]} sequences (padded to {tensor.shape[1]}) to {args.output}")


if __name__ == "__main__":
    main()
