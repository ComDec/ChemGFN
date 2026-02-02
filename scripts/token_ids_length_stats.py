#!/usr/bin/env python3

import argparse
import ast
import csv
import statistics
import sys


def parse_token_ids(cell: str) -> list[int]:
    if cell is None:
        return []
    s = cell.strip()
    if not s:
        return []

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
    except Exception:
        pass

    # Fallback: extract ints from a loose representation.
    s2 = s
    for ch in "[]()":
        s2 = s2.replace(ch, " ")
    parts = [p for p in s2.replace(",", " ").split() if p]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"  # interpreted as [lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute token_ids lengths after removing padding IDs (default: 13)."
    )
    ap.add_argument("csv_path", help="Path to the CSV file")
    ap.add_argument(
        "--pad-id",
        type=int,
        default=13,
        help="Padding token id to remove before counting length (default: 13)",
    )
    ap.add_argument(
        "--column",
        default="token_ids",
        help="Column name containing token ids (default: token_ids)",
    )
    ap.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=[4, 10, 15, 20],
        help="Bin edges, e.g. --bins 4 10 15 20 => [4,10), [10,15), [15,20) (default: 4 10 15 20)",
    )
    ap.add_argument(
        "--only-valid",
        action="store_true",
        help="If set, only keep rows with is_valid == True",
    )
    args = ap.parse_args()

    if len(args.bins) < 2:
        print("--bins must contain at least 2 integers", file=sys.stderr)
        return 2
    if sorted(args.bins) != list(args.bins):
        print("--bins must be sorted ascending", file=sys.stderr)
        return 2

    lengths: list[int] = []
    n_rows = 0
    n_used = 0
    n_bad = 0

    with open(args.csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or args.column not in reader.fieldnames:
            cols = ", ".join(reader.fieldnames or [])
            print(
                f"Column {args.column!r} not found. Available columns: {cols}",
                file=sys.stderr,
            )
            return 2

        for row in reader:
            n_rows += 1
            if args.only_valid:
                v = (row.get("is_valid") or "").strip().lower()
                if v not in {"true", "1", "yes"}:
                    continue

            cell = row.get(args.column, "")
            ids = parse_token_ids(cell)
            if not ids and (cell or "").strip() not in {"", "[]"}:
                n_bad += 1
            ln = sum(1 for t in ids if t != args.pad_id)
            lengths.append(ln)
            n_used += 1

    if not lengths:
        print("No rows processed (empty CSV or filtered out).", file=sys.stderr)
        return 1

    avg = sum(lengths) / len(lengths)
    med = statistics.median(lengths)
    mn = min(lengths)
    mx = max(lengths)

    edges = list(args.bins)
    bin_counts: dict[str, int] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        key = bin_label(lo, hi)
        bin_counts[key] = sum(1 for x in lengths if lo <= x < hi)

    low_edge = edges[0]
    high_edge = edges[-1]
    under = sum(1 for x in lengths if x < low_edge)
    over = sum(1 for x in lengths if x >= high_edge)

    print(f"file: {args.csv_path}")
    print(f"rows: {n_rows}  used: {n_used}  bad_token_ids_parse: {n_bad}")
    print(f"pad_id: {args.pad_id}  column: {args.column}")
    print(f"length: avg={avg:.3f}  median={med}  min={mn}  max={mx}")
    print("bins (interpreted as [lo, hi)):")
    for k in [bin_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]:
        c = bin_counts[k]
        pct = 100.0 * c / len(lengths)
        print(f"  {k}: {c} ({pct:.2f}%)")
    print(f"  <{low_edge}: {under} ({100.0 * under / len(lengths):.2f}%)")
    print(f"  >={high_edge}: {over} ({100.0 * over / len(lengths):.2f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
