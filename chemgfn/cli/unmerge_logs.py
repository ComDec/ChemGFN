#!/usr/bin/env python3
"""
Restore merged logs produced by chemgfn-merge-logs.

Features:
- Split merged CSVs (with source_step) back into per-step CSV files.
- Expand grouped PNG tars (<prefix>.png.tar.gz) back to the original PNG files.
- Expand wandb tarballs (wandb.tar.gz) back to wandb directories.
"""

from __future__ import annotations

import argparse
import glob
import tarfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def strip_csv_gz(path: Path) -> str:
    """Return the base prefix of <name>.csv.gz (drop both suffixes)."""
    name = path.name
    if not name.endswith(".csv.gz"):
        raise ValueError(f"not a csv.gz file: {path}")
    return name[: -len(".csv.gz")]


def expand_csv_file(
    file_path: Path,
    step_col: str = "source_step",
    chunksize: int = 20000,
    overwrite: bool = False,
    delete_archive: bool = False,
) -> None:
    """Split merged csv.gz into per-step CSV files."""
    base = strip_csv_gz(file_path)
    parent = file_path.parent
    header_written: dict[int, bool] = {}

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if step_col not in chunk.columns:
            raise KeyError(f"Column '{step_col}' not found in {file_path}")
        for step, df_step in chunk.groupby(step_col):
            out_path = parent / f"{base}_{int(step)}.csv"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"{out_path} exists (use --overwrite to replace)")
            write_header = not header_written.get(step, False)
            df_step.drop(columns=[step_col]).to_csv(
                out_path, mode="a" if out_path.exists() else "w", index=False, header=write_header
            )
            header_written[step] = True

    if delete_archive:
        file_path.unlink()


def expand_csvs(root: Path, step_col: str, chunksize: int, overwrite: bool, delete_archive: bool):
    csv_files = [p for p in root.rglob("*.csv.gz") if not p.name.endswith(".png.tar.gz")]
    if not csv_files:
        print(f"[info] no merged csv.gz found under {root}")
        return
    for path in sorted(csv_files):
        print(f"[csv] expanding {path}")
        expand_csv_file(
            file_path=path,
            step_col=step_col,
            chunksize=chunksize,
            overwrite=overwrite,
            delete_archive=delete_archive,
        )


def expand_png_tar(path: Path, overwrite: bool, delete_archive: bool) -> None:
    """Extract <prefix>.png.tar.gz back to original png files."""
    parent = path.parent
    with tarfile.open(path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".png")]
        if not members:
            print(f"[png] no png members in {path}")
            return
        for m in members:
            out_path = parent / m.name
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"{out_path} exists (use --overwrite to replace)")
        tar.extractall(path=parent, members=members)
    if delete_archive:
        path.unlink()


def expand_pngs(root: Path, overwrite: bool, delete_archive: bool) -> None:
    png_tars = list(root.rglob("*.png.tar.gz"))
    if not png_tars:
        print(f"[info] no png tarballs under {root}")
        return
    for path in sorted(png_tars):
        print(f"[png] expanding {path}")
        expand_png_tar(path=path, overwrite=overwrite, delete_archive=delete_archive)


def expand_wandb_tar(path: Path, overwrite: bool, delete_archive: bool) -> None:
    """Extract wandb.tar.gz back to wandb/ directory."""
    parent = path.parent
    target = parent / "wandb"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (use --overwrite to replace)")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=parent)
    if delete_archive:
        path.unlink()


def expand_wandb(root: Path, overwrite: bool, delete_archive: bool) -> None:
    wandb_tars = list(root.rglob("wandb.tar.gz"))
    if not wandb_tars:
        print(f"[info] no wandb tarballs under {root}")
        return
    for path in sorted(wandb_tars):
        print(f"[wandb] expanding {path}")
        expand_wandb_tar(path=path, overwrite=overwrite, delete_archive=delete_archive)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore merged logs back to original files.")
    parser.add_argument(
        "--root",
        type=str,
        default="/data1/xw3763/project/gflow/ChemGFN/logs/train",
        help="Root directory to search recursively.",
    )
    parser.add_argument(
        "--root-glob",
        type=str,
        default=None,
        help="Glob pattern to expand multiple roots (e.g., '/.../logs/train/VarExpr24*').",
    )
    parser.add_argument(
        "--step-col",
        type=str,
        default="source_step",
        help="Column name storing original step inside merged CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=20000,
        help="Rows per chunk when reading merged CSVs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing restored files.",
    )
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="Delete merged archives after successful expansion.",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Skip expanding merged CSVs.",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Skip expanding grouped PNG tarballs.",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip expanding wandb tarballs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.root_glob:
        roots = [
            Path(p).expanduser().resolve()
            for p in sorted(glob.glob(args.root_glob))
            if Path(p).exists()
        ]
        if not roots:
            print(f"[info] root_glob matched nothing: {args.root_glob}")
            return
    else:
        roots = [Path(args.root).expanduser().resolve()]

    for root in roots:
        print(f"[root] expanding {root}")
        if not args.skip_csv:
            expand_csvs(
                root=root,
                step_col=args.step_col,
                chunksize=args.chunksize,
                overwrite=args.overwrite,
                delete_archive=args.delete_archive,
            )
        if not args.skip_png:
            expand_pngs(root=root, overwrite=args.overwrite, delete_archive=args.delete_archive)
        if not args.skip_wandb:
            expand_wandb(root=root, overwrite=args.overwrite, delete_archive=args.delete_archive)
    print("[done] expansion complete")


if __name__ == "__main__":
    main()
