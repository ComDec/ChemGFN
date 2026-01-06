#!/usr/bin/env python3
"""
CLI to merge numeric-suffixed CSV files and optionally compress+cleanup wandb.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

GroupKey = Tuple[Path, str]
IndexedFile = Tuple[int, Path]
PngGroupKey = Tuple[Path, str]
PngIndexedFile = Tuple[int, Path]


def find_groups(root: Path) -> dict[GroupKey, list[IndexedFile]]:
    """Find CSV files that end with _<number>.csv and group by prefix."""
    pattern = re.compile(r"(.+)_([0-9]+)\.csv$")
    groups: dict[GroupKey, list[IndexedFile]] = {}

    for path in root.rglob("*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        prefix, index_str = match.group(1), match.group(2)
        index = int(index_str)
        key: GroupKey = (path.parent, prefix)
        groups.setdefault(key, []).append((index, path))

    return groups


def merge_group(
    files: list[IndexedFile],
    out_path: Path,
    chunksize: int = 20000,
    delete_source: bool = False,
    step_col: str | None = "source_step",
) -> None:
    """Merge sorted files into a single gzip-compressed CSV; optionally delete inputs.

    If step_col is provided, add a column with the numeric suffix per source file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    first_header = True
    with gzip.open(out_path, "wt", encoding="utf-8") as f_out:
        for _, file_path in sorted(files):
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                if step_col:
                    chunk[step_col] = _
                chunk.to_csv(f_out, index=False, header=first_header)
                first_header = False
    if delete_source:
        for _, file_path in sorted(files):
            file_path.unlink()


def merge_all(
    root: Path,
    suffix: str,
    chunksize: int,
    delete_source: bool,
    dry_run: bool,
    step_col: str | None,
) -> None:
    groups = find_groups(root)
    if not groups:
        print(f"[info] no matching CSV groups found under {root}")
        return

    for (parent, prefix), files in sorted(groups.items()):
        out_path = parent / f"{prefix}{suffix}"
        print(f"[merge] {len(files):4d} files -> {out_path}")
        if dry_run:
            continue
        merge_group(
            files,
            out_path,
            chunksize=chunksize,
            delete_source=delete_source,
            step_col=step_col,
        )
    print("[done] merging complete")


def compress_wandb(root: Path, delete_source: bool, dry_run: bool) -> None:
    """Tar.gz every 'wandb' directory under the given root; optionally delete originals."""
    wandb_dirs = [p for p in root.rglob("wandb") if p.is_dir()]
    if not wandb_dirs:
        print(f"[info] no wandb directories under {root}")
        return

    for wdir in sorted(wandb_dirs):
        out_path = wdir.parent / f"{wdir.name}.tar.gz"
        print(f"[wandb] {wdir} -> {out_path}")
        if dry_run:
            continue
        if out_path.exists():
            out_path.unlink()
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(wdir, arcname=wdir.name)
        if delete_source:
            shutil.rmtree(wdir)
    print("[done] wandb compression complete")


def find_png_groups(root: Path) -> dict[PngGroupKey, list[PngIndexedFile]]:
    """Find PNG files ending with _<number>.png and group by prefix."""
    pattern = re.compile(r"(.+)_([0-9]+)\.png$")
    groups: dict[PngGroupKey, list[PngIndexedFile]] = {}
    for path in root.rglob("*.png"):
        match = pattern.match(path.name)
        if not match:
            continue
        prefix, index_str = match.group(1), match.group(2)
        index = int(index_str)
        key: PngGroupKey = (path.parent, prefix)
        groups.setdefault(key, []).append((index, path))
    return groups


def compress_png_groups(root: Path, delete_source: bool, dry_run: bool) -> None:
    """Group pngs by numeric suffix and tar.gz each group into <prefix>.png.tar.gz."""
    groups = find_png_groups(root)
    if not groups:
        print(f"[info] no grouped png files under {root}")
        return

    for (parent, prefix), files in sorted(groups.items()):
        out_path = parent / f"{prefix}.png.tar.gz"
        print(f"[png] {len(files):4d} files -> {out_path}")
        if dry_run:
            continue
        if out_path.exists():
            out_path.unlink()
        with tarfile.open(out_path, "w:gz") as tar:
            for _, file_path in sorted(files):
                tar.add(file_path, arcname=file_path.name)
        if delete_source:
            for _, file_path in sorted(files):
                file_path.unlink()
    print("[done] png compression complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge numeric-suffixed CSV files.")
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
        "--chunksize",
        type=int,
        default=20000,
        help="Number of rows per chunk when reading CSVs.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".csv.gz",
        help="Output suffix for merged files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned merges without writing output.",
    )
    parser.add_argument(
        "--compress-wandb",
        action="store_true",
        default=True,
        help="Tar.gz every 'wandb' directory under the roots. Default: on.",
    )
    parser.add_argument(
        "--no-compress-wandb",
        action="store_false",
        dest="compress_wandb",
        help="Disable wandb compression.",
    )
    parser.add_argument(
        "--compress-png",
        action="store_true",
        default=True,
        help="Group pngs with numeric suffix into <prefix>.png.tar.gz. Default: on.",
    )
    parser.add_argument(
        "--no-compress-png",
        action="store_false",
        dest="compress_png",
        help="Disable PNG compression.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        default=True,
        help="Delete source CSVs and wandb/PNG after compression. Default: on.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_false",
        dest="delete_source",
        help="Keep source CSVs and wandb/PNG after compression.",
    )
    parser.add_argument(
        "--step-col",
        type=str,
        default="source_step",
        help="Name of column to store the numeric suffix (set empty to skip).",
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
        print(f"[root] processing {root}")
        merge_all(
            root=root,
            suffix=args.suffix,
            chunksize=args.chunksize,
            delete_source=args.delete_source,
            dry_run=args.dry_run,
            step_col=args.step_col if args.step_col else None,
        )
        if args.compress_wandb:
            compress_wandb(root=root, delete_source=args.delete_source, dry_run=args.dry_run)
        if args.compress_png:
            compress_png_groups(root=root, delete_source=args.delete_source, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
