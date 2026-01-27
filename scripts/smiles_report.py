#!/usr/bin/env python3

"""Entry point: generate SMILES tables + figures.

This wrapper makes the report generator runnable from the repo root:
  python scripts/smiles_report.py --output-dir out/smiles_report --preset L10 --check-tex

Implementation lives in `notebooks/gen_smiles_report.py` (shared with notebooks).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nb_dir = repo_root / "notebooks"
    sys.path.insert(0, str(nb_dir))

    import gen_smiles_report  # type: ignore

    gen_smiles_report.main()


if __name__ == "__main__":
    main()
