"""Molecular property scores used as SMILES rewards.

Each scorer maps an RDKit ``Mol`` to a float and is registered in
:data:`FUNCTION_MAPPING` under the key used by the ``scorer`` field of the SMILES
validator config. Scorers are total functions: a ``None`` molecule or an RDKit
failure yields the scorer's neutral value instead of raising.
"""

from __future__ import annotations

from typing import Callable

from rdkit.Chem import QED, Descriptors
from rdkit.Chem.rdchem import Mol


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce an RDKit descriptor result to ``float``, falling back to ``default``."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def qed(mol: Mol | None) -> float:
    """Quantitative estimate of drug-likeness in [0, 1]; 0.0 for invalid molecules."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(QED.qed(mol), 0.0)
    except Exception:
        return 0.0


def logP(mol: Mol | None) -> float:
    """Crippen octanol/water partition coefficient; -1.0 for invalid molecules."""
    if mol is None:
        return -1.0
    try:
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return -1.0


FUNCTION_MAPPING: dict[str, Callable[[Mol | None], float]] = {
    "qed": qed,
    "logP": logP,
}
