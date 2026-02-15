from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict

import partialsmiles as pa
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, RDConfig, rdMolDescriptors
from rdkit.Chem.FilterCatalog import *
from tqdm import tqdm

try:
    # doesnt work for github CI/pypi install.
    # SA score is only in the contrib dir for RDKit conda install
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    sys.path.append(rdkit.__path__[0])
    import sascorer
except:
    # download SA score with caching
    import requests

    cache_dir = Path.home() / ".cache" / "chemgfn" / "sa_score"
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py",
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz",
    ]

    for file_url in files:
        filename = file_url.split("/")[-1]
        cache_path = cache_dir / filename
        target_path = Path(rdkit.__path__[0]) / filename

        if cache_path.exists():
            if not target_path.exists() or target_path.stat().st_size != cache_path.stat().st_size:
                shutil.copy2(cache_path, target_path)
        else:
            try:
                response = requests.get(file_url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(cache_path, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename}",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                shutil.copy2(cache_path, target_path)
            except Exception as e:
                if cache_path.exists():
                    cache_path.unlink()
                raise RuntimeError(f"Failed to download {filename}: {e}") from e

    sys.path.append(rdkit.__path__[0])

    import sascorer


def verify_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                return True
            except:
                return False
        else:
            return False
    except:
        return False


def verify_smiles_pa(smiles: str):
    try:
        pa.ParseSmiles(smiles)
        return True
    except Exception:
        return False


# -------------------------
# Helpers
# -------------------------
def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _invalid(mol, default: float = 0.0) -> float:
    return float(default) if mol is None else None  # sentinel


# -------------------------
# Core fast properties
# -------------------------
def qed(mol) -> float:
    """Drug-likeness (0..1)."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(QED.qed(mol), 0.0)
    except Exception:
        return 0.0


def tpsa(mol) -> float:
    """Topological polar surface area (A^2)."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(rdMolDescriptors.CalcTPSA(mol), 0.0)
    except Exception:
        return 0.0


def hbd(mol) -> float:
    """H-bond donors."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(Lipinski.NumHDonors(mol), 0.0)
    except Exception:
        return 0.0


def hba(mol) -> float:
    """H-bond acceptors."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(Lipinski.NumHAcceptors(mol), 0.0)
    except Exception:
        return 0.0


def rot_bonds(mol) -> float:
    """Rotatable bonds (Lipinski definition)."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(Lipinski.NumRotatableBonds(mol), 0.0)
    except Exception:
        return 0.0


def ring_count(mol) -> float:
    """Total ring count."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(rdMolDescriptors.CalcNumRings(mol), 0.0)
    except Exception:
        return 0.0


def aromatic_ring_count(mol) -> float:
    """Aromatic ring count."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(rdMolDescriptors.CalcNumAromaticRings(mol), 0.0)
    except Exception:
        return 0.0


def fraction_csp3(mol) -> float:
    """FractionCSP3 (0..1): saturated carbon fraction proxy for 3D-ness."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(rdMolDescriptors.CalcFractionCSP3(mol), 0.0)
    except Exception:
        return 0.0


def heavy_atom_count(mol) -> float:
    if mol is None:
        return 0.0
    try:
        return _safe_float(mol.GetNumHeavyAtoms(), 0.0)
    except Exception:
        return 0.0


def mw(mol) -> float:
    """Molecular weight."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(Descriptors.MolWt(mol), 0.0)
    except Exception:
        return 0.0


def logS_esol_proxy(mol) -> float:
    """
    Very fast ESOL-like solubility proxy (not the full Delaney ESOL model).
    This is a heuristic: higher -> more soluble-ish.
    If you prefer strict ESOL, implement full formula; this is a quick proxy.
    """
    if mol is None:
        return 0.0
    try:
        # simple: -logP - 0.01*(MW-200) - 0.1*(RB) + 0.1*(TPSA/40)
        lp = _safe_float(Crippen.MolLogP(mol), 0.0)
        m = _safe_float(Descriptors.MolWt(mol), 0.0)
        rb = _safe_float(Lipinski.NumRotatableBonds(mol), 0.0)
        ps = _safe_float(rdMolDescriptors.CalcTPSA(mol), 0.0)
        val = (-lp) - 0.01 * (m - 200.0) - 0.1 * rb + 0.1 * (ps / 40.0)
        return float(val)
    except Exception:
        return 0.0


# -------------------------
# Rule-based scores (fast multi-objective components)
# -------------------------
def lipinski_pass(mol) -> float:
    """
    Returns 1.0 if passes (rough) Lipinski Ro5, else 0.0.
    """
    if mol is None:
        return 0.0
    try:
        mw_ = Descriptors.MolWt(mol)
        logp_ = Crippen.MolLogP(mol)
        hbd_ = Lipinski.NumHDonors(mol)
        hba_ = Lipinski.NumHAcceptors(mol)
        ok = (mw_ <= 500.0) and (logp_ <= 5.0) and (hbd_ <= 5) and (hba_ <= 10)
        return 1.0 if ok else 0.0
    except Exception:
        return 0.0


def veber_pass(mol) -> float:
    """
    Veber rule: rotatable bonds <= 10 and TPSA <= 140.
    Returns 1.0 if pass else 0.0.
    """
    if mol is None:
        return 0.0
    try:
        rb = Lipinski.NumRotatableBonds(mol)
        ps = rdMolDescriptors.CalcTPSA(mol)
        ok = (rb <= 10) and (ps <= 140.0)
        return 1.0 if ok else 0.0
    except Exception:
        return 0.0


def size_window_score(mol, mw_lo: float = 200.0, mw_hi: float = 500.0) -> float:
    """
    Soft window score in [0,1] for MW being in [mw_lo, mw_hi].
    Triangular-ish: 1 in window center, 0 outside.
    """
    if mol is None:
        return 0.0
    try:
        x = float(Descriptors.MolWt(mol))
        if x <= mw_lo * 0.5 or x >= mw_hi * 1.5:
            return 0.0
        # piecewise linear to [0,1]
        if x < mw_lo:
            return float((x - mw_lo * 0.5) / max(1e-8, (mw_lo - mw_lo * 0.5)))
        if x > mw_hi:
            return float(((mw_hi * 1.5) - x) / max(1e-8, (mw_hi * 1.5 - mw_hi)))
        return 1.0
    except Exception:
        return 0.0


def logp_window_score(mol, lo: float = 0.0, hi: float = 3.0) -> float:
    """
    Soft window score in [0,1] for logP in [lo, hi].
    """
    if mol is None:
        return 0.0
    try:
        x = float(Crippen.MolLogP(mol))
        # allow margins
        m_lo = lo - 2.0
        m_hi = hi + 2.0
        if x <= m_lo or x >= m_hi:
            return 0.0
        if x < lo:
            return float((x - m_lo) / max(1e-8, (lo - m_lo)))
        if x > hi:
            return float((m_hi - x) / max(1e-8, (m_hi - hi)))
        return 1.0
    except Exception:
        return 0.0


def pains_alert_count(mol) -> float:
    """
    Returns number of PAINS alerts (>=0). If FilterCatalog not available, returns 0.
    """
    if mol is None:
        return 0.0
    try:
        from rdkit.Chem import FilterCatalog

        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        catalog = FilterCatalog.FilterCatalog(params)
        matches = catalog.GetMatches(mol)
        return float(len(matches))
    except Exception:
        # environment may not have contrib; keep it safe
        return 0.0


def pains_pass(mol) -> float:
    """1.0 if no PAINS alerts else 0.0."""
    return 1.0 if pains_alert_count(mol) == 0.0 else 0.0


# -------------------------
# Your existing ones (keep)
# -------------------------
def sa_scorer(mol) -> float:
    if mol is None:
        return 0.0
    try:
        import sascorer

        return float(sascorer.calculateScore(mol))
    except Exception:
        return 0.0


def sa_scorer_normalized(mol) -> float:
    """
    Normalized and inverted synthetic accessibility score.

    RDKit's contrib `sascorer` implements the Ertl & Schuffenhauer (2009)
    SA metric that returns ~1 (easy) to ~10 (hard). We rescale to [0, 1],
    clamp, and invert so values closer to 1 indicate easier synthesis.
    """
    if mol is None:
        return 0.0
    try:
        import sascorer

        raw = float(sascorer.calculateScore(mol))
        min_score, max_score = 1.0, 10.0
        norm = (raw - min_score) / max(1e-8, (max_score - min_score))
        norm = min(1.0, max(0.0, norm))
        return 1.0 - norm
    except Exception:
        return 0.0


def logP(mol) -> float:
    if mol is None:
        return -1.0
    try:
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return -1.0


def logP_drug(mol) -> float:
    if mol is None:
        return -1.0
    try:
        v = float(Descriptors.MolLogP(mol))
        return v if (0.0 <= v <= 3.0) else -1.0
    except Exception:
        return -1.0


def logP_relative(mol) -> float:
    if mol is None:
        return -1.0
    try:
        return float(Descriptors.MolLogP(mol) - 0.8562)
    except Exception:
        return -1.0


def bertz_ct(mol) -> float:
    """Topological complexity (Bertz CT)."""
    if mol is None:
        return 0.0
    try:
        return _safe_float(Descriptors.BertzCT(mol), 0.0)  # type: ignore[attr-defined]
    except Exception:
        return 0.0


def bertz_window_score(mol, lo: float = 200.0, hi: float = 900.0) -> float:
    """
    Soft window score in [0,1] for BertzCT being in [lo, hi].
    Too simple or too complex are down-weighted.
    """
    if mol is None:
        return 0.0
    try:
        x = float(Descriptors.BertzCT(mol))  # type: ignore[attr-defined]
        m_lo = lo * 0.5
        m_hi = hi * 1.5
        if x <= m_lo or x >= m_hi:
            return 0.0
        if x < lo:
            return float((x - m_lo) / max(1e-8, (lo - m_lo)))
        if x > hi:
            return float((m_hi - x) / max(1e-8, (m_hi - hi)))
        return 1.0
    except Exception:
        return 0.0


def stereo_complexity_reward(
    mol,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    bertz_lo: float = 200.0,
    bertz_hi: float = 900.0,
) -> float:
    """
    Composite reward R = QED^alpha * Fsp3^beta * d_complexity^gamma.
    d_complexity is the BertzCT soft window; higher favors balanced complexity.
    """
    if mol is None:
        return 0.0
    try:
        q = max(0.0, min(1.0, qed(mol)))
        f = max(0.0, min(1.0, fraction_csp3(mol)))
        d = max(0.0, min(1.0, bertz_window_score(mol, lo=bertz_lo, hi=bertz_hi)))
        return float((q**alpha) * (f**beta) * (d**gamma))
    except Exception:
        return 0.0


FUNCTION_MAPPING: dict[str, Callable] = {
    # original
    "sa": sa_scorer,
    "sa_norm": sa_scorer_normalized,
    "logP": logP,
    "logP_drug": logP_drug,
    "logP_relative": logP_relative,
    "bertz_ct": bertz_ct,
    # recommended fast objectives
    "qed": qed,
    "tpsa": tpsa,
    "hbd": hbd,
    "hba": hba,
    "rot_bonds": rot_bonds,
    "ring_count": ring_count,
    "aromatic_ring_count": aromatic_ring_count,
    "fraction_csp3": fraction_csp3,
    "heavy_atom_count": heavy_atom_count,
    "mw": mw,
    "logS_proxy": logS_esol_proxy,
    # rule passes (binary)
    "lipinski_pass": lipinski_pass,
    "veber_pass": veber_pass,
    # soft window scores (0..1)
    "mw_window": size_window_score,
    "logp_window": logp_window_score,
    "bertz_window": bertz_window_score,
    # optional filters
    "pains_alerts": pains_alert_count,
    "pains_pass": pains_pass,
    # composites
    "stereo_complexity_reward": stereo_complexity_reward,
}
