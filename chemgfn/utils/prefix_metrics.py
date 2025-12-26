from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

Token = Hashable


def _entropy_from_counter(c: Counter) -> float:
    n = sum(c.values())
    if n <= 0:
        return 0.0
    ent = 0.0
    for v in c.values():
        p = v / n
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def _to_bool_list(x) -> list[bool] | None:
    """Accept list/bool-seq/torch.Tensor/np.array and convert to List[bool]."""
    if x is None:
        return None
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [bool(v) for v in x]


@dataclass
class PrefixCollapseResult:
    top1_mass: list[float]
    entropy: list[float]
    eff_support: list[float]
    unique: list[int]
    support: list[int]
    top1_auc: float
    collapse_depth: int
    collapse_thr: float

    # ---- NEW: only prefixes from ultimately-correct sequences (valid SMILES) ----
    top1_mass_correct: list[float] = field(default_factory=list)
    entropy_correct: list[float] = field(default_factory=list)
    eff_support_correct: list[float] = field(default_factory=list)
    unique_correct: list[int] = field(default_factory=list)
    support_correct: list[int] = field(default_factory=list)

    top1_auc_correct: float = 0.0
    collapse_depth_correct: int = 0

    # optional: fraction of correct samples among active samples at each position
    correct_frac: list[float] = field(default_factory=list)


def prefix_collapse_by_position(
    seqs: Sequence[Sequence[Token]],
    active_before: Sequence[Sequence[bool]] | None = None,
    *,
    max_T: int | None = None,
    collapse_thr: float = 0.95,
    # NEW: invalid[i]=True means the whole sequence is invalid (最终不是合法SMILES)
    invalid: Sequence[bool] | None = None,
) -> PrefixCollapseResult:
    """
    Compute prefix-collapse metrics per position t.
    Additionally compute collapse metrics restricted to prefixes from sequences with (not invalid[i]).

    Args:
      seqs: list of token sequences.
      active_before: same shape as seqs, True if token at t is active (not past EOS).
      max_T: optional cap on time steps.
      collapse_thr: threshold for collapse_depth.
      invalid: per-sample invalid flags from your validator. True => invalid SMILES.

    Returns:
      PrefixCollapseResult with both all-samples curves and correct-only curves.
    """
    if len(seqs) == 0:
        return PrefixCollapseResult([], [], [], [], [], 0.0, 0, float(collapse_thr))

    invalid = _to_bool_list(invalid)
    want_correct = invalid is not None

    T = max(len(s) for s in seqs)
    if max_T is not None:
        T = min(T, int(max_T))

    top1_mass: list[float] = []
    entropy: list[float] = []
    eff_support: list[float] = []
    unique: list[int] = []
    support: list[int] = []

    top1_mass_c: list[float] = []
    entropy_c: list[float] = []
    eff_support_c: list[float] = []
    unique_c: list[int] = []
    support_c: list[int] = []
    correct_frac: list[float] = []

    for t in range(T):
        toks_all: list[Token] = []
        toks_cor: list[Token] = []

        for i, s in enumerate(seqs):
            if len(s) <= t:
                continue
            if active_before is not None:
                if (
                    i >= len(active_before)
                    or t >= len(active_before[i])
                    or (not active_before[i][t])
                ):
                    continue

            tok = s[t]
            toks_all.append(tok)

            if want_correct:
                # correct prefix ⇔ this sequence is ultimately valid SMILES ⇔ invalid[i] == False
                if i < len(invalid) and (not invalid[i]):
                    toks_cor.append(tok)

        n_all = len(toks_all)
        if n_all == 0:
            break

        c_all = Counter(toks_all)
        ent_all = _entropy_from_counter(c_all)
        eff_all = math.exp(ent_all)
        top1_all = max(c_all.values()) / n_all

        top1_mass.append(float(top1_all))
        entropy.append(float(ent_all))
        eff_support.append(float(eff_all))
        unique.append(int(len(c_all)))
        support.append(int(n_all))

        if want_correct:
            n_cor = len(toks_cor)
            support_c.append(int(n_cor))
            correct_frac.append(float(n_cor / n_all) if n_all > 0 else 0.0)

            if n_cor > 0:
                c_cor = Counter(toks_cor)
                ent_cor = _entropy_from_counter(c_cor)
                eff_cor = math.exp(ent_cor)
                top1_cor = max(c_cor.values()) / n_cor

                top1_mass_c.append(float(top1_cor))
                entropy_c.append(float(ent_cor))
                eff_support_c.append(float(eff_cor))
                unique_c.append(int(len(c_cor)))
            else:
                # keep curve lengths aligned
                top1_mass_c.append(0.0)
                entropy_c.append(0.0)
                eff_support_c.append(0.0)
                unique_c.append(0)

    # scalar summaries (all)
    top1_auc = float(sum(top1_mass) / max(1, len(top1_mass)))
    depth = 0
    for x in top1_mass:
        if x >= collapse_thr:
            depth += 1
        else:
            break

    # scalar summaries (correct-only): average over positions where support_correct > 0
    top1_auc_c = 0.0
    depth_c = 0
    if want_correct and len(top1_mass_c) > 0:
        idx = [j for j, n in enumerate(support_c) if n > 0]
        if len(idx) > 0:
            top1_auc_c = float(sum(top1_mass_c[j] for j in idx) / len(idx))

        # depth: consecutive from t=0 with support_correct>0 and top1>=thr
        for j, x in enumerate(top1_mass_c):
            if j < len(support_c) and support_c[j] > 0 and x >= collapse_thr:
                depth_c += 1
            else:
                break

    return PrefixCollapseResult(
        top1_mass=top1_mass,
        entropy=entropy,
        eff_support=eff_support,
        unique=unique,
        support=support,
        top1_auc=top1_auc,
        collapse_depth=depth,
        collapse_thr=float(collapse_thr),
        top1_mass_correct=top1_mass_c if want_correct else [],
        entropy_correct=entropy_c if want_correct else [],
        eff_support_correct=eff_support_c if want_correct else [],
        unique_correct=unique_c if want_correct else [],
        support_correct=support_c if want_correct else [],
        top1_auc_correct=float(top1_auc_c) if want_correct else 0.0,
        collapse_depth_correct=int(depth_c) if want_correct else 0,
        correct_frac=correct_frac if want_correct else [],
    )


def prefix_collapse_by_k(
    seqs: Sequence[Sequence[Token]],
    active_before: Sequence[Sequence[bool]] | None = None,
    *,
    k_list: Iterable[int] = (1, 2, 3, 4, 5, 6),
    # NEW: invalid[i]=True means the whole sequence is invalid
    invalid: Sequence[bool] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute collapse metrics over k-prefixes.
    Additionally, compute correct-only collapse restricted to sequences with (not invalid[i]).

    Returns dict: k -> {
      "n","unique","top1","top5","entropy","eff",
      "n_correct","unique_correct","top1_correct","top5_correct","entropy_correct","eff_correct"
    }
    """
    invalid = _to_bool_list(invalid)
    want_correct = invalid is not None

    out: dict[int, dict[str, float]] = {}

    for k in k_list:
        k = int(k)
        if k <= 0:
            continue

        prefixes_all: list[tuple[Token, ...]] = []
        prefixes_cor: list[tuple[Token, ...]] = []

        for i, s in enumerate(seqs):
            if len(s) < k:
                continue

            if active_before is not None:
                if i >= len(active_before) or len(active_before[i]) < k:
                    continue
                ok = True
                for t in range(k):
                    if not active_before[i][t]:
                        ok = False
                        break
                if not ok:
                    continue

            pref = tuple(s[:k])
            prefixes_all.append(pref)

            if want_correct:
                if i < len(invalid) and (not invalid[i]):
                    prefixes_cor.append(pref)

        # all
        n = len(prefixes_all)
        if n == 0:
            base = {"n": 0.0, "unique": 0.0, "top1": 0.0, "top5": 0.0, "entropy": 0.0, "eff": 0.0}
        else:
            c = Counter(prefixes_all)
            ent = _entropy_from_counter(c)
            eff = math.exp(ent)
            vals = sorted(c.values(), reverse=True)
            top1 = vals[0] / n
            top5 = sum(vals[: min(5, len(vals))]) / n
            base = {
                "n": float(n),
                "unique": float(len(c)),
                "top1": float(top1),
                "top5": float(top5),
                "entropy": float(ent),
                "eff": float(eff),
            }

        # correct-only
        if want_correct:
            nc = len(prefixes_cor)
            if nc == 0:
                base.update(
                    {
                        "n_correct": 0.0,
                        "unique_correct": 0.0,
                        "top1_correct": 0.0,
                        "top5_correct": 0.0,
                        "entropy_correct": 0.0,
                        "eff_correct": 0.0,
                    }
                )
            else:
                cc = Counter(prefixes_cor)
                entc = _entropy_from_counter(cc)
                effc = math.exp(entc)
                valsc = sorted(cc.values(), reverse=True)
                top1c = valsc[0] / nc
                top5c = sum(valsc[: min(5, len(valsc))]) / nc
                base.update(
                    {
                        "n_correct": float(nc),
                        "unique_correct": float(len(cc)),
                        "top1_correct": float(top1c),
                        "top5_correct": float(top5c),
                        "entropy_correct": float(entc),
                        "eff_correct": float(effc),
                    }
                )

        out[k] = base

    return out
