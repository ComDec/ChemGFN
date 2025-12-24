# metrics_prefix.py
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

Token = Union[int, str]


@dataclass
class PrefixCollapseResult:
    # per position (t)
    top1_mass: List[float]
    entropy: List[float]
    eff_support: List[float]
    unique: List[int]
    support: List[int]  # number of active samples at t

    # scalar summaries
    top1_auc: float
    collapse_depth: int  # consecutive from t=0 with top1>=thr
    collapse_thr: float


def _entropy_from_counter(c: Counter) -> float:
    n = sum(c.values())
    if n <= 0:
        return 0.0
    ent = 0.0
    for v in c.values():
        p = v / n
        ent -= p * math.log(p + 1e-12)
    return ent


def prefix_collapse_by_position(
    seqs: Sequence[Sequence[Token]],
    active_before: Optional[Sequence[Sequence[bool]]] = None,
    *,
    max_T: Optional[int] = None,
    collapse_thr: float = 0.95,
) -> PrefixCollapseResult:
    """
    Compute prefix-collapse metrics per position t.

    Args:
      seqs: list of token sequences (already prompt-stripped if you want).
      active_before: same shape as seqs, True if token at t is "active_before" (not past EOS).
                     If None, treat all available positions as active.
      max_T: optional cap on time steps.
      collapse_thr: threshold for collapse_depth.

    Returns:
      PrefixCollapseResult with per-position curves and scalar summaries.
    """
    if len(seqs) == 0:
        return PrefixCollapseResult([], [], [], [], [], 0.0, 0, collapse_thr)

    T = max(len(s) for s in seqs)
    if max_T is not None:
        T = min(T, int(max_T))

    top1_mass: List[float] = []
    entropy: List[float] = []
    eff_support: List[float] = []
    unique: List[int] = []
    support: List[int] = []

    for t in range(T):
        toks: List[Token] = []
        for i, s in enumerate(seqs):
            if len(s) <= t:
                continue
            if active_before is not None:
                if t >= len(active_before[i]) or (not active_before[i][t]):
                    continue
            toks.append(s[t])

        n = len(toks)
        if n == 0:
            break

        c = Counter(toks)
        ent = _entropy_from_counter(c)
        eff = math.exp(ent)
        top1 = max(c.values()) / n

        top1_mass.append(float(top1))
        entropy.append(float(ent))
        eff_support.append(float(eff))
        unique.append(int(len(c)))
        support.append(int(n))

    # scalar summaries
    top1_auc = float(sum(top1_mass) / max(1, len(top1_mass)))
    depth = 0
    for x in top1_mass:
        if x >= collapse_thr:
            depth += 1
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
    )


def prefix_collapse_by_k(
    seqs: Sequence[Sequence[Token]],
    active_before: Optional[Sequence[Sequence[bool]]] = None,
    *,
    k_list: Iterable[int] = (1, 2, 3, 4, 5, 6),
) -> Dict[int, Dict[str, float]]:
    """
    Compute collapse metrics over k-prefixes.
    For each k, we collect prefix tokens[0:k] from samples that are active for all positions < k.

    Returns dict: k -> {"n":..., "unique":..., "top1":..., "top5":..., "entropy":..., "eff":...}
    """
    out: Dict[int, Dict[str, float]] = {}

    for k in k_list:
        k = int(k)
        if k <= 0:
            continue

        prefixes: List[Tuple[Token, ...]] = []
        for i, s in enumerate(seqs):
            if len(s) < k:
                continue
            if active_before is not None:
                # require all positions 0..k-1 active_before True
                ok = True
                m = active_before[i]
                if len(m) < k:
                    ok = False
                else:
                    for t in range(k):
                        if not m[t]:
                            ok = False
                            break
                if not ok:
                    continue
            prefixes.append(tuple(s[:k]))

        n = len(prefixes)
        if n == 0:
            out[k] = {
                "n": 0.0,
                "unique": 0.0,
                "top1": 0.0,
                "top5": 0.0,
                "entropy": 0.0,
                "eff": 0.0,
            }
            continue

        c = Counter(prefixes)
        ent = _entropy_from_counter(c)
        eff = math.exp(ent)
        vals = sorted(c.values(), reverse=True)
        top1 = vals[0] / n
        top5 = sum(vals[:5]) / n

        out[k] = {
            "n": float(n),
            "unique": float(len(c)),
            "top1": float(top1),
            "top5": float(top5),
            "entropy": float(ent),
            "eff": float(eff),
        }

    return out
