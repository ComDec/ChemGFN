"""Levenshtein-based diversity and novelty metrics.

Implements Eq. 2 (Diversity) and Eq. 3 (Novelty) from
Jain et al. 2022, "Biological Sequence Design with GFlowNets".
Distance function: raw (unnormalized) Levenshtein edit distance via polyleven.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

from polyleven import levenshtein


def select_topk(
    sequences: Sequence[str],
    scores: Sequence[float],
    k: int,
) -> tuple[list[str], list[float]]:
    """Select top-k sequences by score (descending)."""
    if not sequences:
        return [], []
    paired = sorted(zip(scores, sequences), reverse=True)
    paired = paired[: min(k, len(paired))]
    top_scores, top_seqs = zip(*paired) if paired else ([], [])
    return list(top_seqs), list(top_scores)


def levenshtein_diversity(sequences: Sequence[str]) -> float:
    """Mean pairwise Levenshtein edit distance (Eq. 2).

    Diversity(D) = sum_{i!=j} d(x_i, x_j) / (|D| * (|D| - 1))

    Since d is symmetric, sum over unordered pairs and divide by C(n,2).
    """
    n = len(sequences)
    if n <= 1:
        return 0.0
    total = sum(levenshtein(a, b) for a, b in combinations(sequences, 2))
    # combinations gives n*(n-1)/2 pairs; paper divides by n*(n-1) [ordered pairs]
    return total / (n * (n - 1) / 2)


def levenshtein_novelty(
    generated: Sequence[str],
    training_set: Sequence[str],
) -> float:
    """Mean minimum Levenshtein distance to training set (Eq. 3).

    Novelty(D) = sum_{x_i in D} min_{s_j in D0} d(x_i, s_j) / |D|
    """
    if not generated or not training_set:
        return 0.0
    total = 0.0
    for seq in generated:
        min_dist = min(levenshtein(seq, ref) for ref in training_set)
        total += min_dist
    return total / len(generated)
