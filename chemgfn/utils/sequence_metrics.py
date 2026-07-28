"""Levenshtein-based diversity and novelty metrics for generated sequences.

Implements the Diversity (Eq. 2) and Novelty (Eq. 3) metrics of Jain et al.
(2022), "Biological Sequence Design with GFlowNets", using the raw
(unnormalised) Levenshtein edit distance. Used to score the top-k samples of the
AMP task.
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
    """Select the ``k`` highest-scoring sequences.

    Args:
        sequences: Candidate sequences.
        scores: Score of each sequence, aligned with ``sequences``.
        k: Number of sequences to keep.

    Returns:
        The selected sequences and their scores, in descending score order.
    """
    if not sequences:
        return [], []
    paired = sorted(zip(scores, sequences), reverse=True)
    paired = paired[: min(k, len(paired))]
    top_scores, top_seqs = zip(*paired) if paired else ([], [])
    return list(top_seqs), list(top_scores)


def levenshtein_diversity(sequences: Sequence[str]) -> float:
    """Mean pairwise Levenshtein distance over a set of sequences.

    ``Diversity(D) = sum_{i != j} d(x_i, x_j) / (|D| * (|D| - 1))``. Distance is
    symmetric, so the sum runs over unordered pairs and is divided by ``C(n, 2)``.
    Returns 0.0 for fewer than two sequences.
    """
    n = len(sequences)
    if n <= 1:
        return 0.0
    total = sum(levenshtein(a, b) for a, b in combinations(sequences, 2))
    return total / (n * (n - 1) / 2)


def levenshtein_novelty(
    generated: Sequence[str],
    training_set: Sequence[str],
) -> float:
    """Mean minimum Levenshtein distance from each generated sequence to the training set.

    ``Novelty(D) = sum_{x_i in D} min_{s_j in D0} d(x_i, s_j) / |D|``. Returns 0.0
    when either collection is empty.
    """
    if not generated or not training_set:
        return 0.0
    total = 0.0
    for seq in generated:
        min_dist = min(levenshtein(seq, ref) for ref in training_set)
        total += min_dist
    return total / len(generated)
