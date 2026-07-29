"""Tests for Levenshtein-based diversity and novelty metrics."""
import pytest


class TestDiversity:
    def test_identical_sequences(self):
        from chemgfn.utils.sequence_metrics import levenshtein_diversity

        seqs = ["AAAA", "AAAA", "AAAA"]
        assert levenshtein_diversity(seqs) == 0.0

    def test_completely_different(self):
        from chemgfn.utils.sequence_metrics import levenshtein_diversity

        seqs = ["AAAA", "CCCC"]
        div = levenshtein_diversity(seqs)
        assert div == 4.0  # 4 substitutions

    def test_single_sequence(self):
        from chemgfn.utils.sequence_metrics import levenshtein_diversity

        assert levenshtein_diversity(["AAAA"]) == 0.0

    def test_empty(self):
        from chemgfn.utils.sequence_metrics import levenshtein_diversity

        assert levenshtein_diversity([]) == 0.0

    def test_known_value(self):
        from chemgfn.utils.sequence_metrics import levenshtein_diversity

        # "AB" vs "CD" = 2, "AB" vs "AC" = 1, "CD" vs "AC" = 2
        # sum = 5, n_pairs = C(3,2) = 3, mean = 5/3 ≈ 1.6667
        seqs = ["AB", "CD", "AC"]
        div = levenshtein_diversity(seqs)
        assert abs(div - 5.0 / 3.0) < 1e-6


class TestNovelty:
    def test_identical_to_training(self):
        from chemgfn.utils.sequence_metrics import levenshtein_novelty

        generated = ["AAAA", "CCCC"]
        training = ["AAAA", "CCCC", "GGGG"]
        assert levenshtein_novelty(generated, training) == 0.0

    def test_completely_novel(self):
        from chemgfn.utils.sequence_metrics import levenshtein_novelty

        generated = ["WWWW"]
        training = ["AAAA"]
        assert levenshtein_novelty(generated, training) == 4.0

    def test_known_value(self):
        from chemgfn.utils.sequence_metrics import levenshtein_novelty

        # gen "AB": min to {"AA","CC"} = min(1,2) = 1
        # gen "CD": min to {"AA","CC"} = min(2,1) = 1
        # novelty = (1+1)/2 = 1.0
        generated = ["AB", "CD"]
        training = ["AA", "CC"]
        assert levenshtein_novelty(generated, training) == 1.0

    def test_empty_generated(self):
        from chemgfn.utils.sequence_metrics import levenshtein_novelty

        assert levenshtein_novelty([], ["AAAA"]) == 0.0


class TestTopKSelection:
    def test_topk_basic(self):
        from chemgfn.utils.sequence_metrics import select_topk

        seqs = ["A", "B", "C", "D"]
        scores = [0.1, 0.9, 0.5, 0.7]
        top_seqs, top_scores = select_topk(seqs, scores, k=2)
        assert top_seqs == ["B", "D"]
        assert top_scores == [0.9, 0.7]

    def test_topk_k_larger_than_n(self):
        from chemgfn.utils.sequence_metrics import select_topk

        seqs = ["A", "B"]
        scores = [0.3, 0.7]
        top_seqs, top_scores = select_topk(seqs, scores, k=10)
        assert len(top_seqs) == 2
