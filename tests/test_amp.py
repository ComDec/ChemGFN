"""Tests for the AMP oracle and validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chemgfn.models.amp_oracle import AMPMLP, AMPOracle
from chemgfn.models.validators import AMPValidator
from tests.test_validators import batch_from_strings

ORACLE_WEIGHTS = Path(__file__).resolve().parents[1] / "data" / "AMP" / "oracle_weights.pt"
PROT_ALBERT = "Rostlab/prot_albert"


class _StubOracle:
    """Oracle returning a fixed score per sequence, avoiding the ProtTrans download."""

    def __init__(self, score: float = 0.75) -> None:
        self.score = score
        self.seen: list[list[str]] = []

    def score_sequences(self, sequences):
        self.seen.append(list(sequences))
        return torch.full((len(sequences),), self.score)


@pytest.fixture
def validator():
    """AMP validator whose oracle is replaced by a deterministic stub."""
    validator = AMPValidator(oracle_weights_path=None)
    validator.oracle = _StubOracle()
    return validator


class TestAMPMLP:
    """Classifier head of the oracle."""

    def test_output_shape(self):
        logits = AMPMLP()(torch.randn(4, 4096))
        assert logits.shape == (4, 2)

    def test_scores_are_probabilities(self):
        probs = torch.softmax(AMPMLP()(torch.randn(8, 4096)), dim=1)

        assert (probs >= 0).all() and (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_dropout_is_disabled_in_eval_mode(self):
        mlp = AMPMLP().eval()
        features = torch.randn(2, 4096)

        assert torch.allclose(mlp(features), mlp(features))

    def test_mc_dropout_perturbs_the_output(self):
        mlp = AMPMLP().eval()
        features = torch.randn(2, 4096)

        assert not torch.allclose(mlp(features, mc_dropout=True), mlp(features, mc_dropout=True))


class TestAMPOracle:
    """Oracle wrapper around the classifier head."""

    def test_released_weights_load(self):
        if not ORACLE_WEIGHTS.exists():
            pytest.skip(f"Oracle weights not found at {ORACLE_WEIGHTS}")

        oracle = AMPOracle(weights_path=ORACLE_WEIGHTS, device="cpu")
        assert not oracle.mlp.training

    def test_the_encoder_is_loaded_lazily(self):
        # Constructing the oracle must not pull the ProtTrans encoder.
        oracle = AMPOracle(weights_path=None, device="cpu")
        assert oracle._encoder is None

    def test_scoring_an_empty_batch(self):
        assert AMPOracle(weights_path=None, device="cpu").score_sequences([]).numel() == 0

    @pytest.mark.requires_model
    def test_scores_lie_in_the_unit_interval(self):
        from tests.conftest import load_tokenizer_or_skip

        load_tokenizer_or_skip(PROT_ALBERT)
        oracle = AMPOracle(weights_path=None, device="cpu")
        scores = oracle.score_sequences(["AAAAAKKKKKK", "RRRRRGGGGG"])

        assert scores.shape == (2,)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestAMPValidator:
    """Peptide validity and the absorbed terminal oracle score."""

    def test_output_shapes(self, validator, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["ACDEF", "MKTAY"])
        out = validator(sentences, gpt2_tokenizer)

        batch_size, seq_len = sentences.shape
        assert out["invalid"].shape == (batch_size, seq_len + 1)
        assert out["local_score"].shape == (batch_size, seq_len + 1)
        assert out["global_score"].shape == (batch_size,)
        assert len(out["full_tokens"]) == batch_size

    def test_every_amino_acid_prefix_is_legal(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["MKTAY"]), gpt2_tokenizer)

        # State 0 is the empty prefix; states 1..5 each add one amino acid.
        assert (out["invalid"][0, 1:6] == 0.0).all()

    def test_the_oracle_score_is_placed_at_the_terminal_state(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["AKLWF"]), gpt2_tokenizer)

        assert out["global_score"][0].item() == pytest.approx(0.75)
        assert out["local_score"][0, 5].item() == pytest.approx(0.75)
        assert (out["local_score"][0, :5] == 0.0).all()
        assert (out["local_score"][0, 6:] == 0.0).all()

    def test_non_peptide_output_is_not_scored(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["123"]), gpt2_tokenizer)

        assert out["global_score"][0].item() == pytest.approx(0.0)
        assert validator.oracle.seen == []

    def test_the_oracle_sees_the_decoded_peptides(self, validator, gpt2_tokenizer):
        validator(batch_from_strings(gpt2_tokenizer, ["ACD", "EFG"]), gpt2_tokenizer)
        assert validator.oracle.seen == [["ACD", "EFG"]]

    def test_accuracy_reports_the_paper_metrics(self, validator, gpt2_tokenizer, tmp_path):
        training_set = tmp_path / "train.txt"
        training_set.write_text("ACDEF\nMKTAY\n")
        validator._training_sequences = ["ACDEF", "MKTAY"]

        metrics = validator.accuracy(
            batch_from_strings(gpt2_tokenizer, ["AKLWF", "ACDEF"]), gpt2_tokenizer
        )

        for key in ("acc", "amp_score", "diversity"):
            assert key in metrics
