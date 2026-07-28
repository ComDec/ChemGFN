"""Tests for the task validators in ``chemgfn.models.validators``."""

from __future__ import annotations

import pytest
import torch

from chemgfn.models.validators import (
    CommonGenValidator,
    Expr24Validator,
    RDKitValidator,
    Validator,
)


def encode_chars(tokenizer, text: str, pad_to: int | None = None) -> list[int]:
    """Encode ``text`` one character per token, padding with EOS.

    Generation is vocabulary-constrained to single-character tokens, so this mirrors the token
    layout the validators see at training time.
    """
    ids = [tokenizer.encode(ch, add_special_tokens=False)[0] for ch in text]
    if pad_to is not None:
        ids += [tokenizer.eos_token_id] * (pad_to - len(ids))
    return ids


def batch_from_strings(tokenizer, texts: list[str]) -> torch.Tensor:
    """Build a padded ``(B, T)`` token batch from character strings."""
    width = max(len(t) for t in texts) + 1
    return torch.tensor([encode_chars(tokenizer, t, pad_to=width) for t in texts])


class TestValidatorBase:
    """Contract of the shared base class."""

    def test_scoring_is_abstract(self, gpt2_tokenizer):
        with pytest.raises(NotImplementedError):
            Validator()(torch.zeros(1, 1, dtype=torch.long), gpt2_tokenizer)

    def test_accuracy_defaults_to_no_metrics(self, gpt2_tokenizer):
        assert Validator().accuracy(torch.zeros(1, 1, dtype=torch.long), gpt2_tokenizer) == {}


class TestExpr24Validator:
    """Arithmetic expressions targeting a fixed value."""

    @pytest.fixture
    def validator(self):
        return Expr24Validator()

    def test_output_shapes(self, validator, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["8*3", "1+1"])
        out = validator(sentences, gpt2_tokenizer)

        batch_size, seq_len = sentences.shape
        assert out["invalid"].shape == (batch_size, seq_len + 1)
        assert out["local_score"].shape == (batch_size, seq_len + 1)
        assert out["global_score"].shape == (batch_size,)
        assert len(out["full_tokens"]) == batch_size

    def test_hitting_the_target_scores_one(self, validator, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["8*3"])
        out = validator(sentences, gpt2_tokenizer)

        assert out["global_score"][0].item() == pytest.approx(1.0)
        assert out["invalid"][0, -1].item() == 0.0
        assert out["full_tokens"] == ["8*3"]

    def test_missing_the_target_scores_zero(self, validator, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["8*4"])
        out = validator(sentences, gpt2_tokenizer)

        assert out["global_score"][0].item() == pytest.approx(0.0)
        assert out["invalid"][0, -1].item() == 1.0

    def test_prefixes_are_scored_densely(self, validator, gpt2_tokenizer):
        # "2" -> 2, "2*" -> unparseable, "2*3" -> 6, "2*3*" -> unparseable, "2*3*4" -> 24
        sentences = batch_from_strings(gpt2_tokenizer, ["2*3*4"])
        out = validator(sentences, gpt2_tokenizer)

        assert out["local_score"][0, :6].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_precedence_is_respected(self, validator, gpt2_tokenizer):
        # 2 + 2 * 11 = 24 under standard precedence, 44 if evaluated left to right.
        out = validator(batch_from_strings(gpt2_tokenizer, ["2+2*11"]), gpt2_tokenizer)
        assert out["global_score"][0].item() == pytest.approx(1.0)

    def test_division_by_zero_is_invalid(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["8/0"]), gpt2_tokenizer)
        assert out["global_score"][0].item() == pytest.approx(0.0)
        assert out["invalid"][0, -1].item() == 1.0

    def test_a_trailing_operator_is_invalid(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["8*"]), gpt2_tokenizer)
        assert out["global_score"][0].item() == pytest.approx(0.0)

    def test_the_target_value_is_configurable(self, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["3*4"])
        assert Expr24Validator(target_value=12)(sentences, gpt2_tokenizer)["global_score"][
            0
        ].item() == pytest.approx(1.0)
        assert Expr24Validator(target_value=24)(sentences, gpt2_tokenizer)["global_score"][
            0
        ].item() == pytest.approx(0.0)

    def test_accuracy_is_the_hit_rate(self, validator, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["8*3", "8*4", "4*6", "1+1"])
        assert validator.accuracy(sentences, gpt2_tokenizer)["acc"] == pytest.approx(0.5)

    def test_accuracy_of_an_empty_batch(self, validator, gpt2_tokenizer):
        empty = torch.zeros(0, 4, dtype=torch.long)
        assert validator.accuracy(empty, gpt2_tokenizer)["acc"] == 0.0


class TestRDKitValidator:
    """SMILES molecules scored with an RDKit property function."""

    @pytest.fixture
    def validator(self):
        return RDKitValidator(scorer="qed")

    def test_output_shapes(self, validator, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["CCO", "CCC"])
        out = validator(sentences, gpt2_tokenizer)

        batch_size, seq_len = sentences.shape
        assert out["invalid"].shape == (batch_size, seq_len + 1)
        assert out["local_score"].shape == (batch_size, seq_len + 1)
        assert out["global_score"].shape == (batch_size,)

    def test_a_valid_molecule_scores_and_is_legal(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["CCO"]), gpt2_tokenizer)

        assert out["global_score"][0].item() > 0.0
        # States 1..3 correspond to the prefixes "C", "CC" and "CCO", all valid molecules.
        assert out["invalid"][0, 1:4].tolist() == [0.0, 0.0, 0.0]
        assert out["local_score"][0, 3].item() > 0.0

    def test_an_unparseable_string_scores_zero(self, validator, gpt2_tokenizer):
        out = validator(batch_from_strings(gpt2_tokenizer, ["C))"]), gpt2_tokenizer)
        assert out["global_score"][0].item() == pytest.approx(0.0)

    def test_an_incomplete_prefix_is_not_scored(self, validator, gpt2_tokenizer):
        # "CC(" cannot be read as a molecule yet, though it is still an extendable prefix.
        out = validator(batch_from_strings(gpt2_tokenizer, ["CC("]), gpt2_tokenizer)
        assert out["local_score"][0, 3].item() == pytest.approx(0.0)

    def test_the_scaffold_receives_the_fragment(self, validator, gpt2_tokenizer):
        scaffold = "O=C1Nc2cc(*)ccc2N1"
        out = validator(batch_from_strings(gpt2_tokenizer, ["CC"]), gpt2_tokenizer, scaffold)

        assert out["full_tokens"] == ["O=C1Nc2cc(CC)ccc2N1"]
        assert out["global_score"][0].item() > 0.0

    def test_the_scorer_selects_the_property(self, gpt2_tokenizer):
        sentences = batch_from_strings(gpt2_tokenizer, ["CCCCCCCC"])
        qed = RDKitValidator(scorer="qed")(sentences, gpt2_tokenizer)["global_score"]
        logp = RDKitValidator(scorer="logP")(sentences, gpt2_tokenizer)["global_score"]

        assert not torch.allclose(qed, logp)

    def test_an_unknown_scorer_is_rejected(self):
        with pytest.raises(KeyError):
            RDKitValidator(scorer="not-a-property")

    def test_accuracy_reports_validity_and_diversity(self, validator, gpt2_tokenizer):
        metrics = validator.accuracy(
            batch_from_strings(gpt2_tokenizer, ["CCO", "C))", "CCC"]), gpt2_tokenizer
        )

        assert metrics["acc"] == pytest.approx(2 / 3)
        assert metrics["qed"] > 0.0
        assert "fp_div_internal_valid" in metrics


class TestCommonGenValidator:
    """Concept-covering sentence generation."""

    @pytest.fixture
    def validator(self):
        try:
            return CommonGenValidator()
        except ImportError as exc:
            pytest.skip(str(exc))

    def test_covering_every_concept_scores_higher(self, validator, gpt2_tokenizer):
        scaffold = {"concepts": ["dog", "run", "field"], "references": ["A dog runs in a field."]}
        covered = batch_from_strings(gpt2_tokenizer, ["A dog runs in a field."])
        uncovered = batch_from_strings(gpt2_tokenizer, ["A cat sleeps on a sofa."])

        covered_score = validator(covered, gpt2_tokenizer, scaffold)["global_score"][0]
        uncovered_score = validator(uncovered, gpt2_tokenizer, scaffold)["global_score"][0]

        assert covered_score > uncovered_score

    def test_output_shapes(self, validator, gpt2_tokenizer):
        scaffold = {"concepts": ["dog"], "references": ["A dog runs."]}
        sentences = batch_from_strings(gpt2_tokenizer, ["A dog runs."])
        out = validator(sentences, gpt2_tokenizer, scaffold)

        batch_size, seq_len = sentences.shape
        assert out["invalid"].shape == (batch_size, seq_len + 1)
        assert out["local_score"].shape == (batch_size, seq_len + 1)
        assert out["global_score"].shape == (batch_size,)
