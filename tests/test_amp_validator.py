"""Tests for AMPValidator."""
import pytest
import torch


class TestAMPValidatorCall:
    """Test __call__ output shapes and semantics."""

    def test_output_shapes(self):
        from chemgfn.models.validators import AMPValidator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        validator = AMPValidator(oracle_weights_path=None)
        validator.termination_token_id = tokenizer.eos_token_id

        B, T = 2, 5
        aa_ids = [tokenizer.encode(aa, add_special_tokens=False)[0] for aa in "ACDEF"]
        sentences = torch.tensor([aa_ids, aa_ids])

        result = validator(sentences, tokenizer)

        assert result["invalid"].shape == (B, T + 1)
        assert result["local_score"].shape == (B, T + 1)
        assert result["global_score"].shape == (B,)
        assert len(result["full_tokens"]) == B

    def test_all_positions_valid(self):
        """Any prefix of amino acids should be valid."""
        from chemgfn.models.validators import AMPValidator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        validator = AMPValidator(oracle_weights_path=None)
        validator.termination_token_id = tokenizer.eos_token_id

        aa_ids = [tokenizer.encode(aa, add_special_tokens=False)[0] for aa in "MKTAY"]
        sentences = torch.tensor([aa_ids])

        result = validator(sentences, tokenizer)

        # All positions after step 0 should be valid (invalid == 0)
        assert (result["invalid"][:, 1:] == 0.0).all()

    def test_global_score_in_range(self):
        from chemgfn.models.validators import AMPValidator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        validator = AMPValidator(oracle_weights_path=None)
        validator.termination_token_id = tokenizer.eos_token_id

        aa_ids = [tokenizer.encode(aa, add_special_tokens=False)[0] for aa in "AKLWF"]
        sentences = torch.tensor([aa_ids])
        result = validator(sentences, tokenizer)

        assert (result["global_score"] >= 0).all()
        assert (result["global_score"] <= 1).all()

    def test_local_score_absorbing(self):
        """local_score should be non-zero only at terminal position."""
        from chemgfn.models.validators import AMPValidator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        validator = AMPValidator(oracle_weights_path=None)
        validator.termination_token_id = tokenizer.eos_token_id

        aa_ids = [tokenizer.encode(aa, add_special_tokens=False)[0] for aa in "AKLWF"]
        sentences = torch.tensor([aa_ids])
        result = validator(sentences, tokenizer)

        # local_score at terminal position should equal global_score
        assert result["local_score"][0, -1] == result["global_score"][0]
        # All other positions should be 0
        assert (result["local_score"][:, :-1] == 0).all()


class TestAMPValidatorAccuracy:
    def test_accuracy_keys(self):
        from chemgfn.models.validators import AMPValidator
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        validator = AMPValidator(oracle_weights_path=None)
        validator.termination_token_id = tokenizer.eos_token_id

        aa_ids = [tokenizer.encode(aa, add_special_tokens=False)[0] for aa in "AKLWF"]
        sentences = torch.tensor([aa_ids, aa_ids])

        metrics = validator.accuracy(sentences, tokenizer)

        assert "acc" in metrics
        assert "amp_score" in metrics
        assert "diversity" in metrics
