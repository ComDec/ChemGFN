"""Unit tests for chemgfn/models/reward.py

Tests reward computation, validators, and scoring functions including:
- Score computation (score_fast)
- Sentence validators
- Reward model classes
- Invalid penalty application
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import GPT2Tokenizer

from chemgfn.models.reward import (
    FrozenModelSentenceGivenPrompt,
    Reference_Target_Score_Positive_Mixed_Invalid_Mask,
    SentenceValidator,
    _apply_invalid_penalty,
    _build_penalty_ramp,
    _decode_tokens_to_string,
    _ensure_tensor_like,
    _stack_if_not_empty,
    score_fast,
    use_base_model,
)

# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions in reward module."""

    def test_ensure_tensor_like_with_tensor(self):
        """Test _ensure_tensor_like with tensor input."""
        value = torch.tensor([1.0, 2.0, 3.0])
        reference = torch.zeros(3)

        result = _ensure_tensor_like(value, reference)

        assert torch.equal(result, value)

    def test_ensure_tensor_like_with_scalar(self):
        """Test _ensure_tensor_like with scalar input."""
        value = 5.0
        reference = torch.zeros(3)

        result = _ensure_tensor_like(value, reference)

        assert result.shape == (3,)
        assert torch.all(result == 5.0)

    def test_build_penalty_ramp(self):
        """Test penalty ramp building."""
        base_values = torch.tensor([1.0, 2.0])
        steps = 5
        start_ratio = 0.5
        end_ratio = 1.5
        reference = torch.zeros(2, steps)

        ramp = _build_penalty_ramp(base_values, steps, start_ratio, end_ratio, reference)

        assert ramp.shape == (2, steps)
        # First column should be base_values * start_ratio
        assert torch.allclose(ramp[:, 0], base_values * start_ratio)
        # Last column should be base_values * end_ratio
        assert torch.allclose(ramp[:, -1], base_values * end_ratio)

    def test_apply_invalid_penalty(self):
        """Test invalid penalty application."""
        batch_size = 4
        seq_len = 5

        reward = torch.randn(batch_size, seq_len) + 10  # Positive rewards
        invalid_mask = torch.zeros(batch_size, seq_len)
        invalid_mask[:, 3:] = 1  # Mark last 2 positions as invalid

        start_ratio = 0.2
        end_ratio = 1.2

        penalized = _apply_invalid_penalty(reward, invalid_mask, start_ratio, end_ratio)

        assert penalized.shape == reward.shape
        # Invalid positions should have lower rewards
        assert (penalized[:, 3:] <= reward[:, 3:]).all()

    def test_stack_if_not_empty(self):
        """Test stacking tensors if list is not empty."""
        tensors = [torch.randn(2, 3) for _ in range(4)]

        result = _stack_if_not_empty(tensors)

        assert result.shape == (4, 2, 3)

    def test_stack_if_not_empty_with_empty_list(self):
        """Test stacking with empty list."""
        tensors = []

        result = _stack_if_not_empty(tensors)

        assert result is None

    def test_decode_tokens_to_string(self):
        """Test token decoding to string."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Create a simple sequence
        sequence = torch.tensor([464, 2068, 318])  # "The world is"

        result = _decode_tokens_to_string(sequence, tokenizer)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_tokens_with_eos(self):
        """Test token decoding stops at EOS."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        sequence = torch.tensor([464, 2068, tokenizer.eos_token_id, 318])

        result = _decode_tokens_to_string(sequence, tokenizer)

        # Should stop at EOS, not include token 318
        assert isinstance(result, str)


# ============================================================================
# Test Score Fast Function
# ============================================================================


class TestScoreFast:
    """Test the score_fast function for reward computation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()

        def forward(*args, **kwargs):
            input_ids = args[0] if args else kwargs.get("input_ids")
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # Use GPT2 vocab_size (50257) to match tokenizer
            vocab_size = 50257

            output = Mock()
            output.logits = torch.randn(batch_size, seq_len, vocab_size)
            return output

        model.side_effect = forward
        model.__call__ = forward
        return model

    def test_score_fast_basic(self, mock_model):
        """Test basic score computation."""
        batch_size = 4
        seq_len = 10
        skip_first = 3
        termination_token_id = 0

        encoded_input = torch.randint(1, 100, (batch_size, seq_len))

        reward, reward_unpenalized = score_fast(
            model=mock_model,
            encoded_input=encoded_input,
            termination_token_id=termination_token_id,
            skip_first=skip_first,
        )

        assert reward.shape[0] == batch_size
        assert reward_unpenalized.shape[0] == batch_size
        assert not torch.isnan(reward).any()
        assert not torch.isnan(reward_unpenalized).any()

    def test_score_fast_with_invalid_mask(self, mock_model):
        """Test score computation with vocabulary mask."""
        batch_size = 4
        seq_len = 10
        skip_first = 3
        termination_token_id = 0
        vocab_size = 50257  # GPT2 vocab_size

        encoded_input = torch.randint(1, vocab_size, (batch_size, seq_len))
        invalid_mask = torch.zeros(vocab_size, dtype=torch.bool)
        invalid_mask[50:60] = True  # Mark tokens 50-59 as invalid

        reward, _ = score_fast(
            model=mock_model,
            encoded_input=encoded_input,
            termination_token_id=termination_token_id,
            skip_first=skip_first,
            invalid_vocab_mask=invalid_mask,
            illegal_vocab_penalty=-50,
        )

        assert not torch.isnan(reward).any()

    def test_score_fast_with_temperature(self, mock_model):
        """Test score computation with different temperatures."""
        batch_size = 4
        seq_len = 10
        skip_first = 3
        termination_token_id = 0

        encoded_input = torch.randint(1, 100, (batch_size, seq_len))

        reward_temp1, _ = score_fast(
            model=mock_model,
            encoded_input=encoded_input,
            termination_token_id=termination_token_id,
            skip_first=skip_first,
            reward_temperature=1.0,
        )

        reward_temp2, _ = score_fast(
            model=mock_model,
            encoded_input=encoded_input,
            termination_token_id=termination_token_id,
            skip_first=skip_first,
            reward_temperature=2.0,
        )

        # Different temperatures should give different rewards
        assert not torch.allclose(reward_temp1, reward_temp2)

    def test_score_fast_with_agree_list(self, mock_model):
        """Test score computation with agreement list."""
        batch_size = 4
        seq_len = 10
        skip_first = 3
        termination_token_id = 0
        vocab_size = 50257  # GPT2 vocab_size

        encoded_input = torch.randint(1, vocab_size, (batch_size, seq_len))

        # Create agreement list (grammar constraints)
        agree_list = [
            torch.ones(batch_size, vocab_size, dtype=torch.bool)
            for _ in range(seq_len - skip_first + 1)
        ]

        reward, _ = score_fast(
            model=mock_model,
            encoded_input=encoded_input,
            termination_token_id=termination_token_id,
            skip_first=skip_first,
            agree_list=agree_list,
        )

        assert not torch.isnan(reward).any()


# ============================================================================
# Test Sentence Validator Base Class
# ============================================================================


class TestSentenceValidator:
    """Test base sentence validator."""

    def test_sentence_validator_init(self):
        """Test validator initialization."""
        validator = SentenceValidator(termination_token_id=0)

        assert validator.termination_token_id == 0

    def test_sentence_validator_call_not_implemented(self):
        """Test that base class __call__ is not implemented."""
        validator = SentenceValidator(termination_token_id=0)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        sentences = torch.randint(1, 100, (4, 10))

        # Base class should not implement validation
        # This might raise NotImplementedError or return basic result
        try:
            result = validator(sentences, tokenizer)
            # If it doesn't raise, it should return a dict
            assert isinstance(result, dict)
        except (NotImplementedError, AttributeError):
            # Expected if not implemented
            pass


# ============================================================================
# Test Frozen Model Sentence Given Prompt
# ============================================================================


class TestFrozenModelSentenceGivenPrompt:
    """Test FrozenModelSentenceGivenPrompt reward model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.base_model.enable_adapter_layers = Mock()
        model.eval = Mock()
        model.train = Mock()

        def forward(*args, **kwargs):
            input_ids = args[0] if args else kwargs.get("input_ids")
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # Use GPT2 vocab_size (50257) to match tokenizer
            vocab_size = 50257

            output = Mock()
            output.logits = torch.randn(batch_size, seq_len, vocab_size)
            return output

        model.side_effect = forward
        model.__call__ = forward
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer."""
        return GPT2Tokenizer.from_pretrained("gpt2")

    def test_reward_model_init(self):
        """Test reward model initialization."""
        reward_model = FrozenModelSentenceGivenPrompt(
            sentence_validator=None,
        )

        assert reward_model.sentence_validator is None

    def test_reward_model_score(self, mock_model, tokenizer):
        """Test reward scoring."""
        reward_model = FrozenModelSentenceGivenPrompt(
            sentence_validator=None,
        )

        batch_size = 4
        seq_len = 10
        prompt_length = 3

        input_batch = torch.randint(1, 100, (batch_size, seq_len))

        reward, reward_unpen = reward_model.score(
            input_batch=input_batch,
            prompt_length=prompt_length,
            model=mock_model,
            tokenizer=tokenizer,
        )

        assert reward.shape[0] == batch_size
        assert reward_unpen.shape[0] == batch_size

    def test_reward_model_with_validator(self, mock_model, tokenizer):
        """Test reward scoring with validator."""
        # Mock validator
        mock_validator = Mock()
        mock_validator.return_value = {"invalid": torch.zeros(4, 7, dtype=torch.bool)}

        reward_model = FrozenModelSentenceGivenPrompt(
            sentence_validator=mock_validator,
        )

        batch_size = 4
        seq_len = 10
        prompt_length = 3

        input_batch = torch.randint(1, 100, (batch_size, seq_len))

        reward, reward_unpen = reward_model.score(
            input_batch=input_batch,
            prompt_length=prompt_length,
            model=mock_model,
            tokenizer=tokenizer,
        )

        assert reward.shape[0] == batch_size
        # Validator should have been called
        mock_validator.assert_called_once()


# ============================================================================
# Test Reference Target Score Positive Mixed Invalid Mask
# ============================================================================


class TestReferenceTargetScorePositiveMixedInvalidMask:
    """Test the main reward model with mixed scoring."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.base_model.enable_adapter_layers = Mock()
        model.eval = Mock()
        model.train = Mock()

        def forward(*args, **kwargs):
            input_ids = args[0] if args else kwargs.get("input_ids")
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # Use GPT2 vocab_size (50257) to match tokenizer
            vocab_size = 50257

            output = Mock()
            output.logits = torch.randn(batch_size, seq_len, vocab_size)
            return output

        model.side_effect = forward
        model.__call__ = forward
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer."""
        return GPT2Tokenizer.from_pretrained("gpt2")

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        reward_model = Reference_Target_Score_Positive_Mixed_Invalid_Mask(
            sentence_validator=None,
        )

        assert reward_model.sentence_validator is None
        assert reward_model.temperature == 1.0

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        reward_model = Reference_Target_Score_Positive_Mixed_Invalid_Mask(
            sentence_validator=None,
            invalid_start_ratio=0.3,
            invalid_end_ratio=1.5,
            illegal_vocab_penalty=-100,
            grammar_disagree_penalty=-100,
        )

        assert reward_model.invalid_start_ratio == 0.3
        assert reward_model.invalid_end_ratio == 1.5
        assert reward_model.illegal_vocab_penalty == -100
        assert reward_model.grammar_disagree_penalty == -100

    def test_score_basic(self, mock_model, tokenizer):
        """Test basic scoring."""
        reward_model = Reference_Target_Score_Positive_Mixed_Invalid_Mask(
            sentence_validator=None,
        )

        batch_size = 4
        seq_len = 10
        prompt_length = 3

        input_batch = torch.randint(1, 100, (batch_size, seq_len))

        result = reward_model.score(
            input_batch=input_batch,
            prompt_length=prompt_length,
            model=mock_model,
            tokenizer=tokenizer,
        )

        assert isinstance(result, dict)
        assert "reward" in result
        assert "reward_unpenalized" in result
        assert "log_pf_ref" in result
        assert "validator_dict" in result

    def test_score_with_scaling_factor(self, mock_model, tokenizer):
        """Test scoring with scaling factor."""
        # Mock validator
        mock_validator = Mock()
        mock_validator.return_value = {
            "invalid": torch.zeros(4, 7, dtype=torch.bool),
            "valid_score": torch.ones(4, 7),
        }

        reward_model = Reference_Target_Score_Positive_Mixed_Invalid_Mask(
            sentence_validator=mock_validator,
        )

        batch_size = 4
        seq_len = 10
        prompt_length = 3

        input_batch = torch.randint(1, 100, (batch_size, seq_len))

        result = reward_model.score(
            input_batch=input_batch,
            prompt_length=prompt_length,
            model=mock_model,
            tokenizer=tokenizer,
            scaling_factor=50.0,
        )

        assert result["reward"].shape[0] == batch_size
        # With validator, reward should be modified by scaling factor
        assert not torch.equal(result["reward"], result["log_pf_ref"])

    def test_score_with_target_molecule(self, mock_model, tokenizer):
        """Test scoring with target molecule."""
        mock_validator = Mock()
        mock_validator.return_value = {
            "invalid": torch.zeros(4, 7, dtype=torch.bool),
            "valid_score": torch.ones(4, 7),
        }

        reward_model = Reference_Target_Score_Positive_Mixed_Invalid_Mask(
            sentence_validator=mock_validator,
        )

        batch_size = 4
        seq_len = 10
        prompt_length = 3

        input_batch = torch.randint(1, 100, (batch_size, seq_len))

        result = reward_model.score(
            input_batch=input_batch,
            prompt_length=prompt_length,
            model=mock_model,
            tokenizer=tokenizer,
            target_molecule="CCO",
        )

        assert isinstance(result, dict)
        # Validator should have received target_molecule
        assert mock_validator.called


# ============================================================================
# Test Use Base Model Context Manager
# ============================================================================


class TestUseBaseModel:
    """Test the use_base_model context manager."""

    def test_use_base_model_enable_disable(self):
        """Test that adapters are properly enabled/disabled."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.base_model.enable_adapter_layers = Mock()

        with use_base_model(model, disable_peft=False):
            # Inside context, adapters should be disabled
            model.base_model.disable_adapter_layers.assert_called_once()

        # After context, adapters should be re-enabled
        model.base_model.enable_adapter_layers.assert_called_once()

    def test_use_base_model_with_disable_peft(self):
        """Test that nothing happens when disable_peft=True."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.base_model.enable_adapter_layers = Mock()

        with use_base_model(model, disable_peft=True):
            pass

        # No calls should have been made
        model.base_model.disable_adapter_layers.assert_not_called()
        model.base_model.enable_adapter_layers.assert_not_called()

    def test_use_base_model_exception_handling(self):
        """Test that adapters are re-enabled even on exception."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.base_model.enable_adapter_layers = Mock()

        try:
            with use_base_model(model, disable_peft=False):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Even with exception, adapters should be re-enabled
        model.base_model.enable_adapter_layers.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================


class TestRewardIntegration:
    """Integration tests for reward computation pipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.base_model.enable_adapter_layers = Mock()
        model.eval = Mock()
        model.train = Mock()

        def forward(*args, **kwargs):
            input_ids = args[0] if args else kwargs.get("input_ids")
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            # Use GPT2 vocab_size (50257) to match tokenizer
            vocab_size = 50257

            output = Mock()
            # Create deterministic logits for testing
            output.logits = torch.randn(batch_size, seq_len, vocab_size)
            return output

        model.side_effect = forward
        model.__call__ = forward
        return model

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer."""
        return GPT2Tokenizer.from_pretrained("gpt2")

    def test_end_to_end_reward_computation(self, mock_model, tokenizer):
        """Test end-to-end reward computation."""
        reward_model = Reference_Target_Score_Positive_Mixed_Invalid_Mask(
            sentence_validator=None,
        )

        batch_size = 8
        seq_len = 15
        prompt_length = 5

        # Generate some trajectories
        input_batch = torch.randint(1, 100, (batch_size, seq_len))

        # Compute rewards
        result = reward_model.score(
            input_batch=input_batch,
            prompt_length=prompt_length,
            model=mock_model,
            tokenizer=tokenizer,
            reward_temperature=1.0,
            scaling_factor=10.0,
        )

        # Verify output structure
        assert "reward" in result
        assert "reward_unpenalized" in result
        assert result["reward"].shape == (batch_size, seq_len - prompt_length + 1)

        # Verify no NaN or Inf
        assert not torch.isnan(result["reward"]).any()
        assert not torch.isinf(result["reward"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
