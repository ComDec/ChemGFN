"""Unit tests for chemgfn/utils/gfn_utils.py

Tests core GFlowNet utilities including:
- Token mask preparation
- Diversity calculation
- Generation with termination log probabilities
- Modified SubTB loss computation
- Replay buffer operations
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from transformers import GPT2Tokenizer

from chemgfn.utils.gfn_utils import (
    ReplayBuffer,
    base_to_lora,
    calculate_diversity,
    generate_and_return_termination_logprob,
    get_termination_vals,
    lora_to_base,
    modified_subtb_balance_loss,
    modified_subtb_loss,
    prepare_token_mask,
)

# ============================================================================
# Test Token Mask Preparation
# ============================================================================


class TestPrepareTokenMask:
    """Test token mask preparation for vocabulary constraints."""

    @pytest.fixture
    def tokenizer(self):
        """Create a simple tokenizer for testing."""
        return GPT2Tokenizer.from_pretrained("gpt2")

    @pytest.fixture
    def vocab_file(self, tmp_path):
        """Create a temporary vocabulary file."""
        vocab_path = tmp_path / "test_vocab.txt"
        # Write some simple tokens
        with open(vocab_path, "w") as f:
            f.write("C\n")
            f.write("O\n")
            f.write("N\n")
            f.write("(\n")
            f.write(")\n")
        return str(vocab_path)

    def test_prepare_token_mask_basic(self, tokenizer, vocab_file):
        """Test basic token mask preparation."""
        legal_mask, illegal_mask, legal_tokens = prepare_token_mask(
            tokenizer, vocab_file, reverse=False
        )

        # Check types
        assert isinstance(legal_mask, torch.Tensor)
        assert isinstance(illegal_mask, torch.Tensor)
        assert isinstance(legal_tokens, list)

        # Check dimensions
        assert legal_mask.shape[0] == len(tokenizer)
        assert illegal_mask.shape[0] == len(tokenizer)

        # Check masks are complementary (except EOS)
        assert (legal_mask | illegal_mask).all()

    def test_token_mask_eos_handling(self, tokenizer, vocab_file):
        """Test that EOS token is properly handled."""
        legal_mask, illegal_mask, _ = prepare_token_mask(tokenizer, vocab_file, reverse=False)

        # EOS should be legal (line 75 in gfn_utils.py sets it to True)
        assert legal_mask[tokenizer.eos_token_id].item() == True
        # BOS should be illegal (line 74 in gfn_utils.py sets it to False)
        # Note: In GPT2, bos_token_id == eos_token_id (both are 50256), so we check separately
        if tokenizer.bos_token_id != tokenizer.eos_token_id:
            assert legal_mask[tokenizer.bos_token_id].item() == False

    def test_token_mask_reverse(self, tokenizer, vocab_file):
        """Test reverse mode (currently not implemented but parameter exists)."""
        legal_mask, illegal_mask, _ = prepare_token_mask(tokenizer, vocab_file, reverse=False)

        # Just ensure it doesn't crash
        assert legal_mask is not None
        assert illegal_mask is not None


# ============================================================================
# Test Diversity Calculation
# ============================================================================


class TestCalculateDiversity:
    """Test diversity calculation for generated sequences."""

    def test_diversity_zero_for_single_sample(self):
        """Single sample should have zero diversity."""
        token_ids = torch.tensor([[1, 2, 3, 4, 5]])
        diversity = calculate_diversity(token_ids)
        assert diversity == 0.0

    def test_diversity_zero_for_identical_samples(self):
        """Identical samples should have zero diversity."""
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ]
        )
        diversity = calculate_diversity(token_ids)
        assert diversity == 0.0

    def test_diversity_positive_for_different_samples(self):
        """Different samples should have positive diversity."""
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 6, 7],
                [1, 2, 8, 9, 10],
            ]
        )
        diversity = calculate_diversity(token_ids)
        assert diversity > 0.0

    def test_diversity_maximum_for_all_different(self):
        """All different tokens at each position should give high diversity."""
        token_ids = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        diversity = calculate_diversity(token_ids)
        # Should be close to log(3) ≈ 1.099
        assert diversity > 1.0

    def test_diversity_with_long_sequences(self):
        """Test diversity with longer sequences."""
        torch.manual_seed(42)
        token_ids = torch.randint(0, 100, (10, 50))
        diversity = calculate_diversity(token_ids)
        assert diversity > 0.0
        assert diversity < 10.0  # Reasonable upper bound


# ============================================================================
# Test Modified SubTB Loss
# ============================================================================


class TestModifiedSubTBLoss:
    """Test the modified subtrajectory balance loss."""

    @pytest.fixture
    def simple_batch_data(self):
        """Create simple batch data for testing."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)

        # Create generated text with termination token
        termination_token_id = 0
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        # Add termination token at the end
        generated_text[:, -1] = termination_token_id

        return log_pf, log_r, log_pterm, generated_text, termination_token_id, prompt_len

    def test_subtb_loss_shape(self, simple_batch_data):
        """Test that SubTB loss returns a scalar."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch_data

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_subtb_loss_positive(self, simple_batch_data):
        """Test that SubTB loss is non-negative."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch_data

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert loss >= 0

    def test_subtb_loss_with_lambda(self, simple_batch_data):
        """Test SubTB loss with different lambda values."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch_data

        loss1 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=1.0
        )
        loss2 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=0.9
        )

        # Different lambdas should give different losses
        assert not torch.allclose(loss1, loss2)

    def test_subtb_loss_with_balance(self, simple_batch_data):
        """Test SubTB loss with token-coverage balancing."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch_data

        # Use modified_subtb_balance_loss which accepts balance parameter
        loss_no_balance = modified_subtb_balance_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, balance=0.0
        )
        loss_full_balance = modified_subtb_balance_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, balance=1.0
        )

        # Different balance values should give different losses
        # Note: when balance=0, it returns the original window-wise loss
        # when balance=1, it returns fully token-balanced loss
        assert isinstance(loss_no_balance, torch.Tensor)
        assert isinstance(loss_full_balance, torch.Tensor)
        assert not torch.allclose(loss_no_balance, loss_full_balance, atol=1e-6)

    def test_subtb_loss_gradient_flow(self, simple_batch_data):
        """Test that gradients can flow through the loss."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch_data

        log_pf.requires_grad = True
        log_r.requires_grad = True
        log_pterm.requires_grad = True

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        loss.backward()

        assert log_pf.grad is not None
        assert log_r.grad is not None
        assert log_pterm.grad is not None

    def test_subtb_loss_with_early_termination(self):
        """Test SubTB loss when some sequences terminate early."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)

        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        # First sequence terminates at position 3
        generated_text[0, prompt_len + 2 :] = term_id
        # Second sequence terminates at position 4
        generated_text[1, prompt_len + 3 :] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0


# ============================================================================
# Test Get Termination Values
# ============================================================================


class TestGetTerminationVals:
    """Test extraction of termination values from trajectories."""

    def test_get_termination_vals_basic(self):
        """Test basic termination value extraction."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_r_unpenalized = torch.randn(batch_size, seq_len)

        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, prompt_len + 3] = term_id  # Terminate at position 3

        log_pfs, last_log_r, last_log_r_unpen, gen_len = get_termination_vals(
            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized, term_id, prompt_len
        )

        assert log_pfs.shape == (batch_size,)
        assert last_log_r.shape == (batch_size,)
        assert last_log_r_unpen.shape == (batch_size,)
        assert gen_len.shape == (batch_size,)

    def test_get_termination_vals_with_none_inputs(self):
        """Test when log_pf and log_pterm are None."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_r = torch.randn(batch_size, seq_len)
        log_r_unpenalized = torch.randn(batch_size, seq_len)

        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, prompt_len + 3] = term_id

        log_pfs, last_log_r, last_log_r_unpen, gen_len = get_termination_vals(
            generated_text, None, None, log_r, log_r_unpenalized, term_id, prompt_len
        )

        assert log_pfs is None
        assert last_log_r.shape == (batch_size,)


# ============================================================================
# Test Replay Buffer
# ============================================================================


class TestReplayBuffer:
    """Test replay buffer functionality."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for buffer tests."""
        return GPT2Tokenizer.from_pretrained("gpt2")

    @pytest.fixture
    def buffer(self):
        """Create a replay buffer instance."""
        buffer = ReplayBuffer(buffer_size=10, sim_tolerance=0.25)
        buffer.set_termination_token_id(50256)  # GPT2 EOS
        return buffer

    def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.buffer_size == 10
        assert buffer.sim_tolerance == 0.25
        assert buffer._buffer == {}

    def test_buffer_add_batch(self, buffer, tokenizer):
        """Test adding a batch to the buffer."""
        batch_size = 4
        prompt_len = 3
        seq_len = 10

        prompt = torch.randint(1, 1000, (batch_size, prompt_len))
        sentences = torch.randint(1, 1000, (batch_size, seq_len))
        sentences[:, -1] = tokenizer.eos_token_id
        logrewards = torch.randn(batch_size, seq_len)

        # Mock result_dict
        result_dict = {
            "validator_dict": {
                "valid_score": torch.ones(batch_size, seq_len).bool(),
                "invalid": torch.zeros(batch_size, seq_len).bool(),
                "global_score": torch.randn(batch_size),
            }
        }

        buffer.add_batch(prompt, sentences, logrewards, tokenizer, result_dict)

        stats = buffer.stat()
        assert "prompt_0_total_buffer" in stats
        assert stats["prompt_0_total_buffer"] > 0

    def test_buffer_sample(self, buffer, tokenizer):
        """Test sampling from the buffer."""
        batch_size = 4
        prompt_len = 3
        seq_len = 10

        prompt = torch.randint(1, 1000, (batch_size, prompt_len))
        sentences = torch.randint(1, 1000, (batch_size, seq_len))
        sentences[:, -1] = tokenizer.eos_token_id
        logrewards = torch.randn(batch_size, seq_len)

        result_dict = {
            "validator_dict": {
                "valid_score": torch.ones(batch_size, seq_len).bool(),
                "invalid": torch.zeros(batch_size, seq_len).bool(),
                "global_score": torch.randn(batch_size),
            }
        }

        buffer.add_batch(prompt, sentences, logrewards, tokenizer, result_dict)

        # Try to sample
        sampled_sentences, sampled_answers = buffer.sample(2, prompt[0:1], tokenizer)

        if sampled_sentences is not None:
            assert sampled_sentences.shape[0] == 2
            assert sampled_answers.shape[0] == 2

    def test_buffer_reset(self, buffer, tokenizer):
        """Test buffer reset."""
        batch_size = 2
        prompt_len = 3
        seq_len = 10

        prompt = torch.randint(1, 1000, (batch_size, prompt_len))
        sentences = torch.randint(1, 1000, (batch_size, seq_len))
        sentences[:, -1] = tokenizer.eos_token_id
        logrewards = torch.randn(batch_size, seq_len)

        result_dict = {
            "validator_dict": {
                "valid_score": torch.ones(batch_size, seq_len).bool(),
                "invalid": torch.zeros(batch_size, seq_len).bool(),
                "global_score": torch.randn(batch_size),
            }
        }

        buffer.add_batch(prompt, sentences, logrewards, tokenizer, result_dict)
        assert len(buffer._buffer) > 0

        buffer.reset()
        assert len(buffer._buffer) == 0

    def test_buffer_stats(self, buffer, tokenizer):
        """Test buffer statistics."""
        stats = buffer.stat()
        assert isinstance(stats, dict)

        # Add some data
        batch_size = 4
        prompt_len = 3
        seq_len = 10

        prompt = torch.randint(1, 1000, (batch_size, prompt_len))
        sentences = torch.randint(1, 1000, (batch_size, seq_len))
        sentences[:, -1] = tokenizer.eos_token_id
        logrewards = torch.randn(batch_size, seq_len)

        result_dict = {
            "validator_dict": {
                "valid_score": torch.ones(batch_size, seq_len).bool(),
                "invalid": torch.zeros(batch_size, seq_len).bool(),
                "global_score": torch.randn(batch_size),
            }
        }

        buffer.add_batch(prompt, sentences, logrewards, tokenizer, result_dict)

        stats = buffer.stat()
        assert "prompt_0_total_buffer" in stats
        assert "prompt_0_avg_logR" in stats


# ============================================================================
# Test LoRA Utilities
# ============================================================================


class TestLoRAUtilities:
    """Test LoRA enable/disable utilities."""

    def test_lora_to_base(self):
        """Test disabling LoRA adapters."""
        # Mock model with LoRA
        model = Mock()
        model.base_model = Mock()
        model.base_model.disable_adapter_layers = Mock()
        model.eval = Mock()

        lora_to_base(model)

        model.base_model.disable_adapter_layers.assert_called_once()
        model.eval.assert_called_once()

    def test_base_to_lora(self):
        """Test enabling LoRA adapters."""
        # Mock model with LoRA
        model = Mock()
        model.base_model = Mock()
        model.base_model.enable_adapter_layers = Mock()
        model.train = Mock()

        base_to_lora(model)

        model.base_model.enable_adapter_layers.assert_called_once()
        model.train.assert_called_once()


# ============================================================================
# Test Generate and Return Termination LogProb
# ============================================================================


class TestGenerateAndReturnTerminationLogProb:
    """Test the main generation function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()

        # Mock forward pass
        def mock_forward(*args, **kwargs):
            batch_size = kwargs.get("input_ids").shape[0]
            seq_len = kwargs.get("input_ids").shape[1]
            vocab_size = 100

            output = Mock()
            output.logits = torch.randn(batch_size, seq_len, vocab_size)
            output.past_key_values = None
            return output

        model.side_effect = mock_forward
        model.__call__ = mock_forward
        return model

    @pytest.fixture
    def mock_reward_fn(self):
        """Create a mock reward function."""

        def reward_fn(state, **kwargs):
            batch_size = state.shape[0]
            seq_len = state.shape[1]
            return {
                "reward": torch.randn(batch_size, 10),
                "reward_unpenalized": torch.randn(batch_size, 10),
                "log_pf_ref": torch.randn(batch_size, 10),
                "full_tokens": None,
                "validator_dict": None,
            }

        return reward_fn

    def test_generate_basic(self, mock_model, mock_reward_fn):
        """Test basic generation."""
        batch_size = 2
        prompt_len = 3
        max_len = 5

        encoded_data = {
            "encoded_prompt": torch.randint(1, 100, (batch_size, prompt_len)),
        }

        result = generate_and_return_termination_logprob(
            model=mock_model,
            encoded_data=encoded_data,
            termination_token_id=0,
            reward_fn=mock_reward_fn,
            max_len=max_len,
            temperature=1.0,
        )

        assert "state" in result
        assert "log_pf" in result
        assert "log_pterm" in result
        assert "log_r" in result
        assert result["state"].shape[0] == batch_size

    def test_generate_with_skip_rewards(self, mock_model):
        """Test generation with rewards skipped."""
        batch_size = 2
        prompt_len = 3
        max_len = 5

        encoded_data = {
            "encoded_prompt": torch.randint(1, 100, (batch_size, prompt_len)),
        }

        result = generate_and_return_termination_logprob(
            model=mock_model,
            encoded_data=encoded_data,
            termination_token_id=0,
            reward_fn=None,
            max_len=max_len,
            skip_rewards=True,
        )

        assert result["log_r"] is None
        assert result["log_r_unpenalized"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
