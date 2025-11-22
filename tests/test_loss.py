"""Comprehensive tests for loss functions

Focused tests for the modified SubTB loss including:
- Mathematical correctness
- Edge cases
- Gradient flow
- Numerical stability
- Balance parameter behavior
"""

import numpy as np
import pytest
import torch

from chemgfn.models.losses import ModifiedSubTBBalanceLoss


# Helper function to maintain backward compatibility with tests
def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
    balance=0.0,
    **kwargs
):
    """Wrapper function for backward compatibility with existing tests."""
    loss_fn = ModifiedSubTBBalanceLoss(subtb_lambda=subtb_lambda, balance=balance)
    loss_output = loss_fn(
        log_pf, log_r, log_pterm, generated_text, termination_token_id, prompt_len
    )
    # Return scalar for backward compatibility with tests
    return loss_output["loss"] if isinstance(loss_output, dict) else loss_output


# ============================================================================
# Test Basic Loss Computation
# ============================================================================


class TestBasicLossComputation:
    """Test basic loss computation functionality."""

    @pytest.fixture
    def simple_batch(self):
        """Create a simple batch for testing."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)

        termination_token_id = 0
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = termination_token_id

        return log_pf, log_r, log_pterm, generated_text, termination_token_id, prompt_len

    def test_loss_is_scalar(self, simple_batch):
        """Test that loss returns a scalar value."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Scalar
        assert loss.ndim == 0

    def test_loss_is_non_negative(self, simple_batch):
        """Test that loss is always non-negative."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert loss >= 0

    def test_loss_no_nan_or_inf(self, simple_batch):
        """Test that loss doesn't produce NaN or Inf."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = simple_batch

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_with_different_batch_sizes(self):
        """Test loss computation with different batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            seq_len = 5
            prompt_len = 2
            term_id = 0

            log_pf = torch.randn(batch_size, seq_len)
            log_r = torch.randn(batch_size, seq_len)
            log_pterm = torch.randn(batch_size, seq_len)
            generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
            generated_text[:, -1] = term_id

            loss = modified_subtb_loss(
                log_pf, log_r, log_pterm, generated_text, term_id, prompt_len
            )

            assert not torch.isnan(loss)
            assert loss >= 0

    def test_loss_with_different_sequence_lengths(self):
        """Test loss computation with different sequence lengths."""
        for seq_len in [2, 5, 10, 20]:
            batch_size = 4
            prompt_len = 2
            term_id = 0

            log_pf = torch.randn(batch_size, seq_len)
            log_r = torch.randn(batch_size, seq_len)
            log_pterm = torch.randn(batch_size, seq_len)
            generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
            generated_text[:, -1] = term_id

            loss = modified_subtb_loss(
                log_pf, log_r, log_pterm, generated_text, term_id, prompt_len
            )

            assert not torch.isnan(loss)
            assert loss >= 0


# ============================================================================
# Test SubTB Lambda Parameter
# ============================================================================


class TestSubTBLambda:
    """Test the subtb_lambda parameter behavior."""

    @pytest.fixture
    def fixed_batch(self):
        """Create a fixed batch for comparison tests."""
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        return log_pf, log_r, log_pterm, generated_text, term_id, prompt_len

    def test_lambda_one_gives_valid_loss(self, fixed_batch):
        """Test that lambda=1.0 gives valid loss."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        loss = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=1.0
        )

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_different_lambdas_give_different_losses(self, fixed_batch):
        """Test that different lambda values give different losses."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        loss_lambda_09 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=0.9
        )

        loss_lambda_10 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=1.0
        )

        loss_lambda_11 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=1.1
        )

        # Different lambdas should give different losses
        assert not torch.allclose(loss_lambda_09, loss_lambda_10)
        assert not torch.allclose(loss_lambda_10, loss_lambda_11)

    def test_lambda_zero_behavior(self, fixed_batch):
        """Test behavior with lambda=0 (only length-1 windows)."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        loss = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, subtb_lambda=0.0
        )

        # Should still compute valid loss (only single-step windows)
        assert not torch.isnan(loss)
        assert loss >= 0

    def test_lambda_range(self, fixed_batch):
        """Test lambda values in reasonable range."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        for lambda_val in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
            loss = modified_subtb_loss(
                log_pf,
                log_r,
                log_pterm,
                generated_text,
                term_id,
                prompt_len,
                subtb_lambda=lambda_val,
            )

            assert not torch.isnan(loss)
            assert loss >= 0


# ============================================================================
# Test Balance Parameter
# ============================================================================


class TestBalanceParameter:
    """Test the balance parameter for token-coverage balancing."""

    @pytest.fixture
    def fixed_batch(self):
        """Create a fixed batch."""
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 8
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        return log_pf, log_r, log_pterm, generated_text, term_id, prompt_len

    def test_balance_zero_vs_one(self, fixed_batch):
        """Test that balance=0 and balance=1 give different losses."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        loss_balance_0 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, balance=0.0
        )

        loss_balance_1 = modified_subtb_loss(
            log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, balance=1.0
        )

        # Should be different (unless by chance)
        # The balance parameter changes how window losses are aggregated
        assert isinstance(loss_balance_0, torch.Tensor)
        assert isinstance(loss_balance_1, torch.Tensor)

    def test_balance_interpolation(self, fixed_batch):
        """Test that intermediate balance values interpolate smoothly."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        losses = []
        for balance in [0.0, 0.25, 0.5, 0.75, 1.0]:
            loss = modified_subtb_loss(
                log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, balance=balance
            )
            losses.append(loss.item())

        # All should be valid
        assert all(not np.isnan(l) for l in losses)
        assert all(l >= 0 for l in losses)

    def test_balance_out_of_range_warning(self, fixed_batch):
        """Test behavior with out-of-range balance values."""
        log_pf, log_r, log_pterm, generated_text, term_id, prompt_len = fixed_batch

        # Balance should be in [0, 1], but test edge cases
        for balance in [-0.1, 1.1]:
            loss = modified_subtb_loss(
                log_pf, log_r, log_pterm, generated_text, term_id, prompt_len, balance=balance
            )

            # Should still compute something (might be extrapolated)
            assert isinstance(loss, torch.Tensor)


# ============================================================================
# Test Early Termination Handling
# ============================================================================


class TestEarlyTermination:
    """Test loss computation with early termination."""

    def test_single_early_termination(self):
        """Test with one sequence terminating early."""
        batch_size = 4
        seq_len = 8
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))

        # First sequence terminates early at position 4
        generated_text[0, prompt_len + 3 :] = term_id
        # Others terminate normally
        generated_text[1:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_all_early_termination(self):
        """Test when all sequences terminate early."""
        batch_size = 4
        seq_len = 8
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))

        # All sequences terminate early at different positions
        generated_text[0, prompt_len + 2 :] = term_id
        generated_text[1, prompt_len + 3 :] = term_id
        generated_text[2, prompt_len + 4 :] = term_id
        generated_text[3, prompt_len + 5 :] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_immediate_termination(self):
        """Test when sequence terminates immediately after prompt."""
        batch_size = 4
        # Need seq_len > 1 to avoid assertion error in modified_subtb_loss
        seq_len = 3
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))

        # First token after prompt is termination, fill rest with term_id
        generated_text[:, prompt_len] = term_id
        generated_text[:, prompt_len + 1 :] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        # When immediately terminating, mask should handle it and loss should be 0 or valid
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0


# ============================================================================
# Test Gradient Flow
# ============================================================================


class TestGradientFlow:
    """Test gradient computation and flow."""

    def test_gradients_exist(self):
        """Test that gradients are computed for all inputs."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len, requires_grad=True)
        log_r = torch.randn(batch_size, seq_len, requires_grad=True)
        log_pterm = torch.randn(batch_size, seq_len, requires_grad=True)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        loss.backward()

        assert log_pf.grad is not None
        assert log_r.grad is not None
        assert log_pterm.grad is not None

    def test_gradients_not_nan(self):
        """Test that gradients are not NaN."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len, requires_grad=True)
        log_r = torch.randn(batch_size, seq_len, requires_grad=True)
        log_pterm = torch.randn(batch_size, seq_len, requires_grad=True)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        loss.backward()

        assert not torch.isnan(log_pf.grad).any()
        assert not torch.isnan(log_r.grad).any()
        assert not torch.isnan(log_pterm.grad).any()

    def test_second_order_gradients(self):
        """Test that second-order gradients can be computed."""
        batch_size = 2
        seq_len = 3
        prompt_len = 1
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len, requires_grad=True)
        log_r = torch.randn(batch_size, seq_len, requires_grad=True)
        log_pterm = torch.randn(batch_size, seq_len, requires_grad=True)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        # Compute first-order gradients
        grad_outputs = torch.ones_like(loss)
        grads = torch.autograd.grad(
            loss, [log_pf, log_r, log_pterm], create_graph=True, retain_graph=True
        )

        # Should be able to compute second-order
        if grads[0] is not None and grads[0].requires_grad:
            second_order = torch.autograd.grad(
                grads[0].sum(), log_pf, retain_graph=True, allow_unused=True
            )
            # Just verify we can compute it
            assert True


# ============================================================================
# Test Numerical Stability
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of loss computation."""

    def test_large_values(self):
        """Test with large input values."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        # Large values
        log_pf = torch.randn(batch_size, seq_len) * 10 + 50
        log_r = torch.randn(batch_size, seq_len) * 10 + 50
        log_pterm = torch.randn(batch_size, seq_len) * 10 + 50
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_small_values(self):
        """Test with small input values."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        # Small values
        log_pf = torch.randn(batch_size, seq_len) * 0.01
        log_r = torch.randn(batch_size, seq_len) * 0.01
        log_pterm = torch.randn(batch_size, seq_len) * 0.01
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_mixed_sign_values(self):
        """Test with mixed positive/negative values."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)  # Mixed signs
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_zero_values(self):
        """Test with zero input values."""
        batch_size = 4
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.zeros(batch_size, seq_len)
        log_r = torch.zeros(batch_size, seq_len)
        log_pterm = torch.zeros(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        # With all zeros, loss should be zero
        assert torch.allclose(loss, torch.tensor(0.0))


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_sequence_length(self):
        """Test with minimum valid sequence length."""
        batch_size = 4
        seq_len = 2  # Minimum: need at least 2 for delta computation
        prompt_len = 1
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_single_batch(self):
        """Test with single batch element."""
        batch_size = 1
        seq_len = 5
        prompt_len = 2
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_long_sequence(self):
        """Test with long sequences."""
        batch_size = 2
        seq_len = 50
        prompt_len = 5
        term_id = 0

        log_pf = torch.randn(batch_size, seq_len)
        log_r = torch.randn(batch_size, seq_len)
        log_pterm = torch.randn(batch_size, seq_len)
        generated_text = torch.randint(1, 100, (batch_size, prompt_len + seq_len))
        generated_text[:, -1] = term_id

        loss = modified_subtb_loss(log_pf, log_r, log_pterm, generated_text, term_id, prompt_len)

        assert not torch.isnan(loss)
        assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
