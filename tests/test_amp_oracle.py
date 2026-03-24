"""Tests for AMP oracle module."""
import pytest
import torch


class TestAMPOracleMLP:
    """Test the MLP architecture matches the paper."""

    def test_mlp_output_shape(self):
        from chemgfn.models.amp_oracle import AMPMLP

        mlp = AMPMLP(num_inputs=4096, num_hiddens=1024, num_outputs=2, dropout_rate=0.5)
        x = torch.randn(4, 4096)
        out = mlp(x)
        assert out.shape == (4, 2)

    def test_mlp_score_range(self):
        from chemgfn.models.amp_oracle import AMPMLP

        mlp = AMPMLP(num_inputs=4096, num_hiddens=1024, num_outputs=2, dropout_rate=0.5)
        x = torch.randn(8, 4096)
        logits = mlp(x)
        probs = torch.softmax(logits, dim=1)
        assert (probs >= 0).all() and (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


class TestAMPOracle:
    """Test the full oracle wrapper."""

    def test_score_sequences_returns_tensor(self):
        from chemgfn.models.amp_oracle import AMPOracle

        # Without real weights, just test the interface
        oracle = AMPOracle(weights_path=None)
        sequences = ["AAAAAKKKKKK", "RRRRRGGGGG"]
        scores = oracle.score_sequences(sequences)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (2,)
        assert (scores >= 0).all() and (scores <= 1).all()
