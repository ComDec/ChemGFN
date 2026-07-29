"""Tests for the GFlowNet objectives in ``chemgfn.models.losses``."""

from __future__ import annotations

import pytest
import torch

from chemgfn.models.losses import (
    AvgPrefixTBLoss,
    GFNLoss,
    RapTBLoss,
    RootSubTBLogZLoss,
    SubTBLoss,
    TBLoss,
)

ALL_LOSSES = [TBLoss, SubTBLoss, RootSubTBLogZLoss, RapTBLoss, AvgPrefixTBLoss]


@pytest.mark.parametrize("loss_cls", ALL_LOSSES)
class TestCommonContract:
    """Every objective returns a scalar ``loss`` and is differentiable."""

    def test_returns_scalar_loss(self, loss_cls, trajectory_batch):
        log_pf, log_r, log_pterm, tokens, eos, prompt_len = trajectory_batch()
        out = loss_cls()(log_pf, log_r, log_pterm, tokens, eos, prompt_len)

        assert isinstance(out, dict)
        assert out["loss"].ndim == 0
        assert out["loss"] >= 0
        assert torch.isfinite(out["loss"])

    def test_is_a_gfn_loss(self, loss_cls, trajectory_batch):
        assert issubclass(loss_cls, GFNLoss)

    def test_gradients_reach_every_input(self, loss_cls, trajectory_batch):
        log_pf, log_r, log_pterm, tokens, eos, prompt_len = trajectory_batch(requires_grad=True)
        loss_cls()(log_pf, log_r, log_pterm, tokens, eos, prompt_len)["loss"].backward()

        for tensor in (log_pf, log_r, log_pterm):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_batch_sizes(self, loss_cls, trajectory_batch, batch_size):
        args = trajectory_batch(batch_size=batch_size)
        assert torch.isfinite(loss_cls()(*args)["loss"])

    @pytest.mark.parametrize("seq_len", [2, 5, 20])
    def test_sequence_lengths(self, loss_cls, trajectory_batch, seq_len):
        args = trajectory_batch(seq_len=seq_len)
        assert torch.isfinite(loss_cls()(*args)["loss"])

    @pytest.mark.parametrize("terminate_at", [1, 2, 3])
    def test_early_termination(self, loss_cls, trajectory_batch, terminate_at):
        args = trajectory_batch(seq_len=8, terminate_at=terminate_at)
        out = loss_cls()(*args)
        assert torch.isfinite(out["loss"])
        assert out["loss"] >= 0

    def test_large_magnitudes_stay_finite(self, loss_cls, trajectory_batch):
        log_pf, log_r, log_pterm, tokens, eos, prompt_len = trajectory_batch()
        out = loss_cls()(
            log_pf * 10 - 50, log_r * 10 - 50, log_pterm * 10 - 50, tokens, eos, prompt_len
        )
        assert torch.isfinite(out["loss"])


class TestTBLoss:
    """Terminal Trajectory Balance."""

    def test_zero_residual_gives_zero_loss(self):
        batch_size, seq_len, prompt_len, eos = 3, 4, 2, 0
        tokens = torch.full((batch_size, prompt_len + seq_len), 7, dtype=torch.long)
        tokens[:, prompt_len + 2] = eos

        zeros = torch.zeros(batch_size, seq_len)
        out = TBLoss()(zeros, zeros, zeros, tokens, eos, prompt_len)

        # log Z is initialised to 0, so log Z + log P(traj) - log R(traj) vanishes.
        assert torch.allclose(out["loss"], torch.zeros(()))

    def test_accepts_log_pf_without_terminal_column(self, trajectory_batch):
        log_pf, log_r, log_pterm, tokens, eos, prompt_len = trajectory_batch(seq_len=5)
        out = TBLoss()(log_pf[:, :-1], log_r, log_pterm, tokens, eos, prompt_len)
        assert torch.isfinite(out["loss"])

    def test_reports_residual_diagnostics(self, trajectory_batch):
        out = TBLoss()(*trajectory_batch())
        for key in ("log_z_b", "tb_residual_mean", "tb_residual_std", "logr_traj_mean"):
            assert key in out
            assert not out[key].requires_grad

    def test_log_z_is_learnable(self, trajectory_batch):
        loss_fn = TBLoss()
        loss_fn(*trajectory_batch())["loss"].backward()
        assert loss_fn.log_z.grad is not None


class TestSubTBLoss:
    """Subtrajectory Balance over every window of a trajectory."""

    def test_zero_inputs_give_zero_loss(self, trajectory_batch):
        _, _, _, tokens, eos, prompt_len = trajectory_batch(seq_len=5)
        zeros = torch.zeros(tokens.shape[0], tokens.shape[1] - prompt_len)
        out = SubTBLoss()(zeros, zeros, zeros, tokens, eos, prompt_len)
        assert torch.allclose(out["loss"], torch.zeros(()))

    @pytest.mark.parametrize("subtb_lambda", [0.0, 0.5, 0.9, 1.0])
    def test_lambda_values_are_valid(self, trajectory_batch, subtb_lambda):
        out = SubTBLoss(subtb_lambda=subtb_lambda)(*trajectory_batch())
        assert torch.isfinite(out["loss"])
        assert out["loss"] >= 0

    def test_lambda_changes_the_loss(self, trajectory_batch):
        args = trajectory_batch(seq_len=8)
        assert not torch.allclose(
            SubTBLoss(subtb_lambda=0.5)(*args)["loss"],
            SubTBLoss(subtb_lambda=1.0)(*args)["loss"],
        )

    def test_single_step_sequences_are_rejected(self, trajectory_batch):
        args = trajectory_batch(seq_len=1)
        with pytest.raises(AssertionError):
            SubTBLoss()(*args)


class TestRootSubTBLogZLoss:
    """Rooted SubTB with an explicit learnable log-partition."""

    def test_exposes_log_z(self, trajectory_batch):
        loss_fn = RootSubTBLogZLoss(init_logZ=1.5)
        out = loss_fn(*trajectory_batch())

        assert torch.allclose(out["logZ"], torch.tensor([1.5]))
        assert not out["logZ"].requires_grad

    def test_log_z_receives_gradient(self, trajectory_batch):
        loss_fn = RootSubTBLogZLoss()
        loss_fn(*trajectory_batch())["loss"].backward()
        assert loss_fn.logZ.grad is not None
        assert torch.isfinite(loss_fn.logZ.grad).all()


@pytest.fixture
def sparse_reward_batch():
    """Trajectory whose task reward sits entirely at the terminal state.

    Intermediate prefixes then carry no task reward of their own, which is the condition under
    which :class:`RapTBLoss` applies its absorbed suffix correction.
    """
    batch_size, seq_len, prompt_len, eos = 2, 8, 1, 0

    tokens = torch.full((batch_size, prompt_len + seq_len), 5, dtype=torch.long)
    tokens[:, -1] = eos

    log_r = torch.zeros(batch_size, seq_len)
    log_r[:, -1] = 3.0

    generator = torch.Generator().manual_seed(0)
    log_pf = torch.randn(batch_size, seq_len, generator=generator)
    log_pterm = torch.randn(batch_size, seq_len, generator=generator)

    return log_pf, log_r, log_pterm, tokens, eos, prompt_len


class TestRapTBLoss:
    """Rooted Absorbed Prefix Trajectory Balance."""

    def test_zero_aux_weight_recovers_terminal_tb(self, trajectory_batch):
        args = trajectory_batch(seq_len=8)
        out = RapTBLoss(aux_weight=0.0)(*args)

        assert torch.allclose(out["loss"], out["loss_tb"])
        assert torch.allclose(out["loss_aux"], torch.zeros(()))

    def test_auxiliary_term_changes_the_loss(self, trajectory_batch):
        args = trajectory_batch(seq_len=8)
        assert not torch.allclose(
            RapTBLoss(aux_weight=0.0)(*args)["loss"],
            RapTBLoss(aux_weight=0.5)(*args)["loss"],
        )

    @pytest.mark.parametrize("target_mode", ["future_max", "future_soft", "mix"])
    def test_every_absorbed_target(self, trajectory_batch, target_mode):
        out = RapTBLoss(target_mode=target_mode)(*trajectory_batch(seq_len=8))
        assert torch.isfinite(out["loss"])

    def test_unknown_target_mode_is_rejected(self, trajectory_batch):
        with pytest.raises(ValueError):
            RapTBLoss(target_mode="nonsense")(*trajectory_batch(seq_len=8))

    def test_k_min_beyond_the_horizon_disables_the_auxiliary_term(self, trajectory_batch):
        # Terminating after two tokens leaves no prefix of length >= 6 to supervise.
        args = trajectory_batch(seq_len=8, terminate_at=2)
        out = RapTBLoss(aux_weight=0.5, k_min=6)(*args)

        assert torch.allclose(out["aux_active_rate"], torch.zeros(()))
        assert torch.allclose(out["loss"], out["loss_tb"])

    def test_k_min_can_be_overridden_per_call(self, trajectory_batch):
        args = trajectory_batch(seq_len=8)
        loss_fn = RapTBLoss(aux_weight=0.5, k_min=2)

        assert not torch.allclose(loss_fn(*args)["loss"], loss_fn(*args, k_min=5)["loss"])

    def test_max_prefix_len_bounds_the_absorb_horizon(self, trajectory_batch):
        args = trajectory_batch(seq_len=8)
        loss_fn = RapTBLoss(aux_weight=0.5)

        assert not torch.allclose(loss_fn(*args)["loss"], loss_fn(*args, max_prefix_len=3)["loss"])

    def test_discount_shapes_the_absorbed_correction(self, sparse_reward_batch):
        # The correction only fires at prefixes that earn no task reward of their own, so it is
        # visible exactly on a trajectory whose reward sits entirely at the terminal state.
        assert not torch.allclose(
            RapTBLoss(aux_weight=0.5, gamma=0.5)(*sparse_reward_batch)["loss"],
            RapTBLoss(aux_weight=0.5, gamma=0.95)(*sparse_reward_batch)["loss"],
        )

    def test_reference_prior_is_subtracted_from_the_absorbed_target(self, sparse_reward_batch):
        log_pf, log_r, log_pterm, tokens, eos, prompt_len = sparse_reward_batch
        batch_size, seq_len = log_pf.shape

        loss_fn = RapTBLoss(aux_weight=0.5)
        without_ref = loss_fn(log_pf, log_r, log_pterm, tokens, eos, prompt_len)["loss"]
        with_ref = loss_fn(
            log_pf,
            log_r,
            log_pterm,
            tokens,
            eos,
            prompt_len,
            ref_log_pf=torch.full((batch_size, seq_len - 1), 0.5),
            ref_log_pterm=torch.full((batch_size, seq_len), 0.5),
        )["loss"]

        # Removing the reference prior leaves a non-zero task reward at every prefix, which
        # disqualifies them from the absorbed correction.
        assert not torch.allclose(without_ref, with_ref)

    def test_gamma_must_be_a_contraction(self):
        with pytest.raises(ValueError):
            RapTBLoss(gamma=1.0)

    def test_k_min_must_be_positive(self):
        with pytest.raises(ValueError):
            RapTBLoss(k_min=0)

    def test_log_z_receives_gradient(self, trajectory_batch):
        loss_fn = RapTBLoss()
        loss_fn(*trajectory_batch(seq_len=8))["loss"].backward()
        assert loss_fn.logZ.grad is not None


class TestAvgPrefixTBLoss:
    """Uniform prefix TB baseline."""

    def test_exposes_log_z(self, trajectory_batch):
        out = AvgPrefixTBLoss(init_logZ=-0.5)(*trajectory_batch())
        assert torch.allclose(out["logZ"], torch.tensor([-0.5]))

    def test_detaching_pterm_removes_its_gradient(self, trajectory_batch):
        log_pf, log_r, log_pterm, tokens, eos, prompt_len = trajectory_batch(requires_grad=True)
        loss_fn = AvgPrefixTBLoss(detach_pterm_in_aux=True)
        loss_fn(log_pf, log_r, log_pterm, tokens, eos, prompt_len)["loss"].backward()

        assert log_pterm.grad is None
        assert log_pf.grad is not None

    def test_zero_residual_gives_zero_loss(self):
        batch_size, seq_len, prompt_len, eos = 3, 4, 2, 0
        tokens = torch.full((batch_size, prompt_len + seq_len), 7, dtype=torch.long)
        tokens[:, -1] = eos

        zeros = torch.zeros(batch_size, seq_len)
        out = AvgPrefixTBLoss()(zeros, zeros, zeros, tokens, eos, prompt_len)
        assert torch.allclose(out["loss"], torch.zeros(()))
