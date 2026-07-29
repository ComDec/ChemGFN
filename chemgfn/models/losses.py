"""GFlowNet training objectives for language-model prefix trees.

The module implements the objectives compared in the paper:

* :class:`TBLoss` -- Trajectory Balance (Sec. 3.2).
* :class:`SubTBLoss` -- Subtrajectory Balance (Sec. 3.3).
* :class:`RapTBLoss` -- Rooted Absorbed Prefix Trajectory Balance (Sec. 3.4).
* :class:`RootSubTBLogZLoss` -- rooted-prefix SubTB with an explicit learnable log-partition.
* :class:`AvgPrefixTBLoss` -- uniform prefix Trajectory Balance baseline.

Every loss consumes the tensors produced by :class:`chemgfn.models.gfn.ChemGFNModule`. Writing
``B`` for the batch size, ``L`` for the number of generated positions and ``P`` for the prompt
length:

* ``log_pf`` ``[B, L]`` -- forward log-probabilities; the meaningful token steps are the first
  ``L - 1`` columns, ``log_pf[:, :-1]``.
* ``log_pterm`` ``[B, L]`` -- log-probability of emitting the stop token at prefix ``s_{0:k}``.
* ``log_r`` ``[B, L]`` -- mixed stop-reward ``log R(s_{0:k}^T)`` of every prefix ``s_{0:k}``.
* ``generated_text`` ``[B, P + L]`` -- prompt tokens followed by the generated tokens; the last
  position always holds the forced stop token and is therefore excluded when locating the first
  stop token chosen by the policy.

The trajectory length ``tau`` is the number of generated tokens strictly before the first stop
token, so ``s_{0:tau}`` is the deepest state the policy chose to reach.
"""

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

__all__ = [
    "GFNLoss",
    "TBLoss",
    "SubTBLoss",
    "RootSubTBLogZLoss",
    "RapTBLoss",
    "AvgPrefixTBLoss",
]


class GFNLoss(nn.Module, ABC):
    """Abstract base class shared by every GFlowNet objective.

    Subclasses implement :meth:`forward` with the common signature and return a dictionary whose
    ``"loss"`` entry is the scalar the trainer backpropagates; any additional entries are treated
    as diagnostics and logged automatically.
    """

    @abstractmethod
    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the objective for one batch of sampled trajectories.

        Args:
            log_pf: Forward-policy log-probabilities, shape ``[B, L]``.
            log_r: Log stop-reward of every prefix, shape ``[B, L]``.
            log_pterm: Log stop-probability of every prefix, shape ``[B, L]``.
            generated_text: Prompt and generated token ids, shape ``[B, prompt_len + L]``.
            termination_token_id: Token id of the stop symbol.
            prompt_len: Number of prompt tokens preceding the generated ones.
            **kwargs: Objective-specific arguments such as reference log-probabilities.

        Returns:
            Mapping with the scalar training loss under ``"loss"`` plus optional diagnostics.
        """


def _first_eos_index(gen_tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Index of the first stop token per row, falling back to the last position when absent."""
    _, seq_len = gen_tokens.shape
    is_eos = gen_tokens == eos_id
    has_eos = is_eos.any(dim=1)
    first = is_eos.float().argmax(dim=1)
    return torch.where(has_eos, first, torch.full_like(first, seq_len - 1))


def _gather_by_index(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select ``x[b, idx[b]]`` for every row ``b`` of a ``[B, L]`` tensor."""
    batch_size = x.shape[0]
    return x.gather(1, idx.view(batch_size, 1)).squeeze(1)


def _sum_log_pf_upto_tau(log_pf: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Return ``sum_{t < tau} log_pf[t]``, using the token steps ``log_pf[:, :-1]``."""
    batch_size = log_pf.shape[0]
    steps = log_pf[:, :-1]
    if steps.shape[1] == 0:
        return torch.zeros((batch_size,), device=log_pf.device, dtype=log_pf.dtype)

    pf_cum = steps.cumsum(dim=1)
    idx = (tau - 1).clamp(min=0, max=steps.shape[1] - 1)
    s = pf_cum.gather(1, idx.view(batch_size, 1)).squeeze(1)
    return s * (tau > 0).to(s.dtype)


def _reconstruct_ref_logP(ref_log_pf: torch.Tensor, ref_log_pterm: torch.Tensor) -> torch.Tensor:
    """Log-probability the frozen reference model assigns to every terminated prefix.

    ``ref_logP[:, 0] = ref_log_pterm[:, 0]`` and, for ``k >= 1``,
    ``ref_logP[:, k] = ref_log_pterm[:, k] + sum_{t < k} ref_log_pf[:, t]``.

    Args:
        ref_log_pf: Reference forward log-probabilities, shape ``[B, L - 1]``.
        ref_log_pterm: Reference stop log-probabilities, shape ``[B, L]``.

    Returns:
        Reference log-probabilities of every terminated prefix, shape ``[B, L]``.
    """
    batch_size, num_steps = ref_log_pf.shape
    ref_batch, seq_len = ref_log_pterm.shape
    assert (
        batch_size == ref_batch and seq_len == num_steps + 1
    ), f"shape mismatch: ref_log_pf {ref_log_pf.shape}, ref_log_pterm {ref_log_pterm.shape}"

    prefix = ref_log_pf.cumsum(dim=1)
    ref_logP = ref_log_pterm.clone()
    ref_logP[:, 1:] = ref_logP[:, 1:] + prefix
    return ref_logP


def _delta_cumsum(
    log_pf: torch.Tensor, log_r: torch.Tensor, log_pterm: torch.Tensor
) -> torch.Tensor:
    """Prefix sums of the single-step SubTB residuals.

    The residual of the transition ``s_{0:k} -> s_{0:k+1}`` is
    ``log_r[k] + log_pf[k] + log_pterm[k+1] - log_r[k+1] - log_pterm[k]``. The returned tensor
    ``C`` has ``C[:, 0] = 0`` and ``C[:, k]`` equal to the sum of the first ``k`` residuals, so
    the residual of the subtrajectory ``i -> j`` is ``C[:, j] - C[:, i]``.

    Returns:
        Cumulative residuals, shape ``[B, L]``.
    """
    delta = log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
    return torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)


def _suffix_future_max(u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Absorbed suffix maximum ``u_k^max = max_{j in [k, h]} u_j`` over valid positions."""
    u_mask = torch.where(valid, u, torch.full_like(u, 0))
    rev = torch.flip(u_mask, dims=[1])
    rev_max = torch.cummax(rev, dim=1).values
    return torch.flip(rev_max, dims=[1])


def _suffix_future_soft(
    u: torch.Tensor, valid: torch.Tensor, beta: float, rho: float
) -> torch.Tensor:
    """Absorbed suffix soft-maximum with a distance penalty.

    Computes ``u_k^soft = (1 / beta) * log sum_{j >= k} exp(beta * u_j - beta * rho * (j - k))``
    by a backward recursion in log space; ``beta`` controls the smoothness and ``rho >= 0``
    downweights evidence far ahead of ``k``. Invalid positions are masked to ``-inf``, so callers
    must zero out entries outside the valid horizon before using the result.
    """
    seq_len = u.shape[1]
    u_mask = torch.where(valid, u, torch.full_like(u, -torch.inf))
    out = torch.empty_like(u_mask)

    b = float(beta)
    step_pen = b * float(rho)

    z = b * u_mask[:, -1]
    out[:, -1] = z / b
    for t in range(seq_len - 2, -1, -1):
        z = torch.logaddexp(b * u_mask[:, t], z - step_pen)
        out[:, t] = z / b
    return out


class TBLoss(GFNLoss):
    """Trajectory Balance (Sec. 3.2).

    One residual per sampled trajectory anchors the policy to the terminal stop-reward,

        ``Delta_TB = log Z + sum_{t < tau} log_pf[t] + log_pterm[tau] - log_r[tau]``,

    and the objective is ``E[Delta_TB^2]``. The backward kernel of a prefix tree is deterministic,
    so no backward policy appears in the residual. ``log Z`` is a single learnable scalar.

    Args:
        **kwargs: Ignored, so that a shared configuration block can be reused across objectives.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.log_z = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the squared trajectory-balance residual averaged over the batch."""
        assert log_r.ndim == log_pterm.ndim == 2
        batch_size, seq_len = log_pterm.shape
        assert log_r.shape == (
            batch_size,
            seq_len,
        ), f"log_r shape {log_r.shape} must match log_pterm shape {(batch_size, seq_len)}"

        # allow log_pf to be shorter by 1 (non-terminal actions only)
        if log_pf.shape[1] == seq_len - 1:
            log_pf = torch.cat([log_pf, log_pf.new_zeros((batch_size, 1))], dim=1)
        assert log_pf.shape == (
            batch_size,
            seq_len,
        ), f"log_pf shape {log_pf.shape} must be (B,S) or (B,S-1)"

        gen = generated_text[:, prompt_len : prompt_len + seq_len]
        tau = _first_eos_index(gen, termination_token_id)

        steps = torch.arange(seq_len, device=log_pf.device).view(1, seq_len)
        pre_mask = steps < tau.view(batch_size, 1)

        token_logp = (log_pf * pre_mask.to(log_pf.dtype)).sum(dim=1)
        term_logp = log_pterm.gather(1, tau.view(batch_size, 1)).squeeze(1)
        logp_traj = token_logp + term_logp

        logr_traj = log_r.gather(1, tau.view(batch_size, 1)).squeeze(1)

        log_z_b = self.log_z.expand(batch_size)
        residual = log_z_b + logp_traj - logr_traj
        loss = (residual**2).mean()

        return {
            "loss": loss,
            "log_z_b": log_z_b.mean().detach(),
            "tb_residual_mean": residual.mean().detach(),
            "tb_residual_std": residual.std(unbiased=False).detach(),
            "logp_traj_mean": logp_traj.mean().detach(),
            "logr_traj_mean": logr_traj.mean().detach(),
        }


class SubTBLoss(GFNLoss):
    """Subtrajectory Balance (Sec. 3.3).

    Every subtrajectory ``s_{0:i} -> s_{0:j}`` of a sampled trajectory contributes the square of

        ``sum_{k=i}^{j-1} log_pf[k] + log_r[i] - log_r[j] + log_pterm[j] - log_pterm[i]``,

    weighted by ``subtb_lambda^(j - i - 1)``. The objective is the weighted mean over all windows
    whose final transition occurs strictly before the stop token. Window sums are evaluated with a
    prefix-sum of the single-step residuals, so the cost is linear in the sequence length per
    window size.

    Args:
        subtb_lambda: Geometric weight ``lambda^(len - 1)`` applied per subtrajectory length.
        eps: Lower bound on the weight normaliser, guarding against division by zero.
    """

    def __init__(self, subtb_lambda: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.subtb_lambda = subtb_lambda
        self.eps = eps

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the length-weighted mean squared subtrajectory residual."""
        assert (
            log_pf.shape[1]
            == log_r.shape[1]
            == log_pterm.shape[1]
            == generated_text.shape[1] - prompt_len
        ), (
            f"Shape mismatch: log_pf={log_pf.shape}, log_r={log_r.shape}, "
            f"log_pterm={log_pterm.shape}, generated_text={generated_text.shape}, "
            f"prompt_len={prompt_len}"
        )
        assert log_pf.shape[1] > 1, "Need at least one transition before termination (L > 1)"

        delta_cumsum = _delta_cumsum(log_pf, log_r, log_pterm)

        # True at and after the first stop token, excluding the forced final position.
        mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(dim=-1) >= 1

        batch_loss = log_pf.new_zeros(())
        total_lambda = log_pf.new_zeros(())
        generated_len = generated_text.shape[1] - prompt_len

        for subtraj_len in range(1, generated_len):
            w = self.subtb_lambda ** (subtraj_len - 1)

            subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2

            # Invalidate windows whose last transition lands at or after the stop token.
            subtb_term[mask[:, subtraj_len - 1 :]] = 0

            batch_loss = batch_loss + w * subtb_term.sum()
            total_lambda = total_lambda + w * (~mask[:, subtraj_len - 1 :]).sum()

        return {"loss": batch_loss / total_lambda.clamp_min(self.eps)}


class RootSubTBLogZLoss(GFNLoss):
    """Rooted SubTB with an explicit learnable log-partition.

    Only subtrajectories rooted at ``s_0`` are constrained, and the log-partition is kept as a
    learnable scalar instead of being cancelled between two prefixes. The residual at prefix
    length ``k`` is

        ``res(k) = log Z + sum_{t < k} log_pf[t] + log_pterm[k] - log_r[k]``,

    and the objective is ``sum_k lambda^(k-1) res(k)^2 / sum_k lambda^(k-1)`` over the prefixes
    that end before the stop token. This is the ablation isolating rooted supervision from the
    absorbed suffix targets of :class:`RapTBLoss`.

    Args:
        subtb_lambda: Geometric weight ``lambda^(k-1)`` applied per prefix length.
        eps: Lower bound on the weight normaliser.
        init_logZ: Initial value of the learnable log-partition.
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        eps: float = 1e-8,
        init_logZ: float = 0.0,
    ) -> None:
        super().__init__()
        self.subtb_lambda = float(subtb_lambda)
        self.eps = float(eps)
        self.logZ = nn.Parameter(torch.tensor([float(init_logZ)], dtype=torch.float32))

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the length-weighted mean squared rooted-prefix residual."""
        batch_size, seq_len = log_pf.shape
        assert log_r.shape == (batch_size, seq_len)
        assert log_pterm.shape == (batch_size, seq_len)
        assert generated_text.shape[1] - prompt_len == seq_len
        assert seq_len > 1

        # True at and after the first stop token, excluding the forced final position.
        mask = (generated_text[:, prompt_len:-1] == int(termination_token_id)).cumsum(dim=-1) >= 1

        # prefix_logpf[:, k - 1] = sum_{t < k} log_pf[t]
        prefix_logpf = log_pf[:, :-1].cumsum(dim=1)

        k = torch.arange(1, seq_len, device=log_pf.device, dtype=log_pf.dtype)
        w = (self.subtb_lambda ** (k - 1)).view(1, seq_len - 1)

        valid = (~mask).to(log_pf.dtype)

        logZ = self.logZ.to(device=log_pf.device, dtype=log_pf.dtype).view(1, 1)
        res = logZ + prefix_logpf + log_pterm[:, 1:] - log_r[:, 1:]

        num = (w * valid * (res**2)).sum()
        den = (w * valid).sum().clamp_min(self.eps)

        return {"loss": num / den, "logZ": self.logZ.detach()}


class RapTBLoss(GFNLoss):
    """Rooted Absorbed Prefix Trajectory Balance (Sec. 3.4).

    The objective combines the exact terminal TB anchor with a dense auxiliary term computed on
    rooted prefixes,

        ``L_RapTB = E[ Delta_TB^2 + eta * L_aux ]``,

    where ``L_aux`` is the length-weighted mean of ``(rooted residual + u_k - u_k^tgt)^2`` over
    the eligible prefixes. The rooted residual ``bar_Delta_k = Delta_k^TB - Delta_0^TB`` cancels
    the log-partition and is obtained from the prefix sums of the single-step residuals.

    The absorbed target ``u_k^tgt`` backs up the task-only reward from the observed suffix. With
    ``u_j`` the task-only component of the stop-reward (the mixed reward minus ``ref_scale`` times
    the reference prior), the targets are the suffix maximum ``u_k^max``, the distance-penalised
    suffix soft-maximum ``u_k^soft``, or their ``mix_weight`` blend. The correction is applied
    with weight ``gamma^(h - k)`` only where the prefix earns no task reward of its own
    (``|u_k| <= extra_absorb_eps``), only strictly inside the absorb horizon ``K``, and only for
    trajectories that actually reach ``K``.

    Prefixes shorter than ``k_min`` are excluded from the auxiliary term entirely; a sample with
    no eligible prefix is trained by the TB anchor alone.

    Args:
        subtb_lambda: Geometric weight ``lambda^(k - k_min)`` applied per prefix length.
        aux_weight: Auxiliary weight ``eta``; ``0`` recovers plain terminal TB.
        gamma: Discount ``gamma^(h - k)`` damping the absorbed correction away from the horizon.
        detach_pterm_in_aux: Stop gradients through the stop-probability inside the auxiliary
            term, so that dense supervision cannot be satisfied by shifting termination logits.
        eps: Lower bound on weight normalisers.
        extra_absorb_eps: Magnitude below which ``u_k`` counts as "no task reward at ``k``".
        target_mode: Absorbed target to use, one of ``"future_max"``, ``"future_soft"``, ``"mix"``.
        mix_weight: Weight ``alpha`` of ``u_k^max`` when ``target_mode="mix"``.
        soft_beta: Smoothness ``beta`` of the suffix soft-maximum.
        soft_rho: Distance penalty ``rho`` of the suffix soft-maximum.
        init_logZ: Initial value of the learnable log-partition used by the TB anchor.
        k_min: Shortest prefix length admitted into the auxiliary term.
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        aux_weight: float = 0.25,
        gamma: float = 0.99,
        detach_pterm_in_aux: bool = True,
        eps: float = 1e-8,
        extra_absorb_eps: float = 1e-6,
        target_mode: Literal["future_max", "future_soft", "mix"] = "mix",
        mix_weight: float = 0.5,
        soft_beta: float = 5.0,
        soft_rho: float = 0.0,
        init_logZ: float = 0.0,
        k_min: int = 2,
    ) -> None:
        super().__init__()
        self.subtb_lambda = float(subtb_lambda)
        self.aux_weight = float(aux_weight)
        self.gamma = float(gamma)
        self.detach_pterm_in_aux = bool(detach_pterm_in_aux)
        self.eps = float(eps)

        self.extra_absorb_eps = float(extra_absorb_eps)
        self.target_mode = target_mode
        self.mix_weight = float(mix_weight)
        self.soft_beta = float(soft_beta)
        self.soft_rho = float(soft_rho)

        self.k_min = int(k_min)

        if not (self.gamma < 1.0):
            raise ValueError(f"gamma must be < 1.0, got gamma={self.gamma}")
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got k_min={self.k_min}")

        self.logZ = nn.Parameter(torch.tensor([float(init_logZ)], dtype=torch.float32))

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        ref_log_pf: Optional[torch.Tensor] = None,
        ref_log_pterm: Optional[torch.Tensor] = None,
        ref_scale: float = 1.0,
        max_prefix_len: Optional[int] = None,
        k_min: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the terminal TB anchor plus the absorbed rooted-prefix auxiliary term.

        Args:
            log_pf: Forward-policy log-probabilities, shape ``[B, L]``.
            log_r: Log stop-reward of every prefix, shape ``[B, L]``.
            log_pterm: Log stop-probability of every prefix, shape ``[B, L]``.
            generated_text: Prompt and generated token ids, shape ``[B, prompt_len + L]``.
            termination_token_id: Token id of the stop symbol.
            prompt_len: Number of prompt tokens preceding the generated ones.
            ref_log_pf: Reference forward log-probabilities, shape ``[B, L - 1]``. When given
                together with ``ref_log_pterm``, the reference prior is subtracted from the mixed
                reward to recover the task-only component ``u``.
            ref_log_pterm: Reference stop log-probabilities, shape ``[B, L]``.
            ref_scale: Scale ``kappa`` of the reference prior inside the mixed reward.
            max_prefix_len: Absorb horizon ``K`` in ``[1, L - 1]``; ``None`` uses the full
                trajectory.
            k_min: Per-step override of the shortest admitted prefix length.
            **kwargs: Ignored.

        Returns:
            Mapping with ``"loss"`` and the diagnostics ``"loss_tb"``, ``"loss_aux"``,
            ``"aux_active_rate"`` and ``"logZ"``.
        """
        batch_size, seq_len = log_pf.shape
        assert log_r.shape == (
            batch_size,
            seq_len,
        ), f"log_r {log_r.shape} vs (B,L)={(batch_size, seq_len)}"
        assert log_pterm.shape == (
            batch_size,
            seq_len,
        ), f"log_pterm {log_pterm.shape} vs (B,L)={(batch_size, seq_len)}"
        assert generated_text.shape[0] == batch_size
        assert generated_text.shape[1] - prompt_len == seq_len, (
            f"generated_text post-prompt length must equal L. "
            f"got generated_text.shape={generated_text.shape}, prompt_len={prompt_len}, "
            f"L={seq_len}"
        )
        assert seq_len > 1, "Need L>=2 (at least one step + terminal state)"

        eos_or_after = (generated_text[:, prompt_len:-1] == int(termination_token_id)).cumsum(
            dim=-1
        ) >= 1
        valid_end = ~eos_or_after  # True strictly before the first stop token
        tau = valid_end.sum(dim=1).clamp(0, seq_len - 1)

        # ---------------- terminal TB anchor ----------------
        logZ_b = self.logZ.to(device=log_pf.device, dtype=log_pf.dtype).expand(batch_size)

        sum_log_pf = _sum_log_pf_upto_tau(log_pf, tau)
        log_pterm_tau = _gather_by_index(log_pterm, tau)
        log_r_tau = _gather_by_index(log_r, tau)

        tb_res = logZ_b + sum_log_pf + log_pterm_tau - log_r_tau
        loss_tb_i = tb_res**2
        loss_tb = loss_tb_i.mean()

        # ---------------- absorbed rooted-prefix auxiliary ----------------
        loss_aux = torch.zeros((), device=log_pf.device, dtype=log_pf.dtype)
        aux_active_rate = torch.zeros((), device=log_pf.device, dtype=log_pf.dtype)

        if self.aux_weight > 0.0:
            if max_prefix_len is None:
                K = None
            else:
                K = int(max_prefix_len)
                K = max(1, min(K, seq_len - 1))

            kmin = self.k_min if k_min is None else int(k_min)
            kmin = max(1, min(kmin, seq_len - 1))

            lp_aux = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm
            C_aux = _delta_cumsum(log_pf, log_r, lp_aux)
            Ck = C_aux[:, 1:]  # rooted residuals for k = 1..L-1

            k_idx = torch.arange(1, seq_len, device=log_pf.device).view(1, seq_len - 1)

            # horizon h = min(tau, K)
            if K is None:
                h = tau
            else:
                h = torch.minimum(tau, tau.new_full(tau.shape, K))

            within_tau = k_idx <= tau.view(batch_size, 1)
            within_h = k_idx <= h.view(batch_size, 1)

            m_base = within_tau & within_h & valid_end
            after_kmin = k_idx >= kmin
            m = (m_base & after_kmin).to(log_pf.dtype)

            w_exp = (k_idx - kmin).clamp_min(0).to(log_pf.dtype)
            w = (self.subtb_lambda**w_exp) * after_kmin.to(log_pf.dtype)

            # task-only reward u = log_r - ref_scale * reference log-probability
            if (ref_log_pf is not None) and (ref_log_pterm is not None):
                assert ref_log_pf.shape == (
                    batch_size,
                    seq_len - 1,
                ), f"ref_log_pf {ref_log_pf.shape} vs {(batch_size, seq_len - 1)}"
                assert ref_log_pterm.shape == (
                    batch_size,
                    seq_len,
                ), f"ref_log_pterm {ref_log_pterm.shape} vs {(batch_size, seq_len)}"
                ref_logP = _reconstruct_ref_logP(ref_log_pf, ref_log_pterm)
                u = log_r - float(ref_scale) * ref_logP
            else:
                u = log_r

            u_k = u[:, 1:]
            no_reward = u_k.abs() <= self.extra_absorb_eps

            # trajectories that never reach K get no absorption correction
            if K is None:
                can_absorb_seq = torch.ones(
                    (batch_size, 1), device=log_pf.device, dtype=torch.bool
                )
            else:
                can_absorb_seq = tau.view(batch_size, 1) >= K

            if K is None:
                before_horizon = within_h
            else:
                before_horizon = k_idx < K

            pos = torch.arange(seq_len, device=log_pf.device).view(1, seq_len)
            valid_future = (pos <= h.view(batch_size, 1)) & (pos <= tau.view(batch_size, 1))

            u_max = _suffix_future_max(u.detach(), valid_future)
            u_soft = _suffix_future_soft(
                u.detach(), valid_future, beta=self.soft_beta, rho=self.soft_rho
            )

            if self.target_mode == "future_max":
                u_target = u_max
            elif self.target_mode == "future_soft":
                u_target = u_soft
            elif self.target_mode == "mix":
                mw = float(self.mix_weight)
                u_target = mw * u_max + (1.0 - mw) * u_soft
            else:
                raise ValueError(f"Unknown target_mode: {self.target_mode}")

            u_target = torch.where(valid_future, u_target, torch.zeros_like(u_target))
            u_tk = u_target[:, 1:]

            # alpha_k = gamma^(h - k)
            exp = (h.view(batch_size, 1) - k_idx).clamp_min(0).to(log_pf.dtype)
            alpha = (self.gamma**exp) * within_h.to(log_pf.dtype)

            apply_absorb = (m_base & after_kmin) & no_reward & before_horizon & can_absorb_seq
            alpha_eff = alpha * apply_absorb.to(log_pf.dtype)

            Ck_abs = Ck + alpha_eff * (u_k - u_tk)

            # per-sample auxiliary, so samples with no eligible prefix stay TB-only
            mw_mask = m * w
            num_i = ((Ck_abs**2) * mw_mask).sum(dim=1)
            den_i = mw_mask.sum(dim=1)

            active_i = den_i > self.eps
            aux_active_rate = active_i.to(log_pf.dtype).mean()

            loss_aux_i = torch.zeros_like(num_i)
            loss_aux_i[active_i] = num_i[active_i] / den_i[active_i].clamp_min(self.eps)
            loss_aux = loss_aux_i.mean()

            eta_i = float(self.aux_weight) * active_i.to(log_pf.dtype)
            loss = (loss_tb_i + eta_i * loss_aux_i).mean()
        else:
            loss = loss_tb

        return {
            "loss": loss,
            "loss_tb": loss_tb.detach(),
            "loss_aux": loss_aux.detach(),
            "aux_active_rate": aux_active_rate.detach(),
            "logZ": self.logZ.detach(),
        }


class AvgPrefixTBLoss(GFNLoss):
    """Uniform prefix Trajectory Balance baseline (Appendix, "AvgPrefixTB").

    Every prefix of a sampled trajectory is treated as a terminated sequence and contributes its
    own TB residual,

        ``Delta_k^TB = log Z + sum_{t < k} log_pf[t] + log_pterm[k] - log_r[k]``,

    and the loss is the plain mean of ``Delta_k^TB`` squared over the prefixes ``k = 0..tau``.
    Unlike :class:`RapTBLoss`, each residual keeps the learnable log-partition instead of
    cancelling it by rooting, and uses the raw stop-reward instead of an absorbed suffix target.
    It isolates the effect of simply densifying TB supervision across prefixes.

    Args:
        detach_pterm_in_aux: Stop gradients through the stop-probability inside every prefix
            residual.
        init_logZ: Initial value of the learnable log-partition.
    """

    def __init__(
        self,
        detach_pterm_in_aux: bool = False,
        init_logZ: float = 0.0,
    ) -> None:
        super().__init__()
        self.detach_pterm_in_aux = bool(detach_pterm_in_aux)
        self.logZ = nn.Parameter(torch.tensor([float(init_logZ)], dtype=torch.float32))

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the mean squared TB residual over all prefixes of each trajectory."""
        batch_size, seq_len = log_pf.shape
        assert log_r.shape == (
            batch_size,
            seq_len,
        ), f"log_r {log_r.shape} vs (B,L)={(batch_size, seq_len)}"
        assert log_pterm.shape == (
            batch_size,
            seq_len,
        ), f"log_pterm {log_pterm.shape} vs (B,L)={(batch_size, seq_len)}"
        assert generated_text.shape[0] == batch_size
        assert generated_text.shape[1] - prompt_len == seq_len, (
            f"generated_text post-prompt length must equal L. "
            f"got generated_text.shape={generated_text.shape}, prompt_len={prompt_len}, "
            f"L={seq_len}"
        )
        assert seq_len > 1, "Need L>=2 (at least one step + terminal state)"

        eos_or_after = (generated_text[:, prompt_len:-1] == int(termination_token_id)).cumsum(
            dim=-1
        ) >= 1
        valid_end = ~eos_or_after
        tau = valid_end.sum(dim=1).clamp(0, seq_len - 1)

        logZ_b = self.logZ.to(device=log_pf.device, dtype=log_pf.dtype).expand(batch_size)

        # prefix_logpf[:, k] = sum_{t < k} log_pf[t], zero for k = 0
        zeros_col = torch.zeros(batch_size, 1, device=log_pf.device, dtype=log_pf.dtype)
        prefix_logpf = torch.cat([zeros_col, log_pf[:, :-1].cumsum(dim=1)], dim=1)

        pterm = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm
        residuals = logZ_b.unsqueeze(1) + prefix_logpf + pterm - log_r

        k_idx = torch.arange(seq_len, device=log_pf.device).unsqueeze(0)
        valid_mask = k_idx <= tau.unsqueeze(1)

        masked_sq = (residuals**2) * valid_mask.to(log_pf.dtype)
        count = valid_mask.to(log_pf.dtype).sum(dim=1).clamp_min(1.0)
        loss = (masked_sq.sum(dim=1) / count).mean()

        return {"loss": loss, "logZ": self.logZ.detach()}
