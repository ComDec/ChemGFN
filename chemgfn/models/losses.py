"""
Loss functions for GFlowNet training.

This module provides loss function implementations for training GFlowNets,
including SubTrajectory Balance (SubTB) losses with various enhancements.
"""

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chemgfn.models.mcmc_prior import (
    CoarseTokenIndexer,
    load_q_mcmc,
    pack_q_mcmc_to_device,
)

__all__ = [
    "GFNLoss",
    "ModifiedSubTBLoss",
    "ModifiedSubTBLossSplitReward",
    "ModifiedSubTBBalanceLoss",
    "SubTBMCMCPriorLoss",
    "SubTBBatchDiversityLoss",
    "SubTBBatchDiversityPriorLoss",
]


class GFNLoss(nn.Module, ABC):
    """
    Abstract base class for GFlowNet loss functions.

    All GFlowNet loss implementations should inherit from this class and implement
    the forward method with the standard signature.

    This base class enables:
    - Type checking in training code
    - Consistent interface across loss functions
    - Easy extension with new loss variants
    """

    def __init__(self):
        super().__init__()
        self.global_step: int | None = None
        self.weight_schedulers: dict[str, callable] = {}
        self.requires_policy_logits: bool = False

    def set_global_step(self, step: int):
        self.global_step = int(step)

    def set_weight_schedulers(self, schedulers: dict[str, callable]):
        self.weight_schedulers = schedulers or {}

    def _resolve_weight(
        self,
        name: str,
        base_value: float,
        overrides: dict | None = None,
        step: Optional[int] = None,
    ) -> float:
        step = self.global_step if step is None else step
        if overrides is not None and name in overrides and overrides[name] is not None:
            # overrides 直接覆盖为常数
            return float(overrides[name])
        if self.weight_schedulers and name in self.weight_schedulers and step is not None:
            try:
                return float(self.weight_schedulers[name](step))
            except Exception:
                pass
        return float(base_value)

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
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Compute the GFlowNet loss.

        Args:
            log_pf: Log forward policy probabilities at each step, shape [B, L]
            log_r: Log reward prefix accumulator, shape [B, L]
            log_pterm: Log termination probabilities, shape [B, L]
            generated_text: Token IDs including prompt, shape [B, prompt_len + L]
            termination_token_id: EOS token ID
            prompt_len: Length of prompt
            **kwargs: Additional loss-specific arguments

        Returns:
            Dictionary with 'loss' key for backward and optional additional losses for logging,
            or a scalar tensor for backward compatibility.
            Example: {"loss": total_loss, "loss_reference": ref_loss, "loss_target": target_loss}
        """
        pass


class ModifiedSubTBLoss(GFNLoss):
    """
    Modified SubTrajectory Balance (SubTB) Loss.

    This loss implements the SubTB objective for GFlowNet training, which balances
    subtrajectories of different lengths to improve training stability and sample quality.

    Args:
        subtb_lambda (float): Length decay weight for subtrajectories. Default: 1.0
        balance (float): Token-level balancing degree in [0,1]. 0 keeps original window-sum;
                        1 re-weights so each token contributes equally. Default: 0.0
        eps (float): Numerical stabilizer for division. Default: 1e-8

    References:
        SubTB: https://arxiv.org/abs/2209.12782
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        balance: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.subtb_lambda = subtb_lambda
        self.balance = balance
        self.eps = eps

    def forward(
        self,
        log_pf: torch.Tensor,  # [B, L]
        log_r: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute SubTB loss.

        Args:
            log_pf: Log forward policy probabilities at each step, shape [B, L]
            log_r: Log reward prefix accumulator, shape [B, L+1]
            log_pterm: Log termination probabilities, shape [B, L+1]
            generated_text: Token IDs including prompt, shape [B, prompt_len + L+1]
            termination_token_id: EOS token ID
            prompt_len: Length of prompt
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Scalar loss tensor
        """
        # Ensure the dimensions of log probabilities, rewards, and generated text match
        assert (
            log_pf.shape[1]
            == log_r.shape[1]
            == log_pterm.shape[1]
            == generated_text.shape[1] - prompt_len
        ), f"Shape mismatch: log_pf={log_pf.shape}, log_r={log_r.shape}, log_pterm={log_pterm.shape}, generated_text={generated_text.shape}, prompt_len={prompt_len}"

        # Ensure there is at least one transition before termination
        assert log_pf.shape[1] > 1, "Need at least one transition before termination (L > 1)"
        # Calculate the change in expected reward and probability at each step
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
        )

        # Compute cumulative sum of delta for subtrajectory balance calculation
        delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)

        # Create a mask for tokens after the termination token
        mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(dim=-1) >= 1

        batch_loss = 0.0
        total_lambda = 0.0
        generated_len = generated_text.shape[1] - prompt_len

        for subtraj_len in range(1, generated_len):
            # Calculate the subtrajectory balance term
            subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
            # Apply mask to ignore invalid parts of the sequence
            subtb_term[mask[:, subtraj_len - 1 :]] = 0
            # Accumulate weighted subtrajectory balance term
            batch_loss += self.subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
            # Accumulate total weight for normalization
            total_lambda += (
                self.subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
            )

        # Normalize the loss by the total weight
        batch_loss /= total_lambda
        return {"loss": batch_loss}


class RootAbsorbExtraSubTBLoss(GFNLoss):
    """
    L = (1-eta)*L_TB_terminal + eta*L_aux_rooted_absorb_extra(K)

    Key semantics you requested:
    - K is an absorption horizon (scheduler-controlled).
      For prefixes k < K with (approximately) no extra reward signal, we absorb a future target
      computed from extra rewards in [k..K] (or [k..tau] if K is None).
    - If a trajectory terminates before K (tau < K), we do NOT absorb (keep original).
    - Length decay is mandatory: alpha_k = gamma^(horizon - k), gamma < 1.
    - Target supports: future-max and discounted softmax; default is a mix.
    - Aux detaches log_pterm to avoid short-length bias.
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,  # w_k = lambda^(k-1)
        aux_weight: float = 0.25,  # eta in [0,1] if you use convex combo
        gamma: float = 0.99,  # MUST be < 1 for length decay
        detach_pterm_in_aux: bool = True,
        eps: float = 1e-8,
        # absorption behavior
        extra_absorb_eps: float = 1e-6,  # treat |extra_k| <= eps as "no reward"
        target_mode: Literal["future_max", "future_soft", "mix"] = "mix",
        mix_weight: float = 0.5,  # mix: w*max + (1-w)*soft
        soft_beta: float = 5.0,  # softmax temperature in logsumexp
        soft_rho: float = 0.0,  # per-step distance penalty (>=0)
    ):
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

        # enforce "length decay is mandatory"
        if not (self.gamma < 1.0):
            raise ValueError(
                f"gamma must be < 1.0 for mandatory length decay, got gamma={self.gamma}"
            )

    def _delta_cumsum(self, log_pf, log_r, log_pterm):
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
        )  # [B, L-1]
        return torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)  # [B, L]

    @staticmethod
    def _reconstruct_ref_logP(ref_log_pf: torch.Tensor, ref_log_pterm: torch.Tensor):
        B, T = ref_log_pf.shape
        B2, L = ref_log_pterm.shape
        assert (
            B == B2 and L == T + 1
        ), f"shape mismatch: pf {ref_log_pf.shape}, pterm {ref_log_pterm.shape}"
        prefix = ref_log_pf.cumsum(dim=-1)  # [B, T]
        ref_logP = ref_log_pterm.clone()
        ref_logP[:, 1:] = ref_logP[:, 1:] + prefix
        return ref_logP

    @staticmethod
    def _suffix_future_max(u: torch.Tensor, valid: torch.Tensor):
        """
        u: [B, L]
        valid: [B, L] bool, True means position is allowed in the suffix set.
        returns v: [B, L] where v[b,k] = max_{t>=k, valid[b,t]} u[b,t]
        """
        dtype = u.dtype
        device = u.device
        neg_inf = torch.finfo(dtype).min
        u_mask = torch.where(valid, u, torch.full_like(u, neg_inf))
        rev = torch.flip(u_mask, dims=[1])
        rev_max = torch.cummax(rev, dim=1).values
        return torch.flip(rev_max, dims=[1])

    @staticmethod
    def _suffix_future_soft(u: torch.Tensor, valid: torch.Tensor, beta: float, rho: float):
        """
        v_k = (1/beta) logsumexp_{t>=k, valid[t]} exp(beta*(u_t - rho*(t-k))).
        Implemented by reverse scan:
          Z_k = logaddexp(beta*u_k, Z_{k+1} - beta*rho)
        """
        B, L = u.shape
        dtype = u.dtype
        device = u.device
        neg_inf = torch.finfo(dtype).min

        u_mask = torch.where(valid, u, torch.full_like(u, neg_inf))
        out = torch.empty_like(u_mask)

        b = float(beta)
        r = float(rho)

        Z = b * u_mask[:, -1]  # [B]
        out[:, -1] = Z / b

        step_pen = b * r
        for t in range(L - 2, -1, -1):
            Z = torch.logaddexp(b * u_mask[:, t], Z - step_pen)
            out[:, t] = Z / b
        return out

    def forward(
        self,
        log_pf: torch.Tensor,  # [B, L]
        log_r: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        ref_log_pf: Optional[torch.Tensor] = None,  # [B, L-1]
        ref_log_pterm: Optional[torch.Tensor] = None,  # [B, L]
        ref_scale: float = 1.0,
        max_prefix_len: Optional[
            int
        ] = None,  # <-- K: absorb horizon (scheduler). None => horizon=tau
        **kwargs,
    ):
        B, L = log_pf.shape
        assert log_r.shape == (B, L) and log_pterm.shape == (B, L)
        assert generated_text.shape[1] - prompt_len == L
        assert L > 1

        # ---- tau from EOS mask ----
        eos_or_after = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(
            dim=-1
        ) >= 1  # [B, L-1]
        valid_end = ~eos_or_after  # [B, L-1] valid prefix ends (k=1..)
        tau = valid_end.sum(dim=1).clamp(0, L - 1)  # [B] first EOS state index

        # ---- main TB at terminal (keeps pterm grads) ----
        C_main = self._delta_cumsum(log_pf, log_r, log_pterm)  # [B, L]
        C_tau = C_main.gather(1, tau.view(B, 1)).squeeze(1)
        loss_tb = (C_tau**2).mean()

        # ---- auxiliary rooted absorbed prefixes (extra-only, horizon K) ----
        loss_aux = torch.zeros((), device=log_pf.device, dtype=log_pf.dtype)
        if self.aux_weight > 0.0:
            # horizon K: if None => per-sample horizon = tau (absorb everywhere until terminal)
            if max_prefix_len is None:
                K = None
            else:
                K = int(max_prefix_len)
                K = max(1, min(K, L - 1))

            lp_aux = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm
            C_aux = self._delta_cumsum(log_pf, log_r, lp_aux)  # [B, L]
            Ck = C_aux[:, 1:]  # [B, L-1], k=1..L-1

            # prefix indices k=1..L-1
            k_idx = torch.arange(1, L, device=log_pf.device).view(1, L - 1)  # [1, L-1]

            # aux is only computed on reachable prefixes and within horizon h:
            # - if K is None: h = tau
            # - else: h = min(K, tau)  (so we only use prefixes <= K; if tau<K, we only have <=tau)
            if K is None:
                h = tau
            else:
                h = torch.minimum(tau, tau.new_full(tau.shape, K))

            within_tau = k_idx <= tau.view(B, 1)  # [B, L-1]
            within_h = k_idx <= h.view(B, 1)  # [B, L-1]
            m = (within_tau & within_h & valid_end).to(log_pf.dtype)  # [B, L-1]

            # weights w_k = lambda^(k-1)
            w = (
                self.subtb_lambda
                ** torch.arange(0, L - 1, device=log_pf.device, dtype=log_pf.dtype)
            ).view(1, -1)

            # ---------- compute extra u ----------
            if (ref_log_pf is not None) and (ref_log_pterm is not None):
                ref_logP = self._reconstruct_ref_logP(ref_log_pf, ref_log_pterm)  # [B, L]
                u = log_r - float(ref_scale) * ref_logP  # [B, L]
            else:
                # fallback: treat log_r itself as "extra" (less principled, but keeps training running)
                u = log_r

            u_k = u[:, 1:]  # [B, L-1]

            # "no reward" prefixes: only these get absorption correction
            no_reward = u_k.abs() <= self.extra_absorb_eps  # [B, L-1]

            # If K is given, sequences with tau < K should remain unchanged (no absorption).
            if K is None:
                can_absorb_seq = torch.ones((B, 1), device=log_pf.device, dtype=torch.bool)
            else:
                can_absorb_seq = tau.view(B, 1) >= K

            # We only absorb for k < K (strictly before horizon) when K is given.
            # If K is None (absorb to terminal), we absorb for k <= tau; strictness isn't important.
            if K is None:
                before_horizon = within_h
            else:
                before_horizon = k_idx < K  # [1, L-1], broadcast

            # ---------- build per-position target u_target[k] from future in [k..h] ----------
            # valid states for future aggregation: state index t is valid iff t <= h (per sample) and t <= tau.
            pos = torch.arange(L, device=log_pf.device).view(1, L)  # [1, L]
            valid_future = (pos <= h.view(B, 1)) & (pos <= tau.view(B, 1))  # [B, L]

            # Compute two targets over suffix (k..):
            u_max = self._suffix_future_max(u.detach(), valid_future)  # [B, L]
            u_soft = self._suffix_future_soft(
                u.detach(), valid_future, beta=self.soft_beta, rho=self.soft_rho
            )  # [B, L]

            if self.target_mode == "future_max":
                u_target = u_max
            elif self.target_mode == "future_soft":
                u_target = u_soft
            elif self.target_mode == "mix":
                mw = float(self.mix_weight)
                u_target = mw * u_max + (1.0 - mw) * u_soft
            else:
                raise ValueError(f"Unknown target_mode: {self.target_mode}")

            u_tk = u_target[:, 1:]  # [B, L-1] aligned with k=1..L-1

            # ---------- mandatory length decay alpha_k = gamma^(horizon - k) ----------
            # horizon for decay: if K is given, use K; else use h (which is tau)
            if K is None:
                horizon_for_decay = h
            else:
                horizon_for_decay = h  # equals K for sequences with tau>=K; equals tau otherwise (but those won't absorb)

            exp = (horizon_for_decay.view(B, 1) - k_idx).clamp_min(0).to(log_pf.dtype)  # [B, L-1]
            alpha = (self.gamma**exp) * within_h.to(log_pf.dtype)  # [B, L-1]

            # ---------- apply absorption correction only where you asked ----------
            # condition:
            #   (1) prefix is in aux mask (reachable & <=h)
            #   (2) prefix has no reward signal
            #   (3) prefix is before horizon (k<K when K given)
            #   (4) sequence can absorb (tau>=K when K given)
            apply_absorb = m.bool() & no_reward & before_horizon & can_absorb_seq  # [B, L-1]

            alpha_eff = alpha * apply_absorb.to(log_pf.dtype)

            # correction: replace boundary extra u_k by future target u_tk (stopgrad target)
            corr = alpha_eff * (u_k - u_tk)  # [B, L-1]
            Ck_abs = Ck + corr

            num = ((Ck_abs**2) * m * w).sum()
            den = (m * w).sum().clamp_min(self.eps)
            loss_aux = num / den

        loss = (1 - self.aux_weight) * loss_tb + self.aux_weight * loss_aux
        return {"loss": loss, "loss_tb": loss_tb.detach(), "loss_aux": loss_aux.detach()}


class RootAbsorbExtraSubTBLossFixTBLogZ(GFNLoss):
    """
    L = (1-eta)*L_TB_terminal + eta*L_aux_rooted_absorb_extra(K)

    Alignment (MATCH your existing pipeline):
      - log_pf:    [B, L]   where meaningful token steps are log_pf[:, :-1]  (length L-1)
      - log_pterm: [B, L]   eos logprob at each state k=0..L-1
      - log_r:     [B, L]   per-state reward (your mixed reward), k=0..L-1
      - generated_text post-prompt length == L
      - generated_text[:, prompt_len:-1] excludes the forced last EOS token (by max length)

    tau definition (EXACTLY like your previous loss):
      eos_or_after = cumsum( token==EOS )>=1 over generated_text[:, prompt_len:-1]
      valid_end = ~eos_or_after
      tau = valid_end.sum() in [0, L-1]
      - if EOS appears early, tau is number of tokens strictly before first EOS
      - if no EOS before last forced EOS, tau = L-1

    Standard TB (terminal):
      res_tau = logZ + sum_{t < tau} log_pf[t] + log_pterm[tau] - log_r[tau]
      where the sum uses log_pf[:, :-1] steps.
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
        init_logZ: float = 0.0,  # NEW: explicit learnable logZ
    ):
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

        if not (self.gamma < 1.0):
            raise ValueError(f"gamma must be < 1.0, got gamma={self.gamma}")

        # global learnable logZ
        self.logZ = torch.nn.Parameter(torch.tensor([float(init_logZ)], dtype=torch.float32))

    # ----------------- helpers -----------------

    @staticmethod
    def _gather_by_index(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        return x.gather(1, idx.view(B, 1)).squeeze(1)

    @staticmethod
    def _sum_log_pf_upto_tau(log_pf: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        sum_{t < tau} log_pf[t], but only meaningful steps are log_pf[:, :-1].
        log_pf: [B, L], use steps = log_pf[:, :-1] of length L-1.
        tau: [B] in [0, L-1]
        """
        B, L = log_pf.shape
        steps = log_pf[:, :-1]  # [B, L-1]
        if steps.shape[1] == 0:
            return torch.zeros((B,), device=log_pf.device, dtype=log_pf.dtype)

        pf_cum = steps.cumsum(dim=1)  # [B, L-1]
        idx = (tau - 1).clamp(min=0, max=steps.shape[1] - 1)  # [B]
        s = pf_cum.gather(1, idx.view(B, 1)).squeeze(1)  # [B]
        s = s * (tau > 0).to(s.dtype)  # tau=0 => 0
        return s

    @staticmethod
    def _reconstruct_ref_logP(
        ref_log_pf: torch.Tensor, ref_log_pterm: torch.Tensor
    ) -> torch.Tensor:
        """
        Matches score_fast:
          ref_logP[0] = ref_log_pterm[0]
          ref_logP[k] = ref_log_pterm[k] + sum_{t<k} ref_log_pf[t], for k>=1
        Expected shapes:
          ref_log_pf:   [B, L-1]
          ref_log_pterm:[B, L]
        """
        B, T = ref_log_pf.shape
        B2, L = ref_log_pterm.shape
        assert (
            B == B2 and L == T + 1
        ), f"shape mismatch: ref_log_pf {ref_log_pf.shape}, ref_log_pterm {ref_log_pterm.shape}"

        prefix = ref_log_pf.cumsum(dim=1)  # [B, L-1]
        ref_logP = ref_log_pterm.clone()  # [B, L]
        ref_logP[:, 1:] = ref_logP[:, 1:] + prefix
        return ref_logP

    def _delta_cumsum(
        self, log_pf: torch.Tensor, log_r: torch.Tensor, log_pterm: torch.Tensor
    ) -> torch.Tensor:
        """
        Keep EXACTLY your previous delta definition (now shapes are aligned):
          delta[k] for k=1..L-1 uses log_pf[:, :-1] as steps.
        returns C: [B, L] with C[:,0]=0
        """
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
        )  # [B, L-1]
        return torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)  # [B, L]

    @staticmethod
    def _suffix_future_max(u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        dtype = u.dtype
        neg_inf = torch.finfo(dtype).min
        u_mask = torch.where(valid, u, torch.full_like(u, neg_inf))
        rev = torch.flip(u_mask, dims=[1])
        rev_max = torch.cummax(rev, dim=1).values
        return torch.flip(rev_max, dims=[1])

    @staticmethod
    def _suffix_future_soft(
        u: torch.Tensor, valid: torch.Tensor, beta: float, rho: float
    ) -> torch.Tensor:
        B, L = u.shape
        dtype = u.dtype
        neg_inf = torch.finfo(dtype).min
        u_mask = torch.where(valid, u, torch.full_like(u, neg_inf))
        out = torch.empty_like(u_mask)

        b = float(beta)
        step_pen = b * float(rho)

        Z = b * u_mask[:, -1]  # [B]
        out[:, -1] = Z / b
        for t in range(L - 2, -1, -1):
            Z = torch.logaddexp(b * u_mask[:, t], Z - step_pen)
            out[:, t] = Z / b
        return out

    # ----------------- forward -----------------

    def forward(
        self,
        log_pf: torch.Tensor,  # [B, L]  (NOTE: L = T_tok+1 in your setup)
        log_r: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        ref_log_pf: Optional[torch.Tensor] = None,  # [B, L-1]
        ref_log_pterm: Optional[torch.Tensor] = None,  # [B, L]
        ref_scale: float = 1.0,
        max_prefix_len: Optional[int] = None,  # K in [1, L-1]
        logZ: Optional[torch.Tensor] = None,  # optional override scalar/[B]
        **kwargs,
    ):
        B, L = log_pf.shape
        assert log_r.shape == (B, L), f"log_r {log_r.shape} vs (B,L)={(B,L)}"
        assert log_pterm.shape == (B, L), f"log_pterm {log_pterm.shape} vs (B,L)={(B,L)}"
        assert generated_text.shape[0] == B
        assert generated_text.shape[1] - prompt_len == L, (
            f"generated_text post-prompt length must equal L. "
            f"got generated_text.shape={generated_text.shape}, prompt_len={prompt_len}, L={L}"
        )
        assert L > 1, "Need L>=2 (at least one step + terminal state)"

        # ---------------- tau (exactly your previous logic) ----------------
        # exclude the forced last EOS: generated_text[:, prompt_len:-1] has length L-1
        eos_or_after = (generated_text[:, prompt_len:-1] == int(termination_token_id)).cumsum(
            dim=-1
        ) >= 1  # [B, L-1]
        valid_end = ~eos_or_after  # [B, L-1], True before first EOS
        tau = valid_end.sum(dim=1).clamp(0, L - 1)  # [B], in [0, L-1]

        # ================== 1) Standard terminal TB with explicit logZ ==================
        if logZ is None:
            logZ_b = self.logZ.to(device=log_pf.device, dtype=log_pf.dtype).expand(B)
        else:
            z = logZ.to(device=log_pf.device, dtype=log_pf.dtype)
            logZ_b = z.expand(B) if z.ndim == 0 else z
            assert logZ_b.shape == (B,), f"logZ must be scalar or [B], got {tuple(logZ.shape)}"

        sum_log_pf = self._sum_log_pf_upto_tau(log_pf, tau)  # [B]
        log_pterm_tau = self._gather_by_index(log_pterm, tau)  # [B]
        log_r_tau = self._gather_by_index(log_r, tau)  # [B]

        tb_res = logZ_b + sum_log_pf + log_pterm_tau - log_r_tau
        loss_tb = (tb_res**2).mean()

        # ================== 2) AUX rooted absorbed prefixes ==================
        loss_aux = torch.zeros((), device=log_pf.device, dtype=log_pf.dtype)
        if self.aux_weight > 0.0:
            # K clamp
            if max_prefix_len is None:
                K = None
            else:
                K = int(max_prefix_len)
                K = max(1, min(K, L - 1))

            lp_aux = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm
            C_aux = self._delta_cumsum(log_pf, log_r, lp_aux)  # [B, L]
            Ck = C_aux[:, 1:]  # [B, L-1], k=1..L-1

            k_idx = torch.arange(1, L, device=log_pf.device).view(1, L - 1)  # [1, L-1]

            # horizon h
            if K is None:
                h = tau
            else:
                h = torch.minimum(tau, tau.new_full(tau.shape, K))

            within_tau = k_idx <= tau.view(B, 1)
            within_h = k_idx <= h.view(B, 1)
            m = (within_tau & within_h & valid_end).to(log_pf.dtype)  # [B, L-1]

            # w_k = lambda^(k-1)
            w = (
                self.subtb_lambda
                ** torch.arange(0, L - 1, device=log_pf.device, dtype=log_pf.dtype)
            ).view(1, -1)

            # extra u = log_r - ref_scale * ref_logP
            if (ref_log_pf is not None) and (ref_log_pterm is not None):
                assert ref_log_pf.shape == (
                    B,
                    L - 1,
                ), f"ref_log_pf {ref_log_pf.shape} vs (B,L-1)={(B,L-1)}"
                assert ref_log_pterm.shape == (
                    B,
                    L,
                ), f"ref_log_pterm {ref_log_pterm.shape} vs (B,L)={(B,L)}"
                ref_logP = self._reconstruct_ref_logP(ref_log_pf, ref_log_pterm)  # [B, L]
                u = log_r - float(ref_scale) * ref_logP  # [B, L]
            else:
                u = log_r

            u_k = u[:, 1:]  # [B, L-1]
            no_reward = u_k.abs() <= self.extra_absorb_eps

            # tau < K => no absorption correction
            if K is None:
                can_absorb_seq = torch.ones((B, 1), device=log_pf.device, dtype=torch.bool)
            else:
                can_absorb_seq = tau.view(B, 1) >= K

            if K is None:
                before_horizon = within_h
            else:
                before_horizon = k_idx < K

            # suffix targets on states (0..L-1)
            pos = torch.arange(L, device=log_pf.device).view(1, L)
            valid_future = (pos <= h.view(B, 1)) & (pos <= tau.view(B, 1))

            u_max = self._suffix_future_max(u.detach(), valid_future)
            u_soft = self._suffix_future_soft(
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

            u_tk = u_target[:, 1:]  # [B, L-1]

            # alpha_k = gamma^(h - k)
            exp = (h.view(B, 1) - k_idx).clamp_min(0).to(log_pf.dtype)
            alpha = (self.gamma**exp) * within_h.to(log_pf.dtype)

            apply_absorb = m.bool() & no_reward & before_horizon & can_absorb_seq
            alpha_eff = alpha * apply_absorb.to(log_pf.dtype)

            corr = alpha_eff * (u_k - u_tk)
            Ck_abs = Ck + corr

            num = ((Ck_abs**2) * m * w).sum()
            den = (m * w).sum().clamp_min(self.eps)
            loss_aux = num / den

        loss = (1.0 - self.aux_weight) * loss_tb + self.aux_weight * loss_aux
        return {
            "loss": loss,
            "loss_tb": loss_tb.detach(),
            "loss_aux": loss_aux.detach(),
            "logZ": self.logZ.detach(),
        }


def _first_eos_index(gen_tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    B, S = gen_tokens.shape
    is_eos = gen_tokens == eos_id
    has_eos = is_eos.any(dim=1)
    first = is_eos.float().argmax(dim=1)  # 若全 False -> 0，需要修正
    first = torch.where(has_eos, first, torch.full_like(first, S - 1))
    return first


class LLMTrajectoryBalanceLoss(nn.Module):
    """
    Trajectory Balance (TB) loss for LLM-GFlowNet with separate (log_pf, log_pterm, log_r grid).

    兼容你的“log_r 是 terminate-at-t 的 logR 表格”设计：
      - 终止发生在 τ（首次 EOS）：
         logP(traj) = sum_{t<τ} log_pf[t] + log_pterm[τ]
         logR(traj) = log_r[τ]
      - PB=1（tree/backward deterministic）时不需要 log_pb 项
    """

    def __init__(self, learn_log_z: bool = True, **kwargs):
        super().__init__()
        self.log_z = nn.Parameter(torch.zeros(())) if learn_log_z else None

    def forward(
        self,
        log_pf: torch.Tensor,  # (B, S) 或 (B, S-1)
        log_r: torch.Tensor,  # (B, S)  terminate-at-t 的 logR 表
        log_pterm: torch.Tensor,  # (B, S)  每个 state 的 EOS logprob
        generated_text: torch.Tensor,  # (B, prompt_len + S)
        termination_token_id: int,
        prompt_len: int,
        log_z: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        assert log_r.ndim == log_pterm.ndim == 2
        B, S = log_pterm.shape
        assert log_r.shape == (
            B,
            S,
        ), f"log_r shape {log_r.shape} must match log_pterm shape {(B,S)}"

        # 允许 log_pf 少 1（有人只存非终止动作的 pf）
        if log_pf.shape[1] == S - 1:
            log_pf = torch.cat([log_pf, log_pf.new_zeros((B, 1))], dim=1)
        assert log_pf.shape == (B, S), f"log_pf shape {log_pf.shape} must be (B,S) or (B,S-1)"

        # 从 generated_text 找 τ：prompt 后前 S 个 token
        gen = generated_text[:, prompt_len : prompt_len + S]
        tau = _first_eos_index(gen, termination_token_id)  # (B,)

        # mask: t < τ 的 token 步（非终止前缀）
        t = torch.arange(S, device=log_pf.device).view(1, S)
        pre_mask = t < tau.view(B, 1)  # (B, S)

        # logP(traj) = sum_{t<τ} log_pf[t] + log_pterm[τ]
        token_logp = (log_pf * pre_mask.to(log_pf.dtype)).sum(dim=1)  # (B,)
        term_logp = log_pterm.gather(1, tau.view(B, 1)).squeeze(1)  # (B,)
        logp_traj = token_logp + term_logp

        # logR(traj) = log_r[τ] （你的 scorer 已经把 prefix+eos 概率塞进 log_r[t] 了）
        logr_traj = log_r.gather(1, tau.view(B, 1)).squeeze(1)

        # logZ：外部传入优先，否则学一个全局标量
        if log_z is None:
            if self.log_z is None:
                raise ValueError("No log_z provided and learn_log_z=False.")
            log_z = self.log_z
        log_z_b = log_z.expand(B)

        residual = log_z_b + logp_traj - logr_traj
        loss = (residual**2).mean()

        return {
            "loss": loss,
            "tb_residual_mean": residual.mean().detach(),
            "tb_residual_std": residual.std(unbiased=False).detach(),
            "logp_traj_mean": logp_traj.mean().detach(),
            "logr_traj_mean": logr_traj.mean().detach(),
        }


class ModifiedSubTBLossSplitReward(GFNLoss):
    """
    Modified SubTrajectory Balance (SubTB) Loss with split reward.
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        alpha_reference: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.subtb_lambda = subtb_lambda
        self.alpha_reference = float(alpha_reference)
        assert 0.0 <= self.alpha_reference <= 1.0, "alpha_reference must be between 0 and 1"
        self.eps = eps

    def set_alpha_reference(self, alpha_reference: float):
        self.alpha_reference = float(alpha_reference)
        assert 0.0 <= self.alpha_reference <= 1.0, "alpha_reference must be between 0 and 1"

    def forward(
        self,
        log_pf: torch.Tensor,  # [B, L]
        log_r_reference: torch.Tensor,  # [B, L]
        log_r_target: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute SubTB loss.

        Args:
            log_pf: Log forward policy probabilities at each step, shape [B, L]
            log_r: Log reward prefix accumulator, shape [B, L]
            log_pterm: Log termination probabilities, shape [B, L]
            generated_text: Token IDs including prompt, shape [B, prompt_len + L]
            termination_token_id: EOS token ID
            prompt_len: Length of prompt
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Scalar loss tensor
        """
        # Ensure the dimensions of log probabilities, rewards, and generated text match
        assert (
            log_pf.shape[1]
            == log_r_reference.shape[1]
            == log_r_target.shape[1]
            == log_pterm.shape[1]
            == generated_text.shape[1] - prompt_len
        ), f"Shape mismatch: log_pf={log_pf.shape}, log_r_reference={log_r_reference.shape}, log_r_target={log_r_target.shape}, log_pterm={log_pterm.shape}, generated_text={generated_text.shape}, prompt_len={prompt_len}"

        # Ensure there is at least one transition before termination
        assert log_pf.shape[1] > 1, "Need at least one transition before termination (L > 1)"

        # Calculate the change in expected reward and probability at each step
        delta_reference = (
            log_r_reference[:, :-1]
            + log_pf[:, :-1]
            + log_pterm[:, 1:]
            - log_r_reference[:, 1:]
            - log_pterm[:, :-1]
        )
        delta_target = (
            log_r_target[:, :-1]
            + log_pf[:, :-1]
            + log_pterm[:, 1:]
            - log_r_target[:, 1:]
            - log_pterm[:, :-1]
        )

        # Compute cumulative sum of delta for subtrajectory balance calculation
        delta_cumsum_reference = torch.cat(
            [torch.zeros_like(delta_reference[:, :1]), delta_reference], dim=1
        ).cumsum(dim=1)
        delta_cumsum_target = torch.cat(
            [torch.zeros_like(delta_target[:, :1]), delta_target], dim=1
        ).cumsum(dim=1)

        # Create a mask for tokens after the termination token
        mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(dim=-1) >= 1

        batch_loss_reference = 0.0
        batch_loss_target = 0.0
        total_lambda = 0.0
        generated_len = generated_text.shape[1] - prompt_len

        for subtraj_len in range(1, generated_len):
            # Calculate the subtrajectory balance term
            subtb_term_reference = (
                delta_cumsum_reference[:, subtraj_len:] - delta_cumsum_reference[:, :-subtraj_len]
            ) ** 2
            subtb_term_target = (
                delta_cumsum_target[:, subtraj_len:] - delta_cumsum_target[:, :-subtraj_len]
            ) ** 2
            # Apply mask to ignore invalid parts of the sequence
            subtb_term_reference[mask[:, subtraj_len - 1 :]] = 0
            subtb_term_target[mask[:, subtraj_len - 1 :]] = 0
            # Accumulate weighted subtrajectory balance term
            batch_loss_reference += (
                self.subtb_lambda ** (subtraj_len - 1) * subtb_term_reference.sum()
            )
            batch_loss_target += self.subtb_lambda ** (subtraj_len - 1) * subtb_term_target.sum()
            # Accumulate total weight for normalization
            total_lambda += (
                self.subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
            )

        # Normalize the loss by the total weight
        batch_loss_reference /= total_lambda
        batch_loss_target /= total_lambda

        total_loss = (
            self.alpha_reference * batch_loss_reference
            + (1.0 - self.alpha_reference) * batch_loss_target
        )

        return {
            "loss": total_loss,
            "loss_reference": batch_loss_reference.detach(),
            "loss_target": batch_loss_target.detach(),
        }


class ModifiedSubTBBalanceLoss(GFNLoss):
    """
    Modified SubTrajectory Balance (SubTB) Loss with token-coverage balancing.

    This is an enhanced version of SubTB that supports token-level balancing to ensure
    all tokens contribute more evenly to the loss, improving training stability.

    Args:
        subtb_lambda (float): Per-window-length weight, ^(len-1). Default: 1.0
        balance (float): Token-level balancing degree in [0,1].
                        0 keeps original window-sum;
                        1 re-weights so each token contributes equally.
                        Default: 0.0
        eps (float): Numerical stabilizer for division. Default: 1e-8

    References:
        SubTB: https://arxiv.org/abs/2209.12782
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        balance: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.subtb_lambda = subtb_lambda
        self.balance = balance
        self.eps = eps

    def forward(
        self,
        log_pf: torch.Tensor,  # [B, L]
        log_r: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute SubTB loss with token-coverage balancing.

        Args:
            log_pf: Log P_F at each step (forward policy), shape [B, L]
            log_r: Log reward prefix accumulator used in SubTB, shape [B, L]
            log_pterm: Log termination (or other TB buffers), shape [B, L]
            generated_text: Token IDs incl. prompt, shape [B, prompt_len+L]
            termination_token_id: EOS id to build mask
            prompt_len: Length of prompt
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Scalar loss tensor
        """
        # Basic checks
        B, L = log_pf.shape
        assert (
            L == log_r.shape[1] == log_pterm.shape[1] == generated_text.shape[1] - prompt_len
        ), "Shapes must match: [B, L] and generated_text len - prompt_len == L"
        assert L > 1, "Need at least one transition before termination (L > 1)."

        # ----- Standard SubTB core (unchanged) -----
        # delta has length L-1 (per-step TB residuals)
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
        )  # [B, L-1]

        # prefix-sum trick to get window sums quickly
        delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(
            dim=1
        )  # [B, L]

        # mask out everything at/after first EOS inside the generated suffix (exclude last token)
        # shape: [B, L-1]
        mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(dim=-1) >= 1

        # ---------- (A) Original design: window-wise accumulation ----------
        loss_num_win = delta.new_zeros(())
        loss_den_win = delta.new_zeros(())

        # ---------- (B) New addition: token-coverage "amortized" accumulation ----------
        # We amortize each window loss (sum(delta_i))^2 of length s evenly to the s delta positions it covers.
        # Then average over all tokens (delta positions). This makes all tokens contribute more evenly to the loss.
        # contributions: accumulated amortized loss for each sample and each delta position
        contributions = torch.zeros(B, L - 1, device=delta.device, dtype=delta.dtype)
        # denom_tokens: effective coverage count for each delta position (also amortized by 1/s)
        denom_tokens = torch.zeros(B, L - 1, device=delta.device, dtype=delta.dtype)

        for s in range(1, L):  # window length from 1..L-1
            # Window sum: delta_cumsum[:, s:] - delta_cumsum[:, :-s]  -> [B, L-s]
            window_sum = delta_cumsum[:, s:] - delta_cumsum[:, :-s]  # [B, L-s]
            subtb_term = window_sum.pow(2)  # [B, L-s]

            # Apply mask (at the window start index dimension)
            # mask has shape [B, L-1], slice to [B, L-s]
            m_slice = mask[:, s - 1 :]  # [B, L-s]
            subtb_term = subtb_term.masked_fill(m_slice, 0.0)

            # Original design accumulation (window-wise) with length weight subtb_lambda^(s-1)
            w = self.subtb_lambda ** (s - 1)
            loss_num_win = loss_num_win + w * subtb_term.sum()
            loss_den_win = loss_den_win + w * (~m_slice).sum()

            # ------- Amortize to token (delta index) level contribution -------
            # Distribute each window value evenly to the s positions it covers (by 1/s):
            # This is equivalent to a 1D transposed convolution on [B, 1, L-s] with an all-ones kernel of length s,
            # resulting in [B, 1, (L-s) + s - 1] = [B, 1, L-1], then divide by s.
            if self.balance > 0:
                v = (w * subtb_term).unsqueeze(1)  # [B, 1, L-s]
                k = torch.ones(1, 1, s, device=delta.device, dtype=delta.dtype)
                contrib_add = F.conv_transpose1d(v, k)[:, 0, :] / float(s)  # [B, L-1]
                contributions.add_(contrib_add)

                # Coverage count denominator: use the same "transposed convolution" to compute how many
                # windows of length s cover each token, amortized by 1/s with the same weight w.
                cover = (~m_slice).to(dtype=delta.dtype).unsqueeze(1)  # [B, 1, L-s]
                cover_add = F.conv_transpose1d(cover, k)[:, 0, :] / float(s)  # [B, L-1]
                denom_tokens.add_(w * cover_add)

        # Safety clamp
        loss_den_win = torch.clamp(loss_den_win, min=self.eps)

        # (A) Original window-wise loss
        loss_win = loss_num_win / loss_den_win

        if self.balance <= 0:
            return loss_win

        # (B) Token-balanced loss: average the contribution allocated to each token
        denom_tokens = torch.clamp(denom_tokens, min=self.eps)
        loss_tok = (contributions / denom_tokens).mean()

        # Linear interpolation: balance=0 -> pure original design; balance=1 -> fully balanced to token
        loss = (1.0 - self.balance) * loss_win + self.balance * loss_tok
        return {"loss": loss}


def _entropy_from_counts(counts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = counts / counts.sum().clamp_min(eps)
    return -(probs * probs.add(eps).log()).sum()


class SubTBBatchDiversityLoss(GFNLoss):
    """
    SubTB + batch-level diversity loss.

    Args:
        subtb_lambda: SubTB lambda
        balance: Balance between SubTB and diversity loss
        eps: Epsilon
        seq_diversity_weight: Sequence-level diversity weight
        pos_diversity_weight: Position-level diversity weight
        pos_max: Maximum position
        use_logits: Use logits to compute diversity loss
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        balance: float = 0.0,
        eps: float = 1e-8,
        seq_diversity_weight: float = 0.0,
        pos_diversity_weight: float = 0.0,
        pos_max: Optional[int] = None,
        use_logits: bool = False,
    ):
        super().__init__()
        self.base_subtb = ModifiedSubTBBalanceLoss(
            subtb_lambda=subtb_lambda, balance=balance, eps=eps
        )
        self.seq_diversity_weight = seq_diversity_weight
        self.pos_diversity_weight = pos_diversity_weight
        self.pos_max = pos_max
        self.eps = eps
        self.use_logits = use_logits
        self.requires_policy_logits = use_logits

    def _batch_seq_entropy(
        self, suffix: torch.Tensor, termination_token_id: int
    ) -> Optional[torch.Tensor]:
        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        alive_mask = eos_cum == 0
        if not bool(alive_mask.any()):
            return None
        tokens = suffix[alive_mask]
        vocab = int(tokens.max().item()) + 1
        counts = torch.bincount(tokens, minlength=vocab).to(torch.float32)
        return _entropy_from_counts(counts, eps=self.eps)

    def _batch_pos_entropy_tokens(
        self, suffix: torch.Tensor, termination_token_id: int, max_pos: Optional[int]
    ) -> Optional[torch.Tensor]:
        B, L = suffix.shape
        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        ent_acc = torch.zeros((), device=suffix.device, dtype=torch.float32)
        ent_cnt = torch.zeros((), device=suffix.device, dtype=torch.float32)
        limit = L if max_pos is None else min(L, int(max_pos))
        vocab = int(suffix.max().item()) + 1
        for t in range(limit):
            mask = eos_cum[:, t] == 0
            if not bool(mask.any()):
                continue
            toks = suffix[mask, t]
            counts = torch.bincount(toks, minlength=vocab).to(torch.float32)
            ent_acc = ent_acc + _entropy_from_counts(counts, eps=self.eps)
            ent_cnt = ent_cnt + 1.0
        if ent_cnt.item() == 0:
            return None
        return ent_acc / ent_cnt.clamp_min(1.0)

    def _batch_pos_entropy_logits(
        self,
        policy_logits: torch.Tensor,
        suffix: torch.Tensor,
        termination_token_id: int,
        max_pos: Optional[int],
    ) -> Optional[torch.Tensor]:
        if policy_logits is None:
            return None
        B, L = suffix.shape
        logits = policy_logits[:, :L, :]
        ptheta = logits.log_softmax(dim=-1).exp()
        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        ent_acc = torch.zeros((), device=suffix.device, dtype=ptheta.dtype)
        ent_cnt = torch.zeros((), device=suffix.device, dtype=ptheta.dtype)
        limit = L if max_pos is None else min(L, int(max_pos))
        for t in range(limit):
            mask = eos_cum[:, t] == 0
            if not bool(mask.any()):
                continue
            p_bar = ptheta[mask, t].mean(dim=0)
            ent_acc = ent_acc + (-(p_bar * p_bar.add(self.eps).log())).sum()
            ent_cnt = ent_cnt + 1.0
        if ent_cnt.item() == 0:
            return None
        return ent_acc / ent_cnt.clamp_min(1.0)

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        policy_logits: Optional[torch.Tensor] = None,
        weight_overrides: Optional[dict] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        suffix = generated_text[:, prompt_len:]
        subtb_out = self.base_subtb(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=termination_token_id,
            prompt_len=prompt_len,
        )
        subtb_loss = subtb_out["loss"] if isinstance(subtb_out, dict) else subtb_out
        loss_total = subtb_loss
        logs = {"loss_subtb": subtb_loss.detach()}

        seq_w = self._resolve_weight(
            "seq_diversity_weight", self.seq_diversity_weight, weight_overrides
        )
        pos_w = self._resolve_weight(
            "pos_diversity_weight", self.pos_diversity_weight, weight_overrides
        )

        if seq_w >= 0.0:
            seq_ent = self._batch_seq_entropy(suffix, termination_token_id)
            if seq_ent is not None:
                loss_total = loss_total - seq_w * seq_ent
                logs["loss_batch_seq_entropy"] = (-seq_w * seq_ent).detach()

        if pos_w >= 0.0:
            if self.use_logits:
                pos_ent = self._batch_pos_entropy_logits(
                    policy_logits, suffix, termination_token_id, self.pos_max
                )
            else:
                pos_ent = self._batch_pos_entropy_tokens(
                    suffix, termination_token_id, self.pos_max
                )
            if pos_ent is not None:
                loss_total = loss_total - pos_w * pos_ent
                logs["loss_batch_pos_entropy"] = (-pos_w * pos_ent).detach()

        logs["loss"] = loss_total
        return logs


class SubTBMCMCPriorLoss(GFNLoss):
    """
    SubTB loss with optional MCMC n-gram regularization.

    Adds two regularizers to the base SubTB objective:
    - Sequence-level prior NLL against a pre-computed q_MCMC.
    - Position-wise KL between the model policy and q_MCMC.
    Coarse-grained variants are supported via a token grouping map.
    """

    def __init__(
        self,
        subtb_lambda: float = 1.0,
        balance: float = 0.0,
        eps: float = 1e-8,
        # MCMC priors
        q_mcmc_path: Optional[str] = None,
        q_mcmc: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = None,
        prior_temperature: float = 1.0,
        base_n: int = 2,
        bos_id: int = -1,
        vocab_size: Optional[int] = None,
        # Sequence-level prior
        seq_weight: float = 0.0,
        seq_include_eos: bool = False,
        seq_normalize_by_len: bool = True,
        seq_max_pos: Optional[int] = None,
        # Logits-level KL
        kl_weight: float = 0.0,
        kl_max_pos: Optional[int] = 1,
        kl_temperature: float = 1.0,
        # Coarse-grained options
        coarse_token_groups: Optional[list[list[int]]] = None,
        token_to_coarse: Optional[Dict[int, int]] = None,
        coarse_default_id: Optional[int] = None,
        coarse_ngram_n: Optional[int] = None,
        coarse_seq_weight: float = 0.0,
        coarse_seq_include_eos: bool = False,
        coarse_seq_normalize_by_len: bool = True,
        coarse_seq_max_pos: Optional[int] = None,
        coarse_kl_weight: float = 0.0,
        coarse_kl_max_pos: Optional[int] = 1,
        coarse_prior_temperature: Optional[float] = None,
        coarse_q_mcmc_path: Optional[str] = None,
        coarse_q_mcmc: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = None,
        # Batch-level distribution regularizer
        batch_kl_weight: float = 0.0,
        batch_kl_max_pos: Optional[int] = None,
        coarse_batch_kl_weight: float = 0.0,
        coarse_batch_kl_max_pos: Optional[int] = None,
        # Diversity (entropy) bonus
        diversity_weight: float = 0.0,
        diversity_max_pos: Optional[int] = None,
        diversity_batch_level: bool = True,
        diversity_use_logits: bool = True,
        # Coarse reward shaping
        coarse_reward_weight: float = 0.0,
    ):
        super().__init__()
        self.base_subtb = ModifiedSubTBBalanceLoss(
            subtb_lambda=subtb_lambda, balance=balance, eps=eps
        )
        self.eps = eps
        self.base_n = base_n
        self.bos_id = bos_id
        self.vocab_size = vocab_size

        self.seq_weight = seq_weight
        self.seq_include_eos = seq_include_eos
        self.seq_normalize_by_len = seq_normalize_by_len
        self.seq_max_pos = seq_max_pos

        self.kl_weight = kl_weight
        self.kl_max_pos = kl_max_pos
        self.kl_temperature = kl_temperature
        self.prior_temperature = prior_temperature

        self.coarse_seq_weight = coarse_seq_weight
        self.coarse_seq_include_eos = coarse_seq_include_eos
        self.coarse_seq_normalize_by_len = coarse_seq_normalize_by_len
        self.coarse_seq_max_pos = coarse_seq_max_pos
        self.coarse_kl_weight = coarse_kl_weight
        self.coarse_kl_max_pos = coarse_kl_max_pos
        self.coarse_prior_temperature = (
            prior_temperature if coarse_prior_temperature is None else coarse_prior_temperature
        )
        self.coarse_ngram_n = coarse_ngram_n or base_n

        # Batch-level distribution KL
        self.batch_kl_weight = batch_kl_weight
        self.batch_kl_max_pos = batch_kl_max_pos
        self.coarse_batch_kl_weight = coarse_batch_kl_weight
        self.coarse_batch_kl_max_pos = coarse_batch_kl_max_pos

        # Diversity/entropy bonus
        self.diversity_weight = diversity_weight
        self.diversity_max_pos = diversity_max_pos
        self.diversity_batch_level = diversity_batch_level
        self.diversity_use_logits = diversity_use_logits

        # Coarse reward shaping
        self.coarse_reward_weight = coarse_reward_weight

        mapping = None
        if token_to_coarse is not None:
            mapping = {int(k): int(v) for k, v in token_to_coarse.items()}
        elif coarse_token_groups:
            mapping = {}
            for cid, group in enumerate(coarse_token_groups):
                for tok in group:
                    mapping[int(tok)] = cid

        self.coarse_indexer = (
            CoarseTokenIndexer(mapping, vocab_size, coarse_default_id)
            if mapping is not None
            else None
        )

        self.q_mcmc_source = q_mcmc or self._load_prior(q_mcmc_path)
        self.q_mcmc_coarse_source = coarse_q_mcmc or self._load_prior(coarse_q_mcmc_path)
        self._q_cache: dict[tuple[torch.device, int], Dict[Tuple[int, ...], torch.Tensor]] = {}
        self._q_coarse_cache: dict[
            tuple[torch.device, int], Dict[Tuple[int, ...], torch.Tensor]
        ] = {}
        self._warned_missing_prior: set[str] = set()
        self.requires_policy_logits = (
            (kl_weight > 0.0)
            or (coarse_kl_weight > 0.0)
            or (diversity_weight > 0.0 and diversity_use_logits)
        )
        self.weight_schedulers: dict[str, callable] = {}

    # Optional: allow external schedulers to adjust weights dynamically
    def set_weight_scheduler(self, name: str, fn):
        self.weight_schedulers[name] = fn
        if name in {"kl_weight", "coarse_kl_weight", "diversity_weight"}:
            self.requires_policy_logits = True

    def set_weight_schedulers(self, schedulers: dict[str, callable]):
        self.weight_schedulers.update(schedulers)
        if any(n in {"kl_weight", "coarse_kl_weight", "diversity_weight"} for n in schedulers):
            self.requires_policy_logits = True

    # ----------------------- helpers ----------------------- #
    @staticmethod
    def _load_prior(path: Optional[str]):
        if path is None:
            return None
        try:
            return load_q_mcmc(path)
        except FileNotFoundError:
            print(f"[SubTBMCMCPriorLoss] q_mcmc file not found: {path}, skipping.")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[SubTBMCMCPriorLoss] Failed to load q_mcmc from {path}: {exc}")
        return None

    @staticmethod
    def _ctx_tuple(row: torch.Tensor, t: int, n: int, bos: int) -> Tuple[int, ...]:
        k = n - 1
        if k <= 0:
            return ()
        ctx = []
        for kk in range(k, 0, -1):
            idx = t - kk
            ctx.append(bos if idx < 0 else int(row[idx].item()))
        return tuple(ctx)

    def _infer_vocab_size(self, policy_logits, generated_text: torch.Tensor) -> int:
        if self.vocab_size is not None:
            return int(self.vocab_size)
        if policy_logits is not None:
            return int(policy_logits.shape[-1])
        return int(generated_text.max().item()) + 1

    def _get_q_packed(
        self,
        source: Optional[Dict[Tuple[int, ...], Dict[int, float]]],
        cache: dict[tuple[torch.device, int], Dict[Tuple[int, ...], torch.Tensor]],
        device: torch.device,
        vocab_size: int,
    ) -> Optional[Dict[Tuple[int, ...], torch.Tensor]]:
        if source is None:
            return None
        key = (device, vocab_size)
        if key not in cache:
            cache[key] = pack_q_mcmc_to_device(
                source, vocab_size=vocab_size, device=device, eps=self.eps
            )
        return cache[key]

    def _get_q_vec(
        self, q_packed: Dict[Tuple[int, ...], torch.Tensor], context: Tuple[int, ...]
    ) -> torch.Tensor:
        h = context
        while True:
            if h in q_packed:
                q = q_packed[h]
                break
            if len(h) == 0:
                q = q_packed[()]
                break
            h = h[1:]
        if self.prior_temperature == 1.0:
            return q
        q_temp = torch.pow(q, 1.0 / max(self.prior_temperature, self.eps))
        return q_temp / q_temp.sum().clamp_min(self.eps)

    def _get_q_vec_coarse(
        self, q_packed: Dict[Tuple[int, ...], torch.Tensor], context: Tuple[int, ...]
    ) -> torch.Tensor:
        h = context
        while True:
            if h in q_packed:
                q = q_packed[h]
                break
            if len(h) == 0:
                q = q_packed[()]
                break
            h = h[1:]
        if self.coarse_prior_temperature == 1.0:
            return q
        q_temp = torch.pow(q, 1.0 / max(self.coarse_prior_temperature, self.eps))
        return q_temp / q_temp.sum().clamp_min(self.eps)

    def _warn_missing(self, key: str, message: str):
        if key in self._warned_missing_prior:
            return
        print(message)
        self._warned_missing_prior.add(key)

    # ----------------------- loss terms ----------------------- #
    def _sequence_prior_loss(
        self,
        suffix: torch.Tensor,
        termination_token_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
        base_n: int,
        include_eos: bool,
        normalize_by_len: bool,
        max_pos: Optional[int],
    ) -> Optional[torch.Tensor]:
        """
        Compute the sequence prior loss.
        Make sure the token choice is not far from the prior.
        More suitable for small vocabulary sizes.
        """
        if q_packed is None:
            return None

        B, L = suffix.shape
        total = suffix.new_zeros(())
        denom = suffix.new_zeros(())

        for b in range(B):
            row = suffix[b]
            T_eff = L
            if not include_eos:
                pos = (row == termination_token_id).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    T_eff = int(pos[0].item())
            if max_pos is not None:
                T_eff = min(T_eff, int(max_pos))
            if T_eff <= 0:
                continue

            logp = 0.0
            for t in range(T_eff):
                ctx = self._ctx_tuple(row, t, n=base_n, bos=self.bos_id)
                q_vec = self._get_q_vec(q_packed, ctx)
                w = int(row[t].item())
                logp += float(torch.log(q_vec[w].clamp_min(self.eps)).item())

            if include_eos and (T_eff < L):
                ctx = self._ctx_tuple(row, T_eff, n=base_n, bos=self.bos_id)
                q_vec = self._get_q_vec(q_packed, ctx)
                logp += float(torch.log(q_vec[termination_token_id].clamp_min(self.eps)).item())

            if normalize_by_len:
                total = total + (-logp / max(1, T_eff))
            else:
                total = total + (-logp)
            denom = denom + 1.0

        return total / denom.clamp_min(1.0)

    def _kl_regularizer(
        self,
        suffix: torch.Tensor,
        policy_logits: torch.Tensor,
        termination_token_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if q_packed is None:
            return None
        B, L = suffix.shape
        logits = policy_logits[:, :L, :]
        if self.kl_temperature != 1.0:
            logits = logits / float(self.kl_temperature)
        ptheta = logits.log_softmax(dim=-1).exp()

        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        alive = eos_cum == 0
        if self.kl_max_pos is not None:
            pos_mask = torch.arange(L, device=suffix.device).unsqueeze(0) < int(self.kl_max_pos)
            alive = alive & pos_mask

        kl_acc = torch.zeros((), device=suffix.device, dtype=ptheta.dtype)
        kl_cnt = torch.zeros((), device=suffix.device, dtype=ptheta.dtype)

        for b in range(B):
            row = suffix[b]
            for t in range(L):
                if not bool(alive[b, t].item()):
                    continue
                ctx = self._ctx_tuple(row, t, n=self.base_n, bos=self.bos_id)
                q_vec = self._get_q_vec(q_packed, ctx).to(ptheta.dtype)
                p = ptheta[b, t]
                kl = (p * (p.add(self.eps).log() - q_vec.add(self.eps).log())).sum()
                kl_acc = kl_acc + kl
                kl_cnt = kl_cnt + 1.0

        return kl_acc / kl_cnt.clamp_min(1.0)

    def _coarse_sequence_loss(
        self,
        coarse_suffix: torch.Tensor,
        termination_token_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
        include_eos: bool,
        normalize_by_len: bool,
        max_pos: Optional[int],
    ) -> Optional[torch.Tensor]:
        if q_packed is None:
            return None

        B, L = coarse_suffix.shape
        total = coarse_suffix.new_zeros(())
        denom = coarse_suffix.new_zeros(())

        for b in range(B):
            row = coarse_suffix[b]
            T_eff = L
            if not include_eos:
                pos = (row == termination_token_id).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    T_eff = int(pos[0].item())
            if max_pos is not None:
                T_eff = min(T_eff, int(max_pos))
            if T_eff <= 0:
                continue

            logp = 0.0
            for t in range(T_eff):
                ctx = self._ctx_tuple(row, t, n=self.coarse_ngram_n, bos=self.bos_id)
                q_vec = self._get_q_vec_coarse(q_packed, ctx)
                w = int(row[t].item())
                logp += float(torch.log(q_vec[w].clamp_min(self.eps)).item())
            if include_eos and (T_eff < L):
                ctx = self._ctx_tuple(row, T_eff, n=self.coarse_ngram_n, bos=self.bos_id)
                q_vec = self._get_q_vec_coarse(q_packed, ctx)
                logp += float(torch.log(q_vec[termination_token_id].clamp_min(self.eps)).item())

            if normalize_by_len:
                total = total + (-logp / max(1, T_eff))
            else:
                total = total + (-logp)
            denom = denom + 1.0

        return total / denom.clamp_min(1.0)

    def _coarse_kl_regularizer(
        self,
        coarse_suffix: torch.Tensor,
        policy_logits: torch.Tensor,
        termination_coarse_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
        projection: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if q_packed is None:
            return None

        B, L = coarse_suffix.shape
        logits = policy_logits[:, :L, :]
        if self.kl_temperature != 1.0:
            logits = logits / float(self.kl_temperature)
        ptheta = logits.log_softmax(dim=-1).exp()
        coarse_probs = torch.matmul(ptheta, projection.to(ptheta.dtype))  # [B, L, C]

        eos_cum = (coarse_suffix == termination_coarse_id).cumsum(dim=1)
        alive = eos_cum == 0
        if self.coarse_kl_max_pos is not None:
            pos_mask = torch.arange(L, device=coarse_suffix.device).unsqueeze(0) < int(
                self.coarse_kl_max_pos
            )
            alive = alive & pos_mask

        kl_acc = torch.zeros((), device=coarse_suffix.device, dtype=coarse_probs.dtype)
        kl_cnt = torch.zeros((), device=coarse_suffix.device, dtype=coarse_probs.dtype)

        for b in range(B):
            row = coarse_suffix[b]
            for t in range(L):
                if not bool(alive[b, t].item()):
                    continue
                ctx = self._ctx_tuple(row, t, n=self.coarse_ngram_n, bos=self.bos_id)
                q_vec = self._get_q_vec_coarse(q_packed, ctx).to(coarse_probs.dtype)
                p = coarse_probs[b, t]
                kl = (p * (p.add(self.eps).log() - q_vec.add(self.eps).log())).sum()
                kl_acc = kl_acc + kl
                kl_cnt = kl_cnt + 1.0

        return kl_acc / kl_cnt.clamp_min(1.0)

    def _batch_distribution_kl(
        self,
        suffix: torch.Tensor,
        termination_token_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
        max_pos: Optional[int],
    ) -> Optional[torch.Tensor]:
        if q_packed is None:
            return None
        B, L = suffix.shape
        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        kl_acc = suffix.new_zeros((), dtype=torch.float32, device=suffix.device)
        kl_cnt = suffix.new_zeros((), dtype=torch.float32, device=suffix.device)

        limit = L if max_pos is None else min(L, int(max_pos))
        for t in range(limit):
            alive = eos_cum[:, t] == 0
            if not bool(alive.any()):
                continue
            active_tokens = suffix[alive, t]
            vocab_size = q_packed.get((), next(iter(q_packed.values()))).numel()
            p_counts = torch.bincount(active_tokens, minlength=vocab_size).to(torch.float32)
            p = p_counts / p_counts.sum().clamp_min(self.eps)

            q_list = []
            idx_alive = active_tokens.shape[0]
            for i in range(idx_alive):
                row = suffix[alive][i]
                ctx = self._ctx_tuple(row, t, n=self.base_n, bos=self.bos_id)
                q_vec = self._get_q_vec(q_packed, ctx).to(torch.float32)
                q_list.append(q_vec)
            q_batch = torch.stack(q_list, dim=0).mean(dim=0)
            kl = (p * (p.add(self.eps).log() - q_batch.add(self.eps).log())).sum()
            kl_acc = kl_acc + kl
            kl_cnt = kl_cnt + 1.0
        return kl_acc / kl_cnt.clamp_min(1.0)

    def _coarse_batch_distribution_kl(
        self,
        coarse_suffix: torch.Tensor,
        termination_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
        max_pos: Optional[int],
    ) -> Optional[torch.Tensor]:
        if q_packed is None:
            return None
        B, L = coarse_suffix.shape
        eos_cum = (coarse_suffix == termination_id).cumsum(dim=1)
        kl_acc = coarse_suffix.new_zeros((), dtype=torch.float32, device=coarse_suffix.device)
        kl_cnt = coarse_suffix.new_zeros((), dtype=torch.float32, device=coarse_suffix.device)
        limit = L if max_pos is None else min(L, int(max_pos))
        for t in range(limit):
            alive = eos_cum[:, t] == 0
            if not bool(alive.any()):
                continue
            active_tokens = coarse_suffix[alive, t]
            vocab_size = q_packed.get((), next(iter(q_packed.values()))).numel()
            p_counts = torch.bincount(active_tokens, minlength=vocab_size).to(torch.float32)
            p = p_counts / p_counts.sum().clamp_min(self.eps)

            q_list = []
            idx_alive = active_tokens.shape[0]
            for i in range(idx_alive):
                row = coarse_suffix[alive][i]
                ctx = self._ctx_tuple(row, t, n=self.coarse_ngram_n, bos=self.bos_id)
                q_vec = self._get_q_vec_coarse(q_packed, ctx).to(torch.float32)
                q_list.append(q_vec)
            q_batch = torch.stack(q_list, dim=0).mean(dim=0)
            kl = (p * (p.add(self.eps).log() - q_batch.add(self.eps).log())).sum()
            kl_acc = kl_acc + kl
            kl_cnt = kl_cnt + 1.0
        return kl_acc / kl_cnt.clamp_min(1.0)

    def _entropy_bonus(
        self,
        policy_logits: torch.Tensor,
        termination_token_id: int,
        suffix: torch.Tensor,
        max_pos: Optional[int],
    ) -> Optional[torch.Tensor]:
        if policy_logits is None and self.diversity_use_logits:
            return None
        B, L = suffix.shape
        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        alive = eos_cum == 0
        if max_pos is not None:
            pos_mask = torch.arange(L, device=suffix.device).unsqueeze(0) < int(max_pos)
            alive = alive & pos_mask

        if self.diversity_use_logits:
            logits = policy_logits[:, :L, :]
            ptheta = logits.log_softmax(dim=-1).exp()
            ent_acc = torch.zeros((), device=suffix.device, dtype=ptheta.dtype)
            ent_cnt = torch.zeros((), device=suffix.device, dtype=ptheta.dtype)
            for t in range(L):
                mask_t = alive[:, t]
                if not bool(mask_t.any()):
                    continue
                p_bar = ptheta[mask_t, t].mean(dim=0)
                ent = -(p_bar * p_bar.add(self.eps).log()).sum()
                ent_acc = ent_acc + ent
                ent_cnt = ent_cnt + 1.0
        else:
            # 用采样 token 的 batch 分布熵作为多样性指标
            ent_acc = torch.zeros((), device=suffix.device, dtype=torch.float32)
            ent_cnt = torch.zeros((), device=suffix.device, dtype=torch.float32)
            vocab_size = int(suffix.max().item()) + 1
            for t in range(L):
                mask_t = alive[:, t]
                if not bool(mask_t.any()):
                    continue
                tokens = suffix[mask_t, t]
                counts = torch.bincount(tokens, minlength=vocab_size).to(torch.float32)
                p = counts / counts.sum().clamp_min(self.eps)
                ent = -(p * p.add(self.eps).log()).sum()
                ent_acc = ent_acc + ent
                ent_cnt = ent_cnt + 1.0
        if ent_cnt.item() == 0:
            return None
        return -(ent_acc / ent_cnt.clamp_min(1.0))

    def _coarse_reward_bonus(
        self,
        coarse_suffix: torch.Tensor,
        termination_coarse_id: int,
        q_packed: Optional[Dict[Tuple[int, ...], torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """基于 coarse n-gram 先验的逐 token log 概率，用于奖励塑形。"""
        if q_packed is None:
            return None
        B, L = coarse_suffix.shape
        bonus = torch.zeros(B, L, device=coarse_suffix.device, dtype=torch.float32)
        eos_cum = (coarse_suffix == termination_coarse_id).cumsum(dim=1)
        alive = eos_cum == 0
        for b in range(B):
            row = coarse_suffix[b]
            for t in range(L):
                if not bool(alive[b, t].item()):
                    continue
                ctx = self._ctx_tuple(row, t, n=self.coarse_ngram_n, bos=self.bos_id)
                q_vec = self._get_q_vec_coarse(q_packed, ctx).to(torch.float32)
                w = int(row[t].item())
                bonus[b, t] = torch.log(q_vec[w].clamp_min(self.eps))
        return bonus

    # ----------------------- main forward ----------------------- #
    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        policy_logits: Optional[torch.Tensor] = None,
        weight_overrides: Optional[dict] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        B, L = log_pf.shape
        suffix = generated_text[:, prompt_len : prompt_len + L]
        device = log_pf.device
        vocab_size = self._infer_vocab_size(policy_logits, generated_text)

        q_packed = self._get_q_packed(self.q_mcmc_source, self._q_cache, device, vocab_size)
        logs = {}

        # Resolve weights (static, override, or scheduler)
        seq_w = self._resolve_weight("seq_weight", self.seq_weight, weight_overrides)
        kl_w = self._resolve_weight("kl_weight", self.kl_weight, weight_overrides)
        batch_kl_w = self._resolve_weight(
            "batch_kl_weight", self.batch_kl_weight, weight_overrides
        )
        coarse_seq_w = self._resolve_weight(
            "coarse_seq_weight", self.coarse_seq_weight, weight_overrides
        )
        coarse_kl_w = self._resolve_weight(
            "coarse_kl_weight", self.coarse_kl_weight, weight_overrides
        )
        coarse_batch_kl_w = self._resolve_weight(
            "coarse_batch_kl_weight",
            self.coarse_batch_kl_weight,
            weight_overrides,
        )
        diversity_w = self._resolve_weight(
            "diversity_weight", self.diversity_weight, weight_overrides
        )
        coarse_reward_w = self._resolve_weight(
            "coarse_reward_weight", self.coarse_reward_weight, weight_overrides
        )

        # --------- 可选：coarse 先验奖励塑形（修改 log_r 家族后再算 SubTB） ---------
        if self.coarse_indexer is not None and coarse_reward_w > 0.0:
            self.coarse_indexer.ensure_vocab_size(vocab_size)
            coarse_suffix = self.coarse_indexer.coarse_tokens(suffix)
            coarse_term_id = int(
                self.coarse_indexer.coarse_tokens(
                    torch.tensor([termination_token_id], device=device)
                )[0].item()
            )
            q_coarse_pack = self._get_q_packed(
                self.q_mcmc_coarse_source,
                self._q_coarse_cache,
                device,
                self.coarse_indexer.coarse_vocab_size,
            )
            coarse_bonus = self._coarse_reward_bonus(
                coarse_suffix=coarse_suffix,
                termination_coarse_id=coarse_term_id,
                q_packed=q_coarse_pack,
            )
            if coarse_bonus is not None:
                w = float(coarse_reward_w)
                log_r = log_r + w * coarse_bonus.to(log_r.dtype)
                if "log_r_reference" in kwargs and kwargs["log_r_reference"] is not None:
                    kwargs["log_r_reference"] = kwargs["log_r_reference"] + w * coarse_bonus.to(
                        kwargs["log_r_reference"].dtype
                    )
                if "log_r_target" in kwargs and kwargs["log_r_target"] is not None:
                    kwargs["log_r_target"] = kwargs["log_r_target"] + w * coarse_bonus.to(
                        kwargs["log_r_target"].dtype
                    )

        # --------- 主体 SubTB（使用可能被奖励塑形后的 log_r） ---------
        subtb_out = self.base_subtb(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=termination_token_id,
            prompt_len=prompt_len,
        )
        subtb_loss = subtb_out["loss"] if isinstance(subtb_out, dict) else subtb_out
        loss_total = subtb_loss
        logs["loss_subtb"] = subtb_loss.detach()

        if seq_w > 0.0:
            seq_loss = self._sequence_prior_loss(
                suffix,
                termination_token_id,
                q_packed,
                base_n=self.base_n,
                include_eos=self.seq_include_eos,
                normalize_by_len=self.seq_normalize_by_len,
                max_pos=self.seq_max_pos,
            )
            if seq_loss is None:
                self._warn_missing(
                    "seq", "[SubTBMCMCPriorLoss] Missing q_MCMC for sequence prior; skipping."
                )
            else:
                loss_total = loss_total + seq_w * seq_loss
                logs["loss_seq_prior"] = seq_loss.detach()

        if kl_w > 0.0:
            if policy_logits is None:
                self._warn_missing(
                    "kl", "[SubTBMCMCPriorLoss] policy_logits is None; KL term skipped."
                )
            else:
                kl_loss = self._kl_regularizer(
                    suffix,
                    policy_logits,
                    termination_token_id=termination_token_id,
                    q_packed=q_packed,
                )
                if kl_loss is None:
                    self._warn_missing(
                        "kl_q", "[SubTBMCMCPriorLoss] Missing q_MCMC for KL prior; skipping."
                    )
                else:
                    loss_total = loss_total + kl_w * kl_loss
                    logs["loss_logits_kl"] = kl_loss.detach()

        if batch_kl_w > 0.0:
            batch_kl = self._batch_distribution_kl(
                suffix,
                termination_token_id=termination_token_id,
                q_packed=q_packed,
                max_pos=self.batch_kl_max_pos,
            )
            if batch_kl is None:
                self._warn_missing(
                    "batch_kl",
                    "[SubTBMCMCPriorLoss] Missing q_MCMC for batch KL; skipping.",
                )
            else:
                loss_total = loss_total + batch_kl_w * batch_kl
                logs["loss_batch_kl"] = batch_kl.detach()

        # --------- coarse-grained terms --------- #
        if self.coarse_indexer is not None and (
            coarse_seq_w > 0.0 or coarse_kl_w > 0.0 or coarse_batch_kl_w > 0.0
        ):
            self.coarse_indexer.ensure_vocab_size(vocab_size)

            coarse_suffix = self.coarse_indexer.coarse_tokens(suffix)
            projection = self.coarse_indexer.projection(device)
            coarse_vocab = self.coarse_indexer.coarse_vocab_size
            coarse_term_id = int(
                self.coarse_indexer.coarse_tokens(
                    torch.tensor([termination_token_id], device=device)
                )[0].item()
            )
            q_coarse = self._get_q_packed(
                self.q_mcmc_coarse_source, self._q_coarse_cache, device, coarse_vocab
            )

            if coarse_seq_w > 0.0:
                seq_loss_c = self._coarse_sequence_loss(
                    coarse_suffix,
                    termination_token_id=coarse_term_id,
                    q_packed=q_coarse,
                    include_eos=self.coarse_seq_include_eos,
                    normalize_by_len=self.coarse_seq_normalize_by_len,
                    max_pos=self.coarse_seq_max_pos,
                )
                if seq_loss_c is None:
                    self._warn_missing(
                        "coarse_seq",
                        "[SubTBMCMCPriorLoss] Missing coarse q_MCMC for sequence prior; skipping.",
                    )
                else:
                    loss_total = loss_total + coarse_seq_w * seq_loss_c
                    logs["loss_coarse_seq_prior"] = seq_loss_c.detach()

            if coarse_kl_w > 0.0:
                if policy_logits is None:
                    self._warn_missing(
                        "coarse_kl_logits",
                        "[SubTBMCMCPriorLoss] policy_logits is None; coarse KL term skipped.",
                    )
                else:
                    kl_c = self._coarse_kl_regularizer(
                        coarse_suffix,
                        policy_logits,
                        termination_coarse_id=coarse_term_id,
                        q_packed=q_coarse,
                        projection=projection,
                    )
                    if kl_c is None:
                        self._warn_missing(
                            "coarse_kl_q",
                            "[SubTBMCMCPriorLoss] Missing coarse q_MCMC for KL prior; skipping.",
                        )
                    else:
                        loss_total = loss_total + coarse_kl_w * kl_c
                        logs["loss_coarse_logits_kl"] = kl_c.detach()
            if coarse_batch_kl_w > 0.0:
                batch_kl_c = self._coarse_batch_distribution_kl(
                    coarse_suffix,
                    termination_id=coarse_term_id,
                    q_packed=q_coarse,
                    max_pos=self.coarse_batch_kl_max_pos,
                )
                if batch_kl_c is None:
                    self._warn_missing(
                        "coarse_batch_kl",
                        "[SubTBMCMCPriorLoss] Missing coarse q_MCMC for batch KL; skipping.",
                    )
                else:
                    loss_total = loss_total + coarse_batch_kl_w * batch_kl_c
                    logs["loss_coarse_batch_kl"] = batch_kl_c.detach()
        elif coarse_seq_w > 0.0 or coarse_kl_w > 0.0 or coarse_batch_kl_w > 0.0:
            self._warn_missing(
                "coarse_map",
                "[SubTBMCMCPriorLoss] coarse_token_groups/token_to_coarse not provided; coarse terms skipped.",
            )

        if diversity_w > 0.0:
            div_loss = self._entropy_bonus(
                policy_logits=policy_logits,
                termination_token_id=termination_token_id,
                suffix=suffix,
                max_pos=self.diversity_max_pos,
            )
            if div_loss is not None:
                loss_total = loss_total + diversity_w * div_loss
                logs["loss_diversity"] = div_loss.detach()

        logs["loss"] = loss_total
        return logs


class SubTBBatchDiversityPriorLoss(SubTBBatchDiversityLoss):
    """
    Batch 多样性 + MCMC 先验（序列先验 + batch 位置先验 KL）。
    """

    def __init__(
        self,
        q_mcmc_path: Optional[str] = None,
        q_mcmc: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = None,
        base_n: int = 2,
        bos_id: int = -1,
        seq_prior_weight: float = 0.0,
        batch_prior_weight: float = 0.0,
        batch_prior_max_pos: Optional[int] = None,
        prior_temperature: float = 1.0,
        subtb_lambda: float = 1.0,
        balance: float = 0.0,
        eps: float = 1e-8,
        seq_diversity_weight: float = 0.0,
        pos_diversity_weight: float = 0.0,
        pos_max: Optional[int] = None,
        use_logits: bool = False,
    ):
        super().__init__(
            subtb_lambda=subtb_lambda,
            balance=balance,
            eps=eps,
            seq_diversity_weight=seq_diversity_weight,
            pos_diversity_weight=pos_diversity_weight,
            pos_max=pos_max,
            use_logits=use_logits,
        )
        self.q_mcmc_source = q_mcmc or self._load_prior(q_mcmc_path)
        self.base_n = base_n
        self.bos_id = bos_id
        self.seq_prior_weight = seq_prior_weight
        self.batch_prior_weight = batch_prior_weight
        self.batch_prior_max_pos = batch_prior_max_pos
        self.prior_temperature = prior_temperature
        self._q_cache: dict[tuple[torch.device, int], Dict[Tuple[int, ...], torch.Tensor]] = {}

    @staticmethod
    def _load_prior(path: Optional[str]):
        if path is None:
            return None
        try:
            return load_q_mcmc(path)
        except Exception:
            return None

    def _get_q_packed(
        self,
        source: Optional[Dict[Tuple[int, ...], Dict[int, float]]],
        cache: dict[tuple[torch.device, int], Dict[Tuple[int, ...], torch.Tensor]],
        device: torch.device,
        vocab_size: int,
    ) -> Optional[Dict[Tuple[int, ...], torch.Tensor]]:
        if source is None:
            return None
        key = (device, vocab_size)
        if key not in cache:
            cache[key] = pack_q_mcmc_to_device(
                source, vocab_size=vocab_size, device=device, eps=1e-8
            )
        return cache[key]

    def _get_q_vec(
        self, q_packed: Dict[Tuple[int, ...], torch.Tensor], context: Tuple[int, ...]
    ) -> torch.Tensor:
        h = context
        while True:
            if h in q_packed:
                q = q_packed[h]
                break
            if len(h) == 0:
                q = q_packed[()]
                break
            h = h[1:]
        if self.prior_temperature == 1.0:
            return q
        q_temp = torch.pow(q, 1.0 / max(self.prior_temperature, 1e-8))
        return q_temp / q_temp.sum().clamp_min(1e-8)

    @staticmethod
    def _ctx_tuple(row: torch.Tensor, t: int, n: int, bos: int) -> Tuple[int, ...]:
        k = n - 1
        if k <= 0:
            return ()
        ctx = []
        for kk in range(k, 0, -1):
            idx = t - kk
            ctx.append(bos if idx < 0 else int(row[idx].item()))
        return tuple(ctx)

    def _sequence_prior(self, suffix: torch.Tensor, termination_token_id: int, q_packed):
        if q_packed is None:
            return None
        B, L = suffix.shape
        total = suffix.new_zeros(())
        denom = suffix.new_zeros(())
        for b in range(B):
            row = suffix[b]
            T_eff = L
            pos = (row == termination_token_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                T_eff = int(pos[0].item())
            logp = 0.0
            for t in range(T_eff):
                ctx = self._ctx_tuple(row, t, n=self.base_n, bos=self.bos_id)
                q_vec = self._get_q_vec(q_packed, ctx)
                w = int(row[t].item())
                logp += float(torch.log(q_vec[w].clamp_min(1e-8)).item())
            total = total + (-logp / max(1, T_eff))
            denom = denom + 1.0
        return total / denom.clamp_min(1.0)

    def _batch_prior_kl(self, suffix: torch.Tensor, termination_token_id: int, q_packed):
        if q_packed is None:
            return None
        B, L = suffix.shape
        eos_cum = (suffix == termination_token_id).cumsum(dim=1)
        kl_acc = torch.zeros((), device=suffix.device, dtype=torch.float32)
        kl_cnt = torch.zeros((), device=suffix.device, dtype=torch.float32)
        limit = L if self.batch_prior_max_pos is None else min(L, int(self.batch_prior_max_pos))
        vocab_size = next(iter(q_packed.values())).numel()
        for t in range(limit):
            mask = eos_cum[:, t] == 0
            if not bool(mask.any()):
                continue
            toks = suffix[mask, t]
            p_counts = torch.bincount(toks, minlength=vocab_size).to(torch.float32)
            p = p_counts / p_counts.sum().clamp_min(1e-8)
            q_list = []
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            for i in idxs:
                row = suffix[i]
                ctx = self._ctx_tuple(row, t, n=self.base_n, bos=self.bos_id)
                q_vec = self._get_q_vec(q_packed, ctx).to(torch.float32)
                q_list.append(q_vec)
            q_avg = torch.stack(q_list, dim=0).mean(dim=0)
            kl = (p * (p.add(1e-8).log() - q_avg.add(1e-8).log())).sum()
            kl_acc = kl_acc + kl
            kl_cnt = kl_cnt + 1.0
        if kl_cnt.item() == 0:
            return None
        return kl_acc / kl_cnt.clamp_min(1.0)

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        policy_logits: Optional[torch.Tensor] = None,
        weight_overrides: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        suffix = generated_text[:, prompt_len:]
        device = log_pf.device
        vocab_size = int(
            max(
                int(suffix.max().item()) + 1,
                policy_logits.shape[-1] if policy_logits is not None else 0,
            )
        )
        q_packed = self._get_q_packed(self.q_mcmc_source, self._q_cache, device, vocab_size)

        out = super().forward(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=termination_token_id,
            prompt_len=prompt_len,
            policy_logits=policy_logits,
            weight_overrides=weight_overrides,
            **kwargs,
        )
        loss_total = out["loss"]

        seq_prior_w = self._resolve_weight(
            "seq_prior_weight", self.seq_prior_weight, weight_overrides
        )
        batch_prior_w = self._resolve_weight(
            "batch_prior_weight", self.batch_prior_weight, weight_overrides
        )

        if seq_prior_w > 0.0:
            seq_prior = self._sequence_prior(suffix, termination_token_id, q_packed)
            if seq_prior is not None:
                loss_total = loss_total + seq_prior_w * seq_prior
                out["loss_seq_prior"] = seq_prior.detach()

        if batch_prior_w > 0.0:
            batch_kl = self._batch_prior_kl(suffix, termination_token_id, q_packed)
            if batch_kl is not None:
                loss_total = loss_total + batch_prior_w * batch_kl
                out["loss_batch_prior_kl"] = batch_kl.detach()

        out["loss"] = loss_total
        return out
