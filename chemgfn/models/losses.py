"""
Loss functions for GFlowNet training.

This module provides loss function implementations for training GFlowNets,
including SubTrajectory Balance (SubTB) losses with various enhancements.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GFNLoss",
    "ModifiedSubTBLoss",
    "ModifiedSubTBLossSplitReward",
    "ModifiedSubTBBalanceLoss",
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


class RootAbsorbExtraSubTBLossFixTBLogZv2(GFNLoss):
    """
    L = (1-eta)*L_TB_terminal + eta*L_aux_rooted_absorb_extra(K)

    NEW (k_min semantics you requested):
      - For prefixes with k < k_min: DO NOT count them in AUX at all.
        (i.e., they do not contribute to AUX numerator/denominator; training for those
         prefixes is handled only by TB.)
      - Additionally, if a sample has no eligible AUX prefixes (den_i==0),
        we set eta_i=0 for that sample so its total loss is pure TB (no (1-eta) shrink).

    Alignment (MATCH your existing pipeline):
      - log_pf:    [B, L]   meaningful steps are log_pf[:, :-1] (length L-1)
      - log_pterm: [B, L]
      - log_r:     [B, L]
      - generated_text post-prompt length == L
      - generated_text[:, prompt_len:-1] excludes the forced last EOS token (by max length)
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
        init_logZ: float = 0.0,  # explicit learnable logZ
        k_min: int = 2,  # NEW: prefixes k < k_min are skipped in AUX entirely
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

        self.k_min = int(k_min)

        if not (self.gamma < 1.0):
            raise ValueError(f"gamma must be < 1.0, got gamma={self.gamma}")
        if self.k_min < 1:
            print(f"k_min must be >= 1, got k_min={self.k_min}, setting to 1 automatically")
            self.k_min = 1

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
        log_pf: [B, L], use steps = log_pf[:, :-1] (length L-1).
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
        Shapes:
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
        delta uses steps log_pf[:, :-1], length L-1.
        returns C: [B, L] with C[:,0]=0
        """
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
        )  # [B, L-1]
        return torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)  # [B, L]

    @staticmethod
    def _suffix_future_max(u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        dtype = u.dtype
        u_mask = torch.where(valid, u, torch.full_like(u, 0))
        rev = torch.flip(u_mask, dims=[1])
        rev_max = torch.cummax(rev, dim=1).values
        return torch.flip(rev_max, dims=[1])

    @staticmethod
    def _suffix_future_soft(
        u: torch.Tensor, valid: torch.Tensor, beta: float, rho: float
    ) -> torch.Tensor:
        """
        Backward softmax over suffix with step penalty rho (distance discount).
        Returns u_soft[t] = (1/beta) log sum_{j>=t} exp(beta*u[j] - beta*rho*(j-t))
        """
        B, L = u.shape
        u_mask = torch.where(valid, u, torch.full_like(u, -torch.inf))
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
        log_pf: torch.Tensor,  # [B, L] (steps are log_pf[:, :-1])
        log_r: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        ref_log_pf: Optional[torch.Tensor] = None,  # [B, L-1]
        ref_log_pterm: Optional[torch.Tensor] = None,  # [B, L]
        ref_scale: float = 1.0,
        max_prefix_len: Optional[int] = None,  # K in [1, L-1]; None => h=tau
        logZ: Optional[torch.Tensor] = None,  # scalar or [B]
        k_min: Optional[int] = None,  # override per call
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

        eos_or_after = (generated_text[:, prompt_len:-1] == int(termination_token_id)).cumsum(
            dim=-1
        ) >= 1  # [B, L-1]
        valid_end = ~eos_or_after  # [B, L-1] True before first EOS
        tau = valid_end.sum(dim=1).clamp(0, L - 1)  # [B] in [0, L-1]

        # Terminal TB
        if logZ is None:
            logZ_b = self.logZ.to(device=log_pf.device, dtype=log_pf.dtype).expand(B)
        else:
            z = logZ.to(device=log_pf.device, dtype=log_pf.dtype)
            logZ_b = z.expand(B) if z.ndim == 0 else z
            assert logZ_b.shape == (B,), f"logZ must be scalar or [B], got {tuple(z.shape)}"

        sum_log_pf = self._sum_log_pf_upto_tau(log_pf, tau)  # [B]
        log_pterm_tau = self._gather_by_index(log_pterm, tau)  # [B]
        log_r_tau = self._gather_by_index(log_r, tau)  # [B]

        tb_res = logZ_b + sum_log_pf + log_pterm_tau - log_r_tau
        loss_tb_i = tb_res**2  # [B]
        loss_tb = loss_tb_i.mean()

        # RapTB
        loss_aux = torch.zeros((), device=log_pf.device, dtype=log_pf.dtype)
        aux_active_rate = torch.zeros((), device=log_pf.device, dtype=log_pf.dtype)

        if self.aux_weight > 0.0:
            # K clamp
            if max_prefix_len is None:
                K = None
            else:
                K = int(max_prefix_len)
                K = max(1, min(K, L - 1))

            # k_min clamp
            kmin = self.k_min if k_min is None else int(k_min)
            kmin = max(1, min(kmin, L - 1))

            # aux uses detached pterm if requested
            lp_aux = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm
            C_aux = self._delta_cumsum(log_pf, log_r, lp_aux)  # [B, L]
            Ck = C_aux[:, 1:]  # [B, L-1], k=1..L-1

            # k indices: 1..L-1
            k_idx = torch.arange(1, L, device=log_pf.device).view(1, L - 1)  # [1, L-1]

            # horizon h: min(tau, K) if K given, else tau
            if K is None:
                h = tau
            else:
                h = torch.minimum(tau, tau.new_full(tau.shape, K))

            within_tau = k_idx <= tau.view(B, 1)
            within_h = k_idx <= h.view(B, 1)

            # base eligibility: before EOS and within horizon
            m_base = within_tau & within_h & valid_end  # bool [B, L-1]

            # skip k < k_min from AUX entirely (not even in den)
            after_kmin = k_idx >= kmin  # bool [1, L-1]
            m = (m_base & after_kmin).to(log_pf.dtype)  # [B, L-1] float mask

            # w_k = 1[k>=kmin] * lambda^(k-kmin)
            w_exp = (k_idx - kmin).clamp_min(0).to(log_pf.dtype)  # [1, L-1]
            w = (self.subtb_lambda**w_exp) * after_kmin.to(log_pf.dtype)  # [1, L-1]

            # extra u = log_r - ref_scale * ref_logP  (or fallback u=log_r)
            if (ref_log_pf is not None) and (ref_log_pterm is not None):
                assert ref_log_pf.shape == (
                    B,
                    L - 1,
                ), f"ref_log_pf {ref_log_pf.shape} vs {(B,L-1)}"
                assert ref_log_pterm.shape == (
                    B,
                    L,
                ), f"ref_log_pterm {ref_log_pterm.shape} vs {(B,L)}"
                ref_logP = self._reconstruct_ref_logP(ref_log_pf, ref_log_pterm)  # [B, L]
                u = log_r - float(ref_scale) * ref_logP
            else:
                u = log_r

            u_k = u[:, 1:]  # [B, L-1]
            no_reward = u_k.abs() <= self.extra_absorb_eps

            # tau < K => no absorption correction at all
            if K is None:
                can_absorb_seq = torch.ones((B, 1), device=log_pf.device, dtype=torch.bool)
            else:
                can_absorb_seq = tau.view(B, 1) >= K

            # absorb only for k < K when K is set, else within_h
            if K is None:
                before_horizon = within_h
            else:
                before_horizon = k_idx < K

            # suffix targets (on states 0..L-1)
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

            u_target = torch.where(valid_future, u_target, torch.zeros_like(u_target))
            u_tk = u_target[:, 1:]

            # alpha_k = gamma^(h - k)
            exp = (h.view(B, 1) - k_idx).clamp_min(0).to(log_pf.dtype)
            alpha = (self.gamma**exp) * within_h.to(log_pf.dtype)

            # apply absorption only where m allows (already excludes k<kmin)
            apply_absorb = (
                (m_base & after_kmin) & no_reward & before_horizon & can_absorb_seq
            )  # bool [B,L-1]
            alpha_eff = alpha * apply_absorb.to(log_pf.dtype)

            # correction on Ck
            corr = alpha_eff * (u_k - u_tk)
            Ck_abs = Ck + corr

            # -------- per-sample AUX (so samples with no eligible k are TB-only) --------
            mw_mask = m * w  # [B, L-1]
            num_i = ((Ck_abs**2) * mw_mask).sum(dim=1)  # [B]
            den_i = mw_mask.sum(dim=1)  # [B]

            active_i = den_i > self.eps
            aux_active_rate = active_i.to(log_pf.dtype).mean()

            loss_aux_i = torch.zeros_like(num_i)
            loss_aux_i[active_i] = num_i[active_i] / den_i[active_i].clamp_min(self.eps)
            loss_aux = loss_aux_i.mean()

            # per-sample eta: if no aux terms => eta_i=0 (pure TB)
            eta = float(self.aux_weight)
            eta_i = eta * active_i.to(log_pf.dtype)

            loss_i = loss_tb_i + eta_i * loss_aux_i
            loss = loss_i.mean()
        else:
            loss = loss_tb

        return {
            "loss": loss,
            "loss_tb": loss_tb.detach(),
            "loss_aux": loss_aux.detach(),
            "aux_active_rate": aux_active_rate.detach(),
            "logZ": self.logZ.detach(),
        }


class RootAbsorbExtraSubTBLossFixTBLogZv3(GFNLoss):
    """
    Terminal TB + Rooted Absorb Extra SubTB AUX (v3)

    v3 semantics:
      - Prefixes with k < k_min are SKIPPED in AUX entirely (not in numerator/denominator).
      - If a sample has no eligible AUX prefixes => eta_i = 0 (TB-only for that sample).

    Absorb semantics (as you clarified):
      - Absorb correction is STRICTLY restricted to absorb horizon:
            * only k < K (when K is set),
            * and only if trajectory reaches K (tau >= K) when K is set (hard gate for collapse),
        and THEN, within that horizon, apply absorb at prefix k IFF the specific position k
        has no extra reward: |u_k| <= extra_absorb_eps.

    Mixing (default normalized; affects gradients):
      loss_i = (TB_i + eta_i * AUX_i) / (1 + eta_i)

    Alignment:
      - log_pf:    [B, L]   meaningful steps are log_pf[:, :-1] (length L-1)
      - log_pterm: [B, L]
      - log_r:     [B, L]
      - generated_text post-prompt length == L
      - generated_text[:, prompt_len:-1] excludes the forced last EOS token (by max length)
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
        k_window: int = 4,  # NEW: window size for AUX/absorb (fix kmin attractor)
        normalize_mix: bool = True,  # default ON (gradient-level normalization)
        absorb_requires_tau_ge_K: bool = True,  # default ON (hard gate to mitigate prefix collapse)
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

        self.k_min = int(k_min)
        self.k_window = int(k_window)
        self.normalize_mix = bool(normalize_mix)
        self.absorb_requires_tau_ge_K = bool(absorb_requires_tau_ge_K)

        if not (self.gamma < 1.0):
            raise ValueError(f"gamma must be < 1.0, got gamma={self.gamma}")
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got k_min={self.k_min}")

        self.logZ = torch.nn.Parameter(torch.tensor([float(init_logZ)], dtype=torch.float32))

    # ----------------- helpers -----------------

    @staticmethod
    def _gather_by_index(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x.gather(1, idx.view(-1, 1)).squeeze(1)

    @staticmethod
    def _sum_log_pf_upto_tau(log_pf: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        sum_{t < tau} log_pf[t], meaningful steps are log_pf[:, :-1] (length L-1).
        tau: [B] in [0, L-1]
        """
        B, L = log_pf.shape
        steps = log_pf[:, :-1]  # [B, L-1]
        if steps.shape[1] == 0:
            return torch.zeros((B,), device=log_pf.device, dtype=log_pf.dtype)

        pf_cum = steps.cumsum(dim=1)  # [B, L-1]
        idx = (tau - 1).clamp(min=0, max=steps.shape[1] - 1)
        s = pf_cum.gather(1, idx.view(B, 1)).squeeze(1)
        s = s * (tau > 0).to(s.dtype)
        return s

    @staticmethod
    def _reconstruct_ref_logP(
        ref_log_pf: torch.Tensor, ref_log_pterm: torch.Tensor
    ) -> torch.Tensor:
        """
        ref_logP[0] = ref_log_pterm[0]
        ref_logP[k] = ref_log_pterm[k] + sum_{t<k} ref_log_pf[t]
        Shapes:
          ref_log_pf:   [B, L-1]
          ref_log_pterm:[B, L]
        """
        B, T = ref_log_pf.shape
        B2, L = ref_log_pterm.shape
        if not (B == B2 and L == T + 1):
            raise ValueError(
                f"shape mismatch: ref_log_pf {ref_log_pf.shape}, ref_log_pterm {ref_log_pterm.shape}"
            )
        prefix = ref_log_pf.cumsum(dim=1)  # [B, L-1]
        ref_logP = ref_log_pterm.clone()  # [B, L]
        ref_logP[:, 1:] = ref_logP[:, 1:] + prefix
        return ref_logP

    def _delta_cumsum(
        self, log_pf: torch.Tensor, log_r: torch.Tensor, log_pterm: torch.Tensor
    ) -> torch.Tensor:
        """
        EXACT delta definition (aligned):
          delta uses steps log_pf[:, :-1], length L-1.
        returns C: [B, L] with C[:,0]=0
        """
        delta = (
            log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
        )  # [B, L-1]
        return torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(dim=1)  # [B, L]

    @staticmethod
    def _suffix_future_max(u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        dtype = u.dtype
        u_mask = torch.where(valid, u, torch.full_like(u, 0))
        rev = torch.flip(u_mask, dims=[1])
        rev_max = torch.cummax(rev, dim=1).values
        return torch.flip(rev_max, dims=[1])

    @staticmethod
    def _suffix_future_soft(
        u: torch.Tensor, valid: torch.Tensor, beta: float, rho: float
    ) -> torch.Tensor:
        """
        Backward softmax over suffix with step penalty rho (distance discount).
        Stable implementation under AMP/bf16:
        u_soft[t] = (1/beta) log sum_{j>=t} exp(beta*u[j] - beta*rho*(j-t))

        We compute in u-space recurrence:
        W_t = (1/b) logaddexp(b*u_t, b*(W_{t+1}-rho))
        where W_t == u_soft[t].
        """
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")

        B, L = u.shape
        device = u.device
        out_dtype = u.dtype

        # IMPORTANT: do the recurrence in fp32 and disable autocast,
        # otherwise (Z ~ beta*u ~ O(200)) makes rho step_pen vanish in bf16.
        with torch.autocast(device_type="cuda", enabled=False):
            u32 = u.float()
            valid32 = valid  # bool

            neg_inf = torch.finfo(torch.float32).min
            u_mask = torch.where(valid32, u32, torch.full_like(u32, neg_inf))  # [B, L]
            out = torch.empty_like(u_mask)  # fp32

            b = float(beta)
            r = float(rho)

            # W is u_soft in u-space
            W = u_mask[:, -1]  # [B]
            out[:, -1] = W

            for t in range(L - 2, -1, -1):
                # W = (1/b) log( exp(b*u_t) + exp(b*(W_next - rho)) )
                W = torch.logaddexp(b * u_mask[:, t], b * (W - r)) / b
                out[:, t] = W

            # optional: make invalid positions harmless (since some callers forget to mask)
            out = torch.where(valid32, out, torch.zeros_like(out))

        return out.to(out_dtype)

    # ----------------- forward -----------------

    def forward(
        self,
        log_pf: torch.Tensor,  # [B, L] (steps are log_pf[:, :-1])
        log_r: torch.Tensor,  # [B, L]
        log_pterm: torch.Tensor,  # [B, L]
        generated_text: torch.Tensor,  # [B, prompt_len + L]
        termination_token_id: int,
        prompt_len: int,
        ref_log_pf: Optional[torch.Tensor] = None,  # [B, L-1]
        ref_log_pterm: Optional[torch.Tensor] = None,  # [B, L]
        ref_scale: float = 1.0,
        max_prefix_len: Optional[int] = None,  # K in [1, L-1]
        logZ: Optional[torch.Tensor] = None,  # scalar or [B]
        k_min: Optional[int] = None,  # override per call
        **kwargs,
    ) -> Dict[str, Any]:
        B, L = log_pf.shape
        if log_r.shape != (B, L):
            raise ValueError(f"log_r {log_r.shape} vs (B,L)={(B,L)}")
        if log_pterm.shape != (B, L):
            raise ValueError(f"log_pterm {log_pterm.shape} vs (B,L)={(B,L)}")
        if generated_text.shape[0] != B:
            raise ValueError("generated_text batch mismatch")
        if generated_text.shape[1] - prompt_len != L:
            raise ValueError(
                f"generated_text post-prompt length must equal L. "
                f"got generated_text.shape={generated_text.shape}, prompt_len={prompt_len}, L={L}"
            )
        if L <= 1:
            raise ValueError("Need L>=2")

        device, dtype = log_pf.device, log_pf.dtype

        # ---- tau ----
        eos_or_after = (generated_text[:, prompt_len:-1] == int(termination_token_id)).cumsum(
            dim=-1
        ) >= 1  # [B, L-1]
        valid_end = ~eos_or_after  # [B, L-1]
        tau = valid_end.sum(dim=1).clamp(0, L - 1)  # [B] in [0, L-1]

        # ================== 1) Terminal TB with explicit logZ ==================
        if logZ is None:
            logZ_b = self.logZ.to(device=device, dtype=dtype).expand(B)
        else:
            z = logZ.to(device=device, dtype=dtype)
            logZ_b = z.expand(B) if z.ndim == 0 else z
            if logZ_b.shape != (B,):
                raise ValueError(f"logZ must be scalar or [B], got {tuple(z.shape)}")

        sum_log_pf = self._sum_log_pf_upto_tau(log_pf, tau)  # [B]
        log_pterm_tau = self._gather_by_index(log_pterm, tau)  # [B]
        log_r_tau = self._gather_by_index(log_r, tau)  # [B]

        tb_res = logZ_b + sum_log_pf + log_pterm_tau - log_r_tau
        loss_tb_i = tb_res.square()  # [B]
        loss_tb = loss_tb_i.mean()

        # ================== 2) AUX ==================
        loss_aux = torch.zeros((), device=device, dtype=dtype)
        aux_active_rate = torch.zeros((), device=device, dtype=dtype)

        if self.aux_weight > 0.0:
            # ---- K clamp ----
            if max_prefix_len is None:
                K = None
            else:
                K = int(max_prefix_len)
                K = max(1, min(K, L - 1))

            # ---- k_min clamp ----
            kmin = self.k_min if k_min is None else int(k_min)
            kmin = max(1, min(kmin, L - 1))

            # ---- build Ck (delta-cumsum) ----
            lp_aux = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm
            C_aux = self._delta_cumsum(log_pf, log_r, lp_aux)  # [B, L]
            Ck = C_aux[:, 1:]  # [B, L-1] for k=1..L-1

            # indices k=1..L-1
            k_idx = torch.arange(1, L, device=device).view(1, L - 1)  # [1, L-1]

            # horizon h = min(tau, K) else tau
            if K is None:
                h = tau
            else:
                h = torch.minimum(tau, tau.new_full(tau.shape, K))

            within_tau = k_idx <= tau.view(B, 1)
            within_h = k_idx <= h.view(B, 1)

            # base eligibility: before EOS and within horizon
            m_base_bool = within_tau & within_h & valid_end  # [B, L-1] bool

            # ---- original: m_base_bool is [B, L-1] bool; k_idx is [1, L-1] with values 1..L-1 ----
            after_kmin = k_idx >= kmin  # [1, L-1] bool

            # ---- NEW: windowed AUX to avoid "k=2 attractor" when kmin gets small ----
            W = max(
                1, int(getattr(self, "k_window", 4))
            )  # set self.k_window=4 in __init__ (or keep getattr)

            if W > 1:
                # per-sample window start k0 ∈ [kmin, max_start], ensuring room for W steps
                max_start = (h - (W - 1)).clamp(min=kmin)  # [B]
                # sample k0 per sample (breaks fixed constraint at k=kmin)
                # NOTE: randomness is per batch; good enough for now
                k1 = h  # [B]
                k0 = kmin + torch.floor(
                    torch.rand((B,), device=device, dtype=dtype)
                    * (h - kmin + 1).clamp(min=1).to(dtype)
                ).to(torch.long)
                in_window = (k_idx >= k0.view(B, 1)) & (k_idx <= k1.view(B, 1))
            else:
                k0 = torch.full((B,), int(kmin), device=device, dtype=torch.long)  # [B]
                k1 = h  # [B]
                in_window = after_kmin.expand(B, -1)

            # final AUX mask
            m_bool = m_base_bool & in_window  # [B, L-1] bool
            m = m_bool.to(dtype)  # [B, L-1] float

            # delta weights: w_k = lambda^(k - k0)
            w_exp = (k_idx - k0.view(B, 1)).clamp_min(0).to(dtype)  # [B, L-1]
            w = (self.subtb_lambda**w_exp).to(dtype)  # [B, L-1]

            # u = log_r - ref_scale * ref_logP (or log_r)
            if (ref_log_pf is not None) and (ref_log_pterm is not None):
                if ref_log_pf.shape != (B, L - 1) or ref_log_pterm.shape != (B, L):
                    raise ValueError(
                        f"ref shapes mismatch: {ref_log_pf.shape}, {ref_log_pterm.shape}"
                    )
                ref_logP = self._reconstruct_ref_logP(ref_log_pf, ref_log_pterm)  # [B, L]
                u = log_r - float(ref_scale) * ref_logP
            else:
                u = log_r

            u_k = u[:, 1:]  # [B, L-1]

            # ================== Absorb gating (position-level no_reward) ==================
            # Strict absorb horizon first:
            if K is None:
                before_absorb_horizon = within_h  # [B, L-1]
                can_absorb_seq = torch.ones((B, 1), device=device, dtype=torch.bool)
            else:
                before_absorb_horizon = k_idx < K  # [1, L-1] (broadcast)
                if self.absorb_requires_tau_ge_K:
                    can_absorb_seq = tau.view(B, 1) >= K  # [B,1]
                else:
                    can_absorb_seq = torch.ones((B, 1), device=device, dtype=torch.bool)

            # Position-level: only when THIS k has no extra reward
            no_reward_at_k = u_k.abs() <= self.extra_absorb_eps  # [B, L-1]

            # suffix targets on states 0..L-1
            pos = torch.arange(L, device=device).view(1, L)
            valid_future = (pos <= h.view(B, 1)) & (pos <= tau.view(B, 1))

            u_det = u.detach()
            u_max = self._suffix_future_max(u_det, valid_future)
            u_soft = self._suffix_future_soft(
                u_det, valid_future, beta=self.soft_beta, rho=self.soft_rho
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

            # mask out invalid future positions to avoid inf * 0 -> nan when apply_absorb==0
            u_target = torch.where(valid_future, u_target, torch.zeros_like(u_target))

            u_tk = u_target[:, 1:]  # [B, L-1]
            # alpha_k = gamma^(h-k)
            exp = (h.view(B, 1) - k_idx).clamp_min(0).to(dtype)
            alpha = (self.gamma**exp) * within_h.to(dtype)  # [B, L-1]

            apply_absorb = (
                m_base_bool & in_window & before_absorb_horizon & can_absorb_seq & no_reward_at_k
            )  # [B, L-1] bool

            alpha_eff = alpha * apply_absorb.to(dtype)
            Ck_abs = Ck + alpha_eff * (u_k - u_tk)

            # ================== per-sample AUX ==================
            mw_mask = m * w  # [B, L-1]
            den_i = mw_mask.sum(dim=1)  # [B]
            active_i = den_i > self.eps
            aux_active_rate = active_i.to(dtype).mean()

            # normalized weights inside each sample (stabilize AUX scale)
            mw_norm = mw_mask / den_i.clamp_min(self.eps).view(B, 1)
            loss_aux_i = (Ck_abs.square() * mw_norm).sum(dim=1)
            loss_aux_i = torch.where(active_i, loss_aux_i, torch.zeros_like(loss_aux_i))
            loss_aux = loss_aux_i.mean()

            # eta_i: if no aux => 0
            eta = float(self.aux_weight)
            eta_i = eta * active_i.to(dtype)
            # ================== TB + eta*AUX, normalized (gradient-level) ==================
            if self.normalize_mix:
                denom = (1.0 + eta_i).clamp_min(self.eps).detach()
                loss_i = (loss_tb_i + eta_i * loss_aux_i) / denom
            else:
                loss_i = loss_tb_i + eta_i * loss_aux_i
            loss = loss_i.mean()
        else:
            loss = loss_tb

        return {
            "loss": loss,
            "loss_tb": loss_tb.detach(),
            "loss_aux": loss_aux.detach(),
            "aux_active_rate": aux_active_rate.detach(),
            "logZ": self.logZ.detach(),
            "tb_res_mean": tb_res.detach().mean(),
            "tb_res_std": tb_res.detach().std(unbiased=False),
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
