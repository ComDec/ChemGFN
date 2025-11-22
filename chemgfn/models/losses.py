"""
Loss functions for GFlowNet training.

This module provides loss function implementations for training GFlowNets,
including SubTrajectory Balance (SubTB) losses with various enhancements.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GFNLoss",
    "ModifiedSubTBLoss",
    "ModifiedSubTBLossSplitReward",
    "ModifiedSubTBBalanceLoss",
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
