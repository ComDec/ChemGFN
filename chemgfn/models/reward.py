"""Log-reward models used to train the GFlowNet policy.

Every reward in this module is the sum of two terms:

* a **reference prior**, the per-state log reward of the frozen pre-trained model, computed by
  :func:`score_fast`; and
* a **task score**, produced by a :class:`~chemgfn.models.validators.Validator` for each state on
  the trajectory.

The classes differ only in how the two terms are combined:

* :class:`MixedReferenceReward` weights the task score by the magnitude of the terminal
  reference log reward.
* :class:`PrefixShapedReward` subtracts the terminal reference log reward instead, turning the
  task score into a per-prefix shaping signal (Expr24).
* :class:`AbsorbingPrefixReward` builds the absorbed suffix target: the score of the terminating
  state is copied forward over every state past termination (SMILES, AMP).
* :class:`CommonGenReward` adds the task score to the reference prior state by state, leaving the
  validator's own prefix shaping untouched (CommonGen).

A reward exposes a single entry point, ``score``, which the sampler calls once per batch of
trajectories and which returns a dict with at least the keys ``reward``, ``reward_unpenalized``,
``log_pf_ref``, ``log_pterm_ref`` and ``validator_dict``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from chemgfn.models.validators import Validator

__all__ = [
    "AbsorbingPrefixReward",
    "CommonGenReward",
    "MixedReferenceReward",
    "PrefixShapedReward",
    "compute_active_before",
    "score_fast",
]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def compute_active_before(gen_tokens: Tensor, eos: int) -> Tensor:
    """Mask of states that are still on-trajectory before each action.

    ``active_before[:, t] = prod_{j < t} 1[token_j != eos]``, i.e. position ``t`` is active only
    if the trajectory had not already terminated when the token at ``t`` was emitted.

    Args:
        gen_tokens: ``(B, T)`` generated token ids, excluding the prompt.
        eos: Termination token id.

    Returns:
        ``(B, T)`` boolean tensor.
    """

    B, T = gen_tokens.shape
    active_before = torch.ones((B, T), device=gen_tokens.device, dtype=torch.bool)
    if T > 1:
        alive_after = (gen_tokens != eos).to(torch.long).cumprod(dim=1).to(torch.bool)
        active_before[:, 1:] = alive_after[:, :-1]
    return active_before


# --------------------------------------------------------------------------- #
# Reference prior
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_fast(
    model,
    encoded_input: Tensor,
    termination_token_id: int,
    skip_first: int,
    reward_temperature: float = 1.0,
    invalid_vocab_mask: Tensor | None = None,
    illegal_vocab_penalty: float = -99,
    **_: Any,
) -> dict[str, Tensor]:
    """Evaluate a batch of trajectories under the frozen reference model.

    The reference log reward of a state is the log probability the frozen model assigns to
    terminating there, plus the log probability of the tokens that lead to it. Terminated states
    are zeroed out so that padding past the first EOS contributes nothing.

    Args:
        model: Frozen reference language model.
        encoded_input: ``(B, P + T)`` prompt tokens followed by generated tokens.
        termination_token_id: Token id that terminates a trajectory.
        skip_first: Prompt length ``P``; scoring starts at this position.
        reward_temperature: Temperature applied to the reference logits.
        invalid_vocab_mask: Optional ``(V,)`` mask of tokens the task forbids.
        illegal_vocab_penalty: Logit offset added to the masked tokens.

    Returns:
        Dict with ``ref_log_pf`` ``(B, T)``, ``ref_log_pterm`` ``(B, T + 1)`` and ``reward``
        ``(B, T + 1)``.
    """

    logits = model(encoded_input).logits
    logits = logits[:, skip_first - 1 :]

    if invalid_vocab_mask is not None:
        logits = logits.clone()  # clone only when mutating
        logits[:, :, invalid_vocab_mask] += illegal_vocab_penalty

    logits /= reward_temperature
    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)

    log_pf = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
    log_p = log_pf.cumsum(dim=-1)

    reward = logprob[:, :, termination_token_id].clone()
    reward[:, 1:] = reward[:, 1:] + log_p
    non_term_mask = (encoded_input != termination_token_id)[:, skip_first:]
    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )

    reward = reward.clone()
    reward[~non_term_mask] = 0.0

    return {
        "ref_log_pf": log_pf,
        "ref_log_pterm": logprob[:, :, termination_token_id],
        "reward": reward,
    }


ScoreFunctionName = Literal["score_fast"]

SCORE_FUNCTIONS: dict[str, Callable[..., dict[str, Tensor]]] = {"score_fast": score_fast}


def _resolve_score_function(name: str) -> Callable[..., dict[str, Tensor]]:
    """Look up a reference-prior scoring function by its configuration name."""

    try:
        return SCORE_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown score_function {name!r}; expected one of {sorted(SCORE_FUNCTIONS)}."
        ) from None


# --------------------------------------------------------------------------- #
# Reward models
# --------------------------------------------------------------------------- #


class MixedReferenceReward:
    """Reference prior mixed with a per-state task score.

    The log reward at every state is the frozen reference log reward plus the validator's local
    score, weighted both by the magnitude of the terminal reference log reward and by the
    ``scaling_factor`` schedule.
    """

    def __init__(
        self,
        sentence_validator: Validator | None,
        illegal_vocab_penalty: float = -99,
        score_function: ScoreFunctionName = "score_fast",
        **kwargs: Any,
    ) -> None:
        """Configure the reward.

        Args:
            sentence_validator: Validator producing the per-state task score, or ``None`` to use
                the reference prior alone.
            illegal_vocab_penalty: Logit offset applied to task-forbidden tokens in the
                reference prior.
            score_function: Name of the reference-prior scoring function.
        """

        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.temperature = 1.0
        self.score_function = score_function

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model,
        tokenizer: PreTrainedTokenizer,
        reward_temperature: float = 1.0,
        vocab_invalid_mask: Tensor | None = None,
        scaling_factor: float = 0.5,
        scaffold: str | None = None,
        termination_token_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the per-state log reward for a batch of trajectories.

        Args:
            input_batch: ``(B, P + T)`` prompt tokens followed by generated tokens.
            prompt_length: Prompt length ``P``.
            model: Frozen reference language model.
            tokenizer: Tokenizer used to decode trajectories for the validator.
            reward_temperature: Temperature applied to the reference logits.
            vocab_invalid_mask: Optional ``(V,)`` mask of task-forbidden tokens.
            scaling_factor: Weight of the validator's local score.
            scaffold: Optional task conditioning passed through to the validator.
            termination_token_id: Termination token id; defaults to the tokenizer's EOS.

        Returns:
            Dict with the per-state log reward and the reference-model diagnostics.
        """

        eos = int(
            termination_token_id if termination_token_id is not None else tokenizer.eos_token_id
        )
        reference_results = _resolve_score_function(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
        )

        validator_dict = None
        reference_logP = reference_results["reward"]
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

        reward_mixed = reference_logP

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"]
            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * local_score
                + scaling_factor * local_score
            )

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reward_mixed,
            "full_tokens": None if validator_dict is None else validator_dict.get("full_tokens"),
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


class PrefixShapedReward:
    """Reference prior shaped by a per-prefix task score.

    The terminal reference log reward is subtracted from every state in proportion to the local
    score, so that a prefix is rewarded for the score it already achieves rather than for the
    reference model's opinion of the completed object. Used for the Expr24 task.
    """

    def __init__(
        self,
        sentence_validator: Validator | None,
        illegal_vocab_penalty: float = -99,
        score_function: ScoreFunctionName = "score_fast",
        **kwargs: Any,
    ) -> None:
        """Configure the reward.

        Args:
            sentence_validator: Validator producing the per-state task score, or ``None`` to use
                the reference prior alone.
            illegal_vocab_penalty: Logit offset applied to task-forbidden tokens in the
                reference prior.
            score_function: Name of the reference-prior scoring function.
        """

        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.score_function = score_function

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer: PreTrainedTokenizer | None = None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask: Tensor | None = None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        termination_token_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the per-state log reward for a batch of trajectories.

        Args:
            input_batch: ``(B, P + T)`` prompt tokens followed by generated tokens.
            prompt_length: Prompt length ``P``.
            model: Frozen reference language model.
            tokenizer: Tokenizer used to decode trajectories for the validator.
            reward_temperature: Temperature applied to the reference logits.
            vocab_invalid_mask: Optional ``(V,)`` mask of task-forbidden tokens.
            scaling_factor: Weight of the validator's local score.
            reference_logits_scale: Weight of the reference prior.
            scaffold: Optional task conditioning passed through to the validator.
            termination_token_id: Termination token id; defaults to the tokenizer's EOS.

        Returns:
            Dict with the per-state log reward and the reference-model diagnostics.
        """

        eos = int(
            termination_token_id if termination_token_id is not None else tokenizer.eos_token_id
        )
        reference_results = _resolve_score_function(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
        )

        reference_logP = reference_results["reward"] * float(reference_logits_scale)
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

        validator_dict = None
        reward_mixed = reference_logP

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(reference_logP.dtype)

            reward_mixed = (
                reference_logP
                + (-1) * (reference_logP[..., -1]).unsqueeze(-1) * local_score
                + scaling_factor * local_score
            )

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


class AbsorbingPrefixReward:
    """Reference prior plus an absorbed suffix target.

    The trajectory terminates at the first EOS. The validator score of that terminating state is
    copied forward onto every later state, so the states past termination form an absorbing
    region carrying the terminal score, and unreachable states carry no score at all. The log
    reward is the reference prior plus ``scaling_factor`` times this absorbed score. Used for the
    SMILES and AMP tasks.
    """

    def __init__(
        self,
        sentence_validator: Validator | None,
        illegal_vocab_penalty: float = 0,
        score_function: ScoreFunctionName = "score_fast",
        **kwargs: Any,
    ) -> None:
        """Configure the reward.

        Args:
            sentence_validator: Validator producing the per-state task score, or ``None`` to use
                the reference prior alone.
            illegal_vocab_penalty: Logit offset applied to task-forbidden tokens in the
                reference prior.
            score_function: Name of the reference-prior scoring function.
        """

        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.score_function = score_function

    @torch.no_grad()
    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer: PreTrainedTokenizer | None = None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask: Tensor | None = None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the per-state log reward for a batch of trajectories.

        Args:
            input_batch: ``(B, P + T)`` prompt tokens followed by generated tokens.
            prompt_length: Prompt length ``P``.
            model: Frozen reference language model.
            tokenizer: Tokenizer used to decode trajectories for the validator; its EOS defines
                termination.
            reward_temperature: Temperature applied to the reference logits.
            vocab_invalid_mask: Optional ``(V,)`` mask of task-forbidden tokens.
            scaling_factor: Weight of the absorbed task score.
            reference_logits_scale: Weight of the reference prior.
            scaffold: Optional task conditioning passed through to the validator.

        Returns:
            Dict with the per-state log reward and the reference-model diagnostics.
        """

        eos = tokenizer.eos_token_id
        reference_results = _resolve_score_function(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
        )

        reference_logP = reference_results["reward"] * float(reference_logits_scale)  # (B, L)
        ref_log_pf = reference_results["ref_log_pf"]  # (B, T)
        ref_log_pterm = reference_results["ref_log_pterm"]  # (B, L)

        validator_dict = None
        reward_mixed = reference_logP

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(reference_logP.dtype)

            # Align the validator's per-state score to (B, L_state).
            if local_score.shape[1] == reference_logP.shape[1] - 1:
                local_score = torch.cat(
                    [local_score.new_zeros(local_score.shape[0], 1), local_score], dim=1
                )
            assert (
                local_score.shape == reference_logP.shape
            ), f"local_score {local_score.shape} vs reference_logP {reference_logP.shape}"

            dtype = reference_logP.dtype
            device = reference_logP.device
            B, L_state = reference_logP.shape
            T_tok = L_state - 1

            gen_tokens = input_batch[:, prompt_length : prompt_length + T_tok]  # (B, T_tok)
            active_before = compute_active_before(gen_tokens, eos=eos)  # (B, T_tok)

            # Terminating state: number of non-EOS tokens emitted before the first EOS.
            non_eos = gen_tokens.ne(int(eos))
            len_mask = active_before & non_eos
            L_term = len_mask.sum(dim=1).to(torch.long)  # (B,) in [0, T_tok]
            terminal_score = local_score.gather(1, L_term.view(B, 1)).squeeze(1).to(dtype)  # (B,)

            # Reachable states: state 0 always, state t + 1 iff the trajectory was still active.
            active_state = torch.cat(
                [torch.ones((B, 1), device=device, dtype=dtype), active_before.to(dtype)],
                dim=1,
            )  # (B, L_state)

            # Terminal state index, counting the EOS action itself.
            tau_state = active_before.to(torch.long).sum(dim=1).clamp(0, L_state - 1)  # (B,)

            # Absorb the terminal score over every state from tau_state onwards, then zero out
            # the states the trajectory never reached.
            pos = torch.arange(L_state, device=device).view(1, L_state)
            score_state = torch.where(
                pos >= tau_state.view(B, 1), terminal_score.view(B, 1), local_score
            )
            score_state = score_state * active_state

            reward_mixed = reference_logP + float(scaling_factor) * score_state

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


class CommonGenReward:
    """Reference prior plus a dense per-state task score.

    The log reward at every state is the frozen reference log reward plus ``scaling_factor``
    times the validator's score for that state, which combines structural validity, concept
    coverage and an n-gram quality term. The score is added exactly where the validator places
    it, so a prefix is credited with the coverage and quality it has already achieved and the
    terminating state additionally carries the hard-coverage bonus. Used for the CommonGen task.
    """

    def __init__(
        self,
        sentence_validator: Validator | None,
        illegal_vocab_penalty: float = -99,
        score_function: ScoreFunctionName = "score_fast",
        **kwargs: Any,
    ) -> None:
        """Configure the reward.

        Args:
            sentence_validator: Validator producing the per-state task score, or ``None`` to use
                the reference prior alone.
            illegal_vocab_penalty: Logit offset applied to task-forbidden tokens in the
                reference prior.
            score_function: Name of the reference-prior scoring function.
        """

        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.score_function = score_function

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer: PreTrainedTokenizer | None = None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask: Tensor | None = None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: Any = None,
        termination_token_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute the per-state log reward for a batch of trajectories.

        Args:
            input_batch: ``(B, P + T)`` prompt tokens followed by generated tokens.
            prompt_length: Prompt length ``P``.
            model: Frozen reference language model.
            tokenizer: Tokenizer used to decode trajectories for the validator.
            reward_temperature: Temperature applied to the reference logits.
            vocab_invalid_mask: Optional ``(V,)`` mask of task-forbidden tokens.
            scaling_factor: Weight of the validator's per-state score.
            reference_logits_scale: Weight of the reference prior.
            scaffold: Concept set and reference sentences passed through to the validator.
            termination_token_id: Termination token id; defaults to the tokenizer's EOS.

        Returns:
            Dict with the per-state log reward and the reference-model diagnostics.
        """

        eos = int(
            termination_token_id if termination_token_id is not None else tokenizer.eos_token_id
        )
        reference_results = _resolve_score_function(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
        )

        reference_logP = reference_results["reward"] * float(reference_logits_scale)
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

        validator_dict = None
        reward_mixed = reference_logP

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(reference_logP.dtype)
            reward_mixed = reference_logP + float(scaling_factor) * local_score

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }
