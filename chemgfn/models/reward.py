from __future__ import annotations

import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from chemgfn.models.validators import (
    BracketValidator,
    Expr24Validator,
    NumberValidator,
    ParenthesesValidator,
    RDKitValidator,
    SentenceValidator,
)
from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base
from chemgfn.utils.phi_utils import compute_prefix_diagnostics

ScorePair = Tuple[Tensor, Tensor]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _repeat_past_key_values(
    past_key_values: tuple[tuple[Tensor, ...], ...], batch_size: int
) -> tuple[tuple[Tensor, ...], ...]:
    """Broadcast cached KV tensors to match the current batch size."""

    return tuple(
        tuple(value.repeat(batch_size, 1, 1, 1) for value in layer) for layer in past_key_values
    )


def _ensure_tensor_like(value: Any, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.full(
        (reference.shape[0],),
        float(value),
        device=reference.device,
        dtype=reference.dtype,
    )


def _build_penalty_ramp(
    base_values: Tensor,
    steps: int,
    start_ratio: float,
    end_ratio: float,
    reference: Tensor,
) -> Tensor:
    increments = torch.linspace(
        0,
        1,
        steps,
        device=reference.device,
        dtype=reference.dtype,
    )
    start = base_values * start_ratio
    end = base_values * end_ratio
    return start.unsqueeze(1) + (end - start).unsqueeze(1) * increments.unsqueeze(0)


def _apply_invalid_penalty(
    reward: Tensor,
    invalid_mask: Tensor,
    start_ratio: float,
    end_ratio: float,
    base_override: Any | None = None,
) -> Tensor:
    base_values = (
        reward.min(dim=-1).values
        if base_override is None
        else _ensure_tensor_like(base_override, reward)
    )
    penalty = _build_penalty_ramp(
        base_values,
        reward.shape[1],
        start_ratio,
        end_ratio,
        reward,
    )
    return torch.min(reward, penalty * invalid_mask)


def _stack_if_not_empty(entries: Iterable[Tensor]) -> Tensor | None:
    entries = list(entries)
    if not entries:
        return None
    return torch.stack(entries, dim=0)


def _init_prefix_collapse(
    prefix_collapse_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], PrefixNoveltyTracker | None]:
    cfg = prefix_collapse_kwargs or {}
    k_list = tuple(int(k) for k in cfg.get("k_list", (1, 2, 3, 4, 5, 6)))
    top1_thr = float(cfg.get("top1_thr", 0.95))
    max_steps = cfg.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)
    novelty_window = int(cfg.get("novelty_window", 200))
    novelty_max_prefixes = int(cfg.get("novelty_max_prefixes", 200_000))

    tracker = None
    if novelty_window > 0:
        tracker = PrefixNoveltyTracker(
            k_list=k_list,
            window_size=novelty_window,
            max_prefixes=novelty_max_prefixes,
        )

    return {"k_list": k_list, "top1_thr": top1_thr, "max_steps": max_steps}, tracker


def _merge_prefix_diag(
    prefix_diag: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not extra:
        return prefix_diag
    if prefix_diag is None:
        return dict(extra)
    prefix_diag.update(extra)
    return prefix_diag


@contextmanager
def use_base_model(model, disable_peft: bool = False) -> None:
    """Temporarily swap LoRA adapters off to operate on the base model."""

    if disable_peft:
        yield
        return
    else:
        lora_to_base(model)
        try:
            yield
        finally:
            base_to_lora(model)


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_fast(
    model,
    encoded_input: Tensor,
    termination_token_id: int,
    skip_first: int,
    reward_temperature: float = 1.0,
    invalid_vocab_mask: Tensor | None = None,
    agree_list: list[Tensor] | None = None,
    illegal_vocab_penalty: float = -99,
    grammar_disagree_penalty: float = -99,
    prompt_cache: tuple[Any, tuple[tuple[Tensor, ...], ...]] | None = None,
    **_: Any,
) -> ScorePair:
    """Compute per-step log rewards for a batch of trajectories."""

    if prompt_cache is None:
        logits = model(encoded_input).logits
    else:
        batched_cache = _repeat_past_key_values(prompt_cache[1], encoded_input.shape[0])
        logits = model(encoded_input, past_key_values=batched_cache).logits

    logits = logits.detach()[:, skip_first - 1 :]

    # TODO: remove all penalty from the logits
    if invalid_vocab_mask is not None:
        logits = logits.clone()
        logits[:, :, invalid_vocab_mask] += illegal_vocab_penalty

    if agree_list is not None:
        # convert list of tensor to tensor
        agree_tensor = _stack_if_not_empty(agree_list).permute(
            1, 0, 2
        )  # (batch_size, seq_len, vocab_size)
        try:
            logits[~agree_tensor] += grammar_disagree_penalty
        except Exception as e:
            # If shapes don't match, skip disagree penalty
            pass

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


# --------------------------------------------------------------------------- #
# Reward models
# --------------------------------------------------------------------------- #


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator | None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model,
        tokenizer: PreTrainedTokenizer,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        illegal_vocab_penalty: float = -99,
        **kwargs,
    ) -> ScorePair:
        with use_base_model(model):
            reward, reward_unpenalized = score_fast(
                model=model,
                encoded_input=input_batch,
                termination_token_id=tokenizer.eos_token_id,
                skip_first=prompt_length,
                reward_temperature=reward_temperature,
                invalid_vocab_mask=vocab_invalid_mask,
                illegal_vocab_penalty=illegal_vocab_penalty,
            )

        if self.sentence_validator is not None:
            invalid_mask = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)[
                "invalid"
            ]
            reward = _apply_invalid_penalty(
                reward, invalid_mask, self.invalid_start_ratio, self.invalid_end_ratio
            )

        return reward, reward_unpenalized


class Reference_Target_Score_Positive_Mixed_Invalid_Mask:
    def __init__(
        self,
        sentence_validator: SentenceValidator | None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        disable_peft: bool = False,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        **kwargs,
    ) -> None:
        """Initialize reward class with penalty values."""
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)

        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio
        self.disable_peft = disable_peft
        self.temperature = 1.0
        self.score_function = score_function

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model,
        tokenizer: PreTrainedTokenizer,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        reference_results = eval(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )

        validator_dict = None
        reference_logP = reference_results["reward"]
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

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

            reward_penalized = reward_mixed
        else:
            reward_penalized = reward_mixed

        return {
            "reward": reward_penalized,
            "reward_unpenalized": reward_mixed,
            "full_tokens": None if validator_dict is None else validator_dict.get("full_tokens"),
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


class Target_Score_Positive:
    def __init__(
        self,
        sentence_validator: SentenceValidator | None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        disable_peft: bool = False,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        **kwargs,
    ) -> None:
        """Initialize reward class with penalty values."""
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)

        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio
        self.disable_peft = disable_peft
        self.temperature = 1.0

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model,
        tokenizer: PreTrainedTokenizer,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        reference_results = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )

        validator_dict = None
        reward_unpenalized = reference_results["reward"]
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            global_score = validator_dict["global_score"].bool()  # [B]

            # Keep intermediate steps as reference reward; only shape terminal step per SubTB design
            reward_penalized = reward_unpenalized.clone()
            terminal_idx = reward_penalized.shape[1] - 1
            eps = 1e-6
            reward_penalized[global_score, terminal_idx] = 0.0  # log(1) for valid
            reward_penalized[~global_score, terminal_idx] = torch.log(
                torch.full_like(reward_penalized[~global_score, terminal_idx], eps)
            )
        else:
            reward_penalized = reward_unpenalized

        return {
            "reward": reward_penalized,
            "reward_unpenalized": reward_unpenalized,
            "full_tokens": None if validator_dict is None else validator_dict.get("full_tokens"),
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


class Target_Score_Positive_MCMCPrior:
    """
    Reward that uses an n-gram MCMC prior q(x) (trained offline) to supply dense log_r,
    while shaping only the terminal step based on Expr24 validity/length.

    - Middle steps: log_r is log q(token | context) from the prior (with backoff).
    - Terminal step: if first EOS arrives after exactly target_length generated tokens,
      set log_r to 0 (log(1)); otherwise set to log_eps (strong penalty).
    """

    def __init__(
        self,
        sentence_validator: SentenceValidator | None,
        q_mcmc_path: str,
        target_length: int = 7,
        eps: float = 1e-6,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.q_mcmc_path = q_mcmc_path
        self.target_length = target_length
        self.eps = eps

        self._q_mcmc = None
        self._q_packed_per_device: dict[torch.device, dict[tuple[int, ...], torch.Tensor]] = {}
        self._max_ctx_len: int | None = None
        self.temperature = 1.0

    def _ensure_q_packed(self, tokenizer: PreTrainedTokenizer, device: torch.device):
        if self._q_mcmc is None:
            self._q_mcmc = load_q_mcmc(self.q_mcmc_path)
            self._max_ctx_len = max((len(k) for k in self._q_mcmc.keys()), default=0)
        if device not in self._q_packed_per_device:
            self._q_packed_per_device[device] = pack_q_mcmc_to_device(
                self._q_mcmc,
                vocab_size=len(tokenizer),
                device=device,
                eps=self.eps,
                dtype=torch.float32,
            )
        return self._q_packed_per_device[device]

    def _log_q(
        self, q_packed: dict[tuple[int, ...], torch.Tensor], context: tuple[int, ...], token: int
    ):
        q_vec = _get_q_vec_with_backoff(q_packed, context)
        return torch.log(q_vec[token].clamp_min(self.eps))

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model,
        tokenizer: PreTrainedTokenizer,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        device = input_batch.device
        q_packed = self._ensure_q_packed(tokenizer, device)

        eos_id = tokenizer.eos_token_id
        generated = input_batch[:, prompt_length:]
        batch_size, gen_len = generated.shape
        log_q_steps = torch.full_like(generated, fill_value=float("-inf"), dtype=torch.float32)

        # Build log q for each position (context includes prompt + previous generated tokens)
        for b in range(batch_size):
            prefix_tokens = input_batch[b, :prompt_length].tolist()
            for t in range(gen_len):
                tok = int(generated[b, t].item())
                # Context: last max_ctx_len tokens from full history
                if self._max_ctx_len and self._max_ctx_len > 0:
                    ctx = tuple(prefix_tokens[-self._max_ctx_len :])
                else:
                    ctx = ()
                log_q_steps[b, t] = self._log_q(q_packed, ctx, tok)
                prefix_tokens.append(tok)

        reward_unpenalized = log_q_steps.cumsum(dim=-1)
        # Terminal shaping based on Expr24 length and validator
        reward_penalized = reward_unpenalized.clone()
        validator_dict = None
        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )

        for b in range(batch_size):
            seq = generated[b]
            eos_positions = (seq == eos_id).nonzero(as_tuple=False)
            eos_pos = int(eos_positions[0].item()) if eos_positions.numel() > 0 else None
            gen_tokens_before_eos = eos_pos if eos_pos is not None else gen_len

            valid_len = gen_tokens_before_eos == self.target_length
            valid_expr = False
            if validator_dict is not None:
                valid_expr = bool(validator_dict["global_score"][b].item())
            is_valid = valid_len and valid_expr

            if eos_pos is not None:
                if is_valid:
                    # keep cumulative log_q up to EOS (best estimate of log p(x))
                    reward_penalized[b, eos_pos] = reward_unpenalized[b, eos_pos]
                else:
                    reward_penalized[b, eos_pos] = reward_unpenalized[b, eos_pos] + math.log(
                        self.eps
                    )

        return {
            "reward": reward_penalized,
            "reward_unpenalized": reward_unpenalized,
            "full_tokens": None if validator_dict is None else validator_dict.get("full_tokens"),
            "log_pf_ref": None,
            "log_pterm_ref": None,
            "validator_dict": validator_dict,
        }


from chemgfn.utils.phi_utils import (
    PrefixValueMemory,
    PrefixValueMemoryNoBackoff,
    apply_phi_shaping,
    batch_prefix_value_kgram,
    build_prefix_potential,
    compute_active_before,
    compute_prefix_diagnostics,
)


@torch.no_grad()
class Reference_Target_Score_Positive_Mixed_Invalid_Mask_PrefixShaping:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
        prefix_collapse_kwargs: dict[str, Any] | None = None,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)
        self.phi_weight = phi_weight
        self.score_function = score_function

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer=None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,  # (B,T) 生成的动作 token（不含 prompt）
        **kwargs,
    ) -> dict[str, Any]:
        eos = tokenizer.eos_token_id
        reference_results = eval(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )

        reference_logP = reference_results["reward"] * float(reference_logits_scale)
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

        validator_dict = None
        reward_mixed = reference_logP
        prefix_diag = None

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(
                reference_logP.dtype
            )  # (B,T), 000..1 or 000..0

            y = local_score[:, -1].clamp(0.0, 1.0)  # (B,)

            reward_mixed = (
                reference_logP
                + (-1) * (reference_logP[..., -1]).unsqueeze(-1) * local_score
                + scaling_factor * local_score
            )

            gen_tokens = input_batch[:, prompt_length:]
            active_before = compute_active_before(gen_tokens, eos=eos)

            if self.phi_weight > 0:
                sentences = input_batch[:, prompt_length:]
                non_term_mask = (input_batch != tokenizer.eos_token_id)[:, prompt_length:]

                pv = batch_prefix_value_kgram(
                    sentences, y, k=6, alpha=1.0, min_count=2, backoff=True
                )
                phi = build_prefix_potential(pv, ref_log_pf, non_term_mask, eta=1.0, clamp=4.0)
                reward_mixed[:, 1:] = reward_mixed[:, 1:] + phi * self.phi_weight

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
            "prefix_diag": prefix_diag,
        }


class Reference_Target_Score_Positive_Mixed_Prefix_Potential_Differential_Shaping:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
        pv_k: int = 6,
        pv_alpha: float = 1.0,
        pv_min_count: int = 2,
        pv_backoff: bool = True,
        phi_eta: float = 1.0,
        phi_clamp: float = 4.0,
        dphi_clip: float | None = None,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)
        self.phi_weight = float(phi_weight)

        self.pv_k = int(pv_k)
        self.pv_alpha = float(pv_alpha)
        self.pv_min_count = int(pv_min_count)
        self.pv_backoff = bool(pv_backoff)

        self.phi_eta = float(phi_eta)
        self.phi_clamp = float(phi_clamp)
        self.dphi_clip = dphi_clip

        self.score_function = score_function

    @torch.no_grad()
    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer=None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        eos = tokenizer.eos_token_id

        reference_results = eval(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )

        reference_logP = reference_results["reward"] * float(
            reference_logits_scale
        )  # (B, L_state)
        ref_log_pf = reference_results["ref_log_pf"]  # (B, T_tok)
        ref_log_pterm = reference_results["ref_log_pterm"]  # (B, L_state)

        validator_dict = None
        reward_mixed = reference_logP
        prefix_diag = None

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(reference_logP.dtype)

            if local_score.shape[1] == reference_logP.shape[1] - 1:
                local_score = torch.cat(
                    [local_score.new_zeros(local_score.shape[0], 1), local_score], dim=1
                )
            assert local_score.shape == reference_logP.shape

            y = local_score[:, -1].clamp(0.0, 1.0)

            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * local_score
                + float(scaling_factor) * local_score
            )

            B, L_state = reward_mixed.shape
            T_tok = L_state - 1
            gen_tokens = input_batch[:, prompt_length : prompt_length + T_tok]  # (B, T_tok)
            active_before = compute_active_before(gen_tokens, eos=eos)  # (B,T_tok)

            pv = batch_prefix_value_kgram(
                gen_tokens,
                y,
                k=self.pv_k,
                alpha=self.pv_alpha,
                min_count=self.pv_min_count,
                backoff=self.pv_backoff,
            )  # (B,T_tok)

            phi_tok = build_prefix_potential(
                pv=pv,
                ref_log_pf=ref_log_pf,
                non_term_mask=active_before,
                eta=self.phi_eta,
                clamp=self.phi_clamp,
            )  # (B,T_tok)

            reward_mixed, phi_state, dphi = apply_phi_shaping(
                reward_mixed=reward_mixed,
                phi_tok=phi_tok,
                active_before=active_before,
                phi_weight=self.phi_weight,
                mode="differential",
                anchor_start=0.0,
                anchor_end=0.0,
                dphi_clip=self.dphi_clip,
            )

            prefix_diag = compute_prefix_diagnostics(
                pv=pv,
                phi_state=phi_state,
                phi_tok=phi_tok,
                active_before=active_before,
                pv_sat_lo=0.05,
                pv_sat_hi=0.95,
            )

        out = {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }
        if prefix_diag is not None:
            out["prefix_value_diag"] = prefix_diag
        return out


class Reference_Target_Score_Positive_Mixed_Invalid_Mask_PrefixShapingWithMemory:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = 0,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
        phi_warmup: int = 800,
        phi_ramp_steps: int = 800,
        phi_decay_start: int | None = None,
        phi_decay_gamma: float = 0.999,
        dphi_clip: float | None = 2.0,
        # entropy gate
        use_entropy_gate: bool = True,
        ent_lo: float = 0.10,
        ent_hi: float = 0.55,
        # phi build params
        phi_eta: float = 1.0,
        phi_clamp: float = 2.0,
        pv_memory_kwargs: dict[str, Any] | None = None,
        debug_shapes: bool | None = None,
        debug_shapes_steps: int = 1,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)
        self.score_function = score_function

        self.phi_weight_max = float(phi_weight)
        self.phi_warmup = int(phi_warmup)
        self.phi_ramp_steps = int(phi_ramp_steps)
        self.phi_decay_start = (
            phi_decay_start if (phi_decay_start is None) else int(phi_decay_start)
        )
        self.phi_decay_gamma = float(phi_decay_gamma)
        self.dphi_clip = dphi_clip

        self.use_entropy_gate = bool(use_entropy_gate)
        self.ent_lo = float(ent_lo)
        self.ent_hi = float(ent_hi)

        self.phi_eta = float(phi_eta)
        self.phi_clamp = float(phi_clamp)

        self.global_step = 0
        self.pv_mem = PrefixValueMemory(**(pv_memory_kwargs or {}))

    def set_step(self, step: int) -> None:
        self.global_step = int(step)

    def _phi_weight_schedule(self, step: int) -> float:
        step = int(step)
        if step < self.phi_warmup:
            return 0.0

        t = step - self.phi_warmup
        if self.phi_ramp_steps > 0:
            w = self.phi_weight_max * min(1.0, t / float(self.phi_ramp_steps))
        else:
            w = self.phi_weight_max

        if (self.phi_decay_start is not None) and (step >= self.phi_decay_start):
            w = w * (self.phi_decay_gamma ** (step - self.phi_decay_start))
        return float(w)

    @staticmethod
    def _entropy_gate(
        pv: Tensor,
        active_before: Tensor,
        ent_lo: float,
        ent_hi: float,
    ) -> Tensor:
        p = pv.clamp(1e-6, 1.0 - 1e-6)
        ent = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        gate = (ent - float(ent_lo)) / max(1e-8, float(ent_hi) - float(ent_lo))
        gate = gate.clamp(0.0, 1.0)
        return gate * active_before.to(gate.dtype)

    @torch.no_grad()
    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer=None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,
        global_step: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        eos = tokenizer.eos_token_id

        reference_results = eval(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )

        reference_logP = reference_results["reward"] * float(
            reference_logits_scale
        )  # (B, L_state)
        ref_log_pf = reference_results["ref_log_pf"]  # (B, T_tok)
        ref_log_pterm = reference_results["ref_log_pterm"]  # (B, L_state)

        validator_dict = None
        reward_mixed = reference_logP
        prefix_diag = None

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(reference_logP.dtype)

            if local_score.shape[1] == reference_logP.shape[1] - 1:
                local_score = torch.cat(
                    [local_score.new_zeros(local_score.shape[0], 1), local_score], dim=1
                )
            assert (
                local_score.shape == reference_logP.shape
            ), f"local_score shape {local_score.shape} vs reference_logP {reference_logP.shape}"

            y = local_score[:, -1].clamp(0.0, 1.0)  # (B,)

            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * local_score
                + float(scaling_factor) * local_score
            )

            B, L_state = reward_mixed.shape
            T_tok = L_state - 1
            gen_tokens = input_batch[:, prompt_length : prompt_length + T_tok]  # (B, T_tok)

            active_before = compute_active_before(gen_tokens, eos=eos)  # (B,T_tok)

            self.pv_mem.set_step(self.global_step)
            phi_w = self._phi_weight_schedule(self.global_step)

            pv = None
            counts = None
            phi_tok = None
            phi_state = None
            dphi = None

            if phi_w > 0.0:
                p0 = self.pv_mem.get_base_rate()

                pv, counts = self.pv_mem.query_pv(gen_tokens, active_before)

                phi_tok = build_prefix_potential(
                    pv=pv,
                    ref_log_pf=ref_log_pf,
                    non_term_mask=active_before,
                    counts=counts,
                    eta=self.phi_eta,
                    clamp=self.phi_clamp,
                    tau_conf=self.pv_mem.tau_conf,
                    base_rate=p0,
                    center_by_base=True,
                    conf_mode="inv_sqrt",
                )

                # entropy gate
                if self.use_entropy_gate:
                    g_ent = self._entropy_gate(pv, active_before, self.ent_lo, self.ent_hi)
                    phi_tok = phi_tok * g_ent

                # differential shaping
                reward_mixed, phi_state, dphi = apply_phi_shaping(
                    reward_mixed=reward_mixed,
                    phi_tok=phi_tok,
                    active_before=active_before,
                    phi_weight=phi_w,
                    mode="differential",
                    anchor_start=0.0,
                    anchor_end=0.0,
                    dphi_clip=self.dphi_clip,
                )

                prefix_diag = compute_prefix_diagnostics(
                    pv=pv,
                    phi_state=phi_state,
                    phi_tok=phi_tok,
                    active_before=active_before,
                    pv_sat_lo=0.05,
                    pv_sat_hi=0.95,
                )

            else:
                prefix_diag = None

            self.pv_mem.update(gen_tokens, y, active_before)

        out = {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
            "prefix_value_diag": prefix_diag if prefix_diag is not None else None,
            "phi_state": phi_state,
            "phi_tok": phi_tok,
            "pv": pv,
        }
        return out


class Reference_Target_Score_Positive_Memory_PrefixShaping_NoBackoff:
    """
    PrefixValueMemoryNoBackoff + PrefixShaping (+ optional Entropy Gate)
      - "legacy": expr24
      - "smiles_absorbing": SMILES absorbing reward
    """

    RewardStrategy = Literal["legacy", "smiles_absorbing"]
    PVUpdateStrategy = Literal["legacy_last_valid_score", "smiles_global_score"]

    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = 0,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
        phi_warmup: int = 800,
        phi_ramp_steps: int = 800,
        phi_decay_start: int | None = None,
        phi_decay_gamma: float = 0.995,
        dphi_clip: float | None = 2.0,
        # entropy gate
        use_entropy_gate: bool = False,
        use_token_entropy_gate: bool = True,
        ent_lo: float = 0.10,
        ent_hi: float = 0.55,
        # phi build params
        phi_eta: float = 1.0,
        phi_clamp: float = 2.0,
        # split logic
        pv_split: int = 2,
        pv_split_inclusive: bool = True,
        pv_memory_kwargs: dict[str, Any] | None = None,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        # ---- NEW: reward strategy ----
        reward_strategy: RewardStrategy = "legacy",
        # SMILES absorbing reward options
        smiles_len_weight: float = 0.0,  # length prior (optional)
        smiles_score_clip: tuple[float, float] = (0.0, 1.0),  # global_score 归一化到 [0,1] 的 clip
        pv_update_strategy: PVUpdateStrategy | None = None,  # None => follow reward_strategy
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = illegal_vocab_penalty
        self.grammar_disagree_penalty = grammar_disagree_penalty
        self.score_function = score_function

        self.phi_weight_max = float(phi_weight)
        self.phi_warmup = int(phi_warmup)
        self.phi_ramp_steps = int(phi_ramp_steps)
        self.phi_decay_start = (
            phi_decay_start if (phi_decay_start is None) else int(phi_decay_start)
        )
        self.phi_decay_gamma = float(phi_decay_gamma)
        self.dphi_clip = dphi_clip if dphi_clip is not None else None

        self.use_entropy_gate = bool(use_entropy_gate)
        self.use_token_entropy_gate = bool(use_token_entropy_gate)
        self.ent_lo = float(ent_lo)
        self.ent_hi = float(ent_hi)

        self.phi_eta = float(phi_eta)
        self.phi_clamp = float(phi_clamp)

        self.pv_split = int(pv_split)
        self.pv_split_inclusive = bool(pv_split_inclusive)

        self.reward_strategy: Reference_Target_Score_Positive_Memory_PrefixShaping_NoBackoff.RewardStrategy = (
            reward_strategy
        )
        self.smiles_len_weight = float(smiles_len_weight)
        self.smiles_score_clip = (float(smiles_score_clip[0]), float(smiles_score_clip[1]))

        if pv_update_strategy is None:
            self.pv_update_strategy: Reference_Target_Score_Positive_Memory_PrefixShaping_NoBackoff.PVUpdateStrategy = (
                "legacy_last_valid_score" if reward_strategy == "legacy" else "smiles_global_score"
            )
        else:
            self.pv_update_strategy = pv_update_strategy

        self.cur_step = 0
        self.pv_mem = PrefixValueMemoryNoBackoff(**(pv_memory_kwargs or {}))

        self._reward_mixers = {
            "legacy": self._mix_reward_legacy,
            "smiles_absorbing": self._mix_reward_smiles_absorbing,
        }
        if self.reward_strategy not in self._reward_mixers:
            raise ValueError(f"Unknown reward_strategy={self.reward_strategy}")

    def set_step(self, step: int) -> None:
        self.cur_step = int(step)

    def _phi_weight_schedule(self, cur_step: int) -> float:
        if cur_step < self.phi_warmup:
            return 0.0
        t = cur_step - self.phi_warmup
        if self.phi_ramp_steps > 0:
            w = self.phi_weight_max * min(1.0, t / float(self.phi_ramp_steps))
        else:
            w = self.phi_weight_max
        if (self.phi_decay_start is not None) and (cur_step >= self.phi_decay_start):
            w = w * (self.phi_decay_gamma ** (cur_step - self.phi_decay_start))
        return float(w)

    @staticmethod
    def _entropy_gate(pv: Tensor, active_before: Tensor, ent_lo: float, ent_hi: float) -> Tensor:
        p = pv.clamp(1e-6, 1.0 - 1e-6)
        ent = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        gate = (ent - float(ent_lo)) / max(1e-8, float(ent_hi) - float(ent_lo))
        gate = gate.clamp(0.0, 1.0)
        return gate * active_before.to(gate.dtype)

    @staticmethod
    def token_entropy_gate_from_batch_normalized(
        gen_tokens: Tensor,  # (B,T) int64
        active_before: Tensor,  # (B,T) bool
        ent_lo: float = 0.15,
        ent_hi: float = 0.75,
        eps: float = 1e-12,
    ) -> Tensor:
        """
        Normalized batch token-frequency entropy gate.
        gate[t] = normalize_entropy( token_freq_at_t ) mapped to [0,1].
        """
        B, T = gen_tokens.shape
        device = gen_tokens.device
        gate_t = torch.zeros((T,), device=device, dtype=torch.float32)

        for t in range(T):
            m = active_before[:, t]
            n_active = int(m.sum().item())
            if n_active <= 1:
                gate_t[t] = 0.0
                continue

            toks = gen_tokens[m, t]
            uniq, cnt = toks.unique(return_counts=True)
            U = int(uniq.numel())
            if U <= 1:
                gate_t[t] = 0.0
                continue

            q = cnt.to(torch.float32) / cnt.sum().to(torch.float32)
            H = -(q * (q + eps).log()).sum()  # nats
            H_norm = H / (torch.log(torch.tensor(float(U), device=device)) + eps)  # in [0,1]

            gate = (H_norm - float(ent_lo)) / max(1e-8, float(ent_hi) - float(ent_lo))
            gate_t[t] = gate.clamp(0.0, 1.0)

        gate = gate_t.view(1, T).expand(B, T)
        return gate * active_before.to(torch.float32)

    # ----------------------------- Reward mixers -----------------------------

    def _mix_reward_legacy(
        self,
        *,
        reference_logP: Tensor,  # (B, L_state)
        local_score: Tensor,  # (B, L_state)
        scaling_factor: float,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        reward = reference_logP + |reference_logP_last| * local_score + scaling_factor * local_score
        """
        y = local_score[:, -1].clamp(0.0, 1.0)

        reward_mixed = (
            reference_logP
            + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * local_score
            + float(scaling_factor) * local_score
        )
        return reward_mixed, y

    def _mix_reward_smiles_absorbing(
        self,
        *,
        reference_logP: Tensor,  # (B, L_state)
        local_score: Tensor,  # (B, L_state) prefix score, invalid -> -1 (or any sentinel)
        scaling_factor: float,  # beta
        gen_tokens: Tensor,  # (B, T_tok) prompt后 tokens（用于 eos 判定）
        eos: int,  # eos token id
        active_before: Tensor,  # (B, T_tok) bool, 你定义的 compute_active_before 输出
        validator_dict: dict[str, Any] | None = None,  # 可留作未来扩展，这里不依赖
    ) -> tuple[Tensor, Tensor]:
        dtype = reference_logP.dtype
        device = reference_logP.device

        B, L_state = reference_logP.shape
        T_tok = L_state - 1

        assert gen_tokens.shape == (
            B,
            T_tok,
        ), f"gen_tokens {gen_tokens.shape} vs (B,T)={(B,T_tok)}"
        assert active_before.shape == (
            B,
            T_tok,
        ), f"active_before {active_before.shape} vs (B,T)={(B,T_tok)}"
        assert local_score.shape == (
            B,
            T_tok + 1,
        ), f"local_score {local_score.shape} vs (B,L)={(B, T_tok+1)}"

        # -------------------------
        # 1) L_term: #non-eos tokens before termination (first EOS semantics)
        # -------------------------
        non_eos = gen_tokens.ne(int(eos))  # (B, T_tok)
        len_mask = active_before & non_eos  # (B, T_tok)
        L_term = len_mask.sum(dim=1).to(torch.long)  # (B,) in [0, T_tok]

        # y: score at termination-length state (NO clip)
        y = local_score.gather(1, L_term.view(B, 1)).squeeze(1).to(dtype)  # (B,)

        # -------------------------
        # 2) active_state: reachable state mask (B, L_state)
        #   state 0 always reachable
        #   state (t+1) reachable iff active_before[t] is True
        # -------------------------
        active_state = torch.cat(
            [torch.ones((B, 1), device=device, dtype=dtype), active_before.to(dtype)],
            dim=1,
        )  # (B, L_state)

        # -------------------------
        # 3) tau_state: terminal state index (includes EOS action)
        #    sum(active_before) == first_eos_pos+1 ; if no eos => T_tok
        # -------------------------
        tau_state = active_before.to(torch.long).sum(dim=1).clamp(0, L_state - 1)  # (B,)

        # -------------------------
        # 4) score_state: per-state score, absorbing after tau_state
        # -------------------------
        score_state = local_score.to(dtype)  # (B, L_state)

        pos = torch.arange(L_state, device=device).view(1, L_state)  # (1, L_state)
        fill_after_tau = pos >= tau_state.view(B, 1)  # (B, L_state)
        score_state = torch.where(fill_after_tau, y.view(B, 1), score_state)  # absorb after tau

        score_state = score_state * active_state

        # -------------------------
        # 5) optional length prior
        # -------------------------
        if getattr(self, "smiles_len_weight", 0.0) != 0.0:
            len_state = pos.to(dtype) / float(max(1, T_tok))  # 0..1
            length_term = (-float(self.smiles_len_weight)) * len_state
            length_term = length_term * active_state
        else:
            length_term = 0.0

        beta = float(scaling_factor)
        reward_mixed = reference_logP + beta * score_state + length_term

        return reward_mixed, y

    # ----------------------------- Main score -----------------------------

    @torch.no_grad()
    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer=None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        eos = tokenizer.eos_token_id
        reference_results = eval(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=eos,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
        )

        reference_logP = reference_results["reward"] * float(
            reference_logits_scale
        )  # (B, L_state)
        ref_log_pf = reference_results["ref_log_pf"]  # (B, T_tok)
        ref_log_pterm = reference_results["ref_log_pterm"]  # (B, L_state)

        validator_dict = None
        reward_mixed = reference_logP
        prefix_value_diag = None

        pv_raw = None
        pv_used = None
        counts = None
        phi_tok = None
        phi_state = None
        dphi = None

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, scaffold
            )
            local_score = validator_dict["local_score"].to(reference_logP.dtype)

            # align to (B, L_state)
            if local_score.shape[1] == reference_logP.shape[1] - 1:
                local_score = torch.cat(
                    [local_score.new_zeros(local_score.shape[0], 1), local_score], dim=1
                )
            assert (
                local_score.shape == reference_logP.shape
            ), f"local_score {local_score.shape} vs reference_logP {reference_logP.shape}"

            # tokens / masks
            B, L_state = reference_logP.shape
            T_tok = L_state - 1
            gen_tokens = input_batch[:, prompt_length : prompt_length + T_tok]  # (B, T_tok)
            active_before = compute_active_before(gen_tokens, eos=eos)  # (B, T_tok) bool

            # ---- reward mixing strategy (mapping) ----
            mixer = self._reward_mixers[self.reward_strategy]
            reward_mixed, y = mixer(
                reference_logP=reference_logP,
                local_score=local_score,
                scaling_factor=float(scaling_factor),
                active_before=active_before,
                validator_dict=validator_dict,
                gen_tokens=gen_tokens,
                eos=eos,
            )

            # ---- PV memory / shaping ----
            self.pv_mem.set_step(self.cur_step)
            phi_w = self._phi_weight_schedule(self.cur_step)

            if phi_w > 0.0:
                device = gen_tokens.device
                p0_vec = self.pv_mem.get_base_rate_vec(
                    T_tok, device=device, dtype=reference_logP.dtype
                )
                p0_mat = p0_vec.view(1, T_tok).expand(B, T_tok)

                # raw pv without backoff
                pv_raw, counts = self.pv_mem.query_pv(gen_tokens, active_before)

                pref_len = (
                    (torch.arange(T_tok, device=device, dtype=torch.long) + 1)
                    .view(1, T_tok)
                    .expand(B, T_tok)
                )
                if self.pv_split_inclusive:
                    short_mask = pref_len <= int(self.pv_split)
                else:
                    short_mask = pref_len < int(self.pv_split)

                pv_used = torch.where(short_mask, 1.0 - pv_raw, pv_raw).clamp(1e-6, 1.0 - 1e-6)
                p0_used = torch.where(
                    short_mask,
                    (1.0 - p0_mat).clamp(1e-6, 1.0 - 1e-6),
                    p0_mat.clamp(1e-6, 1.0 - 1e-6),
                )

                phi_tok = build_prefix_potential(
                    pv=pv_used,
                    ref_log_pf=ref_log_pf,
                    non_term_mask=active_before,
                    counts=counts,
                    eta=self.phi_eta,
                    clamp=self.phi_clamp,
                    tau_conf=self.pv_mem.tau_conf,
                    base_rate=p0_used,
                    center_by_base=True,
                    conf_mode="inv_sqrt",
                )

                # gates (optional; keep your current development status)
                # if self.use_entropy_gate:
                #     g_ent = self._entropy_gate(pv_used, active_before, self.ent_lo, self.ent_hi)
                #     phi_tok = phi_tok * g_ent
                #
                # if self.use_token_entropy_gate:
                #     g_ent = self.token_entropy_gate_from_batch_normalized(
                #         gen_tokens, active_before, self.ent_lo, self.ent_hi
                #     )
                #     phi_tok = phi_tok * g_ent

                reward_mixed, phi_state, dphi = apply_phi_shaping(
                    reward_mixed=reward_mixed,
                    phi_tok=phi_tok,
                    active_before=active_before,
                    phi_weight=phi_w,
                    mode="differential",
                    anchor_start=0.0,
                    anchor_end=0.0,
                    dphi_clip=self.dphi_clip,
                )

                # ---- diagnostics ----
                mask = active_before.to(pv_raw.dtype)
                den = mask.sum().clamp_min(1.0)

                short_f = short_mask.to(mask.dtype) * mask
                long_f = (1.0 - short_mask.to(mask.dtype)) * mask
                den_s = short_f.sum().clamp_min(1.0)
                den_l = long_f.sum().clamp_min(1.0)

                def mmean(x: Tensor, m: Tensor, d: Tensor) -> float:
                    return float((x * m).sum().div(d).item())

                ent_raw = -(pv_raw * pv_raw.log() + (1.0 - pv_raw) * (1.0 - pv_raw).log())
                ent_used = -(pv_used * pv_used.log() + (1.0 - pv_used) * (1.0 - pv_used).log())
                sat_used = ((pv_used < 0.05) | (pv_used > 0.95)).to(mask.dtype)

                prefix_value_diag = {
                    "reward_strategy": float(0.0 if self.reward_strategy == "legacy" else 1.0),
                    "pv_split": float(self.pv_split),
                    "short_frac": float(short_f.sum().div(den).item()),
                    "pv_raw_mean": mmean(pv_raw, mask, den),
                    "pv_used_mean": mmean(pv_used, mask, den),
                    "pv_raw_short_mean": mmean(pv_raw, short_f, den_s),
                    "pv_used_short_mean": mmean(pv_used, short_f, den_s),
                    "pv_raw_long_mean": mmean(pv_raw, long_f, den_l),
                    "pv_used_long_mean": mmean(pv_used, long_f, den_l),
                    "entropy_raw_mean": mmean(ent_raw, mask, den),
                    "entropy_used_mean": mmean(ent_used, mask, den),
                    "counts_mean": mmean(counts, mask, den),
                    "counts_short_mean": mmean(counts, short_f, den_s),
                    "counts_long_mean": mmean(counts, long_f, den_l),
                    "phi_abs_mean": mmean(phi_tok.abs(), mask, den),
                    "dphi_abs_mean": mmean(dphi.abs(), mask, den),
                    "sat_ratio_used": mmean(sat_used, mask, den),
                    "p0_len_mean": float(
                        (p0_mat * mask).sum().div(mask.sum().clamp_min(1)).item()
                    ),
                    "p0_len_short_mean": float((p0_mat * short_f).sum().div(den_s).item()),
                    "p0_len_long_mean": float((p0_mat * long_f).sum().div(den_l).item()),
                }

            # update memory after scoring, disable when phi_weight is 0
            if self.phi_weight_max > 0.0:
                self.pv_mem.update(gen_tokens, y, active_before)

        out = {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
            "prefix_value_diag": prefix_value_diag,
            "phi_state": phi_state,
            "phi_tok": phi_tok,
            "pv_raw": pv_raw,
            "pv": pv_used,
            "phi_weight": phi_w,
            "counts": counts,
        }
        return out


class Reference_Target_Score_Positive_Mixed_Invalid_Mask_PrefixShaping_TestInvalid_MASK:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 0,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)
        self.phi_weight = phi_weight
        self.score_function = score_function

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        model=None,
        tokenizer=None,
        reward_temperature: float = 1.0,
        vocab_invalid_mask=None,
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        scaffold: str | None = None,
        agree_list: Tensor | None = None,
        action_seq: Tensor | None = None,  # (B,T) 生成的动作 token（不含 prompt）
        **kwargs,
    ) -> dict[str, Any]:
        reference_results = eval(self.score_function)(
            model=model,
            encoded_input=input_batch,
            termination_token_id=tokenizer.eos_token_id,
            skip_first=prompt_length,
            reward_temperature=reward_temperature,
            invalid_vocab_mask=vocab_invalid_mask,
            agree_list=agree_list,
            illegal_vocab_penalty=self.illegal_vocab_penalty,
            grammar_disagree_penalty=self.grammar_disagree_penalty,
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
            local_score = validator_dict["local_score"].to(
                reference_logP.dtype
            )  # (B,T), 000..1 or 000..0

            y = local_score[:, -1]  # (B,)

            reward_mixed = reference_logP + (-50) * (1 - local_score)

            # sentences = input_batch[:, prompt_length:]
            # non_term_mask = (input_batch != tokenizer.eos_token_id)[:, prompt_length:]

            # pv = batch_prefix_value_kgram(sentences, y, k=6, alpha=1.0, min_count=2, backoff=True)
            # phi = build_prefix_potential(pv, ref_log_pf, non_term_mask, eta=1.0, clamp=4.0)
            # reward_mixed[:, 1:] = reward_mixed[:, 1:] + phi * self.phi_weight

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


class UniformModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_validator: SentenceValidator | None,
        invalid_start_ratio: float = 0.2,
        invalid_end_ratio: float = 1.2,
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.invalid_start_ratio = invalid_start_ratio
        self.invalid_end_ratio = invalid_end_ratio

    def score(
        self,
        input_batch: Tensor,
        prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        reward_temperature: float = 1.0,
        termination_token_id: int = -1,
        agree_list: Tensor | None = None,
        min_len: int = 10,
        **kwargs,
    ) -> ScorePair:
        if agree_list is None:
            raise ValueError("agree_list must be provided for uniform model scoring.")

        sum_along_dim = agree_list.sum(dim=-1, keepdim=True)
        uniform_base = torch.where(agree_list, 1.0 / sum_along_dim.clamp(min=1e-6), 0).detach()

        first_position_counts = sum_along_dim[0, :, :].squeeze(-1)
        log_start = torch.log((1 / first_position_counts).clamp(min=1e-6))
        seq_len = input_batch.size(1) - prompt_length
        steps = torch.linspace(0, 1, seq_len + 1, device=input_batch.device)
        log_space = log_start[None, :] * (1 - steps[:, None])
        length_increase = torch.exp(log_space).unsqueeze(-1)

        eos_mask = torch.zeros_like(agree_list)
        eos_mask[..., termination_token_id] = agree_list[..., termination_token_id]

        uniform_probs = uniform_base * (~eos_mask)
        eos_probs = torch.log_softmax(uniform_base * eos_mask * length_increase, dim=-1)
        logprob = torch.log_softmax(uniform_probs, dim=-1)

        token_ids = input_batch[:, prompt_length:].unsqueeze(-1)
        log_pf = logprob[1:].gather(-1, token_ids.transpose(0, 1)).squeeze(-1)
        log_p = log_pf.cumsum(dim=0)

        reward = eos_probs[..., termination_token_id]
        reward[1:,] += log_p

        non_term_mask = (input_batch != termination_token_id)[:, prompt_length:]
        non_term_mask = torch.cat(
            (
                non_term_mask.new_ones(non_term_mask.shape[0], 1),
                non_term_mask,
            ),
            dim=-1,
        )

        reward_unpenalized = reward.clone()

        if self.sentence_validator is not None:
            invalid_mask = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)[
                "invalid"
            ]
            reward = reward.clone()

            reward_min = torch.min(reward, dim=-1).values
            start_values = reward_min * self.invalid_start_ratio
            end_values = reward_min * self.invalid_end_ratio

            ramp = []
            for i in range(reward.shape[0]):
                seq = torch.linspace(
                    start=start_values[i],
                    end=end_values[i],
                    steps=reward.shape[1],
                    device=reward.device,
                )
                ramp.append(seq)
            invalid_value = torch.stack(ramp, dim=0)
            reward = torch.min(reward.permute(1, 0), invalid_value.permute(1, 0) * invalid_mask)

        return reward, reward_unpenalized.permute(1, 0)
