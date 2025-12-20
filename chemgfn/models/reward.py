from __future__ import annotations

import math
import re
from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import partialsmiles as ps
import torch

# slient the warning
from rdkit import Chem, RDLogger
from torch import Tensor
from transformers import PreTrainedTokenizer

from chemgfn.models.mcmc_prior import (
    _get_q_vec_with_backoff,
    load_q_mcmc,
    pack_q_mcmc_to_device,
)
from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base
from chemgfn.utils.phi_utils import compute_prefix_diagnostics
from chemgfn.utils.rdkit_utils import FUNCTION_MAPPING, verify_smiles, verify_smiles_pa

RDLogger.DisableLog("rdApp.*")

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


def _decode_tokens_to_string(sequence: Tensor, tokenizer: PreTrainedTokenizer) -> str:
    pieces = []
    for token in sequence:
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        pieces.append(tokenizer.decode(token, skip_special_tokens=False))
    return "".join("".join(pieces).split())


def _merge_target(template: str | None, fragment: str) -> str:
    return fragment if template is None else template.replace("*", fragment)


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
# Validators
# --------------------------------------------------------------------------- #


class SentenceValidator:
    def __init__(self, termination_token_id: int = -1) -> None:
        self.termination_token_id = termination_token_id

    def __call__(self, sentences: Tensor, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        raise NotImplementedError


class Expr24Validator(SentenceValidator):
    """
    24-points validator (CFG-guaranteed valid prefixes).
    Format: d op d op d op d, where d ∈ {0..9}, op ∈ {+,-,*,/}, no parentheses.
    Evaluation: standard precedence (*/ before +-). Equals 24 -> 1, else 0.
    """

    TOKEN_RE = re.compile(r"[0-9]|[+\-*/]")

    def __init__(self, scorer: str = "hit24", amortize_valid_state: bool = False) -> None:
        super().__init__(scorer)

        # Whether to amortize the valid state of the expression to the entire batch
        self.amortize_valid_state = amortize_valid_state

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _decode_expr(self, tokens: Tensor, tokenizer: PreTrainedTokenizer) -> str | None:
        try:
            decoded = _decode_tokens_to_string(tokens, tokenizer)
        except Exception:
            return None
        decoded = decoded.strip()
        return decoded or None

    def expression_accuracy(
        self,
        generated_tokens: Tensor,
        tokenizer: PreTrainedTokenizer,
    ) -> dict[str, float]:
        if generated_tokens is None or generated_tokens.ndim == 0:
            return {"acc": 0.0}
        total = generated_tokens.shape[0]
        if total == 0:
            return {"acc": 0.0}

        hits = 0
        for sample in generated_tokens:
            expr = self._decode_expr(sample, tokenizer)
            if expr is None:
                continue
            hits += self._eval_full_expr_to_01(expr)
        return {"acc": hits / total}

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
        **kwargs,
    ) -> dict[str, float]:
        return self.expression_accuracy(sentences, tokenizer)

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        if sentences is None or sentences.ndim < 1:
            return {
                "invalid": torch.zeros(1, 1),
                "global_score": torch.zeros(1),
                "valid_score": torch.zeros(1, 1),
                "full_tokens": [],
            }

        termination_token_id = tokenizer.eos_token_id
        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        valid_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.zeros(batch_size, device=device)
        full_tokens_list: list[str] = []

        invalid[:, 0] = 1.0  # empty prefix

        for i in range(batch_size):
            for pos in range(seq_len):
                if sentences[i, pos] == termination_token_id:
                    break

            final_expr = self._decode_expr(sentences[i], tokenizer)
            if final_expr is None:
                full_tokens_list.append("")
                continue
            score = self._eval_full_expr_to_01(final_expr)

            valid_score[i, -1] = float(score)
            global_score[i] = float(score)
            full_tokens_list.append(final_expr)

        if self.amortize_valid_state:
            # Vectorized: create mask for entries where global_score > 0
            mask = global_score > 0
            valid_score[mask, 1:] = 1.0

        return {
            "invalid": None,
            "global_score": global_score,  # 0 or 1
            "valid_score": valid_score,  # placeholder for future shaping
            "full_tokens": full_tokens_list,
        }

    def _eval_full_expr_to_01(self, s: str) -> int:
        s = s.replace("×", "*").replace("÷", "/")
        s = "".join(s.split())
        toks = self.TOKEN_RE.findall(s)
        if "".join(toks) != s or len(toks) != 7:
            return 0
        for k, tk in enumerate(toks):
            if k % 2 == 0:
                if not (len(tk) == 1 and tk.isdigit()):
                    return 0
            else:
                if tk not in "+-*/":
                    return 0

        nums = [Fraction(int(toks[i])) for i in (0, 2, 4, 6)]
        ops = [toks[i] for i in (1, 3, 5)]

        try:
            v = nums[:]
            o = ops[:]
            i = 0
            while i < len(o):
                if o[i] in "*/":
                    a, b = v[i], v[i + 1]
                    if o[i] == "*":
                        res = a * b
                    else:
                        if b == 0:
                            return 0
                        res = a / b
                    v[i : i + 2] = [res]
                    o.pop(i)
                else:
                    i += 1
            acc = v[0]
            for op, b in zip(o, v[1:]):
                acc = acc + b if op == "+" else acc - b
            return 1 if acc == 24 else 0
        except Exception:
            return 0


class RDKitValidator(SentenceValidator):
    def __init__(self, scorer: str = "sa", backend: Literal["rdkit", "pa"] = "rdkit") -> None:
        super().__init__(scorer)
        self.score_function = FUNCTION_MAPPING[scorer]
        self.backend = backend
        self.scorer_name = scorer

    @staticmethod
    def rdkit_validate(smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    @staticmethod
    def pa_validate(smiles: str) -> bool:
        try:
            ps.ParseSmiles(smiles)
            return True
        except Exception:
            return False

    def _is_valid_smiles(self, smiles: str) -> bool:
        if self.backend == "rdkit":
            return self.rdkit_validate(smiles)
        if self.backend == "pa":
            return self.pa_validate(smiles)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _decode_batch(self, generated_tokens: Tensor, tokenizer: PreTrainedTokenizer) -> list[str]:
        return [_decode_tokens_to_string(sample, tokenizer) for sample in generated_tokens]

    def smiles_accuracy(
        self,
        generated_tokens: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
    ) -> dict[str, float]:
        decoded = self._decode_batch(generated_tokens, tokenizer)
        scores = []
        valid_flags = []

        for tokens in decoded:
            candidate = _merge_target(target_molecule, tokens)
            if self._is_valid_smiles(candidate):
                mol = Chem.MolFromSmiles(candidate)
                valid_flags.append(bool(mol))
                scores.append(self.score_function(mol) if mol else 0.0)
            else:
                valid_flags.append(False)
                scores.append(0.0)

        total_valid = sum(valid_flags)
        num_samples = generated_tokens.shape[0]
        avg_score = sum(scores) / num_samples if num_samples else 0.0
        filtered_score = (
            sum(score for score, flag in zip(scores, valid_flags) if flag) / total_valid
            if total_valid
            else 0.0
        )

        return {
            "acc": total_valid / num_samples if num_samples else 0.0,
            f"{self.scorer_name}": avg_score,
            f"{self.scorer_name}_filter": filtered_score,
        }

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
    ) -> dict[str, float]:
        return self.smiles_accuracy(sentences, tokenizer, target_molecule)

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
    ) -> dict[str, Tensor]:
        termination_token_id = tokenizer.eos_token_id
        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        valid_score = torch.full((batch_size, seq_len + 1), -1.0, device=device)
        valid_score[:, 0] = 0.0
        global_score = torch.zeros(batch_size, device=device)
        full_tokens_list = []

        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                if sentences[batch_idx, pos] == termination_token_id:
                    break

                prefix = _decode_tokens_to_string(sentences[batch_idx, : pos + 1], tokenizer)
                candidate = _merge_target(target_molecule, prefix)
                valid = self._is_valid_smiles(candidate)

                if valid:
                    mol = Chem.MolFromSmiles(candidate)
                    valid_score[batch_idx, pos + 1] = self.score_function(mol) if mol else -1.0
                    invalid[batch_idx, pos + 1] = 0.0
                else:
                    invalid[batch_idx, pos + 1] = 1.0

            final_tokens = _decode_tokens_to_string(sentences[batch_idx], tokenizer)
            full_smiles = _merge_target(target_molecule, final_tokens)
            try:
                mol = Chem.MolFromSmiles(full_smiles)
                global_score[batch_idx] = self.score_function(mol) if mol else 0.0
            except Exception:
                global_score[batch_idx] = 0.0
            full_tokens_list.append(full_smiles)

        return {
            "invalid": invalid,
            "global_score": global_score,
            "valid_score": valid_score,
            "full_tokens": full_tokens_list,
        }


def number_reward(sampled_numbers: Iterable[int | str]) -> int:
    values = [int(number) for number in sampled_numbers]
    if not all(values[i] % 2 != values[i + 1] % 2 for i in range(len(values) - 1)):
        return 0
    return 1


class BracketValidator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.bracket_map = {")": "(", "]": "[", ">": "<"}
        self.left_brackets = set(self.bracket_map.values())
        self.right_brackets = set(self.bracket_map.keys())

    def preprocess(self, tokens: Tensor) -> tuple[str, bool]:
        decoded = []
        total_length = len(tokens)
        for idx, token in enumerate(tokens):
            if token.item() == self.tokenizer.eos_token_id:
                break
            decoded.append(self.tokenizer.decode(token))
        decoded_string = "".join(decoded)
        return decoded_string, total_length > idx

    def is_valid(self, tokens: Tensor) -> bool:
        sequence, _ = self.preprocess(tokens)
        stack = []
        for char in sequence:
            if char in self.left_brackets:
                stack.append(char)
            elif char in self.right_brackets:
                if not stack or stack[-1] != self.bracket_map[char]:
                    return False
                stack.pop()
            else:
                return False
        return not stack

    def is_valid_prefix(self, tokens: Tensor) -> bool:
        sequence, _ = self.preprocess(tokens)
        stack = []
        for char in sequence:
            if char in self.left_brackets:
                stack.append(char)
            elif char in self.right_brackets:
                if not stack or stack[-1] != self.bracket_map[char]:
                    return False
                stack.pop()
            else:
                return False
        return True

    def has_multiple_nesting(self, tokens: Tensor) -> bool:
        sequence, _ = self.preprocess(tokens)
        stack = []
        has_nesting = False
        for char in sequence:
            if char in self.left_brackets:
                stack.append(char)
            elif char in self.right_brackets:
                if not stack or stack[-1] != self.bracket_map[char]:
                    return False
                stack.pop()
                if len(stack) >= 1:
                    has_nesting = True
            else:
                return False
        return not stack and has_nesting


def number_accuracy(generated_tokens: Tensor, tokenizer: PreTrainedTokenizer) -> float:
    correct = 0
    for sample in generated_tokens:
        decoded = []
        for token in sample:
            if token.item() == tokenizer.eos_token_id:
                break
            decoded.append(int(tokenizer.decode(token)))
        correct += number_reward(decoded)
    return correct / generated_tokens.shape[0]


class NumberValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def accuracy(self, sentences: Tensor, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
        return {"acc": number_accuracy(sentences, tokenizer)}

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        termination_token_id = tokenizer.eos_token_id
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        invalid[:, 0] = 1

        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    break
                tokens = [tokenizer.decode(t) for t in sentences[i, : j + 1]]
                numbers = [int(number) for number in tokens]
                invalid[i, j + 1] = 0 if number_reward(numbers) else 1

        return {"invalid": invalid}


class ParenthesesValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def accuracy(self, sentences: Tensor, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
        validator = BracketValidator(tokenizer)
        correct = 0
        nested = 0
        for sample in sentences:
            valid = validator.is_valid(sample)
            correct += valid
            nested += int(validator.has_multiple_nesting(sample))

        total = sentences.shape[0]
        return {"acc": correct / total, "nest": nested / total}

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        termination_token_id = tokenizer.eos_token_id
        validator = BracketValidator(tokenizer)

        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            device=sentences.device,
        )
        invalid[:, 0] = 1

        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == termination_token_id:
                    break
                invalid[i, j + 1] = 0 if validator.is_valid(sentences[i, : j + 1]) else 1

        return {"invalid": invalid}


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
        target_molecule: str | None = None,
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
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"]
            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * valid_score
                + scaling_factor * valid_score
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
        target_molecule: str | None = None,
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
                input_batch[:, prompt_length:], tokenizer, target_molecule
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
        target_molecule: str | None = None,
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
                input_batch[:, prompt_length:], tokenizer, target_molecule
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


class Reference_Target_Score_Positive_Mixed_Invalid_Mask_Entropy:
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
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        target_molecule: str | None = None,
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
        reference_logP = reference_results["reward"]
        ref_log_pf = reference_results["ref_log_pf"]
        ref_log_pterm = reference_results["ref_log_pterm"]

        # Add entropy over the generated sentence tokens to encourage diversity
        sentences = input_batch[:, prompt_length:]
        entropies: list[Tensor] = []
        eos_token_id = tokenizer.eos_token_id
        for seq in sentences:
            # Trim at EOS to avoid padding impact
            eos_positions = (seq == eos_token_id).nonzero(as_tuple=False)
            end_idx = eos_positions[0, 0].item() if eos_positions.numel() > 0 else seq.size(0)
            trimmed = seq[:end_idx]
            unique_tokens, counts = torch.unique(trimmed, return_counts=True)
            probs = counts.float() / counts.sum().clamp(min=1e-12)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12))
            entropies.append(entropy)
        entropy_tensor = torch.stack(entropies, dim=0).unsqueeze(-1)

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"]
            # set valid sample's logR[-1] to 0 + fixed reward + entropy reward
            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * valid_score
                + scaling_factor * valid_score
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


def batch_prefix_value_kgram(
    sentences: torch.Tensor,  # (B,T) 生成部分 tokens（不含 prompt）
    y: torch.Tensor,  # (B,) 终局 0/1
    k: int = 6,
    alpha: float = 1.0,
    min_count: int = 2,
    backoff: bool = True,
) -> torch.Tensor:
    """
    返回 pv: (B,T), pv[i,t] ≈ P(y=1 | suffix-kgram(prefix_{i,t}))
    用 batch 内统计，Beta-Binomial 平滑。
    """
    B, T = sentences.shape
    y = y.to(torch.float32).view(B)

    # counts[(len, tuple)] = (N,S)
    N = defaultdict(float)
    S = defaultdict(float)

    toks_list = sentences.tolist()
    for i in range(B):
        yi = float(y[i].item())
        toks = toks_list[i]
        for t in range(T):
            for kk in range(1, k + 1):
                s = max(0, t - kk + 1)
                key = (kk, tuple(toks[s : t + 1]))
                N[key] += 1.0
                S[key] += yi

    pv = torch.empty((B, T), device=sentences.device, dtype=torch.float32)

    for i in range(B):
        toks = toks_list[i]
        for t in range(T):
            chosen = None
            # 优先用最长 k-gram；太稀疏就 backoff
            for kk in range(k, 0, -1):
                s = max(0, t - kk + 1)
                key = (kk, tuple(toks[s : t + 1]))
                if (not backoff) or (N[key] >= min_count) or kk == 1:
                    chosen = key
                    break
            n = N[chosen]
            ssum = S[chosen]
            pv[i, t] = (ssum + alpha) / (n + 2.0 * alpha)

    return pv.clamp(1e-4, 1.0 - 1e-4)


class PrefixValueMemory:
    """
    Cross-batch EMA Beta-Binomial PV estimator.

    Keeps EMA counts (N,S) for suffix k-grams across training steps.
    - update(): uses (sentences, y) to update global stats
    - query_pv(): returns pv(B,T) and decayed counts using backoff from kmax..1
    """

    def __init__(
        self,
        kmax: int = 4,  # 建议先用4，6太稀疏
        alpha: float = 1.0,  # Beta prior
        gamma: float = 0.999,  # EMA decay (~1000 steps memory)
        min_count: float = 10.0,  # N>=min_count 才信这个key
        max_keys: int = 500_000,  # 内存上限，够用
        prune_every: int = 2000,
        prune_threshold: float = 1.0,
        tau_conf: float = 20.0,  # 信心权重平滑
    ):
        self.kmax = int(kmax)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.min_count = float(min_count)

        self.max_keys = int(max_keys)
        self.prune_every = int(prune_every)
        self.prune_threshold = float(prune_threshold)
        self.tau_conf = float(tau_conf)

        # stats[(k, tuple(tokens))] = [N, S, last_step]
        self.stats: dict[tuple[int, tuple[int, ...]], list[float]] = {}
        self.global_N = 0.0
        self.global_S = 0.0
        self.global_last_step = 0
        self.step = 0

    def set_step(self, step: int) -> None:
        """Update current step (non-decreasing) for decay bookkeeping."""
        step = int(step)
        if step > self.step:
            self.step = step
            self._decay_global()

    def _decay_global(self) -> None:
        delta = max(0, self.step - self.global_last_step)
        if delta > 0:
            factor = self.gamma**delta
            self.global_N *= factor
            self.global_S *= factor
            self.global_last_step = self.step

    def _decay_key(self, key: tuple[int, tuple[int, ...]]) -> tuple[float, float]:
        N, S, last = self.stats[key]
        if self.step > last:
            factor = self.gamma ** (self.step - last)
            N *= factor
            S *= factor
            self.stats[key] = [N, S, float(self.step)]
        return N, S

    @torch.no_grad()
    def update(self, sentences: torch.Tensor, y: torch.Tensor, non_term_mask: torch.Tensor):
        """
        sentences: (B,T) tokens (no prompt)
        y: (B,) float in {0,1}
        non_term_mask: (B,T) bool, only count positions where True
        """
        B, T = sentences.shape
        y = y.to(torch.float32).view(B)

        toks_list = sentences.detach().to("cpu").tolist()
        mask_list = non_term_mask.detach().to("cpu").tolist()
        y_list = y.detach().to("cpu").tolist()

        batch_N = defaultdict(float)
        batch_S = defaultdict(float)

        n_global = 0.0
        s_global = 0.0

        for i in range(B):
            yi = float(y_list[i])
            toks = toks_list[i]
            msk = mask_list[i]
            for t in range(T):
                if not msk[t]:
                    continue
                n_global += 1.0
                s_global += yi
                for kk in range(1, self.kmax + 1):
                    s = max(0, t - kk + 1)
                    key = (kk, tuple(toks[s : t + 1]))
                    batch_N[key] += 1.0
                    batch_S[key] += yi

        # EMA update global base rate
        self.global_N = self.global_N + n_global
        self.global_S = self.global_S + s_global
        self.global_last_step = self.step

        # EMA update per-key
        for key, n in batch_N.items():
            s = batch_S[key]
            if key in self.stats:
                N, S = self._decay_key(key)
                N = N + n
                S = S + s
                self.stats[key] = [N, S, float(self.step)]
            else:
                self.stats[key] = [n, s, float(self.step)]

        if (self.step % self.prune_every) == 0:
            self._prune()

    def _prune(self):
        # 1) drop tiny N
        to_del = []
        for k in list(self.stats.keys()):
            N, S, last = self.stats[k]
            if self.step > last:
                factor = self.gamma ** (self.step - last)
                N *= factor
                S *= factor
                self.stats[k] = [N, S, float(self.step)]
            if N < self.prune_threshold:
                to_del.append(k)
        for k in to_del:
            del self.stats[k]

        # 2) cap by recency
        if len(self.stats) > self.max_keys:
            items = sorted(self.stats.items(), key=lambda kv: kv[1][2], reverse=True)
            self.stats = dict(items[: self.max_keys])

    @torch.no_grad()
    def query_pv(
        self, sentences: torch.Tensor, non_term_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns pv, counts: both (B,T) float32
        """
        B, T = sentences.shape
        toks_list = sentences.detach().to("cpu").tolist()
        mask_list = non_term_mask.detach().to("cpu").tolist()

        # fallback base rate
        if self.global_N > 0:
            p0 = (self.global_S + self.alpha) / (self.global_N + 2.0 * self.alpha)
        else:
            p0 = 0.5
        p0 = float(min(max(p0, 1e-4), 1.0 - 1e-4))

        pv = torch.empty((B, T), dtype=torch.float32, device=sentences.device)
        counts = torch.zeros((B, T), dtype=torch.float32, device=sentences.device)

        for i in range(B):
            toks = toks_list[i]
            msk = mask_list[i]
            for t in range(T):
                if not msk[t]:
                    pv[i, t] = 1e-4
                    counts[i, t] = 0.0
                    continue

                chosen_p = None
                chosen_n = 0.0
                # backoff: kmax -> 1
                for kk in range(self.kmax, 0, -1):
                    s = max(0, t - kk + 1)
                    key = (kk, tuple(toks[s : t + 1]))
                    if key in self.stats:
                        N, S = self._decay_key(key)
                        if (N >= self.min_count) or (kk == 1):
                            chosen_p = (S + self.alpha) / (N + 2.0 * self.alpha)
                            chosen_n = N
                            break

                if chosen_p is None:
                    chosen_p = p0
                    chosen_n = 0.0

                chosen_p = float(min(max(chosen_p, 1e-4), 1.0 - 1e-4))
                pv[i, t] = chosen_p
                counts[i, t] = float(chosen_n)

        return pv, counts


def build_prefix_potential(
    pv: torch.Tensor,  # (B,T) in (0,1)
    ref_log_pf: torch.Tensor,  # (B,T) 每步 token logprob（来自 score_fast 返回）
    non_term_mask: torch.Tensor,  # (B,T) True 表示还没终止的位置（你可以用 score_fast 里的 mask）
    counts: torch.Tensor | None = None,  # (B,T) decayed counts for confidence
    eta: float = 1.0,
    clamp: float = 2.0,
    tau_conf: float = 20.0,
) -> torch.Tensor:
    """
    生成势函数 Phi(prefix_t)，并锚定终点 Phi[:, -1]=0（避免直接改终局尺度）。
    eta 控制强度；clamp 防止爆炸。
    """
    # logit(p)
    logit = torch.log(pv) - torch.log1p(-pv)
    logit = logit.clamp(-clamp, clamp)  # 让它成为“温和的信号”

    # 用每步 logprob 的平均幅度来对齐尺度（典型 1~5）
    step_scale = ref_log_pf.abs().mean(dim=-1, keepdim=True).clamp_min(0.5)  # (B,1)
    phi = eta * step_scale * logit  # (B,T)

    if counts is not None:
        conf = counts.to(phi.dtype) / (counts.to(phi.dtype) + float(tau_conf))
        phi = phi * conf.clamp(0.0, 1.0)

    # 只在未终止位置有效
    phi = phi * non_term_mask.to(phi.dtype)

    # 关键：锚定终点（不直接改变终局 logR 的绝对值）
    phi[:, -1] = 0.0
    return phi


@torch.no_grad()
def apply_potential_shaping(reward_mixed: torch.Tensor, phi: torch.Tensor, phi_weight: float):
    """
    Apply differential potential shaping to reward.

    reward_mixed: (B,T+1) or (B,T) depending on caller; we align with reward[:, 1:].
    """
    # 差分：shaping_t = phi_t - phi_{t+1}
    shaping = phi[:, :-1] - phi[:, 1:]  # (B,T-1)

    # 对齐 reward 索引：reward_mixed[:, 1:] 对应 prefix_t，差分长度减 1
    reward_mixed[:, 1 : 1 + shaping.shape[1]] += phi_weight * shaping
    return reward_mixed


class Reference_Target_Score_Positive_Mixed_Invalid_Mask_PrefixShaping:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
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
        target_molecule: str | None = None,
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
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"].to(
                reference_logP.dtype
            )  # (B,T), 000..1 or 000..0

            y = valid_score[:, -1].clamp(0.0, 1.0)  # (B,)

            reward_mixed = (
                reference_logP
                + (-1) * (reference_logP[..., -1]).unsqueeze(-1) * valid_score
                + scaling_factor * valid_score
            )

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
        }


class Reference_Target_Score_Positive_Mixed_Prefix_Potential_Differential_Shaping:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
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
        target_molecule: str | None = None,
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

        reference_logP = reference_results["reward"] * float(reference_logits_scale)  # (B, L)
        ref_log_pf = reference_results["ref_log_pf"]  # (B, T_tok)
        ref_log_pterm = reference_results["ref_log_pterm"]  # (B, L)

        validator_dict = None
        reward_mixed = reference_logP

        prefix_diag = None

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"].to(reference_logP.dtype)
            y = valid_score[:, -1].clamp(0.0, 1.0)

            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * valid_score
                + scaling_factor * valid_score
            )

            B, L = reward_mixed.shape
            T_tok = L - 1
            gen_tokens = input_batch[:, prompt_length : prompt_length + T_tok]  # (B, T_tok)

            # active_before[t]：在采样第 t 个 token 之前 state_t 是否仍 active（包含 EOS 那一步）
            # 公式：active_before[:,0]=1；active_before[:,t]=Π_{j< t} 1[token_j != EOS]
            active_before = torch.ones((B, T_tok), device=input_batch.device, dtype=torch.bool)
            if T_tok > 1:
                alive_after_each = (
                    (gen_tokens != eos).to(torch.long).cumprod(dim=1).to(torch.bool)
                )  # Π_{j<=t}
                active_before[:, 1:] = alive_after_each[:, :-1]  # Π_{j< t}

            # ========= prefix value -> phi_tok（注意：不要用 token!=EOS 的 non_term_mask） =========
            pv = batch_prefix_value_kgram(
                gen_tokens, y, k=6, alpha=1.0, min_count=2, backoff=True
            )  # (B, T_tok)

            # build_prefix_potential 里如果会乘 mask，请传 active_before（不要传 token!=EOS）
            # 这样 EOS 那一步的 phi 不会被错误清零
            phi_tok = build_prefix_potential(
                pv=pv,
                ref_log_pf=ref_log_pf,
                non_term_mask=active_before,  # <-- 关键改动
                eta=1.0,
                clamp=4.0,
            )  # (B, T_tok)

            # ========= phi_tok -> phi_state（state轴长度 L） =========
            phi_state = torch.zeros((B, L), device=phi_tok.device, dtype=phi_tok.dtype)
            phi_state[:, 1:] = phi_tok  # phi_state[t+1] 对应 prefix ending at token t

            # 你之前锚定两端：保留（保证“净奖励改变量”是常数，不改最优分布）
            phi_state[:, 0] = 0.0
            phi_state[:, -1] = 0.0

            # ========= differential shaping：Δphi 作用在每个 transition 上 =========
            dphi = phi_state[:, 1:] - phi_state[:, :-1]  # (B, T_tok)
            dphi = dphi * active_before.to(dphi.dtype)  # <-- 关键：mask 用 active_before

            reward_mixed = reward_mixed.clone()
            reward_mixed[:, 1:] = reward_mixed[:, 1:] + self.phi_weight * dphi

            # ========= diagnostics（你已有函数，放到 phi_state/active_before 之后） =========
            prefix_diag = compute_prefix_diagnostics(
                pv=pv,
                phi_state=phi_state,
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
            out["prefix_diag"] = prefix_diag
        return out


class Reference_Target_Score_Positive_Mixed_Invalid_Mask_PrefixShapingWithMemory:
    def __init__(
        self,
        sentence_validator,
        illegal_vocab_penalty: float = -99,
        grammar_disagree_penalty: float = -99,
        phi_weight: float = 1.0,
        phi_warmup: int = 800,
        phi_ramp_steps: int = 800,
        pv_memory_kwargs: dict[str, Any] | None = None,
        score_function: Literal["score_fast", "score_fast_expr24_pterm_last_only"] = "score_fast",
        **kwargs,
    ) -> None:
        self.sentence_validator = sentence_validator
        self.illegal_vocab_penalty = float(illegal_vocab_penalty)
        self.grammar_disagree_penalty = float(grammar_disagree_penalty)
        self.score_function = score_function
        self.pv_mem = PrefixValueMemory(**(pv_memory_kwargs or {}))

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
        target_molecule: str | None = None,
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
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"].to(
                reference_logP.dtype
            )  # (B,T), 000..1 or 000..0

            y = valid_score[:, -1].clamp(0.0, 1.0)  # (B,)

            reward_mixed = (
                reference_logP
                + torch.abs(reference_logP[..., -1]).unsqueeze(-1) * valid_score
                + scaling_factor * valid_score
            )
            sentences = input_batch[:, prompt_length:]
            non_term_mask = (input_batch != tokenizer.eos_token_id)[:, prompt_length:]

            phi_weight = self.phi_weight_schedule(getattr(self, "global_step", 0))
            if phi_weight > 0.0:
                pv, counts = self.pv_mem.query_pv(sentences, non_term_mask)
                phi = build_prefix_potential(
                    pv,
                    ref_log_pf,
                    non_term_mask,
                    counts=counts,
                    eta=1.0,
                    clamp=2.0,
                    tau_conf=self.pv_mem.tau_conf,
                )
                reward_mixed = apply_potential_shaping(reward_mixed, phi, phi_weight)

            # Update memory after querying to avoid same-batch leakage.
            self.pv_mem.update(sentences, y, non_term_mask)

        return {
            "reward": reward_mixed,
            "reward_unpenalized": reference_logP,
            "log_pf_ref": ref_log_pf,
            "log_pterm_ref": ref_log_pterm,
            "validator_dict": validator_dict,
        }


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
        target_molecule: str | None = None,
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
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"].to(
                reference_logP.dtype
            )  # (B,T), 000..1 or 000..0

            y = valid_score[:, -1]  # (B,)

            reward_mixed = reference_logP + (-50) * (1 - valid_score)

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


class Reference_Target_Score_Split_Reward:
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
        scaling_factor: float = 0.5,
        reference_logits_scale: float = 0.5,
        target_molecule: str | None = None,
        agree_list: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        with use_base_model(model, disable_peft=self.disable_peft):
            reference_logits, _ = score_fast(
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

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"]
            max_sum_logpf = torch.max(abs(reference_logits), dim=-1).values.unsqueeze(-1)
            reward_reference = reference_logits
            reward_target = scaling_factor * max_sum_logpf * valid_score
            reward_penalized = reward_target
        else:
            reward_penalized = reward_target

        return {
            "reward": reward_target + reference_logits_scale * reward_reference,
            "reward_unpenalized": reward_reference,
            "reward_target": reward_target,
            "reward_reference": reward_reference,
            "full_tokens": None if validator_dict is None else validator_dict.get("full_tokens"),
            "log_pf_ref": reference_logits,
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
