from __future__ import annotations

import re
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import partialsmiles as ps
import torch

# slient the warning
from rdkit import Chem, RDLogger
from torch import Tensor
from transformers import PreTrainedTokenizer

from chemgfn.utils.gfn_utils import base_to_lora, lora_to_base
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

    reward = logprob[:, :, termination_token_id]
    reward[:, 1:] += log_p

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

    return reward, reward.clone()


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
        reward_mixed = reference_logits

        if self.sentence_validator is not None:
            validator_dict = self.sentence_validator(
                input_batch[:, prompt_length:], tokenizer, target_molecule
            )
            valid_score = validator_dict["valid_score"]
            max_sum_logpf = torch.max(abs(reference_logits), dim=-1).values.unsqueeze(-1)
            reward_mixed = (
                reference_logits * reference_logits_scale
                + scaling_factor * max_sum_logpf * valid_score
            )
            reward_penalized = reward_mixed
        else:
            reward_penalized = reward_mixed

        return {
            "reward": reward_penalized,
            "reward_unpenalized": reward_mixed,
            "full_tokens": None if validator_dict is None else validator_dict.get("full_tokens"),
            "log_pf_ref": reference_logits,
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
