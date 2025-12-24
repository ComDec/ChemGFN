from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from typing import Iterable, Literal

import partialsmiles as ps
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import Tensor
from transformers import PreTrainedTokenizer

from chemgfn.utils.rdkit_utils import FUNCTION_MAPPING

RDLogger.DisableLog("rdApp.*")


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


class ValidatorBase:
    name: str = "validator"
    supports_prefix: bool = True
    requires_target_molecule: bool = False
    returns_invalid_mask: bool = True
    returns_global_score: bool = True
    returns_valid_score: bool = True
    returns_full_tokens: bool = True

    def __init__(self, name: str | None = None, termination_token_id: int = -1) -> None:
        self.name = name or self.name
        self.termination_token_id = termination_token_id

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        return {}

    def __call__(self, sentences: Tensor, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        raise NotImplementedError


class SentenceValidator(ValidatorBase):
    pass


class Expr24Validator(SentenceValidator):
    """
    24-points validator (CFG-guaranteed valid prefixes).
    Format: d op d op d op d, where d in {0..9}, op in {+,-,*,/}, no parentheses.
    Evaluation: standard precedence (*/ before +-). Equals 24 -> 1, else 0.
    """

    TOKEN_RE = re.compile(r"[0-9]|[+\-*/]")

    def __init__(self, scorer: str = "hit24", amortize_valid_state: bool = False) -> None:
        super().__init__(scorer)

        # Whether to amortize the valid state of the expression to the entire batch
        self.amortize_valid_state = amortize_valid_state

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
        s = s.replace("\u00d7", "*").replace("\u00f7", "/")
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
    requires_target_molecule = False

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

    @staticmethod
    def _murcko_scaffold_smiles(mol: Chem.Mol | None) -> str | None:
        """
        Return Bemis-Murcko scaffold as SMILES (canonical). None if fails.
        Note: Murcko scaffold can be empty for some molecules; treat empty as None.
        """
        if mol is None:
            return None
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if scaf is None or scaf.GetNumAtoms() == 0:
                return None
            # canonical scaffold smiles
            return Chem.MolToSmiles(scaf, isomericSmiles=False)
        except Exception:
            return None

    def _decode_batch(self, generated_tokens: Tensor, tokenizer: PreTrainedTokenizer) -> list[str]:
        return [_decode_tokens_to_string(sample, tokenizer) for sample in generated_tokens]

    @staticmethod
    def _entropy_from_counts(counter: Counter) -> float:
        n = sum(counter.values())
        if n <= 0:
            return 0.0
        ent = 0.0
        for c in counter.values():
            p = c / n
            ent -= p * math.log(p + 1e-12)
        return float(ent)

    def smiles_accuracy(
        self,
        generated_tokens: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
    ) -> dict[str, float]:
        decoded = self._decode_batch(generated_tokens, tokenizer)

        scores: list[float] = []
        valid_flags: list[bool] = []

        # scaffold tracking
        scaffolds_all: list[str | None] = []  # include invalid as None
        scaffolds_valid: list[str] = []  # valid only

        for s in decoded:
            candidate = _merge_target(scaffold, s)

            mol = None
            if self._is_valid_smiles(candidate):
                mol = Chem.MolFromSmiles(candidate)

            is_valid = bool(mol)
            valid_flags.append(is_valid)

            if is_valid:
                scores.append(float(self.score_function(mol)))
            else:
                scores.append(0.0)

            scaf = self._murcko_scaffold_smiles(mol)
            scaffolds_all.append(scaf)
            if (scaf is not None) and is_valid:
                scaffolds_valid.append(scaf)

        num_samples = int(generated_tokens.shape[0])
        total_valid = int(sum(valid_flags))

        avg_score = (sum(scores) / num_samples) if num_samples else 0.0
        filtered_score = (
            sum(score for score, flag in zip(scores, valid_flags) if flag) / total_valid
            if total_valid
            else 0.0
        )

        # ---- scaffold diversity metrics ----
        # Unique fraction (all) treats None as "no scaffold"; often you'd rather exclude None.
        # Here we provide both all + valid-only.
        uniq_all = len(set(scaffolds_all)) if num_samples else 0
        uniq_valid = len(set(scaffolds_valid)) if total_valid else 0

        scaffold_unique_frac_all = (uniq_all / num_samples) if num_samples else 0.0
        scaffold_unique_frac_valid = (uniq_valid / total_valid) if total_valid else 0.0

        # Valid-only distribution stats (more meaningful)
        if total_valid > 0 and len(scaffolds_valid) > 0:
            c = Counter(scaffolds_valid)
            ent = self._entropy_from_counts(c)
            eff = math.exp(ent)
            top1 = max(c.values()) / max(1, len(scaffolds_valid))
        else:
            ent, eff, top1 = 0.0, 0.0, 0.0

        return {
            "acc": (total_valid / num_samples) if num_samples else 0.0,
            f"{self.scorer_name}": float(avg_score),
            f"{self.scorer_name}_filter": float(filtered_score),
            # scaffold diversity
            "scaffold_unique_frac_all": float(scaffold_unique_frac_all),
            "scaffold_unique_frac_valid": float(scaffold_unique_frac_valid),
            "scaffold_entropy_valid": float(ent),
            "scaffold_eff_valid": float(eff),
            "scaffold_top1_valid": float(top1),
        }

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
    ) -> dict[str, float]:
        return self.smiles_accuracy(sentences, tokenizer, scaffold)

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
    ) -> dict[str, Tensor]:
        termination_token_id = tokenizer.eos_token_id
        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        valid_score = torch.full((batch_size, seq_len + 1), -1.0, device=device)
        valid_score[:, 0] = 0.0
        global_score = torch.zeros(batch_size, device=device)
        full_tokens_list: list[str] = []

        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                if sentences[batch_idx, pos] == termination_token_id:
                    break

                prefix = _decode_tokens_to_string(sentences[batch_idx, : pos + 1], tokenizer)
                candidate = _merge_target(scaffold, prefix)
                valid = self._is_valid_smiles(candidate)

                if valid:
                    mol = Chem.MolFromSmiles(candidate)
                    valid_score[batch_idx, pos + 1] = self.score_function(mol) if mol else -1.0
                    invalid[batch_idx, pos + 1] = 0.0
                else:
                    invalid[batch_idx, pos + 1] = 1.0

            final_tokens = _decode_tokens_to_string(sentences[batch_idx], tokenizer)
            full_smiles = _merge_target(scaffold, final_tokens)
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
