from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Literal

import partialsmiles as ps
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
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
    Target-sum validator (default target=24).
    Supports variable-length expressions with digits and +-*/,
    evaluated using standard precedence. Scoring modes:
      - hit24        : 1.0 if expression == target else 0.0
      - near_24      : 1.0 at target, linearly decays with distance to 0.0
      - hit24_dense  : hit24 but local_score filled for every valid prefix
      - near_24_dense: near_24 but local_score filled for every valid prefix
    """

    def __init__(
        self,
        scorer: str = "hit24",
        amortize_valid_state: bool = False,
        target_value: int | float = 24,
        allow_parentheses: bool = False,
    ) -> None:
        if scorer not in {"hit24", "near_24", "hit24_dense", "near_24_dense"}:
            raise ValueError(f"Unsupported scorer for Expr24Validator: {scorer}")

        super().__init__(scorer)
        self.scorer = scorer
        self.target_value = Fraction(target_value)

        self.amortize_valid_state = amortize_valid_state
        self.allow_parentheses = bool(allow_parentheses)
        token_pattern = r"\d+|[+\-*/]" if not self.allow_parentheses else r"\d+|[+\-*/()]"
        self.token_re = re.compile(token_pattern)

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

        score_sum = 0.0
        for sample in generated_tokens:
            expr = self._decode_expr(sample, tokenizer)
            if expr is None:
                continue
            is_valid, _, value = self._score_expression(expr)
            if is_valid and value == self.target_value:
                score_sum += 1.0
        return {"acc": score_sum / total}

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
                "local_score": torch.zeros(1, 1),
                "full_tokens": [],
            }

        termination_token_id = tokenizer.eos_token_id
        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        local_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.zeros(batch_size, device=device)
        full_tokens_list: list[str] = []

        invalid[:, 0] = 1.0

        dense_mode = self.scorer in {"hit24_dense", "near_24_dense"}

        for i in range(batch_size):
            stop_pos = seq_len
            for pos in range(seq_len):
                if sentences[i, pos] == termination_token_id:
                    stop_pos = pos
                    break

            for pos in range(stop_pos):
                prefix_expr = _decode_tokens_to_string(sentences[i, : pos + 1], tokenizer)
                is_valid_prefix, prefix_score, prefix_value = self._score_expression(prefix_expr)
                is_hit_target_prefix = is_valid_prefix and prefix_value == self.target_value
                invalid[i, pos + 1] = 0.0 if is_hit_target_prefix else 1.0
                if dense_mode:
                    local_score[i, pos + 1] = float(prefix_score)

            final_expr = self._decode_expr(sentences[i], tokenizer)
            if final_expr is None:
                full_tokens_list.append("")
                continue
            is_valid, score, value = self._score_expression(final_expr)

            last_pos = stop_pos if stop_pos < seq_len else seq_len
            if last_pos >= 1:
                local_score[i, last_pos] = float(score)
                invalid[i, last_pos] = 0.0 if (is_valid and value == self.target_value) else 1.0

            global_score[i] = float(score)
            invalid[i, -1] = 0.0 if (is_valid and value == self.target_value) else 1.0
            full_tokens_list.append(final_expr)

        if self.amortize_valid_state:
            mask = global_score > 0
            local_score[mask, 1:] = global_score[mask].unsqueeze(1)
            invalid[mask, 1:] = 0.0

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens_list,
        }

    def _tokenize_expr(self, expr: str) -> list[str] | None:
        normalized = expr.replace("\u00d7", "*").replace("\u00f7", "/").replace(" ", "")
        if not normalized:
            return None
        tokens = self.token_re.findall(normalized)
        if "".join(tokens) != normalized:
            return None
        return tokens

    def _parse_and_eval(self, tokens: list[str]) -> Fraction | None:
        if len(tokens) == 0:
            return None

        if not self.allow_parentheses:
            if len(tokens) % 2 == 0:
                return None
            for idx, tk in enumerate(tokens):
                if idx % 2 == 0:
                    if not tk.isdigit():
                        return None
                else:
                    if tk not in "+-*/":
                        return None
            try:
                values: list[Fraction] = [Fraction(int(tokens[0]))]
            except Exception:
                return None

            pending_ops: list[str] = []
            try:
                for idx in range(1, len(tokens), 2):
                    op = tokens[idx]
                    nxt = Fraction(int(tokens[idx + 1]))

                    if op in "*/":
                        prev = values.pop()
                        if op == "*":
                            values.append(prev * nxt)
                        else:
                            if nxt == 0:
                                return None
                            values.append(prev / nxt)
                    else:
                        values.append(nxt)
                        pending_ops.append(op)
            except Exception:
                return None

            try:
                result = values[0]
                for op, val in zip(pending_ops, values[1:]):
                    result = result + val if op == "+" else result - val
                return result
            except Exception:
                return None

        def precedence(op: str) -> int:
            return 2 if op in "*/" else 1

        def apply_op(op: str, b: Fraction, a: Fraction) -> Fraction | None:
            try:
                if op == "+":
                    return a + b
                if op == "-":
                    return a - b
                if op == "*":
                    return a * b
                if op == "/":
                    if b == 0:
                        return None
                    return a / b
            except Exception:
                return None
            return None

        values: list[Fraction] = []
        ops: list[str] = []

        def pop_and_apply() -> bool:
            if not ops or len(values) < 2:
                return False
            op = ops.pop()
            b = values.pop()
            a = values.pop()
            res = apply_op(op, b, a)
            if res is None:
                return False
            values.append(res)
            return True

        prev_token_type = None
        for tk in tokens:
            if tk.isdigit():
                values.append(Fraction(int(tk)))
                prev_token_type = "num"
            elif tk in "+-*/":
                if prev_token_type not in {"num", ")"}:
                    return None
                while ops and ops[-1] not in "(" and precedence(ops[-1]) >= precedence(tk):
                    if not pop_and_apply():
                        return None
                ops.append(tk)
                prev_token_type = "op"
            elif tk == "(":
                if prev_token_type not in {None, "op", "("}:
                    return None
                ops.append(tk)
                prev_token_type = "("
            elif tk == ")":
                if prev_token_type not in {"num", ")"}:
                    return None
                found_lparen = False
                while ops:
                    top = ops.pop()
                    if top == "(":
                        found_lparen = True
                        break
                    if len(values) < 2:
                        return None
                    b = values.pop()
                    a = values.pop()
                    res = apply_op(top, b, a)
                    if res is None:
                        return None
                    values.append(res)
                if not found_lparen:
                    return None
                prev_token_type = ")"
            else:
                return None

        if prev_token_type not in {"num", ")"}:
            return None

        while ops:
            op = ops.pop()
            if op == "(":
                return None
            if len(values) < 2:
                return None
            b = values.pop()
            a = values.pop()
            res = apply_op(op, b, a)
            if res is None:
                return None
            values.append(res)

        if len(values) != 1:
            return None
        return values[0]

    def _score_value(self, value: Fraction) -> float:
        base_scorer = self.scorer.replace("_dense", "")
        target = self.target_value
        if base_scorer == "hit24":
            return 1.0 if value == target else 0.0
        if base_scorer == "near_24":
            diff = abs(value - target)
            denom = float(max(abs(target), 1))
            score = 1.0 - float(diff) / denom
            return float(max(0.0, score))
        raise ValueError(f"Unsupported scorer for Expr24Validator: {self.scorer}")

    def _score_expression(self, expr: str) -> tuple[bool, float, Fraction | None]:
        tokens = self._tokenize_expr(expr)
        if tokens is None:
            return False, 0.0, None

        value = self._parse_and_eval(tokens)
        if value is None:
            return False, 0.0, None

        return True, self._score_value(value), value


class RDKitValidator(SentenceValidator):
    requires_target_molecule = False

    def __init__(
        self,
        scorer: str = "sa",
        backend: Literal["rdkit", "pa"] = "pa",
        fp_radius: int = 2,
        fp_nbits: int = 2048,
        topk_diversity: int = 20,
    ) -> None:
        super().__init__(scorer)
        self.score_function = FUNCTION_MAPPING[scorer]
        self.backend = backend
        self.scorer_name = scorer

        self.fp_radius = int(fp_radius)
        self.fp_nbits = int(fp_nbits)
        self.topk_diversity = int(topk_diversity)

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

    def _morgan_fp(self, mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_nbits)

    @staticmethod
    def _first_eos_pos(tokens: Tensor, eos_id: int) -> tuple[Tensor, Tensor]:
        B, T = tokens.shape
        device = tokens.device
        pos = torch.arange(T, device=device).view(1, T).expand(B, T)
        is_eos = tokens.eq(eos_id)
        has_eos = is_eos.any(dim=1)
        eos_pos = torch.where(is_eos, pos, torch.full_like(pos, T))
        first_eos = eos_pos.min(dim=1).values
        return first_eos, has_eos

    @staticmethod
    def _stats_1d(vals: list[int]) -> dict[str, float]:
        if len(vals) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        x = torch.tensor(vals, dtype=torch.float32)
        x_sorted, _ = torch.sort(x)
        n = x_sorted.numel()

        def q(p: float) -> float:
            idx = int(round((n - 1) * p))
            idx = max(0, min(n - 1, idx))
            return float(x_sorted[idx].item())

        mean = float(x.mean().item())
        std = float(x.std(unbiased=False).item()) if n > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "min": float(x_sorted[0].item()),
            "max": float(x_sorted[-1].item()),
            "p50": q(0.50),
            "p90": q(0.90),
            "p95": q(0.95),
            "p99": q(0.99),
        }

    @staticmethod
    def _len_bins(vals: list[int], bins: list[tuple[int, int]]) -> dict[str, float]:
        n = len(vals)
        if n == 0:
            return {f"bin_{lo}_{hi}": 0.0 for (lo, hi) in bins}
        out: dict[str, float] = {}
        for lo, hi in bins:
            cnt = sum(1 for v in vals if lo <= v <= hi)
            out[f"bin_{lo}_{hi}"] = float(cnt / n)
        return out

    @staticmethod
    def _mean_pairwise_tanimoto(
        fps: list[DataStructs.cDataStructs.ExplicitBitVect],
    ) -> float:
        n = len(fps)
        if n <= 1:
            return 1.0
        s = 0.0
        cnt = 0
        for i in range(n - 1):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            s += float(sum(sims))
            cnt += len(sims)
        return s / max(1, cnt)

    def smiles_accuracy(
        self,
        generated_tokens: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
        *,
        return_hist: bool = False,
    ) -> dict[str, Any]:
        num_samples = int(generated_tokens.shape[0])
        if num_samples == 0:
            out: dict[str, Any] = {
                "acc": 0.0,
                f"{self.scorer_name}": 0.0,
                f"{self.scorer_name}_filter": 0.0,
                "fp_div_internal_valid": 0.0,
                "fp_div_topk_valid": 0.0,
                "len_tok_mean": 0.0,
                "len_tok_std": 0.0,
                "len_tok_min": 0.0,
                "len_tok_max": 0.0,
                "len_tok_p50": 0.0,
                "len_tok_p90": 0.0,
                "len_tok_p95": 0.0,
                "len_tok_p99": 0.0,
                "len_tok_eos_rate": 0.0,
                "len_tok_valid_mean": 0.0,
                "len_tok_valid_std": 0.0,
                "len_tok_valid_min": 0.0,
                "len_tok_valid_max": 0.0,
                "len_tok_valid_p50": 0.0,
                "len_tok_valid_p90": 0.0,
                "len_tok_valid_p95": 0.0,
                "len_tok_valid_p99": 0.0,
                "len_char_mean": 0.0,
                "len_char_valid_mean": 0.0,
            }
            if return_hist:
                out["len_tok_hist"] = []
                out["len_tok_valid_hist"] = []
            return out

        eos_id = int(tokenizer.eos_token_id)
        first_eos, has_eos = self._first_eos_pos(generated_tokens, eos_id)
        tok_lens = first_eos.to(torch.int64).tolist()
        eos_rate = float(has_eos.float().mean().item())

        decoded = self._decode_batch(generated_tokens, tokenizer)

        valid_flags: list[bool] = [False] * num_samples
        scores: list[float] = [0.0] * num_samples
        valid_mols: list[Chem.Mol] = []
        valid_scores: list[float] = []

        char_lens: list[int] = [0] * num_samples
        valid_tok_lens: list[int] = []
        valid_char_lens: list[int] = []

        for i, s in enumerate(decoded):
            candidate = _merge_target(scaffold, s)
            char_lens[i] = int(len(candidate))

            mol = None
            if self._is_valid_smiles(candidate):
                mol = Chem.MolFromSmiles(candidate)

            is_valid = bool(mol)
            valid_flags[i] = is_valid

            if is_valid:
                sc = float(self.score_function(mol))
                scores[i] = sc
                valid_mols.append(mol)
                valid_scores.append(sc)
                valid_tok_lens.append(int(tok_lens[i]))
                valid_char_lens.append(int(char_lens[i]))
            else:
                scores[i] = 0.0

        total_valid = int(sum(valid_flags))
        avg_score = float(sum(scores) / num_samples)
        filtered_score = float(sum(valid_scores) / total_valid) if total_valid else 0.0

        if total_valid >= 2:
            fps = [self._morgan_fp(m) for m in valid_mols]
            mean_sim = self._mean_pairwise_tanimoto(fps)
            fp_div_internal_valid = 1.0 - float(mean_sim)

            k = min(self.topk_diversity, total_valid)
            if k >= 2:
                top_idx = sorted(range(total_valid), key=lambda j: valid_scores[j], reverse=True)[
                    :k
                ]
                top_fps = [fps[j] for j in top_idx]
                top_mean_sim = self._mean_pairwise_tanimoto(top_fps)
                fp_div_topk_valid = 1.0 - float(top_mean_sim)
            else:
                fp_div_topk_valid = 0.0
        else:
            fp_div_internal_valid = 0.0
            fp_div_topk_valid = 0.0

        tok_stats = self._stats_1d(tok_lens)
        vt_stats = self._stats_1d(valid_tok_lens)
        char_stats = self._stats_1d(char_lens)
        vchar_stats = self._stats_1d(valid_char_lens)

        bins = [(0, 2), (3, 5), (6, 8), (9, 10), (11, -1)]
        tok_bins = self._len_bins(tok_lens, bins)
        vt_bins = self._len_bins(valid_tok_lens, bins)

        out: dict[str, Any] = {
            "acc": float(total_valid / num_samples),
            f"{self.scorer_name}": float(avg_score),
            f"{self.scorer_name}_filter": float(filtered_score),
            "fp_div_internal_valid": float(fp_div_internal_valid),
            "fp_div_topk_valid": float(fp_div_topk_valid),
            "len_tok_mean": tok_stats["mean"],
            "len_tok_std": tok_stats["std"],
            "len_tok_min": tok_stats["min"],
            "len_tok_max": tok_stats["max"],
            "len_tok_p50": tok_stats["p50"],
            "len_tok_p90": tok_stats["p90"],
            "len_tok_p95": tok_stats["p95"],
            "len_tok_p99": tok_stats["p99"],
            "len_tok_eos_rate": float(eos_rate),
            "len_tok_valid_mean": vt_stats["mean"],
            "len_tok_valid_std": vt_stats["std"],
            "len_tok_valid_min": vt_stats["min"],
            "len_tok_valid_max": vt_stats["max"],
            "len_tok_valid_p50": vt_stats["p50"],
            "len_tok_valid_p90": vt_stats["p90"],
            "len_tok_valid_p95": vt_stats["p95"],
            "len_tok_valid_p99": vt_stats["p99"],
            "len_char_mean": char_stats["mean"],
            "len_char_valid_mean": vchar_stats["mean"],
        }

        def _bin_suffix(bin_key: Any) -> str:
            if isinstance(bin_key, tuple) and len(bin_key) == 2:
                return f"{bin_key[0]}_{bin_key[1]}"
            if isinstance(bin_key, str) and bin_key.startswith("bin_"):
                return bin_key[4:]
            return str(bin_key)

        for bin_key, frac in tok_bins.items():
            suffix = _bin_suffix(bin_key)
            out[f"len_tok_{suffix}_frac"] = float(frac)
        for bin_key, frac in vt_bins.items():
            suffix = _bin_suffix(bin_key)
            out[f"len_tok_valid_{suffix}_frac"] = float(frac)

        if return_hist:
            out["len_tok_hist"] = tok_lens
            out["len_tok_valid_hist"] = valid_tok_lens
            out["len_char_hist"] = char_lens
            out["len_char_valid_hist"] = valid_char_lens
            out["score_hist"] = scores

        return out

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
        *,
        return_hist: bool = False,
    ) -> dict[str, float]:
        return self.smiles_accuracy(sentences, tokenizer, scaffold, return_hist=return_hist)

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
        local_score = torch.full((batch_size, seq_len + 1), 0.0, device=device)
        local_score[:, 0] = 0.0
        global_score = torch.zeros(batch_size, device=device)
        full_tokens_list: list[str] = []

        for b in range(batch_size):
            prefix_cache: dict[int, str] = {}

            for pos in range(seq_len):
                tok = int(sentences[b, pos].item())
                if tok == termination_token_id:
                    break

                if pos not in prefix_cache:
                    prefix_cache[pos] = _decode_tokens_to_string(
                        sentences[b, : pos + 1], tokenizer
                    )

                candidate = _merge_target(scaffold, prefix_cache[pos])

                if self._is_valid_smiles(candidate):
                    mol = Chem.MolFromSmiles(candidate)
                    if mol:
                        local_score[b, pos + 1] = float(self.score_function(mol))
                        invalid[b, pos + 1] = 0.0
                    else:
                        invalid[b, pos + 1] = 1.0
                else:
                    invalid[b, pos + 1] = 1.0

            final_tokens = _decode_tokens_to_string(sentences[b], tokenizer)
            full_smiles = _merge_target(scaffold, final_tokens)
            try:
                mol = Chem.MolFromSmiles(full_smiles)
                global_score[b] = float(self.score_function(mol)) if mol else 0.0
            except Exception:
                global_score[b] = 0.0

            full_tokens_list.append(full_smiles)

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens_list,
        }


class AMPValidator(SentenceValidator):
    """Validator for AMP (Antimicrobial Peptide) generation.

    Uses a pre-trained oracle (ProtTrans AlBert + MLP) to score sequences.
    Per-position reward uses absorbing pattern: score placed only at terminal position.
    All amino acid prefixes are considered valid.
    """

    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        oracle_weights_path: str | None = None,
        oracle_device: str = "cpu",
        mc_samples: int = 0,
        training_set_path: str | None = None,
        topk: int = 100,
    ) -> None:
        super().__init__("amp_validator")
        from chemgfn.models.amp_oracle import AMPOracle

        self.oracle = AMPOracle(
            weights_path=oracle_weights_path,
            device=oracle_device,
            mc_samples=mc_samples,
        )
        self.topk = topk

        # Load training set for novelty computation
        self._training_sequences: list[str] | None = None
        if training_set_path is not None:
            with open(training_set_path) as f:
                self._training_sequences = [line.strip() for line in f if line.strip()]

    def _decode_sequence(self, tokens: Tensor, tokenizer) -> str:
        """Decode token IDs to amino acid string, stopping at EOS."""
        return _decode_tokens_to_string(tokens, tokenizer)

    def _is_valid_aa_sequence(self, seq: str) -> bool:
        return len(seq) > 0 and all(c in self.AMINO_ACIDS for c in seq)

    def __call__(
        self,
        sentences: Tensor,
        tokenizer,
        scaffold: str | None = None,
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        if sentences is None or sentences.ndim < 1:
            return {
                "invalid": torch.zeros(1, 1),
                "global_score": torch.zeros(1),
                "local_score": torch.zeros(1, 1),
                "full_tokens": [],
            }

        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        local_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.zeros(batch_size, device=device)
        full_tokens_list: list[str] = []

        # Decode all sequences and find stop positions
        decoded_seqs: list[str] = []
        stop_positions: list[int] = []
        for i in range(batch_size):
            seq = self._decode_sequence(sentences[i], tokenizer)
            decoded_seqs.append(seq)
            full_tokens_list.append(seq)

            # Find stop position (first EOS or end)
            stop_pos = seq_len
            for pos in range(seq_len):
                if sentences[i, pos].item() == tokenizer.eos_token_id:
                    stop_pos = pos
                    break
            stop_positions.append(stop_pos)

            # All AA positions are valid (any prefix of AAs is valid)
            for pos in range(min(stop_pos, seq_len)):
                invalid[i, pos + 1] = 0.0

        # Batch oracle scoring for valid sequences
        valid_indices = [i for i, s in enumerate(decoded_seqs) if self._is_valid_aa_sequence(s)]
        if valid_indices:
            valid_seqs = [decoded_seqs[i] for i in valid_indices]
            scores = self.oracle.score_sequences(valid_seqs)
            scores = scores.to(device)
            for j, i in enumerate(valid_indices):
                score_val = scores[j].item()
                global_score[i] = score_val
                # Absorbing: place score at terminal position
                term_pos = min(stop_positions[i], seq_len)
                local_score[i, term_pos] = score_val

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens_list,
        }

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer,
        scaffold: str | None = None,
        *,
        return_hist: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        from chemgfn.utils.sequence_metrics import (
            levenshtein_diversity,
            levenshtein_novelty,
            select_topk,
        )

        num_samples = int(sentences.shape[0])
        if num_samples == 0:
            return {"acc": 0.0, "amp_score": 0.0, "diversity": 0.0, "novelty": 0.0}

        # Decode and score
        decoded: list[str] = []
        for i in range(num_samples):
            seq = self._decode_sequence(sentences[i], tokenizer)
            decoded.append(seq)

        valid_flags = [self._is_valid_aa_sequence(s) for s in decoded]
        valid_seqs = [s for s, v in zip(decoded, valid_flags) if v]
        total_valid = len(valid_seqs)

        scores_list = [0.0] * num_samples
        if valid_seqs:
            oracle_scores = self.oracle.score_sequences(valid_seqs)
            j = 0
            for i, v in enumerate(valid_flags):
                if v:
                    scores_list[i] = float(oracle_scores[j].item())
                    j += 1

        # Top-K metrics (paper Eq. 1, 2, 3)
        topk_seqs, topk_scores = select_topk(decoded, scores_list, k=self.topk)
        performance = sum(topk_scores) / len(topk_scores) if topk_scores else 0.0
        diversity = levenshtein_diversity(topk_seqs) if len(topk_seqs) >= 2 else 0.0
        novelty = (
            levenshtein_novelty(topk_seqs, self._training_sequences)
            if self._training_sequences and topk_seqs
            else 0.0
        )

        # Length stats
        eos_id = int(tokenizer.eos_token_id)
        tok_lens = []
        for i in range(num_samples):
            length = int(sentences.shape[1])
            for pos in range(sentences.shape[1]):
                if sentences[i, pos].item() == eos_id:
                    length = pos
                    break
            tok_lens.append(length)

        out: dict[str, Any] = {
            "acc": float(total_valid / num_samples),
            "amp_score": float(sum(scores_list) / num_samples),
            "amp_score_filter": float(
                sum(s for s, v in zip(scores_list, valid_flags) if v) / total_valid
            ) if total_valid else 0.0,
            "performance_topk": float(performance),
            "diversity": float(diversity),
            "novelty": float(novelty),
            "len_tok_mean": float(sum(tok_lens) / len(tok_lens)) if tok_lens else 0.0,
        }

        if return_hist:
            out["len_tok_hist"] = tok_lens
            out["score_hist"] = scores_list

        return out
