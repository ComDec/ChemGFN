"""Task validators.

A validator is the task-specific half of a reward: it decodes a batch of generated token
sequences and reports, for every state on every trajectory, whether the state is a legal prefix
and what it scores. :mod:`chemgfn.models.reward` combines that signal with the frozen reference
model's prior to form the GFlowNet log reward.

One validator is provided per released task: :class:`Expr24Validator` (arithmetic expressions),
:class:`RDKitValidator` (SMILES), :class:`AMPValidator` (antimicrobial peptides) and
:class:`CommonGenValidator` (concept-to-sentence generation).
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from typing import Any, Literal

import partialsmiles as ps
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch import Tensor
from transformers import PreTrainedTokenizer

from chemgfn.utils.molecule_scores import FUNCTION_MAPPING

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


class Validator:
    """Base class for task validators.

    A validator turns a batch of generated token sequences into the per-state quantities the
    reward module needs:

    ``invalid``
        ``(B, T + 1)`` mask over states; ``1.0`` marks a state that is not a legal prefix of a
        terminal object.
    ``local_score``
        ``(B, T + 1)`` task score attached to each state, used for prefix and absorbed-suffix
        shaping.
    ``global_score``
        ``(B,)`` task score of the complete trajectory.
    ``full_tokens``
        Decoded string for each trajectory, used for logging and replay-buffer keys.

    Subclasses implement :meth:`__call__` for scoring and may override :meth:`accuracy` to report
    task-specific evaluation metrics.
    """

    name: str = "validator"

    def __init__(self, name: str | None = None, termination_token_id: int = -1) -> None:
        """Store the validator's name and the token id that terminates a trajectory."""

        self.name = name or self.name
        self.termination_token_id = termination_token_id

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Return evaluation metrics for a batch of generated sequences."""

        return {}

    def __call__(
        self, sentences: Tensor, tokenizer: PreTrainedTokenizer, *args, **kwargs
    ) -> dict[str, Any]:
        """Score a batch of generated sequences."""

        raise NotImplementedError


class Expr24Validator(Validator):
    """Validator for the variable-length arithmetic task ("make 24").

    An expression is a parenthesis-free alternation of non-negative integers and the operators
    ``+ - * /``, evaluated exactly over the rationals with standard precedence. A state scores
    ``1.0`` when the prefix read so far already evaluates to the target value and ``0.0``
    otherwise, so ``local_score`` is dense over every parseable prefix.
    """

    def __init__(self, target_value: int | float = 24) -> None:
        """Configure the validator.

        Args:
            target_value: Value an expression must evaluate to in order to score ``1.0``.
        """

        super().__init__("expr24")
        self.target_value = Fraction(target_value)
        self.token_re = re.compile(r"\d+|[+\-*/]")

    def _decode_expr(self, tokens: Tensor, tokenizer: PreTrainedTokenizer) -> str | None:
        try:
            decoded = _decode_tokens_to_string(tokens, tokenizer)
        except Exception:
            return None
        decoded = decoded.strip()
        return decoded or None

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Return the fraction of sequences that evaluate exactly to the target value."""

        if sentences is None or sentences.ndim == 0:
            return {"acc": 0.0}
        total = sentences.shape[0]
        if total == 0:
            return {"acc": 0.0}

        score_sum = 0.0
        for sample in sentences:
            expr = self._decode_expr(sample, tokenizer)
            if expr is None:
                continue
            is_valid, _, value = self._score_expression(expr)
            if is_valid and value == self.target_value:
                score_sum += 1.0
        return {"acc": score_sum / total}

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Score expressions and every legal prefix of them."""

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
        """Evaluate a token list exactly, or return ``None`` if it is not a valid expression.

        Tokens must alternate integer, operator, integer, ... Multiplication and division bind
        tighter than addition and subtraction; division by zero makes the expression invalid.
        """

        if len(tokens) == 0:
            return None

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

    def _score_expression(self, expr: str) -> tuple[bool, float, Fraction | None]:
        """Return ``(is_valid, score, value)`` for a decoded expression string."""

        tokens = self._tokenize_expr(expr)
        if tokens is None:
            return False, 0.0, None

        value = self._parse_and_eval(tokens)
        if value is None:
            return False, 0.0, None

        return True, 1.0 if value == self.target_value else 0.0, value


class RDKitValidator(Validator):
    """SMILES validator scoring molecules with an RDKit property function.

    Prefixes are checked with ``partialsmiles``, which accepts a string that can still be
    extended into a valid molecule. Every state on a trajectory therefore receives a validity
    flag and, once the prefix parses as a complete molecule, a property score.
    """

    def __init__(
        self,
        scorer: str = "sa",
        fp_radius: int = 2,
        fp_nbits: int = 2048,
        topk_diversity: int = 20,
    ) -> None:
        """Configure the validator.

        Args:
            scorer: Key into :data:`chemgfn.utils.molecule_scores.FUNCTION_MAPPING` selecting the
                molecular property to optimise.
            fp_radius: Morgan fingerprint radius used for the diversity metrics.
            fp_nbits: Morgan fingerprint length used for the diversity metrics.
            topk_diversity: Number of highest-scoring molecules in the top-k diversity metric.
        """

        super().__init__(scorer)
        self.score_function = FUNCTION_MAPPING[scorer]
        self.scorer_name = scorer

        self.fp_radius = int(fp_radius)
        self.fp_nbits = int(fp_nbits)
        self.topk_diversity = int(topk_diversity)

    @staticmethod
    def _is_valid_smiles(smiles: str) -> bool:
        """Whether ``smiles`` is a valid SMILES prefix."""

        try:
            ps.ParseSmiles(smiles)
            return True
        except Exception:
            return False

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

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
        *,
        return_hist: bool = False,
    ) -> dict[str, Any]:
        """Report validity, property score, fingerprint diversity and length statistics."""

        num_samples = int(sentences.shape[0])
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
        first_eos, has_eos = self._first_eos_pos(sentences, eos_id)
        tok_lens = first_eos.to(torch.int64).tolist()
        eos_rate = float(has_eos.float().mean().item())

        decoded = self._decode_batch(sentences, tokenizer)

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

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: str | None = None,
    ) -> dict[str, Any]:
        """Score complete molecules and every parseable SMILES prefix."""

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


class AMPValidator(Validator):
    """Validator for antimicrobial peptide (AMP) generation.

    Sequences are scored with a pre-trained oracle (ProtTrans ALBERT encoder plus an MLP head).
    Every amino-acid prefix is a legal state, and the oracle score is attached to the terminal
    state only, giving the absorbed suffix target its sparse terminal signal.
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
        """Configure the validator and load the oracle.

        Args:
            oracle_weights_path: Path to the oracle MLP state dict. ``None`` leaves the head
                randomly initialised, which is only useful for smoke tests.
            oracle_device: Torch device the oracle runs on.
            mc_samples: Number of MC-dropout passes; ``0`` scores deterministically.
            training_set_path: Optional file of one training peptide per line, used as the
                reference set for the novelty metric.
            topk: Number of highest-scoring peptides used by the top-k metrics.
        """

        super().__init__("amp_validator")
        from chemgfn.models.amp_oracle import AMPOracle

        self.oracle = AMPOracle(
            weights_path=oracle_weights_path,
            device=oracle_device,
            mc_samples=mc_samples,
        )
        self.topk = topk

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
    ) -> dict[str, Any]:
        """Score peptides with the oracle, placing the score at the terminal state."""

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
        """Report oracle score plus top-k performance, diversity and novelty."""

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
            )
            if total_valid
            else 0.0,
            "performance_topk": float(performance),
            "diversity": float(diversity),
            "novelty": float(novelty),
            "len_tok_mean": float(sum(tok_lens) / len(tok_lens)) if tok_lens else 0.0,
        }

        if return_hist:
            out["len_tok_hist"] = tok_lens
            out["score_hist"] = scores_list

        return out


class CommonGenValidator(Validator):
    """Validator for the CommonGen concept-to-sentence task.

    Each prompt carries a concept set and, on the validation split, one or more human reference
    sentences. Both are passed through the ``scaffold`` argument as a dict with keys ``concepts``
    and ``references``.

    The score has three parts, matching the reward definition used in the paper:

    * structural validity -- a terminal sentence is legal only if it is ASCII, uses an allowed
      character set, parses as a single sentence and contains a verb;
    * concept coverage -- a soft coverage ratio at every prefix, plus a lemma-matched hard
      coverage bonus at the terminal state when all concepts are present;
    * linguistic quality -- an n-gram overlap term between the prefix and the reference sentence,
      acting as a step-wise proxy for fluency.

    Requires the optional CommonGen dependencies: ``spacy`` (with the ``en_core_web_sm`` model)
    for parsing and lemmatisation, and ``nltk`` / ``bert-score`` / ``pycocoevalcap`` for the
    reported BLEU and BERTScore metrics.
    """

    name = "common_gen"

    def __init__(
        self,
        termination_token_id: int = -1,
        spacy_model: str = "en_core_web_sm",
        coverage_weight: float = 1.0,
        quality_weight: float = 0.5,
        hard_coverage_bonus: float = 1.0,
        validity_mode: Literal["english", "keywords"] = "english",
        strict_ascii: bool = True,
        allowed_punctuation: str = " .,!?;:'\"()/-",
        require_capital_start: bool = True,
        require_verb: bool = True,
        min_alpha_tokens: int = 2,
        single_sentence: bool = True,
        require_terminal_punct: bool = False,
        valid_step_weight: float = 0.0,
        valid_step_every: int = 1,
        valid_terminal_bonus: float = 0.0,
        ngram_n: int = 2,
        compute_bleu: bool = False,
        compute_bertscore: bool = False,
        bertscore_model_name: str = "microsoft/deberta-xlarge-mnli",
        bertscore_lang: str = "en",
        bertscore_rescale: bool = True,
    ) -> None:
        """Configure the validator.

        Args:
            termination_token_id: Token id that terminates a sentence; ``-1`` falls back to the
                tokenizer's EOS at scoring time.
            spacy_model: spaCy pipeline used for parsing and lemmatisation.
            coverage_weight: Weight of the soft concept-coverage ratio in the prefix score.
            quality_weight: Weight of the n-gram overlap term in the prefix score.
            hard_coverage_bonus: Bonus added at the terminal state when every concept is covered
                under lemma matching.
            validity_mode: ``"english"`` marks a terminal sentence legal when it is
                sentence-like; ``"keywords"`` marks it legal when it covers every concept.
            strict_ascii: Whether non-ASCII characters make a state illegal.
            allowed_punctuation: Punctuation characters permitted in a legal sentence.
            require_capital_start: Whether a legal sentence must start with a capital letter.
            require_verb: Whether a legal sentence must contain a verb.
            min_alpha_tokens: Minimum number of alphabetic tokens in a legal sentence.
            single_sentence: Whether the output must parse as exactly one sentence.
            require_terminal_punct: Whether a legal sentence must end in terminal punctuation.
            valid_step_weight: Weight of the step-wise terminability bonus.
            valid_step_every: Stride at which the terminability bonus is evaluated.
            valid_terminal_bonus: Bonus added at a legal terminal state.
            ngram_n: Order of the n-gram overlap used by the quality term.
            compute_bleu: Whether :meth:`accuracy` reports BLEU.
            compute_bertscore: Whether :meth:`accuracy` reports BERTScore.
            bertscore_model_name: Model backing BERTScore.
            bertscore_lang: Language passed to BERTScore.
            bertscore_rescale: Whether BERTScore is rescaled with its baseline.
        """

        super().__init__(self.name, termination_token_id=termination_token_id)
        self.coverage_weight = float(coverage_weight)
        self.quality_weight = float(quality_weight)
        self.hard_coverage_bonus = float(hard_coverage_bonus)

        self.validity_mode = str(validity_mode)
        self.strict_ascii = bool(strict_ascii)
        self.allowed_punctuation = str(allowed_punctuation)
        self.require_capital_start = bool(require_capital_start)
        self.require_verb = bool(require_verb)
        self.min_alpha_tokens = int(min_alpha_tokens)
        self.single_sentence = bool(single_sentence)
        self.require_terminal_punct = bool(require_terminal_punct)

        self.valid_step_weight = float(valid_step_weight)
        self.valid_step_every = max(1, int(valid_step_every))
        self.valid_terminal_bonus = float(valid_terminal_bonus)

        self._allowed_char_set = set(self.allowed_punctuation)
        self.ngram_n = int(ngram_n)

        self.compute_bleu = bool(compute_bleu)

        self.compute_bertscore = bool(compute_bertscore)
        self.bertscore_model_name = str(bertscore_model_name)
        self.bertscore_lang = str(bertscore_lang)
        self.bertscore_rescale = bool(bertscore_rescale)
        self._bertscorer = None

        self._treebank_tokenizer = None

        self.scorer_name = "coverage"

        self._nlp = self._load_spacy_model(spacy_model)

    @staticmethod
    def _load_spacy_model(spacy_model: str):
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "CommonGenValidator requires the optional CommonGen dependencies. Install them "
                "with `pip install spacy nltk bert-score pycocoevalcap`, then run "
                f"`python -m spacy download {spacy_model}`."
            ) from exc
        try:
            return spacy.load(spacy_model)
        except OSError as exc:
            raise ImportError(
                f"The spaCy model '{spacy_model}' is not installed. Run "
                f"`python -m spacy download {spacy_model}`."
            ) from exc

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _sanitize_caption(text: str) -> str:
        """Strip control characters that would otherwise break the PTB tokenizer."""

        s = text or ""
        s = s.replace("�", " ")
        s = "".join((ch if (ch == "\n" or ord(ch) >= 32) else " ") for ch in s)
        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _chars_ok(self, text: str) -> bool:
        s = text or ""
        if not s:
            return False
        if self.strict_ascii and any(ord(ch) > 127 for ch in s):
            return False
        for ch in s:
            if ch.isalnum() or ch.isspace() or (ch in self._allowed_char_set):
                continue
            return False
        return True

    def _capital_start_ok(self, text: str) -> bool:
        if not self.require_capital_start:
            return True
        s = (text or "").lstrip()
        # Skip leading punctuation and quotes before looking for the first letter.
        for ch in s:
            if ch.isalpha():
                return "A" <= ch <= "Z"
            if ch.isspace():
                continue
            if ch in self._allowed_char_set:
                continue
            break
        return False

    def _is_english_sentence_like(self, text: str) -> bool:
        s = self._sanitize_caption(text)
        if not s:
            return False
        if self.require_terminal_punct and not s.endswith((".", "!", "?")):
            return False
        if not self._chars_ok(s):
            return False
        if not self._capital_start_ok(s):
            return False

        doc = self._nlp(s)
        if self.single_sentence:
            try:
                if len(list(doc.sents)) != 1:
                    return False
            except Exception:
                pass

        alpha_toks = [t for t in doc if (not t.is_space and not t.is_punct and t.is_alpha)]
        if len(alpha_toks) < max(0, self.min_alpha_tokens):
            return False
        if self.require_verb:
            if not any(t.pos_ in {"VERB", "AUX"} for t in doc if not t.is_space):
                return False
        return True

    def _would_be_valid_if_terminated(self, prefix: str) -> bool:
        """Check whether a prefix would be a legal sentence if terminated right now.

        The task uses ``.`` as the end-of-sentence token, so a period is appended before the
        legality check unless the prefix already ends in terminal punctuation.
        """

        s = self._sanitize_caption(prefix)
        if not s:
            return False
        if not self._chars_ok(s):
            return False
        if s.endswith((".", "!", "?")):
            cand = s
        else:
            cand = s + "."
        return self._is_english_sentence_like(cand)

    @staticmethod
    def _word_tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    def _tokenize_for_metrics(self, text: str) -> list[str]:
        """Tokenize with NLTK's Treebank tokenizer, approximating COCO-style PTB tokenization."""

        if self._treebank_tokenizer is None:
            try:
                from nltk.tokenize import TreebankWordTokenizer
            except ImportError as exc:
                raise ImportError(
                    "CommonGen metrics require NLTK. Install it with `pip install nltk`."
                ) from exc

            self._treebank_tokenizer = TreebankWordTokenizer()
        s = (text or "").replace("\n", " ").strip()
        s = re.sub(r"\s+", " ", s)
        return [t for t in self._treebank_tokenizer.tokenize(s) if t]

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        n = max(1, int(n))
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @classmethod
    def _ngram_f1(cls, cand: str, ref: str, n: int) -> float:
        cand_toks = cls._word_tokens(cand)
        ref_toks = cls._word_tokens(ref)
        cand_ng = cls._ngrams(cand_toks, n)
        ref_ng = cls._ngrams(ref_toks, n)
        if not cand_ng or not ref_ng:
            return 0.0
        c_cnt = Counter(cand_ng)
        r_cnt = Counter(ref_ng)
        overlap = 0
        for k, v in c_cnt.items():
            overlap += min(v, r_cnt.get(k, 0))
        denom = len(cand_ng) + len(ref_ng)
        return float((2.0 * overlap) / denom) if denom > 0 else 0.0

    def _get_bertscorer(self):
        if not self.compute_bertscore:
            return None
        if self._bertscorer is not None:
            return self._bertscorer
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError(
                "BERTScore is required. Install it with `pip install bert-score`."
            ) from exc
        self._bertscorer = BERTScorer(
            model_type=self.bertscore_model_name,
            lang=self.bertscore_lang,
            rescale_with_baseline=self.bertscore_rescale,
            device="cpu",
        )
        return self._bertscorer

    @staticmethod
    def _parse_scaffold(scaffold: Any) -> tuple[list[str], list[str]]:
        """Split the per-prompt scaffold into ``(concepts, references)``."""

        if scaffold is None:
            return [], []
        if isinstance(scaffold, dict):
            concepts = scaffold.get("concepts", [])
            refs = scaffold.get("references", [])
            concepts = (
                [str(x) for x in concepts]
                if isinstance(concepts, (list, tuple))
                else [str(concepts)]
            )
            if isinstance(refs, (list, tuple)):
                references = [str(x) for x in refs if str(x)]
            else:
                references = [str(refs)] if str(refs) else []
            return concepts, references
        if isinstance(scaffold, (list, tuple)):
            references = [str(x) for x in scaffold if str(x)]
            return [], references
        if isinstance(scaffold, str):
            s = scaffold.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    import json

                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        return CommonGenValidator._parse_scaffold(obj)
                except Exception:
                    pass
            return [], [scaffold]
        return [], []

    def _coverage_ratio_surface(self, text: str, concepts_norm: list[str]) -> tuple[float, bool]:
        """Surface-form concept coverage, used for the cheap per-prefix shaping term."""

        if not concepts_norm:
            return 0.0, False
        norm = self._normalize_text(text)
        if not norm:
            return 0.0, False
        hit = 0
        for c in concepts_norm:
            if not c:
                continue
            if re.search(r"\b" + re.escape(c) + r"\b", norm):
                hit += 1
        ratio = float(hit) / float(len(concepts_norm)) if concepts_norm else 0.0
        return ratio, hit == len(concepts_norm)

    def _coverage_ratio_lemma(self, text: str, concepts_norm: list[str]) -> tuple[float, bool]:
        """Lemma-matched concept coverage, used for the terminal score and the hard bonus."""

        if not concepts_norm:
            return 0.0, False
        doc = self._nlp((text or "").strip())
        lemmas = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            lt = (t.lemma_ or t.text).lower()
            lt = re.sub(r"[^a-z0-9]+", "", lt)
            if lt:
                lemmas.append(lt)
        lemma_norm = " ".join(lemmas)
        hit = 0
        for c in concepts_norm:
            if not c:
                continue
            c_lemma = " ".join(
                [
                    re.sub(r"[^a-z0-9]+", "", (tok.lemma_ or tok.text).lower())
                    for tok in self._nlp(c)
                    if not tok.is_space and not tok.is_punct
                ]
            ).strip()
            needle = c_lemma if c_lemma else re.sub(r"[^a-z0-9 ]+", " ", c.lower()).strip()
            needle = re.sub(r"\s+", " ", needle)
            if needle and re.search(r"\b" + re.escape(needle) + r"\b", lemma_norm):
                hit += 1
        ratio = float(hit) / float(len(concepts_norm)) if concepts_norm else 0.0
        return ratio, hit == len(concepts_norm)

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: Any = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Report sentence legality, concept coverage and reference-similarity metrics.

        BLEU and BERTScore are expensive, so they are computed only when gradients are disabled,
        i.e. during validation and test, and skipped inside the training step.
        """

        if sentences is None or sentences.ndim == 0:
            return {"acc": 0.0, "cov_ratio": 0.0, "cov_ratio_filter": 0.0, "ngram_f1": 0.0}

        concepts, references = self._parse_scaffold(scaffold)
        concepts_norm = [self._normalize_text(c) for c in concepts if self._normalize_text(c)]

        eos_id = int(
            self.termination_token_id if self.termination_token_id >= 0 else tokenizer.eos_token_id
        )
        eos_text = tokenizer.decode([eos_id], skip_special_tokens=True).strip()
        include_eos = eos_text in {".", "!", "?"}
        candidates: list[str] = []
        for row in sentences:
            row = row.detach().cpu()
            try:
                stop_pos = (row == eos_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                stop_pos = int(row.shape[0])
            end = min(int(row.shape[0]), stop_pos + 1) if include_eos else stop_pos
            cand = tokenizer.decode(row[:end], skip_special_tokens=True).strip()
            candidates.append(self._sanitize_caption(cand))

        cov_hard = []
        legal_flags = []
        cov_ratio = []
        ngram_f1 = []
        bleu3 = None
        bleu4 = None
        for cand in candidates:
            legal_flags.append(1.0 if self._is_english_sentence_like(cand) else 0.0)
            r, ok = self._coverage_ratio_lemma(cand, concepts_norm)
            cov_ratio.append(r)
            cov_hard.append(1.0 if ok else 0.0)
            if references:
                best_f1 = 0.0
                for ref in references:
                    best_f1 = max(best_f1, self._ngram_f1(cand, ref, self.ngram_n))
                ngram_f1.append(best_f1)
            else:
                ngram_f1.append(0.0)
        # `acc` is the legal-English-sentence rate; full keyword coverage is `keyword_acc`.
        acc = float(sum(legal_flags) / max(1, len(legal_flags)))
        keyword_acc = float(sum(cov_hard) / max(1, len(cov_hard)))
        cov_mean = float(sum(cov_ratio) / max(1, len(cov_ratio)))
        num_valid = int(sum(1 for x in cov_hard if x > 0.5))
        cov_filter = (
            float(sum(r for r, ok in zip(cov_ratio, cov_hard) if ok > 0.5) / num_valid)
            if num_valid
            else 0.0
        )

        out: dict[str, Any] = {
            "acc": acc,
            "keyword_acc": keyword_acc,
            "cov_ratio": cov_mean,
            "cov_ratio_filter": cov_filter,
            "coverage": 100.0 * cov_mean,
            "coverage_filter": 100.0 * cov_filter,
            "ngram_f1": float(sum(ngram_f1) / max(1, len(ngram_f1))),
        }

        if torch.is_grad_enabled():
            return out

        if self.compute_bleu and references and candidates:
            # BLEU-3/4 on a 0-100 scale with PTB tokenization, as reported for CommonGen.
            try:
                import contextlib
                import io

                from pycocoevalcap.bleu.bleu import Bleu
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

                refs_clean = [self._sanitize_caption(r) for r in references]
                gts_raw = {i: [{"caption": r} for r in refs_clean] for i in range(len(candidates))}
                res_raw = {i: [{"caption": candidates[i]}] for i in range(len(candidates))}
                tokenizer_ptb = PTBTokenizer()
                with contextlib.redirect_stdout(io.StringIO()):
                    gts = tokenizer_ptb.tokenize(gts_raw)
                    res = tokenizer_ptb.tokenize(res_raw)
                    score, _ = Bleu(4).compute_score(gts, res)
                bleu3 = 100.0 * float(score[2])
                bleu4 = 100.0 * float(score[3])
            except Exception:
                # Fall back to NLTK corpus BLEU with smoothing when the COCO tools are absent.
                try:
                    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

                    smooth = SmoothingFunction().method1
                    refs_tok = [
                        [self._tokenize_for_metrics(r) for r in references] for _ in candidates
                    ]
                    cands_tok = [self._tokenize_for_metrics(c) for c in candidates]
                    bleu3 = 100.0 * float(
                        corpus_bleu(
                            refs_tok,
                            cands_tok,
                            weights=(1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0),
                            smoothing_function=smooth,
                        )
                    )
                    bleu4 = 100.0 * float(
                        corpus_bleu(
                            refs_tok,
                            cands_tok,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smooth,
                        )
                    )
                except Exception:
                    bleu3 = None
                    bleu4 = None

        if bleu3 is not None:
            out["bleu3"] = float(bleu3)
        if bleu4 is not None:
            out["bleu4"] = float(bleu4)

        if self.compute_bertscore and references:
            scorer = self._get_bertscorer()
            if scorer is not None:
                # Compare against the first reference; max-over-references is too expensive here.
                refs = [references[0]] * len(candidates)
                try:
                    with torch.inference_mode():
                        with torch.autocast(device_type="cuda", enabled=False):
                            _, _, f1 = scorer.score(candidates, refs, verbose=False)
                    out["bertscore_f1"] = float(f1.mean().item())
                except Exception:
                    pass

        return out

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        scaffold: Any = None,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Score sentences and their prefixes against the prompt's concept set."""

        if sentences is None or sentences.ndim < 1:
            empty = torch.zeros(
                1, device=sentences.device if torch.is_tensor(sentences) else "cpu"
            )
            return {
                "invalid": empty.unsqueeze(1),
                "global_score": empty,
                "local_score": empty.unsqueeze(1),
                "full_tokens": [""],
            }

        termination_token_id = (
            self.termination_token_id if self.termination_token_id >= 0 else tokenizer.eos_token_id
        )
        batch_size, seq_len = sentences.shape
        device = sentences.device

        concepts, references = self._parse_scaffold(scaffold)
        concepts_norm = [self._normalize_text(c) for c in concepts if self._normalize_text(c)]
        ref_for_shaping = references[0] if references else None

        # States start out invalid; reachable prefixes are marked valid below.
        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        local_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.zeros(batch_size, device=device)
        full_tokens: list[str] = []

        invalid[:, 0] = 1.0  # empty prefix

        for i in range(batch_size):
            row = sentences[i]
            try:
                stop_pos = (row == termination_token_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                stop_pos = seq_len

            term_state = min(int(stop_pos), seq_len)

            final_text = tokenizer.decode(row[:stop_pos], skip_special_tokens=True).strip()
            final_text = self._sanitize_caption(final_text)
            full_tokens.append(final_text)

            cov_final, hard_ok = self._coverage_ratio_lemma(final_text, concepts_norm)
            global_score[i] = float(cov_final)

            legal_ok = self._is_english_sentence_like(final_text)
            if self.validity_mode == "keywords":
                # Alternative validity notion: a sentence is valid iff it covers every concept.
                legal_ok = bool(hard_ok)

            # Prefix scores: soft coverage plus optional quality and terminability shaping.
            for pos in range(stop_pos):
                prefix_text = tokenizer.decode(row[: pos + 1], skip_special_tokens=True)
                prefix_text = self._sanitize_caption(prefix_text)
                if not self._chars_ok(prefix_text):
                    invalid[i, pos + 1] = 1.0
                    local_score[i, pos + 1] = 0.0
                    continue
                cov_pref, _ = self._coverage_ratio_surface(prefix_text, concepts_norm)
                q_pref = 0.0
                if ref_for_shaping:
                    q_pref = self._ngram_f1(prefix_text, ref_for_shaping, self.ngram_n)

                v_pref = 0.0
                if self.valid_step_weight != 0.0 and ((pos + 1) % self.valid_step_every == 0):
                    v_pref = 1.0 if self._would_be_valid_if_terminated(prefix_text) else 0.0
                local = (self.coverage_weight * cov_pref) + (self.quality_weight * q_pref)
                local = local + (self.valid_step_weight * v_pref)
                local_score[i, pos + 1] = float(local)

                # Character-legal prefixes are reachable states.
                invalid[i, pos + 1] = 0.0

            invalid[i, term_state] = 0.0 if legal_ok else 1.0

            if legal_ok and self.valid_terminal_bonus != 0.0:
                local_score[i, term_state] = local_score[i, term_state] + float(
                    self.valid_terminal_bonus
                )

            if hard_ok:
                local_score[i, term_state] = local_score[i, term_state] + float(
                    self.hard_coverage_bonus
                )

            invalid[i, -1] = invalid[i, term_state].clone()

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens,
        }
