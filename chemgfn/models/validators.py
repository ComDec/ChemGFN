from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable, Literal

import partialsmiles as ps
import spacy
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

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


class RuleSentenceValidator(SentenceValidator):
    """Rule-based validator adapted from the original ``next_sentence`` task."""

    name = "rule_sentence"
    requires_target_molecule = False

    def __init__(
        self,
        termination_token_id: int = -1,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        super().__init__(self.name, termination_token_id=termination_token_id)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as exc:
            raise ImportError(
                f"SpaCy model '{spacy_model}' is not installed. "
                "Run `python -m spacy download en_core_web_sm`."
            ) from exc

    def _is_valid_sentence(self, text: str) -> bool:
        doc = self.nlp(text.strip())
        tokens = [t for sent in doc.sents for t in sent]
        if len(tokens) < 2:
            return False

        # Require the first non-space token to be capitalized
        first_tok = next((t for t in tokens if not t.is_space), None)
        if first_tok is None or not first_tok.is_title:
            return False

        has_noun = any(tok.pos_ in {"NOUN", "PROPN", "PRON"} for tok in tokens)
        has_verb = any(tok.pos_ in {"VERB", "AUX"} for tok in tokens)
        return has_noun and has_verb

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
        *args,
        **kwargs,
    ) -> dict[str, Tensor | list[str]]:
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

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        local_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.zeros(batch_size, device=device)
        full_tokens: list[str] = []

        invalid[:, 0] = 1.0  # empty prefix is considered invalid

        for i in range(batch_size):
            row = sentences[i]
            try:
                stop_pos = (row == termination_token_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                stop_pos = seq_len

            for pos in range(stop_pos):
                prefix_text = tokenizer.decode(row[: pos + 1], skip_special_tokens=False)
                is_valid = self._is_valid_sentence(prefix_text)
                invalid[i, pos + 1] = 0.0 if is_valid else 1.0
                local_score[i, pos + 1] = 1.0 if is_valid else 0.0

            final_text = tokenizer.decode(row[:stop_pos], skip_special_tokens=False)
            full_tokens.append(final_text)
            final_valid = stop_pos > 0 and (invalid[i, stop_pos] == 0)
            global_score[i] = 1.0 if final_valid else 0.0
            invalid[i, -1] = 0.0 if final_valid else 1.0

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens,
        }


class CoLASentenceValidator(SentenceValidator):
    """Model-based grammaticality validator using CoLA acceptability."""

    name = "cola_sentence"
    requires_target_molecule = False

    def __init__(
        self,
        termination_token_id: int = -1,
        model_name: str = "textattack/roberta-base-CoLA",
        invalid_threshold: float = 0.2,
    ) -> None:
        super().__init__(self.name, termination_token_id=termination_token_id)
        self.invalid_threshold = float(invalid_threshold)
        self.model_name = model_name
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def _batch_invalid_prob(self, texts: list[str]) -> torch.Tensor:
        inputs = self.model_tokenizer(texts, padding=True, return_tensors="pt").to(
            self.model.device
        )
        logits = self.model(**inputs).logits
        probs = logits.softmax(dim=-1)
        # label 0 corresponds to "unacceptable" in CoLA
        return probs[:, 0].detach()

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        target_molecule: str | None = None,
        *args,
        **kwargs,
    ) -> dict[str, Tensor | list[str]]:
        termination_token_id = (
            self.termination_token_id if self.termination_token_id >= 0 else tokenizer.eos_token_id
        )
        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.ones(batch_size, seq_len + 1, device=device)
        local_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.zeros(batch_size, device=device)
        full_tokens: list[str] = []

        invalid[:, 0] = 1.0  # empty prefix is invalid
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for pos in range(seq_len):
            slice_tokens = sentences[:, : pos + 1]
            done |= slice_tokens[:, -1] == termination_token_id
            if done.all():
                break

            texts = tokenizer.batch_decode(slice_tokens)
            invalid_prob = self._batch_invalid_prob(texts).to(device)

            invalid_mask = invalid_prob > self.invalid_threshold
            invalid[~done, pos + 1] = invalid_mask[~done].to(dtype=invalid.dtype)
            # Set local score to 0 on invalid positions; otherwise 1 - prob
            safe_score = (1.0 - invalid_prob).clamp(min=0.0)
            safe_score = torch.where(invalid_mask, torch.zeros_like(safe_score), safe_score)
            local_score[~done, pos + 1] = safe_score[~done].to(dtype=local_score.dtype)

        for i in range(batch_size):
            row = sentences[i]
            try:
                stop_pos = (row == termination_token_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                stop_pos = seq_len
            final_text = tokenizer.decode(row[:stop_pos], skip_special_tokens=False)
            full_tokens.append(final_text)

            final_score = local_score[i, min(stop_pos, seq_len)]
            global_score[i] = final_score
            invalid[i, -1] = 0.0 if final_score > 0 else 1.0

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens,
        }


class InfillSentenceValidator(SentenceValidator):
    """Validator + evaluator for story infilling completions."""

    name = "infill_sentence"
    requires_target_molecule = False

    def __init__(
        self,
        termination_token_id: int = -1,
        bertscore_model_name: str = "microsoft/deberta-xlarge-mnli",
        bertscore_lang: str = "en",
        bertscore_rescale: bool = True,
        compute_bleu: bool = True,
        compute_gleu: bool = True,
        compute_bertscore: bool = True,
        compute_glue: bool = False,
        glue_model_name: str = "textattack/roberta-base-CoLA",
        glue_batch_size: int = 32,
        skip_train_metrics: bool = True,
    ) -> None:
        super().__init__(self.name, termination_token_id=termination_token_id)
        self.compute_bleu = bool(compute_bleu)
        self.compute_gleu = bool(compute_gleu)
        self.compute_bertscore = bool(compute_bertscore)
        self.compute_glue = bool(compute_glue)
        self.glue_batch_size = int(glue_batch_size)
        self.skip_train_metrics = bool(skip_train_metrics)

        self.bertscore_model_name = bertscore_model_name
        self.bertscore_lang = bertscore_lang
        self.bertscore_rescale = bool(bertscore_rescale)

        self._bertscorer = None

        self._glue_tokenizer = None
        self._glue_model = None
        if self.compute_glue:
            self._glue_tokenizer = AutoTokenizer.from_pretrained(glue_model_name)
            self._glue_model = AutoModelForSequenceClassification.from_pretrained(
                glue_model_name, device_map="auto"
            )
            self._glue_model.eval()

    def _get_bertscorer(self):
        if not self.compute_bertscore:
            return None
        if self._bertscorer is not None:
            return self._bertscorer
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError(
                "BERTScore is required. Install with `pip install bert-score`."
            ) from exc
        device = "cpu"
        self._bertscorer = BERTScorer(
            model_type=self.bertscore_model_name,
            lang=self.bertscore_lang,
            rescale_with_baseline=self.bertscore_rescale,
            device=device,
        )
        self._bertscore_device = device
        # Ensure stable float32 compute (avoid bf16 autocast issues).
        try:
            model = getattr(self._bertscorer, "_model", None)
            if model is not None:
                model = model.float().to(dtype=torch.float32)
                self._bertscorer._model = model
        except Exception:
            pass
        return self._bertscorer

    def _decode_batch(self, sentences: Tensor, tokenizer: PreTrainedTokenizer) -> list[str]:
        termination_token_id = (
            self.termination_token_id if self.termination_token_id >= 0 else tokenizer.eos_token_id
        )
        decoded: list[str] = []
        for row in sentences:
            try:
                stop_pos = (row == termination_token_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                stop_pos = row.shape[0]
            text = tokenizer.decode(row[:stop_pos], skip_special_tokens=False).strip()
            decoded.append(text)
        return decoded

    @torch.no_grad()
    def _glue_scores(self, candidates: list[str]) -> torch.Tensor:
        if self._glue_model is None or self._glue_tokenizer is None:
            return torch.zeros(len(candidates))
        probs = []
        for start in range(0, len(candidates), self.glue_batch_size):
            batch = candidates[start : start + self.glue_batch_size]
            inputs = self._glue_tokenizer(batch, padding=True, return_tensors="pt").to(
                self._glue_model.device
            )
            logits = self._glue_model(**inputs).logits
            batch_probs = logits.softmax(dim=-1)
            probs.append(batch_probs[:, 1].detach().cpu())
        return torch.cat(probs, dim=0)

    def accuracy(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        reference: str | list[str] | None = None,
        **kwargs,
    ) -> dict[str, float]:
        if self.skip_train_metrics and torch.is_grad_enabled():
            return {}
        if reference is None:
            return {}

        candidates = self._decode_batch(sentences, tokenizer)
        if isinstance(reference, (list, tuple)):
            refs = [str(x) for x in reference]
        else:
            refs = [str(reference)] * len(candidates)

        # Match eval_infill.py: drop the first character of generated text.
        candidates = [cand[1:] if cand else cand for cand in candidates]

        metrics: dict[str, float] = {}

        if self.compute_bleu or self.compute_gleu:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
            from nltk.translate.gleu_score import sentence_gleu

            bleu_scores = []
            gleu_scores = []
            for cand, ref in zip(candidates, refs):
                ref_tokens = ref.split()
                cand_tokens = cand.split()
                if self.compute_bleu:
                    bleu_scores.append(
                        sentence_bleu(
                            [ref_tokens],
                            cand_tokens,
                            smoothing_function=SmoothingFunction().method1,
                        )
                    )
                if self.compute_gleu:
                    gleu_scores.append(sentence_gleu([ref_tokens], cand_tokens))
            if self.compute_bleu:
                metrics["bleu"] = float(sum(bleu_scores) / max(1, len(bleu_scores)))
            if self.compute_gleu:
                metrics["gleu"] = float(sum(gleu_scores) / max(1, len(gleu_scores)))

        if self.compute_bertscore:
            scorer = self._get_bertscorer()
            if scorer is not None:
                use_cpu = (
                    torch.is_autocast_enabled()
                    and torch.get_autocast_gpu_dtype() == torch.bfloat16
                )
                if use_cpu and getattr(scorer, "device", "cuda") != "cpu":
                    try:
                        scorer.device = "cpu"
                        model = getattr(scorer, "_model", None)
                        if model is not None:
                            scorer._model = model.to("cpu")
                    except Exception:
                        pass
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", enabled=False):
                        _, _, f1 = scorer.score(candidates, refs, verbose=False)
                if use_cpu and getattr(self, "_bertscore_device", None) == "cuda":
                    try:
                        scorer.device = "cuda"
                        model = getattr(scorer, "_model", None)
                        if model is not None:
                            scorer._model = model.to("cuda")
                    except Exception:
                        pass
                metrics["bertscore_f1"] = float(f1.mean().item())

        if self.compute_glue:
            glue_probs = self._glue_scores(candidates)
            if glue_probs.numel() > 0:
                metrics["glue"] = float(glue_probs.mean().item())
                metrics["glue_acc"] = float((glue_probs > 0.5).float().mean().item())

        return metrics

    def __call__(
        self,
        sentences: Tensor,
        tokenizer: PreTrainedTokenizer,
        reference: str | None = None,
        *args,
        **kwargs,
    ) -> dict[str, Tensor | list[str]]:
        if sentences is None or sentences.ndim < 1:
            empty = torch.zeros(
                1, device=sentences.device if torch.is_tensor(sentences) else "cpu"
            )
            return {
                "invalid": empty.unsqueeze(1),
                "global_score": empty + 1.0,
                "local_score": empty.unsqueeze(1),
                "full_tokens": [""],
            }

        termination_token_id = (
            self.termination_token_id if self.termination_token_id >= 0 else tokenizer.eos_token_id
        )
        batch_size, seq_len = sentences.shape
        device = sentences.device

        invalid = torch.zeros(batch_size, seq_len + 1, device=device)
        invalid[:, 0] = 0.0
        local_score = torch.zeros(batch_size, seq_len + 1, device=device)
        global_score = torch.ones(batch_size, device=device)

        full_tokens = []
        for row in sentences:
            try:
                stop_pos = (row == termination_token_id).nonzero(as_tuple=True)[0][0].item()
            except Exception:
                stop_pos = seq_len
            text = tokenizer.decode(row[:stop_pos], skip_special_tokens=False).strip()
            full_tokens.append(text)

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,
            "full_tokens": full_tokens,
        }


class Expr24Validator(SentenceValidator):
    """
    Target-sum validator (default target=24).
    Supports variable-length expressions with digits and +-*/, optionally with parentheses,
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

        # Whether to amortize the valid state of the expression to the entire batch
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
            # Accuracy is a strict hit: expression must evaluate exactly to target.
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

        invalid[:, 0] = 1.0  # empty prefix

        dense_mode = self.scorer in {"hit24_dense", "near_24_dense"}

        for i in range(batch_size):
            # determine actual length up to EOS (excluded)
            stop_pos = seq_len
            for pos in range(seq_len):
                if sentences[i, pos] == termination_token_id:
                    stop_pos = pos
                    break

            # prefix evaluation for invalid/local scores
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

            # populate final position (last observed prefix or full length)
            last_pos = stop_pos if stop_pos < seq_len else seq_len
            if last_pos >= 1:
                local_score[i, last_pos] = float(score)
                invalid[i, last_pos] = 0.0 if (is_valid and value == self.target_value) else 1.0

            global_score[i] = float(score)
            invalid[i, -1] = 0.0 if (is_valid and value == self.target_value) else 1.0
            full_tokens_list.append(final_expr)

        if self.amortize_valid_state:
            # Vectorized: create mask for entries where global_score > 0
            mask = global_score > 0
            local_score[mask, 1:] = global_score[mask].unsqueeze(1)
            invalid[mask, 1:] = 0.0

        return {
            "invalid": invalid,
            "global_score": global_score,
            "local_score": local_score,  # placeholder for future shaping
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

        # Fast path: no parentheses allowed/used -> original alternating pattern
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

        # Parentheses-capable parser using shunting-yard + eval
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

        prev_token_type = None  # 'num' | ')' | 'op' | '('
        for tk in tokens:
            if tk.isdigit():
                # number
                values.append(Fraction(int(tk)))
                prev_token_type = "num"
            elif tk in "+-*/":
                # operator cannot follow start, '(' or another operator
                if prev_token_type not in {"num", ")"}:
                    return None
                while ops and ops[-1] not in "(" and precedence(ops[-1]) >= precedence(tk):
                    if not pop_and_apply():
                        return None
                ops.append(tk)
                prev_token_type = "op"
            elif tk == "(":
                # '(' allowed at start or after operator
                if prev_token_type not in {None, "op", "("}:
                    return None
                ops.append(tk)
                prev_token_type = "("
            elif tk == ")":
                if prev_token_type not in {"num", ")"}:
                    return None
                # pop until matching '('
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
        """
        tokens: [B, T]
        returns:
          first_eos: [B]  (0..T)   T means "no EOS"
          has_eos:   [B]  bool
        """
        B, T = tokens.shape
        device = tokens.device
        pos = torch.arange(T, device=device).view(1, T).expand(B, T)  # [B,T]
        is_eos = tokens.eq(eos_id)
        has_eos = is_eos.any(dim=1)
        eos_pos = torch.where(is_eos, pos, torch.full_like(pos, T))
        first_eos = eos_pos.min(dim=1).values  # [B], T if none
        return first_eos, has_eos

    @staticmethod
    def _stats_1d(vals: list[int]) -> dict[str, float]:
        """
        vals: non-empty list of ints
        returns mean/std/min/max/p50/p90/p95/p99
        """
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
            # nearest-rank-ish; stable and simple
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
        """
        bins: list of (lo, hi) inclusive, e.g. [(0,2),(3,5),(6,8),(9,10),(11,10**9)]
        returns fraction per bin
        """
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
        """
        Mean_{i<j} Tanimoto(fp_i, fp_j). Returns 1.0 if <2 fps (so diversity becomes 0.0).
        Uses BulkTanimotoSimilarity for speed.
        """
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
        return_hist: bool = False,  # <-- 新增：需要 wandb 直方图时开
    ) -> dict[str, Any]:
        """
        Returns:
          - scalars for easy logging
          - optional raw arrays for wandb.Histogram if return_hist=True
        """
        num_samples = int(generated_tokens.shape[0])
        if num_samples == 0:
            out: dict[str, Any] = {
                "acc": 0.0,
                f"{self.scorer_name}": 0.0,
                f"{self.scorer_name}_filter": 0.0,
                "fp_div_internal_valid": 0.0,
                "fp_div_topk_valid": 0.0,
                # length scalars
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

        # -------- token lengths (before EOS) --------
        eos_id = int(tokenizer.eos_token_id)
        first_eos, has_eos = self._first_eos_pos(generated_tokens, eos_id)  # [B], [B]
        # length excluding EOS: if eos at pos p => length = p; if no eos => length = T
        tok_lens = first_eos.to(torch.int64).tolist()
        eos_rate = float(has_eos.float().mean().item())

        # decode
        decoded = self._decode_batch(generated_tokens, tokenizer)

        # one pass: build mols, validity, scores, and char lengths
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

        # ---- whole-molecule FP diversity (valid-only) ----
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

        # ---- length scalars (token + char) ----
        tok_stats = self._stats_1d(tok_lens)
        vt_stats = self._stats_1d(valid_tok_lens)
        char_stats = self._stats_1d(char_lens)
        vchar_stats = self._stats_1d(valid_char_lens)

        # optional: simple bins for quick dashboard bar-plots
        bins = [(0, 2), (3, 5), (6, 8), (9, 10), (11, -1)]
        tok_bins = self._len_bins(tok_lens, bins)
        vt_bins = self._len_bins(valid_tok_lens, bins)

        out: dict[str, Any] = {
            "acc": float(total_valid / num_samples),
            f"{self.scorer_name}": float(avg_score),
            f"{self.scorer_name}_filter": float(filtered_score),
            "fp_div_internal_valid": float(fp_div_internal_valid),
            "fp_div_topk_valid": float(fp_div_topk_valid),
            # token length (all)
            "len_tok_mean": tok_stats["mean"],
            "len_tok_std": tok_stats["std"],
            "len_tok_min": tok_stats["min"],
            "len_tok_max": tok_stats["max"],
            "len_tok_p50": tok_stats["p50"],
            "len_tok_p90": tok_stats["p90"],
            "len_tok_p95": tok_stats["p95"],
            "len_tok_p99": tok_stats["p99"],
            "len_tok_eos_rate": float(eos_rate),
            # token length (valid-only)
            "len_tok_valid_mean": vt_stats["mean"],
            "len_tok_valid_std": vt_stats["std"],
            "len_tok_valid_min": vt_stats["min"],
            "len_tok_valid_max": vt_stats["max"],
            "len_tok_valid_p50": vt_stats["p50"],
            "len_tok_valid_p90": vt_stats["p90"],
            "len_tok_valid_p95": vt_stats["p95"],
            "len_tok_valid_p99": vt_stats["p99"],
            # char length (SMILES string length after scaffold merge)
            "len_char_mean": char_stats["mean"],
            "len_char_valid_mean": vchar_stats["mean"],
        }

        # add bin fractions (all + valid-only)
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

        # optional raw lists for wandb histograms
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

        # small micro-opt: cache decoded prefixes per batch element to avoid repeated decode of same span
        # (still O(seq_len^2) tokens processed, but fewer Python calls in practice)
        for b in range(batch_size):
            # prefix decoding cache: pos -> string
            prefix_cache: dict[int, str] = {}

            for pos in range(seq_len):
                tok = int(sentences[b, pos].item())
                if tok == termination_token_id:
                    break

                # decode prefix [0:pos+1]
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
