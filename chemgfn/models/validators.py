from __future__ import annotations

import math
import os
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


class CommonGenValidator(SentenceValidator):
    """CommonGen keyword-to-sentence validator.

    Uses a per-prompt concept set (keywords) and optional reference(s) for quality metrics.
    The `reference` argument is passed through the existing `scaffold` plumbing.
    """

    name = "common_gen"
    requires_target_molecule = False

    def __init__(
        self,
        termination_token_id: int = -1,
        spacy_model: str = "en_core_web_sm",
        # reward shaping weights (fed into local_score)
        coverage_weight: float = 1.0,
        quality_weight: float = 0.5,
        hard_coverage_bonus: float = 1.0,
        # sentence validity (used for invalid mask + acc metric)
        validity_mode: Literal["english", "keywords"] = "english",
        strict_ascii: bool = True,
        allowed_punctuation: str = " .,!?;:'\"()/-",
        require_capital_start: bool = True,
        require_verb: bool = True,
        min_alpha_tokens: int = 2,
        single_sentence: bool = True,
        require_terminal_punct: bool = False,
        # shaping: reward prefixes that could be valid if terminated with '.'
        valid_step_weight: float = 0.0,
        valid_step_every: int = 1,
        valid_terminal_bonus: float = 0.0,
        # quality metric (used for shaping in __call__)
        ngram_n: int = 2,
        # metrics (for reporting)
        compute_bleu: bool = False,
        compute_bertscore: bool = False,
        bertscore_model_name: str = "microsoft/deberta-xlarge-mnli",
        bertscore_lang: str = "en",
        bertscore_rescale: bool = True,
    ) -> None:
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

        # For logging: match CommonGen paper naming.
        self.scorer_name = "coverage"

        try:
            self._nlp = spacy.load(spacy_model)
        except OSError as exc:
            raise ImportError(
                f"SpaCy model '{spacy_model}' is not installed. Run `python -m spacy download {spacy_model}`."
            ) from exc

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _sanitize_caption(text: str) -> str:
        # Remove control chars and replacement char that can break PTBTokenizer.
        s = text or ""
        s = s.replace("\ufffd", " ")
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
        # skip leading punctuation/quotes
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
        # Single sentence constraint (optional).
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
        # Check legality/English-ness using the same logic as `acc`, but for an early termination.
        # We explicitly append '.' because this task uses '.' as EOS.
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
        # Backwards-compatible fallback tokenization.
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    def _tokenize_for_metrics(self, text: str) -> list[str]:
        # COCO-style evaluation uses PTB tokenization; TreebankWordTokenizer is a close approximation.
        if self._treebank_tokenizer is None:
            from nltk.tokenize import TreebankWordTokenizer

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

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den != 0 else 0.0

    @staticmethod
    def _ngram_counts(tokens: list[str], n: int) -> Counter:
        n = max(1, int(n))
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    # (CIDEr / METEOR / SPICE are intentionally not implemented in this task for now.)

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
        self._bertscorer = BERTScorer(
            model_type=self.bertscore_model_name,
            lang=self.bertscore_lang,
            rescale_with_baseline=self.bertscore_rescale,
            device="cpu",
        )
        return self._bertscorer

    @staticmethod
    def _parse_scaffold(scaffold: Any) -> tuple[list[str], list[str]]:
        # Returns (concepts, references)
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
            # assume list of references
            references = [str(x) for x in scaffold if str(x)]
            return [], references
        if isinstance(scaffold, str):
            # maybe a JSON-encoded dict
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
        if sentences is None or sentences.ndim == 0:
            return {"acc": 0.0, "cov_ratio": 0.0, "cov_ratio_filter": 0.0, "ngram_f1": 0.0}

        concepts, references = self._parse_scaffold(scaffold)
        concepts_norm = [self._normalize_text(c) for c in concepts if self._normalize_text(c)]

        # decode candidates
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
        # Replace `acc` to mean "legal English sentence rate".
        # Keep keyword full-coverage as a separate metric (`keyword_acc`).
        acc = float(sum(legal_flags) / max(1, len(legal_flags)))
        keyword_acc = float(sum(cov_hard) / max(1, len(cov_hard)))
        cov_mean = float(sum(cov_ratio) / max(1, len(cov_ratio)))
        # filter = among hard-valid sequences
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
            # Paper reports coverage as percentage.
            "coverage": 100.0 * cov_mean,
            "coverage_filter": 100.0 * cov_filter,
            "ngram_f1": float(sum(ngram_f1) / max(1, len(ngram_f1))),
        }

        # Avoid heavy NLP metrics during training steps.
        if torch.is_grad_enabled():
            return out

        if self.compute_bleu and references and candidates:
            # Match CommonGen paper: report BLEU-3/4 (typically *100) with PTB tokenization.
            # Prefer the standard COCO caption BLEU implementation.
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
                # Fallback: NLTK corpus BLEU with a simple smoothing.
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
                # compare to the first reference for stability; max-over-refs is expensive
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

        concepts, references = self._parse_scaffold(scaffold)
        concepts_norm = [self._normalize_text(c) for c in concepts if self._normalize_text(c)]
        ref_for_shaping = references[0] if references else None

        # Default to invalid everywhere; mark reachable prefixes as valid.
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
                # Backwards-compatible mode: treat "valid" as full concept coverage.
                legal_ok = bool(hard_ok)

            # fill prefix local scores (soft coverage + optional quality)
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

                # Mark prefixes as valid if they are character-legal.
                invalid[i, pos + 1] = 0.0

            # Terminal validity: legal English sentence (default).
            invalid[i, term_state] = 0.0 if legal_ok else 1.0

            if legal_ok and self.valid_terminal_bonus != 0.0:
                local_score[i, term_state] = local_score[i, term_state] + float(
                    self.valid_terminal_bonus
                )

            # Optional terminal bonus for full keyword coverage.
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
