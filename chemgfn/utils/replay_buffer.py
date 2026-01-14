import bisect
import csv
import gzip
import heapq
import math
import os
import pickle
import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import editdistance
import numpy as np
import torch

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(
        self,
        buffer_size,
        sim_tolerance=0.25,
        strict_mode: bool = False,
        buffer_aug_value: float = 0,
    ):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.strict_mode = strict_mode
        self.buffer_aug_value = buffer_aug_value
        self.reset()

    def set_termination_token_id(self, termination_token_id):
        self.termination_token_id = termination_token_id

    def reset(self):
        self._buffer = {}

    def add(self, item, force_add=False):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]

        # Hashable string for prompt+answer
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return

        new_item = (
            item["logreward"],
            item["str_sentence"],
            item["tensor_sentence"],
            item["tensor_answer"],
            item["full_logrewards"],
            force_add,
        )
        buffer = self._buffer[str_prompt]["sentences"]

        for buffer_item in list(buffer):  # Iterate over a copy
            existing_answer = [
                x for x in buffer_item[3].tolist() if x != self.termination_token_id
            ]
            new_answer = [
                x for x in item["tensor_answer"].tolist() if x != self.termination_token_id
            ]
            if (
                editdistance.eval(new_answer, existing_answer)
                < (len(new_answer) + len(existing_answer)) * self.sim_tolerance
            ):
                if buffer_item[0] >= item["logreward"] and (not force_add):
                    return
        # Critical fix: Only add to 'exists' AFTER successful heap insertion
        if len(buffer) >= self.buffer_size:
            # Push off the smallest item if buffer is full
            popped = heapq.heappop(buffer)
            # self._buffer[str_prompt]["exists"].remove(popped[1])
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        else:
            if force_add:
                new_item = list(new_item)
                # ensure validated items are preferred
                new_item[-2][..., -1] = new_item[-2][..., -1] + self.buffer_aug_value
                new_item = tuple(new_item)
            if self.strict_mode:
                if force_add:
                    heapq.heappush(buffer, new_item)
                    self._buffer[str_prompt]["exists"].add(item["str_sentence"])
                else:
                    return
            else:
                heapq.heappush(buffer, new_item)
                self._buffer[str_prompt]["exists"].add(item["str_sentence"])

    def add_batch(self, prompt, sentences, logrewards, tokenizer, result_dict=None):
        """
        add a batch of items to the buffer
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        token_sentences = tokenizer.batch_decode(sentences)
        prompt_len = prompt.shape[1]

        for i in range(sentences.size(0)):
            # str_sentence = token_sentences[i].replace(".", "").strip()
            # there is no such termination token in the SMILES
            str_sentence = token_sentences[i].strip()
            valid_state = (result_dict["validator_dict"]["global_score"].bool())[i]
            self.add(
                {
                    "logreward": logrewards[i, (sentences[i] != self.termination_token_id)]
                    .sum()
                    .item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_answer": sentences[i][prompt_len - 1 :],
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                },
                force_add=valid_state,
            )

    def sample(self, batch_size, prompt, tokenizer):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            return None, None
        if len(self._buffer[str_prompt]["sentences"]) < batch_size:
            return None, None
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    def stat(self):
        """
        statistics of the buffer
        """
        stats = {}
        for idx, key in enumerate(self._buffer):
            total_buffer = len(self._buffer[key]["sentences"])
            avg_logR = sum([item[0] for item in self._buffer[key]["sentences"]])
            stats.update(
                {
                    f"prompt_{idx}_total_buffer": total_buffer,
                    f"prompt_{idx}_avg_logR": avg_logR / total_buffer if total_buffer > 0 else 0,
                }
            )
        return stats

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)

    def save_csv(self, path, tokenizer):
        """
        Save the buffer to a CSV file.
        Each row contains: str_prompt, str_sentence, logreward, tensor_sentence, tensor_answer
        """
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("logr_sum,tensor_answer,answer_logR,answer,force_add\n")
            for key in self._buffer:
                for item in self._buffer[key]["sentences"]:
                    answer_tokens = item[2].tolist()
                    force_add = item[-1].item()
                    f.write(
                        f"{item[0]},{answer_tokens},{item[-2].tolist()},{item[1]},{force_add}\n"
                    )


class ReplayBufferNative(ReplayBuffer):
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, sim_tolerance=0.25, **kwargs):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.reset()

    def set_termination_token_id(self, termination_token_id):
        self.termination_token_id = termination_token_id

    def reset(self):
        self._buffer = {}

    def add(self, item, force_add=False):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]

        # Hashable string for prompt+answer
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return

        new_item = (
            item["logreward"],
            item["str_sentence"],
            item["tensor_sentence"],
            item["tensor_answer"],
            item["full_logrewards"],
            force_add,
        )
        buffer = self._buffer[str_prompt]["sentences"]

        for buffer_item in list(buffer):  # Iterate over a copy
            existing_answer = [
                x for x in buffer_item[3].tolist() if x != self.termination_token_id
            ]
            new_answer = [
                x for x in item["tensor_answer"].tolist() if x != self.termination_token_id
            ]
            if (
                editdistance.eval(new_answer, existing_answer)
                < (len(new_answer) + len(existing_answer)) * self.sim_tolerance
            ):
                if buffer_item[0] >= item["logreward"] and not force_add:
                    return

        # Critical fix: Only add to 'exists' AFTER successful heap insertion
        if len(buffer) >= self.buffer_size:
            # Push off the smallest item if buffer is full
            popped = heapq.heappop(buffer)
            # self._buffer[str_prompt]["exists"].remove(popped[1])
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        else:
            heapq.heappush(buffer, new_item)
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])

    def add_batch(self, prompt, sentences, logrewards, tokenizer, result_dict=None):
        """
        add a batch of items to the buffer
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        token_sentences = tokenizer.batch_decode(sentences)
        prompt_len = prompt.shape[1]

        for i in range(sentences.size(0)):
            # str_sentence = token_sentences[i].replace(".", "").strip()
            # there is no such termination token in the SMILES
            str_sentence = token_sentences[i].strip()
            valid_state = (result_dict["validator_dict"]["global_score"].bool())[i]
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i][prompt_len - 1 :] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_answer": sentences[i][prompt_len - 1 :],
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                },
                force_add=valid_state,
            )

    def sample(self, batch_size, prompt, tokenizer):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            return None, None
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    def stat(self):
        """
        statistics of the buffer
        """
        stats = {}
        for idx, key in enumerate(self._buffer):
            total_buffer = len(self._buffer[key]["sentences"])
            avg_logR = sum([item[0] for item in self._buffer[key]["sentences"]])
            stats.update(
                {
                    f"prompt_{idx}_total_buffer": total_buffer,
                    f"prompt_{idx}_avg_logR": avg_logR / total_buffer if total_buffer > 0 else 0,
                }
            )
        return stats

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)

    def save_csv(self, path, tokenizer):
        """
        Save the buffer to a CSV file.
        Each row contains: str_prompt, str_sentence, logreward, tensor_sentence, tensor_answer
        """
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("logr_sum,tensor_answer,answer_logR,answer,force_add\n")
            for key in self._buffer:
                for item in self._buffer[key]["sentences"]:
                    answer_tokens = item[-3].tolist()
                    answer = "".join(
                        [
                            str(x)
                            for x in tokenizer.batch_decode(
                                answer_tokens, skip_special_tokens=False
                            )
                        ]
                    )
                    force_add = item[-1].item()
                    f.write(
                        f"{item[0]},{answer_tokens},{item[-2].tolist()},{answer},{force_add}\n"
                    )


class SimilarityBackend:
    """
    Similarity in [0, 1], higher => more similar.
    Must support bulk(anchor_rep, other_reps) for speed.
    """

    def prepare(self, candidates: list) -> list:
        raise NotImplementedError

    def bulk(self, anchor_rep, other_reps: list) -> list:
        raise NotImplementedError


class RDKITBulkTanimotoBackend(SimilarityBackend):
    """Default for SMILES: uses cand.fingerprint and RDKit BulkTanimotoSimilarity."""

    def prepare(self, candidates: list) -> list:
        return [getattr(c, "fingerprint", None) for c in candidates]

    def bulk(self, anchor_rep, other_reps: list) -> list:
        return list(DataStructs.BulkTanimotoSimilarity(anchor_rep, other_reps))


class ShingleJaccardBackend(SimilarityBackend):
    """
    General string similarity: k-gram shingles + Jaccard.
    Useful for expr24 (expressions) or any string domain.
    """

    def __init__(self, k: int = 2, tokenizer: Optional[callable] = None):
        self.k = int(k)
        self.tokenizer = tokenizer  # optional: str -> list[str]

    def _rep(self, s: str):
        if not s:
            return None
        if self.tokenizer is not None:
            toks = self.tokenizer(s)
            if not toks:
                return None
            k = max(1, self.k)
            if len(toks) <= k:
                return frozenset([" ".join(toks)])
            return frozenset(" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1))
        # char shingles
        k = max(1, self.k)
        if len(s) <= k:
            return frozenset([s])
        return frozenset(s[i : i + k] for i in range(len(s) - k + 1))

    def prepare(self, candidates: list) -> list:
        reps = []
        for c in candidates:
            # Prefer generic text representation; fall back to canonical_smiles for SMILES tasks.
            s = (
                getattr(c, "canonical_text", None)
                or getattr(c, "text", None)
                or getattr(c, "canonical_smiles", None)
                or getattr(c, "smiles", "")
                or ""
            )
            reps.append(self._rep(str(s)))
        return reps

    def bulk(self, anchor_rep, other_reps: list) -> list:
        if anchor_rep is None:
            return [0.0] * len(other_reps)
        A = anchor_rep
        out = []
        for B in other_reps:
            if B is None:
                out.append(0.0)
                continue
            inter = len(A & B)
            if inter == 0:
                out.append(0.0)
                continue
            union = len(A) + len(B) - inter
            out.append(float(inter) / float(union) if union > 0 else 0.0)
        return out


# =========================
# Set Functions
# =========================


class SetFunction:
    """
    Interface:
      begin(buf, candidates) -> state
      gain(buf, cand, idx, state) -> float
      upper_bound(buf, cand, idx, state) -> float  (for lazy heap)
      on_select(buf, best_idx, state, remaining_idx_list, chunk_size) -> None
    """

    def begin(self, buf, candidates: list) -> dict:
        raise NotImplementedError

    def gain(self, buf, cand, idx: int, state: dict) -> float:
        raise NotImplementedError

    def upper_bound(self, buf, cand, idx: int, state: dict) -> float:
        # Safe default for monotone submodular greedy: current gain is an upper bound on future marginal.
        return float(self.gain(buf, cand, idx, state))

    def on_select(
        self,
        buf,
        best_idx: int,
        state: dict,
        remaining_idx: List[int],
        chunk_size: int = 4096,
    ) -> None:
        raise NotImplementedError


class FacilityLengthSetFunction(SetFunction):
    """
    Objective = static_score
              + weight_div * (1 - max_sim_to_selected)
              + weight_len * alpha_bin(b) * [log(1+count_b+1) - log(1+count_b)]
    where b is the length-bin of the candidate.

    - The facility-location-like term uses cand.max_sim, updated incrementally via similarity backend bulk.
    - The length term is provably monotone submodular (concave over counts in each bin).
    - alpha_bin increases with bin index (bias toward longer bins) to counter length collapse.
    """

    def _item_length(self, buf, cand) -> int:
        """
        CRITICAL FIX:
        Prefer "generated token length" if available (cand.gen_len),
        then fallback to seq_len, then token_ids length, then smiles char length.

        This avoids the classic bug: using len(smiles) as a proxy for model token length.
        """
        gen_len = getattr(cand, "gen_len", None)
        if gen_len is not None:
            try:
                return int(gen_len)
            except Exception:
                pass

        seq_len = getattr(cand, "seq_len", None)
        if seq_len is not None:
            try:
                return int(seq_len)
            except Exception:
                pass

        tok = getattr(cand, "token_ids", None)
        if tok is not None:
            try:
                return int(tok.numel()) if hasattr(tok, "numel") else int(len(tok))
            except Exception:
                pass

        s = getattr(cand, "canonical_smiles", None) or getattr(cand, "smiles", "") or ""
        return int(len(s))

    def begin(self, buf, candidates: list) -> dict:
        sim = buf._get_similarity_backend()
        reps = sim.prepare(candidates)

        # reset max_sim (selector may also do it; but do it here to make SF self-contained)
        for c in candidates:
            c.max_sim = 0.0

        # length bins (bin_size is *size*, not number of bins)
        bin_size = max(1, int(getattr(buf, "length_bin_size", 10)))
        lengths = [self._item_length(buf, c) for c in candidates]
        bin_idx = [min(10**9, (int(L) // bin_size)) for L in lengths]
        nbins = (max(bin_idx) + 1) if bin_idx else 1

        # alpha increases with bin (favor longer to fight short oversampling)
        p = float(getattr(buf, "length_alpha_power", 1.0))
        if nbins <= 1:
            alpha = [1.0]
        else:
            alpha = [((b + 1) / nbins) ** p for b in range(nbins)]

        counts = [0] * nbins

        return {
            "candidates": candidates,
            "sim": sim,
            "reps": reps,
            "len_bin": bin_idx,
            "len_counts": counts,
            "len_alpha": alpha,
        }

    def _len_marginal(self, buf, b: int, state: dict) -> float:
        w_len = float(getattr(buf, "weight_len", 0.0))
        if w_len <= 0:
            return 0.0
        counts = state["len_counts"]
        alpha = state["len_alpha"]
        c = counts[b]
        # alpha_b * (log(1+c+1) - log(1+c)) = alpha_b * log((c+2)/(c+1))
        return w_len * float(alpha[b]) * (math.log1p(c + 1) - math.log1p(c))

    def gain(self, buf, cand, idx: int, state: dict) -> float:
        g = float(getattr(cand, "static_score", 0.0))

        # diversity (facility-style)
        w_div = float(getattr(buf, "weight_div", 0.0))
        if w_div > 0:
            g += w_div * (1.0 - float(getattr(cand, "max_sim", 0.0)))

        # length-collapse mitigation
        b = state["len_bin"][idx]
        g += self._len_marginal(buf, b, state)
        return float(g)

    def upper_bound(self, buf, cand, idx: int, state: dict) -> float:
        return float(self.gain(buf, cand, idx, state))

    def on_select(
        self,
        buf,
        best_idx: int,
        state: dict,
        remaining_idx: List[int],
        chunk_size: int = 4096,
    ) -> None:
        candidates = state["candidates"]

        # update length counts first (selected item belongs to its bin)
        b = state["len_bin"][best_idx]
        state["len_counts"][b] += 1

        # update facility max_sim for remaining via bulk similarity
        w_div = float(getattr(buf, "weight_div", 0.0))
        if w_div <= 0:
            return

        reps = state["reps"]
        rep_sel = reps[best_idx]
        if rep_sel is None:
            return

        # gather remaining with non-null reps
        idxs = []
        other_reps = []
        for j in remaining_idx:
            r = reps[j]
            if r is None:
                continue
            idxs.append(j)
            other_reps.append(r)
        if not idxs:
            return

        sim: SimilarityBackend = state["sim"]
        for start in range(0, len(idxs), chunk_size):
            block_idxs = idxs[start : start + chunk_size]
            block_reps = other_reps[start : start + chunk_size]
            sims = sim.bulk(rep_sel, block_reps)
            for j, s in zip(block_idxs, sims):
                if s > candidates[j].max_sim:
                    candidates[j].max_sim = float(s)


# =========================
# Replay Buffer
# =========================


class ReplayBufferSubmodular:
    """
    A replay buffer that maintains a subset of items maximizing a weighted submodular objective.
    Supports diversity (similarity coverage), validity, reward, and (NEW) length-collapse mitigation
    via a submodular length-bin coverage term.
    """

    class BufferItem:
        """Container for an item in the buffer (e.g., a molecule) with precomputed properties."""

        __slots__ = (
            "smiles",
            "canonical_smiles",
            "text",
            "canonical_text",
            "reward",
            "valid",
            "fingerprint",
            "static_score",
            "max_sim",
            "last_eval",
            "selected",
            # ---- LENGTH (CRITICAL FIX) ----
            "seq_len",  # tokens before EOS (as stored)
            "gen_len",  # generation length proxy (preferred for length SF)
        )

        def __init__(
            self,
            text: str,
            reward: float,
            weight_val: float,
            weight_rew: float,
            is_valid: bool,
        ):
            # NOTE: `text` is the generic decoded string (SMILES or expression, etc.)
            self.text = text
            self.canonical_text = text
            # legacy naming to avoid widespread refactors
            self.smiles = text
            self.canonical_smiles = text
            try:
                self.reward = float(reward)
            except Exception:
                self.reward = float(reward.item()) if hasattr(reward, "item") else float(reward)

            # validity is determined upstream by validators; do not override here
            self.valid = bool(is_valid)
            self.fingerprint = None

            # Optional canonicalization for SMILES-like strings (kept for SMILES tasks)
            if self.valid and _HAVE_RDKIT:
                try:
                    mol = Chem.MolFromSmiles(text)
                    if mol is not None:
                        self.canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        self.canonical_text = self.canonical_smiles
                        self.fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                            mol, radius=2, nBits=2048
                        )
                except Exception:
                    pass

            self.static_score = (weight_rew * self.reward) + (weight_val if self.valid else 0.0)

            self.max_sim = 0.0
            self.last_eval = 0
            self.selected = False

            # default lengths (will be overwritten in add_batch)
            self.seq_len = 0
            self.gen_len = 0

        def __repr__(self):
            return f"BufferItem(smiles={self.canonical_smiles}, reward={self.reward:.4f}, valid={self.valid})"

    def __init__(
        self,
        buffer_size: int,
        per_prompt: bool = False,
        weight_div: float = 1.0,
        diversity_valid_only: bool = False,
        diversity_valid_ratio: float = 1.0,
        weight_val: float = 1.0,
        weight_rew: float = 1.0,
        default_strategy: str = "standard",
        validator_key: str = "local_score",
        validator_type: Literal["mean", "max", "last"] = "max",
        stochastic_epsilon: float = 0.1,
        ltlg_epsilon: float = 0.1,
        similarity_backend: SimilarityBackend = None,
        set_function_obj: SetFunction = None,
        **kwargs,
    ):
        self.buffer_size = buffer_size
        self.per_prompt = per_prompt

        self.weight_div = weight_div
        # Diversity gating: 0 => ignore validity; 1 => only valid; (0,1) => mix.
        self.diversity_valid_only = bool(diversity_valid_only)
        # ratio takes precedence; clamp to [0,1]
        r = float(diversity_valid_ratio)
        r = 0.0 if r < 0 else 1.0 if r > 1 else r
        if self.diversity_valid_only:
            r = 1.0
        self.diversity_valid_ratio = r
        self.weight_val = weight_val
        self.weight_rew = weight_rew

        self.validator_key = validator_key
        self.validator_type = validator_type

        self.stochastic_epsilon = stochastic_epsilon
        self.ltlg_epsilon = ltlg_epsilon

        # weight_len: strength of length-bin submodular coverage term
        self.weight_len = float(kwargs.get("weight_len", 0.0))
        # length_bin_size: bin granularity (in token length; NOT number of bins)
        self.length_bin_size = int(kwargs.get("length_bin_size", 10))
        # length_alpha_power: >1 biases more toward longer bins
        self.length_alpha_power = float(kwargs.get("length_alpha_power", 1.0))

        self._similarity_backend = similarity_backend
        self._set_function_obj = set_function_obj

        if per_prompt:
            self.buffer: Dict[str, Dict[str, Any]] = {}
        else:
            self.buffer = {"items": [], "data": {}}

        self.default_strategy = default_strategy.lower()
        self.termination_token_id: Optional[int] = None

    # --- compatibility helpers ---
    def set_termination_token_id(self, termination_token_id: int):
        self.termination_token_id = int(termination_token_id)

    def set_buffer_size(self, buffer_size: int):
        self.buffer_size = buffer_size

    def set_weight_div(self, weight_div: float):
        self.weight_div = weight_div

    def set_weight_val(self, weight_val: float):
        self.weight_val = weight_val

    def set_weight_rew(self, weight_rew: float):
        self.weight_rew = weight_rew

    def set_weight_len(self, weight_len: float):
        self.weight_len = float(weight_len)

    def set_length_bins(self, bin_size: int, alpha_power: float = 1.0):
        self.length_bin_size = int(bin_size)
        self.length_alpha_power = float(alpha_power)

    def set_similarity_backend(self, backend: SimilarityBackend):
        self._similarity_backend = backend

    def set_set_function(self, sf: SetFunction):
        self._set_function_obj = sf

    def _filter_for_diversity(self, candidates: list, data_map: dict):
        """
        Optional gate: restrict facility/diversity selection to a valid-heavy subset.
        diversity_valid_ratio in [0,1]:
          - 0: keep all samples (no gating).
          - 1: keep only valid.
          - (0,1): keep all valid, plus up to N invalid so that
            valid share >= ratio. If no valid exist, fall back to all.
        """
        ratio = self.diversity_valid_ratio
        if ratio <= 0:
            return candidates, data_map
        valid = [it for it in candidates if getattr(it, "valid", False)]
        if ratio >= 1.0:
            filtered = valid
            filtered_data = {id(it): data_map.get(id(it), {}) for it in filtered}
            return filtered, filtered_data

        # ratio in (0,1)
        if not valid:
            return candidates, data_map  # nothing valid to enforce

        invalid = [it for it in candidates if not getattr(it, "valid", False)]
        # ensure valid proportion >= ratio => max_invalid <= valid_count * (1-r)/r
        max_invalid = int(math.floor(len(valid) * (1.0 - ratio) / max(ratio, 1e-12)))
        if max_invalid >= len(invalid):
            filtered = valid + invalid
        else:
            # pick best invalids by static_score (fallback reward)
            invalid_sorted = sorted(
                invalid,
                key=lambda it: (
                    getattr(it, "static_score", None),
                    getattr(it, "reward", None),
                ),
                reverse=True,
            )
            filtered = valid + invalid_sorted[:max_invalid]

        filtered_data = {id(it): data_map.get(id(it), {}) for it in filtered}
        return filtered, filtered_data

    def _get_similarity_backend(self) -> SimilarityBackend:
        b = self._similarity_backend
        if isinstance(b, SimilarityBackend):
            return b
        if b is None:
            return RDKITBulkTanimotoBackend()
        if callable(b):
            out = b()
            if not isinstance(out, SimilarityBackend):
                raise TypeError(
                    f"similarity_backend factory must return SimilarityBackend, got {type(out)}"
                )
            return out
        raise TypeError(
            f"similarity_backend must be SimilarityBackend|callable|None, got {type(b)}"
        )

    def _get_set_function(self) -> SetFunction:
        sf = self._set_function_obj
        if isinstance(sf, SetFunction):
            return sf
        if sf is None:
            return FacilityLengthSetFunction()
        if callable(sf):
            out = sf()
            if not isinstance(out, SetFunction):
                raise TypeError(
                    f"set_function_obj factory must return SetFunction, got {type(out)}"
                )
            return out
        raise TypeError(f"set_function_obj must be SetFunction|callable|None, got {type(sf)}")

    def reset(self):
        if self.per_prompt:
            self.buffer = {}
        else:
            self.buffer = {"items": [], "data": {}}

    def _get_state(self, prompt_key: str):
        if self.per_prompt:
            if prompt_key not in self.buffer:
                self.buffer[prompt_key] = {"items": [], "data": {}}
            return self.buffer[prompt_key]
        return self.buffer

    # =========================
    # main API
    # =========================

    def add_batch(
        self,
        prompt,
        sentences,
        logrewards,
        tokenizer,
        result_dict=None,
        strategy=None,
        epoch_end=False,
    ):
        assert self.termination_token_id is not None, "Call set_termination_token_id() first."
        strat = (strategy or self.default_strategy).lower()

        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        prompt_key = str_prompt if self.per_prompt else "__global__"
        st = self._get_state(prompt_key)

        eos = self.termination_token_id
        sentences = sentences.clone()
        sentences[(sentences == eos).cumsum(dim=-1) >= 1] = eos

        token_sentences = tokenizer.batch_decode(sentences, skip_special_tokens=True)
        prompt_len = int(prompt.shape[1])

        valid_mask = None
        validator_scores = None
        if result_dict is not None:
            validator = result_dict["validator_dict"]
            valid_mask = validator.get("global_score", None)
            if valid_mask is not None and hasattr(valid_mask, "bool"):
                valid_mask = valid_mask.bool()

            if self.validator_type == "last":
                validator_scores = validator.get("global_score", None)
            elif self.validator_type == "mean":
                validator_scores = torch.mean(validator.get(self.validator_key, None), dim=-1)
            elif self.validator_type == "max":
                validator_scores = torch.max(validator.get(self.validator_key, None), dim=-1)[0]

        new_items = []
        for i in range(sentences.size(0)):
            str_sentence = token_sentences[i].strip()
            is_valid = bool(valid_mask[i].item()) if valid_mask is not None else False

            if validator_scores is not None:
                r_raw = float(validator_scores[i].item())
            else:
                idx = (sentences[i] == eos).nonzero(as_tuple=True)[0][0]
                r_raw = float(logrewards[i, idx].item())

            BufferItem = type(self).BufferItem
            bi = BufferItem(
                str_sentence,
                r_raw,
                self.weight_val,
                self.weight_rew,
                is_valid=is_valid,
            )

            # ---- CRITICAL FIX: compute token lengths once, store in BufferItem + sample_data ----
            ts = sentences[i].detach().cpu()  # 1D
            eos_pos = (ts == eos).nonzero(as_tuple=True)[0]
            seq_len = int(eos_pos[0].item()) if eos_pos.numel() > 0 else int(ts.numel())

            # robust gen_len: if prompt isn't actually included, fallback to seq_len
            gen_len = seq_len - prompt_len
            if gen_len <= 0:
                gen_len = seq_len

            bi.seq_len = int(seq_len)
            bi.gen_len = int(gen_len)

            sample_data = {
                "str_prompt": str_prompt,
                "str_sentence": str_sentence,
                "tensor_sentence": ts,
                "tensor_answer": ts,
                "full_logrewards": logrewards[i].detach().cpu()
                if hasattr(logrewards, "detach")
                else logrewards[i],
                "is_valid": is_valid,
                "reward": r_raw,
                # lengths for stat/debug
                "prompt_len": int(prompt_len),
                "seq_len": int(seq_len),
                "gen_len": int(gen_len),
                "smiles_len": int(len(getattr(bi, "canonical_smiles", "") or str_sentence)),
            }
            new_items.append((bi, sample_data))

        if self.per_prompt:
            self._update_buffer_for_prompt(st, new_items, strat, epoch_end)
        else:
            self._update_buffer_single(st, new_items, strat, epoch_end)

    def sample(self, batch_size, prompt, tokenizer):
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        prompt_key = str_prompt if self.per_prompt else "__global__"
        st = self.buffer.get(prompt_key) if self.per_prompt else self.buffer
        if st is None or len(st["items"]) < batch_size:
            return None, None

        ids = np.random.choice(len(st["items"]), batch_size, replace=True)
        seqs = []
        ans = []
        for idx in ids:
            it = st["items"][idx]
            data = st["data"].get(id(it), {})
            seqs.append(data.get("tensor_sentence"))
            ans.append(data.get("tensor_answer"))
        if any(s is None for s in seqs) or any(a is None for a in ans):
            return None, None

        x = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=self.termination_token_id
        )
        y = torch.nn.utils.rnn.pad_sequence(ans, batch_first=True, padding_value=0)
        return x, y

    # =========================
    # Update buffer logic (unchanged)
    # =========================

    def _update_buffer_single(self, st: dict, new_items_list, strategy: str, epoch_end: bool):
        new_items = []
        new_data = {}
        for item in new_items_list:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], type(self).BufferItem)
            ):
                bi, data = item
            elif isinstance(item, type(self).BufferItem):
                bi, data = item, {}
            else:
                BufferItem = type(self).BufferItem
                smiles, reward = item
                bi = BufferItem(smiles, reward, self.weight_val, self.weight_rew)
                data = {}
            new_items.append(bi)
            new_data[id(bi)] = data

        if not epoch_end and strategy == "none":
            st["items"].extend(new_items)
            st["data"].update(new_data)
            return

        candidates = list(st["items"])
        data_map = dict(st["data"])
        existing_map = {it.canonical_smiles: it for it in candidates}

        for bi in new_items:
            canon = bi.canonical_smiles
            if canon in existing_map:
                existing_item = existing_map[canon]
                if bi.reward > existing_item.reward:
                    existing_item.reward = bi.reward
                    existing_item.static_score = (self.weight_rew * bi.reward) + (
                        self.weight_val if bi.valid else 0.0
                    )
                    # also carry lengths
                    existing_item.seq_len = getattr(bi, "seq_len", existing_item.seq_len)
                    existing_item.gen_len = getattr(bi, "gen_len", existing_item.gen_len)

                    data_map[id(existing_item)] = new_data.get(
                        id(bi), data_map.get(id(existing_item), {})
                    )
            else:
                candidates.append(bi)
                existing_map[canon] = bi
                data_map[id(bi)] = new_data.get(id(bi), {})

        uniq = {}
        for it in candidates:
            c = it.canonical_smiles
            if c not in uniq:
                uniq[c] = it
            else:
                if it.reward > uniq[c].reward:
                    uniq[c] = it
        candidates = list(uniq.values())

        candidates, data_map = self._filter_for_diversity(candidates, data_map)

        if len(candidates) <= self.buffer_size:
            st["items"] = candidates[: self.buffer_size]
            st["data"] = {id(it): data_map.get(id(it), {}) for it in st["items"]}
            return

        import time

        start_time = time.time()

        if strategy == "standard":
            selected_items = self._select_standard(candidates, self.buffer_size)
        elif strategy == "lazy":
            selected_items = self._select_lazy(candidates, self.buffer_size)
        elif strategy == "stochastic":
            selected_items = self._select_stochastic(
                candidates, self.buffer_size, epsilon=getattr(self, "stochastic_epsilon", 0.1)
            )
        elif strategy in ("lazier_than_lazy", "lazierthenlazygreedy", "ltlg"):
            selected_items = self._select_lazier_than_lazy(
                candidates, self.buffer_size, epsilon=getattr(self, "ltlg_epsilon", 0.1)
            )
        elif strategy == "random":
            selected_items = self._select_random(candidates, self.buffer_size)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Choose from 'standard', 'lazy', 'stochastic', "
                f"'lazier_than_lazy', 'random'."
            )

        end_time = time.time()
        print(f"Selection of {strategy} time: {end_time - start_time} seconds")

        st["items"] = selected_items
        st["data"] = {id(it): data_map.get(id(it), {}) for it in selected_items}

    def _update_buffer_for_prompt(self, st: dict, new_items_list, strategy: str, epoch_end: bool):
        # identical to single, just on st
        new_items = []
        new_data = {}
        for item in new_items_list:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], type(self).BufferItem)
            ):
                bi, data = item
            elif isinstance(item, type(self).BufferItem):
                bi, data = item, {}
            else:
                BufferItem = type(self).BufferItem
                smiles, reward = item
                bi = BufferItem(smiles, reward, self.weight_val, self.weight_rew)
                data = {}
            new_items.append(bi)
            new_data[id(bi)] = data

        if not epoch_end and strategy == "none":
            st["items"].extend(new_items)
            st["data"].update(new_data)
            return

        candidates = list(st["items"])
        data_map = dict(st["data"])
        existing_map = {it.canonical_smiles: it for it in candidates}

        for bi in new_items:
            canon = bi.canonical_smiles
            if canon in existing_map:
                existing_item = existing_map[canon]
                if bi.reward > existing_item.reward:
                    existing_item.reward = bi.reward
                    existing_item.static_score = (self.weight_rew * bi.reward) + (
                        self.weight_val if bi.valid else 0.0
                    )
                    existing_item.seq_len = getattr(bi, "seq_len", existing_item.seq_len)
                    existing_item.gen_len = getattr(bi, "gen_len", existing_item.gen_len)

                    data_map[id(existing_item)] = new_data.get(
                        id(bi), data_map.get(id(existing_item), {})
                    )
            else:
                candidates.append(bi)
                existing_map[canon] = bi
                data_map[id(bi)] = new_data.get(id(bi), {})

        uniq = {}
        for it in candidates:
            c = it.canonical_smiles
            if c not in uniq:
                uniq[c] = it
            else:
                if it.reward > uniq[c].reward:
                    uniq[c] = it
        candidates = list(uniq.values())

        candidates, data_map = self._filter_for_diversity(candidates, data_map)

        if len(candidates) <= self.buffer_size:
            st["items"] = candidates[: self.buffer_size]
            st["data"] = {id(it): data_map.get(id(it), {}) for it in st["items"]}
            return

        if strategy == "standard":
            selected_items = self._select_standard(candidates, self.buffer_size)
        elif strategy == "lazy":
            selected_items = self._select_lazy(candidates, self.buffer_size)
        elif strategy == "stochastic":
            selected_items = self._select_stochastic(
                candidates, self.buffer_size, epsilon=getattr(self, "stochastic_epsilon", 0.1)
            )
        elif strategy in ("lazier_than_lazy", "lazierthenlazygreedy", "ltlg"):
            selected_items = self._select_lazier_than_lazy(
                candidates, self.buffer_size, epsilon=getattr(self, "ltlg_epsilon", 0.1)
            )
        elif strategy == "random":
            selected_items = self._select_random(candidates, self.buffer_size)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Choose from 'standard', 'lazy', 'stochastic', "
                f"'lazier_than_lazy', 'random'."
            )

        st["items"] = selected_items
        st["data"] = {id(it): data_map.get(id(it), {}) for it in selected_items}

    # =========================
    # Selection Strategies (unchanged)
    # =========================

    def _select_standard(self, candidates: list, K: int, chunk_size: int = 4096) -> list:
        if K <= 0 or not candidates:
            return []

        n = len(candidates)
        k_eff = min(K, n)

        sf = self._get_set_function()
        state = sf.begin(self, candidates)

        pool = list(range(n))
        pos = list(range(n))
        selected = []

        def remove_from_pool(i: int):
            p = pos[i]
            last = pool[-1]
            pool[p] = last
            pos[last] = p
            pool.pop()

        for _ in range(k_eff):
            best_gain = -1e18
            best_i = -1
            for i in pool:
                g = float(sf.gain(self, candidates[i], i, state))
                if g > best_gain:
                    best_gain = g
                    best_i = i

            if best_i < 0 or best_gain <= 0:
                break

            selected.append(candidates[best_i])
            remove_from_pool(best_i)

            sf.on_select(self, best_i, state, pool, chunk_size=chunk_size)

        return selected

    def _select_lazy(self, candidates: list, K: int, chunk_size: int = 4096) -> list:
        if K <= 0 or not candidates:
            return []

        n = len(candidates)
        k_eff = min(K, n)

        sf = self._get_set_function()
        state = sf.begin(self, candidates)

        remaining = [True] * n

        pool = list(range(n))
        pos = list(range(n))

        def remove_from_pool(i: int):
            p = pos[i]
            last = pool[-1]
            pool[p] = last
            pos[last] = p
            pool.pop()

        heap = []
        for i in range(n):
            ub = float(sf.upper_bound(self, candidates[i], i, state))
            heapq.heappush(heap, (-ub, i))

        selected = []
        while len(selected) < k_eff and heap:
            neg_key, i = heapq.heappop(heap)
            if not remaining[i]:
                continue

            actual = float(sf.gain(self, candidates[i], i, state))
            if (-neg_key) > actual + 1e-12:
                heapq.heappush(heap, (-actual, i))
                continue

            if actual <= 0:
                break

            selected.append(candidates[i])
            remaining[i] = False
            remove_from_pool(i)

            sf.on_select(self, i, state, pool, chunk_size=chunk_size)

        return selected

    def _select_stochastic(
        self, candidates: list, K: int, epsilon: float = 0.1, chunk_size: int = 4096
    ) -> list:
        if K <= 0 or not candidates:
            return []

        n = len(candidates)
        k_eff = min(K, n)

        sf = self._get_set_function()
        state = sf.begin(self, candidates)

        pool = list(range(n))
        pos = list(range(n))

        def remove_from_pool(i: int):
            p = pos[i]
            last = pool[-1]
            pool[p] = last
            pos[last] = p
            pool.pop()

        eps = max(float(epsilon), 1e-12)
        s = int((n / max(1, k_eff)) * math.log(1.0 / eps))
        s = max(1, min(n, s))

        selected = []
        for _ in range(k_eff):
            if not pool:
                break
            R = random.sample(pool, min(s, len(pool)))

            best_gain = -1e18
            best_i = -1
            for i in R:
                g = float(sf.gain(self, candidates[i], i, state))
                if g > best_gain:
                    best_gain = g
                    best_i = i

            if best_i < 0 or best_gain <= 0:
                break

            selected.append(candidates[best_i])
            remove_from_pool(best_i)

            sf.on_select(self, best_i, state, pool, chunk_size=chunk_size)

        return selected

    def _select_lazier_than_lazy(
        self, candidates: list, K: int, epsilon: float = 0.1, chunk_size: int = 4096
    ) -> list:
        return self._select_stochastic(candidates, K, epsilon=epsilon, chunk_size=chunk_size)

    def _select_random(self, candidates: list, K: int) -> list:
        if K <= 0 or not candidates:
            return []
        return random.sample(candidates, min(K, len(candidates)))

    # =========================
    # Stats / Print / CSV  (stat FIXED + length stats)
    # =========================

    def stat(self) -> dict:
        """
        Minimal, safe, O(buffer_size) stats.
        Includes token length stats (gen_len/seq_len) for debugging length SF.
        """

        def _summ(items: list, data_map: dict) -> dict:
            total = len(items)
            if total == 0:
                return {
                    "total": 0,
                    "valid_frac": 0.0,
                    "gen_len_mean": 0.0,
                    "seq_len_mean": 0.0,
                }
            valid = sum(1 for it in items if getattr(it, "valid", False))
            gen_lens = []
            seq_lens = []
            for it in items:
                d = data_map.get(id(it), {}) or {}
                gl = d.get("gen_len", getattr(it, "gen_len", 0))
                sl = d.get("seq_len", getattr(it, "seq_len", 0))
                try:
                    gen_lens.append(int(gl))
                    seq_lens.append(int(sl))
                except Exception:
                    pass
            gen_mean = float(sum(gen_lens) / len(gen_lens)) if gen_lens else 0.0
            seq_mean = float(sum(seq_lens) / len(seq_lens)) if seq_lens else 0.0
            return {
                "total": total,
                "valid_frac": float(valid) / float(total),
                "gen_len_mean": gen_mean,
                "seq_len_mean": seq_mean,
            }

        if self.per_prompt:
            stats = {}
            # per prompt
            for idx, (k, st) in enumerate(self.buffer.items()):
                items = list(st.get("items", []) or [])
                data_map = st.get("data", {}) or {}
                s = _summ(items, data_map)
                stats.update(
                    {
                        f"prompt_{idx}_total_buffer": s["total"],
                        f"prompt_{idx}_valid_frac": s["valid_frac"],
                        f"prompt_{idx}_gen_len_mean": s["gen_len_mean"],
                        f"prompt_{idx}_seq_len_mean": s["seq_len_mean"],
                    }
                )
            # global aggregate
            all_items = []
            all_data = {}
            for _, st in self.buffer.items():
                all_items.extend(list(st.get("items", []) or []))
                all_data.update(st.get("data", {}) or {})
            s = _summ(all_items, all_data)
            stats.update(
                {
                    "buffer_total": s["total"],
                    "buffer_valid_frac": s["valid_frac"],
                    "buffer_gen_len_mean": s["gen_len_mean"],
                    "buffer_seq_len_mean": s["seq_len_mean"],
                }
            )
            return stats

        items = list(self.buffer.get("items", []) or [])
        data_map = self.buffer.get("data", {}) or {}
        s = _summ(items, data_map)
        return {
            "buffer_total": s["total"],
            "buffer_valid_frac": s["valid_frac"],
            "buffer_gen_len_mean": s["gen_len_mean"],
            "buffer_seq_len_mean": s["seq_len_mean"],
        }

    def print(self):
        if self.per_prompt:
            for k, st in self.buffer.items():
                print(k)
                for it in st["items"]:
                    print(it.canonical_smiles)
                print("")
        else:
            for it in self.buffer["items"]:
                print(it.canonical_smiles)
            print("")

    def save_csv(self, path, tokenizer=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def _ids_to_str(ids):
            if ids is None:
                return ""
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return " ".join(str(int(x)) for x in ids)

        def _ids_len(ids):
            if ids is None:
                return ""
            if hasattr(ids, "numel"):
                return int(ids.numel())
            try:
                return int(len(ids))
            except Exception:
                return ""

        header = [
            "prompt",
            "sentence",
            "token_ids",
            "reward",
            "valid",
            "static_score",
            "max_sim",
            "smiles_len",
            "token_ids_len",
            "seq_len",
            "gen_len",
            "weight_rew",
            "weight_val",
            "weight_div",
            "weight_len",
        ]

        iterator = self.buffer.items() if self.per_prompt else [(None, self.buffer)]

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for prompt_key, st in iterator:
                prompt = "" if prompt_key is None else str(prompt_key)
                data_map = st.get("data", {}) or {}

                for it in st.get("items", []):
                    data = data_map.get(id(it), {}) or {}
                    sentence = (
                        getattr(it, "sentence", None)
                        or data.get("str_sentence")
                        or getattr(it, "smiles", "")
                        or ""
                    )
                    token_ids = getattr(it, "token_ids", None) or data.get("tensor_sentence")
                    if token_ids is None:
                        token_ids = []
                    # remove eos (safe for list/tensor)
                    try:
                        token_ids = [
                            x for x in token_ids if int(x) != int(self.termination_token_id)
                        ]
                    except Exception:
                        pass

                    smiles = getattr(it, "canonical_smiles", "") or ""

                    w.writerow(
                        [
                            prompt,
                            sentence,
                            _ids_to_str(token_ids),
                            getattr(it, "reward", ""),
                            int(getattr(it, "valid", 0)),
                            getattr(it, "static_score", ""),
                            getattr(it, "max_sim", ""),
                            len(smiles),
                            _ids_len(token_ids),
                            int(data.get("seq_len", getattr(it, "seq_len", 0))),
                            int(data.get("gen_len", getattr(it, "gen_len", 0))),
                            float(getattr(self, "weight_rew", 0.0)),
                            float(getattr(self, "weight_val", 0.0)),
                            float(getattr(self, "weight_div", 0.0)),
                            float(getattr(self, "weight_len", 0.0)),
                        ]
                    )
