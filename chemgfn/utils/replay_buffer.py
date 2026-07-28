"""Replay buffers for GFlowNet training.

Two buffers are provided:

* :class:`ReplayBuffer` keeps, per prompt, the highest-reward trajectories in a heap and
  deduplicates near-identical completions by edit distance. Sampling is either uniform or
  reward-prioritised (Shen et al., 2023).
* :class:`ReplayBufferSubmodular` (SubM) keeps the subset of trajectories that maximises a
  monotone submodular objective combining reward, validity, similarity coverage and a
  length-bin coverage term, selected greedily.

The similarity measure and the submodular objective are injected as
:class:`SimilarityBackend` and :class:`SetFunction` implementations, so the same buffer serves
molecular (Tanimoto over Morgan fingerprints) and textual (k-gram Jaccard) tasks.
"""

import csv
import heapq
import math
import os
import re
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
    """Per-prompt heap of the highest-reward trajectories seen so far.

    Newly added trajectories are rejected when an already-stored trajectory is within
    ``sim_tolerance`` normalised edit distance and scores at least as well, unless the item is
    force-added because the validator marked it valid. Sampling draws with replacement, either
    uniformly or with reward prioritisation.
    """

    def __init__(
        self,
        buffer_size: int,
        sim_tolerance: float = 0.25,
        prioritized_replay: bool = False,
        replay_alpha: float = 0.7,
        replay_beta: float = 0.2,
        min_replay_size: int = 1,
    ) -> None:
        """Initialise the buffer.

        Args:
            buffer_size: Maximum number of trajectories kept per prompt.
            sim_tolerance: Normalised edit-distance threshold below which two completions are
                treated as duplicates.
            prioritized_replay: Enable reward-prioritised sampling.
            replay_alpha: Fraction of each sampled batch drawn from the high-reward tier.
            replay_beta: Fraction of the buffer that forms the high-reward tier.
            min_replay_size: Buffer occupancy below which sampling falls back to uniform.
        """
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.prioritized_replay = prioritized_replay
        self.replay_alpha = float(max(0.0, min(1.0, replay_alpha)))
        self.replay_beta = float(max(0.0, min(1.0, replay_beta)))
        self.min_replay_size = max(1, int(min_replay_size))
        self.termination_token_id: Optional[int] = None
        self.reset()

    def set_termination_token_id(self, termination_token_id: int) -> None:
        """Record the EOS token id used to strip padding before comparing completions."""
        self.termination_token_id = termination_token_id

    def reset(self) -> None:
        """Drop every stored trajectory."""
        self._buffer: Dict[str, Dict[str, Any]] = {}

    def add(self, item: Dict[str, Any], force_add: bool = False) -> None:
        """Insert a single trajectory, subject to deduplication and the size limit.

        Args:
            item: Mapping with ``logreward``, ``str_prompt``, ``str_sentence``,
                ``tensor_sentence``, ``tensor_answer`` and ``full_logrewards``.
            force_add: Bypass the "an existing near-duplicate scores at least as well" rejection,
                used to guarantee that validator-approved trajectories enter the buffer.
        """
        if self.termination_token_id is None:
            raise ValueError(
                "termination_token_id is not set. Call set_termination_token_id first."
            )

        str_prompt = item["str_prompt"]
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": item.get("tensor_prompt"),
                "sentences": [],
                "exists": set(),
            }

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

        for buffer_item in list(buffer):
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

        inserted, popped = self._safe_heappush(buffer, new_item)
        if inserted:
            if popped is not None and popped[1] in self._buffer[str_prompt]["exists"]:
                self._buffer[str_prompt]["exists"].remove(popped[1])
            self._buffer[str_prompt]["exists"].add(item["str_sentence"])

    def add_batch(
        self,
        prompt: torch.Tensor,
        sentences: torch.Tensor,
        logrewards: torch.Tensor,
        tokenizer,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a batch of trajectories generated from a single prompt.

        Args:
            prompt: Prompt token ids of shape ``[1, prompt_len]``.
            sentences: Generated token ids of shape ``[batch, gen_len]``.
            logrewards: Per-step log rewards of shape ``[batch, gen_len]``.
            tokenizer: Tokenizer used to decode the prompt and completions.
            result_dict: Forward-pass output; its ``validator_dict['global_score']`` decides which
                trajectories are force-added.
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

    def sample(
        self, batch_size: int, prompt: torch.Tensor, tokenizer
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Draw a padded batch of stored trajectories for the given prompt.

        Args:
            batch_size: Number of trajectories to draw, with replacement.
            prompt: Prompt token ids identifying the per-prompt buffer.
            tokenizer: Tokenizer used to decode the prompt into its buffer key.

        Returns:
            Tuple of padded ``(tensor_sentence, tensor_answer)`` batches, or ``(None, None)`` when
            the prompt is unknown or holds fewer than ``batch_size`` trajectories.
        """
        str_prompt = " ".join(
            [str(x) for x in tokenizer.batch_decode(prompt, skip_special_tokens=True)]
        )
        if str_prompt not in self._buffer:
            return None, None
        if len(self._buffer[str_prompt]["sentences"]) < batch_size:
            return None, None
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        idx = self._prioritized_indices(prompt_buffer, batch_size)
        if idx is None or len(idx) == 0:
            return None, None
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    def stat(self) -> Dict[str, float]:
        """Return occupancy and mean log reward for each prompt's buffer."""
        stats: Dict[str, float] = {}
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

    def save_csv(self, path: str, tokenizer=None) -> None:
        """Write every stored trajectory to ``path`` as CSV, one row per trajectory.

        Args:
            path: Destination file; parent directories are created if missing.
            tokenizer: Accepted for interface symmetry with the other buffers; unused, since the
                decoded completion is already stored alongside its token ids.
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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _safe_heappush(self, heap: list, item: tuple) -> Tuple[bool, Optional[tuple]]:
        """Keep the top-``buffer_size`` items by reward. Returns ``(inserted, popped_item)``."""
        if self.buffer_size <= 0:
            return False, None

        if len(heap) < self.buffer_size:
            heapq.heappush(heap, item)
            return True, None

        popped = heapq.heappushpop(heap, item)
        # If the new item was the smallest, it is returned and not stored.
        if popped is item:
            return False, None
        return True, popped

    def _prioritized_indices(self, buffer: list, batch_size: int) -> Optional[np.ndarray]:
        """Draw ``batch_size`` indices, biasing towards the top-``replay_beta`` reward tier.

        An ``replay_alpha`` fraction of the batch comes from the high-reward tier and the rest
        from the remainder. Falls back to uniform sampling when prioritisation is disabled or the
        buffer is too small.
        """
        n = len(buffer)
        if batch_size <= 0:
            return None
        if (not self.prioritized_replay) or n < self.min_replay_size:
            return np.random.choice(n, batch_size, replace=True)

        rewards = np.array([float(item[0]) for item in buffer], dtype=np.float64)
        if rewards.size == 0:
            return None

        beta = self.replay_beta
        alpha = self.replay_alpha
        if beta <= 0.0:
            return np.random.choice(n, batch_size, replace=True)

        sorted_idx = np.argsort(rewards)  # ascending
        high_count = max(1, int(math.ceil(beta * n)))
        high_idx = sorted_idx[-high_count:]
        low_idx = sorted_idx[: n - high_count] if n - high_count > 0 else high_idx

        num_high = int(round(batch_size * alpha))
        num_high = min(batch_size, max(0, num_high))
        num_low = batch_size - num_high

        def _sample(pool, k):
            if k <= 0:
                return np.array([], dtype=int)
            replace = len(pool) < k
            return np.random.choice(pool, k, replace=replace)

        chosen_high = _sample(high_idx, num_high)
        chosen_low = _sample(low_idx, num_low)

        if chosen_high.size + chosen_low.size == 0:
            return np.random.choice(n, batch_size, replace=True)

        chosen = np.concatenate([chosen_high, chosen_low])
        np.random.shuffle(chosen)
        return chosen


# ============================================================================ #
# Similarity backends
# ============================================================================ #


class SimilarityBackend:
    """Pairwise similarity in ``[0, 1]``, higher meaning more similar.

    Implementations precompute one representation per candidate and expose a bulk comparison so
    that greedy selection can update every remaining candidate in one call.
    """

    def prepare(self, candidates: List["BufferItem"]) -> list:
        """Return one comparison representation per candidate, aligned by index."""
        raise NotImplementedError

    def bulk(self, anchor_rep, other_reps: list) -> List[float]:
        """Return the similarity of ``anchor_rep`` to each element of ``other_reps``."""
        raise NotImplementedError


class RDKITBulkTanimotoBackend(SimilarityBackend):
    """Tanimoto similarity over Morgan fingerprints, used for the SMILES tasks."""

    def prepare(self, candidates: List["BufferItem"]) -> list:
        """Return the precomputed Morgan fingerprint of each candidate."""
        return [c.fingerprint for c in candidates]

    def bulk(self, anchor_rep, other_reps: list) -> List[float]:
        """Return the Tanimoto similarity of ``anchor_rep`` to each fingerprint."""
        return list(DataStructs.BulkTanimotoSimilarity(anchor_rep, other_reps))


class ShingleJaccardBackend(SimilarityBackend):
    """Jaccard similarity over k-gram shingles, used for the text and expression tasks."""

    _WORD_MODES = {"word", "words", "whitespace"}

    def __init__(
        self,
        k: int = 2,
        tokenizer_mode: Optional[str] = None,
        lowercase: bool = True,
    ) -> None:
        """Initialise the backend.

        Args:
            k: Shingle width, in words when ``tokenizer_mode`` selects word shingles and in
                characters otherwise.
            tokenizer_mode: ``"word"``/``"words"``/``"whitespace"`` for word shingles; any other
                value (including ``None``) uses character shingles.
            lowercase: Lowercase the text before shingling.
        """
        self.k = int(k)
        self.tokenizer_mode = None if tokenizer_mode is None else str(tokenizer_mode)
        self.lowercase = bool(lowercase)
        self._word_shingles = (
            self.tokenizer_mode is not None and self.tokenizer_mode.lower() in self._WORD_MODES
        )

    @staticmethod
    def _word_tokenize(s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", s)

    def _rep(self, s: str) -> Optional[frozenset]:
        if not s:
            return None
        s0 = str(s)
        if self.lowercase:
            s0 = s0.lower()

        k = max(1, self.k)
        if self._word_shingles:
            toks = self._word_tokenize(s0)
            if not toks:
                return None
            if len(toks) <= k:
                return frozenset([" ".join(toks)])
            return frozenset(" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1))

        if len(s0) <= k:
            return frozenset([s0])
        return frozenset(s0[i : i + k] for i in range(len(s0) - k + 1))

    def prepare(self, candidates: List["BufferItem"]) -> list:
        """Return the shingle set of each candidate's canonical text."""
        return [self._rep(str(c.canonical_text or c.text or "")) for c in candidates]

    def bulk(self, anchor_rep, other_reps: list) -> List[float]:
        """Return the Jaccard similarity of ``anchor_rep`` to each shingle set."""
        if anchor_rep is None:
            return [0.0] * len(other_reps)
        out = []
        for other in other_reps:
            if other is None:
                out.append(0.0)
                continue
            inter = len(anchor_rep & other)
            if inter == 0:
                out.append(0.0)
                continue
            union = len(anchor_rep) + len(other) - inter
            out.append(float(inter) / float(union) if union > 0 else 0.0)
        return out


# ============================================================================ #
# Set functions
# ============================================================================ #


class SetFunction:
    """Monotone submodular objective driving greedy buffer selection.

    ``begin`` allocates the state shared across one selection pass, ``gain`` returns the marginal
    value of adding a candidate to the current selection, and ``on_select`` folds the chosen
    candidate into that state.
    """

    def begin(self, buf: "ReplayBufferSubmodular", candidates: List["BufferItem"]) -> dict:
        """Create the selection state for one greedy pass over ``candidates``."""
        raise NotImplementedError

    def gain(
        self, buf: "ReplayBufferSubmodular", cand: "BufferItem", idx: int, state: dict
    ) -> float:
        """Return the marginal gain of adding ``cand`` to the current selection."""
        raise NotImplementedError

    def on_select(
        self,
        buf: "ReplayBufferSubmodular",
        best_idx: int,
        state: dict,
        remaining_idx: List[int],
        chunk_size: int = 4096,
    ) -> None:
        """Update ``state`` after ``best_idx`` has been selected."""
        raise NotImplementedError


class FacilityLengthSetFunction(SetFunction):
    """Reward and validity, plus facility-location diversity and length-bin coverage.

    The marginal gain of a candidate is::

        static_score
            + weight_div * (1 - max_similarity_to_already_selected)
            + weight_len * alpha(b) * [log(2 + n_b) - log(1 + n_b)]

    where ``static_score`` combines reward and validity, ``b`` is the candidate's generated-length
    bin and ``n_b`` the number of already-selected items in that bin. Both extra terms are
    monotone submodular: the first is the facility-location objective, the second is concave in
    the per-bin counts. ``alpha(b)`` grows with the bin index, which counteracts the tendency of
    reward-greedy selection to fill the buffer with short sequences.
    """

    def begin(self, buf: "ReplayBufferSubmodular", candidates: List["BufferItem"]) -> dict:
        """Precompute similarity representations, length bins and per-bin weights."""
        sim = buf._get_similarity_backend()
        reps = sim.prepare(candidates)

        for c in candidates:
            c.max_sim = 0.0

        bin_size = max(1, int(buf.length_bin_size))
        bin_idx = [int(c.gen_len) // bin_size for c in candidates]
        nbins = (max(bin_idx) + 1) if bin_idx else 1

        p = float(buf.length_alpha_power)
        if nbins <= 1:
            alpha = [1.0]
        else:
            alpha = [((b + 1) / nbins) ** p for b in range(nbins)]

        return {
            "candidates": candidates,
            "sim": sim,
            "reps": reps,
            "len_bin": bin_idx,
            "len_counts": [0] * nbins,
            "len_alpha": alpha,
        }

    def gain(
        self, buf: "ReplayBufferSubmodular", cand: "BufferItem", idx: int, state: dict
    ) -> float:
        """Return reward/validity plus the diversity and length-coverage marginals."""
        g = float(cand.static_score)

        w_div = float(buf.weight_div)
        if w_div > 0:
            g += w_div * (1.0 - float(cand.max_sim))

        g += self._len_marginal(buf, state["len_bin"][idx], state)
        return float(g)

    def on_select(
        self,
        buf: "ReplayBufferSubmodular",
        best_idx: int,
        state: dict,
        remaining_idx: List[int],
        chunk_size: int = 4096,
    ) -> None:
        """Charge the selected item to its length bin and refresh similarities of the rest."""
        candidates = state["candidates"]

        state["len_counts"][state["len_bin"][best_idx]] += 1

        w_div = float(buf.weight_div)
        if w_div <= 0:
            return

        reps = state["reps"]
        rep_sel = reps[best_idx]
        if rep_sel is None:
            return

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

    def _len_marginal(self, buf: "ReplayBufferSubmodular", b: int, state: dict) -> float:
        """Return the concave length-coverage marginal for bin ``b``."""
        w_len = float(buf.weight_len)
        if w_len <= 0:
            return 0.0
        c = state["len_counts"][b]
        # alpha_b * (log(1+c+1) - log(1+c)) = alpha_b * log((c+2)/(c+1))
        return w_len * float(state["len_alpha"][b]) * (math.log1p(c + 1) - math.log1p(c))


# ============================================================================ #
# Submodular replay buffer
# ============================================================================ #


class BufferItem:
    """A single trajectory held by :class:`ReplayBufferSubmodular`.

    Stores the decoded text, its reward and validity, the similarity representation and the
    generated length, together with the static part of the submodular objective.
    """

    __slots__ = (
        "text",
        "canonical_text",
        "reward",
        "valid",
        "fingerprint",
        "static_score",
        "max_sim",
        "seq_len",
        "gen_len",
    )

    def __init__(
        self,
        text: str,
        reward: float,
        weight_val: float,
        weight_rew: float,
        is_valid: bool,
    ) -> None:
        """Initialise the item and precompute its static score and fingerprint.

        Args:
            text: Decoded completion (a SMILES string, an expression, a peptide or a sentence).
            reward: Scalar reward or validator score for the completion.
            weight_val: Weight of the validity bonus in the static score.
            weight_rew: Weight of the reward in the static score.
            is_valid: Validity as decided upstream by the task validator.
        """
        self.text = text
        self.canonical_text = text
        try:
            self.reward = float(reward)
        except Exception:
            self.reward = float(reward.item()) if hasattr(reward, "item") else float(reward)

        self.valid = bool(is_valid)
        self.fingerprint = None

        # Molecular tasks canonicalise the text and cache a Morgan fingerprint; for the other
        # tasks the raw text is already canonical and similarity is computed from shingles.
        if self.valid and _HAVE_RDKIT:
            try:
                mol = Chem.MolFromSmiles(text)
                if mol is not None:
                    self.canonical_text = Chem.MolToSmiles(mol, canonical=True)
                    self.fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=2, nBits=2048
                    )
            except Exception:
                pass

        self.static_score = (weight_rew * self.reward) + (weight_val if self.valid else 0.0)

        self.max_sim = 0.0
        self.seq_len = 0
        self.gen_len = 0

    def __repr__(self) -> str:
        return (
            f"BufferItem(text={self.canonical_text}, reward={self.reward:.4f}, "
            f"valid={self.valid})"
        )


class ReplayBufferSubmodular:
    """Replay buffer whose contents maximise a submodular reward/diversity/length objective.

    Every update pools the stored items with the incoming batch, deduplicates by canonical text
    keeping the higher reward, optionally restricts the pool to validator-approved items, and —
    when the pool exceeds ``buffer_size`` — re-selects the buffer greedily under the injected
    :class:`SetFunction`. Greedy selection of a monotone submodular objective under a cardinality
    constraint is within ``1 - 1/e`` of the optimum.
    """

    def __init__(
        self,
        buffer_size: int,
        per_prompt: bool = False,
        weight_div: float = 1.0,
        diversity_valid_only: bool = False,
        diversity_valid_ratio: float = 1.0,
        weight_val: float = 1.0,
        weight_rew: float = 1.0,
        validator_key: str = "local_score",
        validator_type: Literal["mean", "max", "last"] = "max",
        weight_len: float = 0.0,
        length_bin_size: int = 10,
        length_alpha_power: float = 1.0,
        similarity_backend: Optional[SimilarityBackend] = None,
        set_function_obj: Optional[SetFunction] = None,
    ) -> None:
        """Initialise the buffer.

        Args:
            buffer_size: Number of trajectories retained after each selection pass.
            per_prompt: Maintain an independent buffer per prompt instead of one global buffer.
            weight_div: Weight of the facility-location diversity term.
            diversity_valid_only: Restrict selection to validator-approved trajectories.
            diversity_valid_ratio: Minimum share of valid trajectories in the selection pool;
                ``1.0`` keeps only valid ones, ``0.0`` disables the gate.
            weight_val: Weight of the validity bonus in an item's static score.
            weight_rew: Weight of the reward in an item's static score.
            validator_key: Key of the per-step validator score used as the item reward.
            validator_type: How to reduce that per-step score to a scalar: ``"mean"``, ``"max"``,
                or ``"last"`` (the terminal global score).
            weight_len: Weight of the length-bin coverage term.
            length_bin_size: Width of a length bin, in generated tokens.
            length_alpha_power: Exponent shaping how strongly longer bins are favoured.
            similarity_backend: Similarity measure; defaults to Tanimoto over fingerprints.
            set_function_obj: Submodular objective; defaults to
                :class:`FacilityLengthSetFunction`.
        """
        self.buffer_size = buffer_size
        self.per_prompt = per_prompt

        self.weight_div = weight_div
        # Diversity gating: 0 => ignore validity; 1 => only valid; (0,1) => mix.
        self.diversity_valid_only = bool(diversity_valid_only)
        r = float(diversity_valid_ratio)
        r = 0.0 if r < 0 else 1.0 if r > 1 else r
        if self.diversity_valid_only:
            r = 1.0
        self.diversity_valid_ratio = r
        self.weight_val = weight_val
        self.weight_rew = weight_rew

        self.validator_key = validator_key
        self.validator_type = validator_type

        self.weight_len = float(weight_len)
        self.length_bin_size = int(length_bin_size)
        self.length_alpha_power = float(length_alpha_power)

        self._similarity_backend = similarity_backend
        self._set_function_obj = set_function_obj

        self.termination_token_id: Optional[int] = None
        self.reset()

    def set_termination_token_id(self, termination_token_id: int) -> None:
        """Record the EOS token id used to locate the end of each generated sequence."""
        self.termination_token_id = int(termination_token_id)

    def reset(self) -> None:
        """Drop every stored trajectory."""
        if self.per_prompt:
            self.buffer: Dict[str, Dict[str, Any]] = {}
        else:
            self.buffer = {"items": [], "data": {}}

    def add_batch(
        self,
        prompt: torch.Tensor,
        sentences: torch.Tensor,
        logrewards: torch.Tensor,
        tokenizer,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fold a batch of trajectories into the buffer and re-run submodular selection.

        Args:
            prompt: Prompt token ids of shape ``[1, prompt_len]``.
            sentences: Generated token ids of shape ``[batch, gen_len]``.
            logrewards: Per-step log rewards of shape ``[batch, gen_len]``, used as the item
                reward only when no validator scores are available.
            tokenizer: Tokenizer used to decode the prompt and completions.
            result_dict: Forward-pass output supplying ``validator_dict``.
        """
        assert self.termination_token_id is not None, "Call set_termination_token_id() first."

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

        new_items: List[Tuple[BufferItem, Dict[str, Any]]] = []
        for i in range(sentences.size(0)):
            str_sentence = token_sentences[i].strip()
            is_valid = bool(valid_mask[i].item()) if valid_mask is not None else False

            if validator_scores is not None:
                r_raw = float(validator_scores[i].item())
            else:
                idx = (sentences[i] == eos).nonzero(as_tuple=True)[0][0]
                r_raw = float(logrewards[i, idx].item())

            bi = BufferItem(
                str_sentence,
                r_raw,
                self.weight_val,
                self.weight_rew,
                is_valid=is_valid,
            )

            ts = sentences[i].detach().cpu()
            eos_pos = (ts == eos).nonzero(as_tuple=True)[0]
            seq_len = int(eos_pos[0].item()) if eos_pos.numel() > 0 else int(ts.numel())

            # Fall back to the full sequence length when the prompt is not part of `sentences`.
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
                "prompt_len": int(prompt_len),
                "seq_len": int(seq_len),
                "gen_len": int(gen_len),
            }
            new_items.append((bi, sample_data))

        self._update_buffer(st, new_items)

    def sample(
        self, batch_size: int, prompt: torch.Tensor, tokenizer
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Draw a padded batch of stored trajectories uniformly, with replacement.

        Args:
            batch_size: Number of trajectories to draw.
            prompt: Prompt token ids, used as the buffer key when ``per_prompt`` is set.
            tokenizer: Tokenizer used to decode the prompt into its buffer key.

        Returns:
            Tuple of padded ``(tensor_sentence, tensor_answer)`` batches, or ``(None, None)`` when
            the buffer holds fewer than ``batch_size`` trajectories.
        """
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

    def stat(self) -> Dict[str, float]:
        """Return occupancy, valid fraction and mean generated/sequence length of the buffer."""

        def _summ(items: list, data_map: dict) -> Dict[str, float]:
            total = len(items)
            if total == 0:
                return {
                    "total": 0,
                    "valid_frac": 0.0,
                    "gen_len_mean": 0.0,
                    "seq_len_mean": 0.0,
                }
            valid = sum(1 for it in items if it.valid)
            gen_lens = []
            seq_lens = []
            for it in items:
                d = data_map.get(id(it), {}) or {}
                gen_lens.append(int(d.get("gen_len", it.gen_len)))
                seq_lens.append(int(d.get("seq_len", it.seq_len)))
            gen_mean = float(sum(gen_lens) / len(gen_lens)) if gen_lens else 0.0
            seq_mean = float(sum(seq_lens) / len(seq_lens)) if seq_lens else 0.0
            return {
                "total": total,
                "valid_frac": float(valid) / float(total),
                "gen_len_mean": gen_mean,
                "seq_len_mean": seq_mean,
            }

        if self.per_prompt:
            stats: Dict[str, float] = {}
            for idx, (_, st) in enumerate(self.buffer.items()):
                s = _summ(list(st.get("items", []) or []), st.get("data", {}) or {})
                stats.update(
                    {
                        f"prompt_{idx}_total_buffer": s["total"],
                        f"prompt_{idx}_valid_frac": s["valid_frac"],
                        f"prompt_{idx}_gen_len_mean": s["gen_len_mean"],
                        f"prompt_{idx}_seq_len_mean": s["seq_len_mean"],
                    }
                )
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

        s = _summ(
            list(self.buffer.get("items", []) or []),
            self.buffer.get("data", {}) or {},
        )
        return {
            "buffer_total": s["total"],
            "buffer_valid_frac": s["valid_frac"],
            "buffer_gen_len_mean": s["gen_len_mean"],
            "buffer_seq_len_mean": s["seq_len_mean"],
        }

    def save_csv(self, path: str, tokenizer=None) -> None:
        """Write every stored trajectory to ``path`` as CSV, one row per trajectory.

        Rows carry the item's reward, validity, static score, similarity to the rest of the
        buffer and lengths, together with the selection weights in force.

        Args:
            path: Destination file; parent directories are created if missing.
            tokenizer: Accepted for interface symmetry with the other buffers; unused, since the
                decoded completion is already stored alongside its token ids.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def _ids_to_str(ids) -> str:
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
                    sentence = data.get("str_sentence") or it.text or ""
                    token_ids = data.get("tensor_sentence")
                    if token_ids is None:
                        token_ids = []
                    try:
                        token_ids = [
                            x for x in token_ids if int(x) != int(self.termination_token_id)
                        ]
                    except Exception:
                        pass

                    text = it.canonical_text or ""

                    w.writerow(
                        [
                            prompt,
                            sentence,
                            _ids_to_str(token_ids),
                            it.reward,
                            int(it.valid),
                            it.static_score,
                            it.max_sim,
                            len(text),
                            _ids_len(token_ids),
                            int(data.get("seq_len", it.seq_len)),
                            int(data.get("gen_len", it.gen_len)),
                            float(self.weight_rew),
                            float(self.weight_val),
                            float(self.weight_div),
                            float(self.weight_len),
                        ]
                    )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_state(self, prompt_key: str) -> Dict[str, Any]:
        """Return the ``{"items", "data"}`` state for ``prompt_key``, creating it if needed."""
        if self.per_prompt:
            if prompt_key not in self.buffer:
                self.buffer[prompt_key] = {"items": [], "data": {}}
            return self.buffer[prompt_key]
        return self.buffer

    def _get_similarity_backend(self) -> SimilarityBackend:
        """Return the configured similarity backend, defaulting to bulk Tanimoto."""
        if self._similarity_backend is None:
            return RDKITBulkTanimotoBackend()
        return self._similarity_backend

    def _get_set_function(self) -> SetFunction:
        """Return the configured set function, defaulting to the facility/length objective."""
        if self._set_function_obj is None:
            return FacilityLengthSetFunction()
        return self._set_function_obj

    def _filter_for_diversity(
        self, candidates: List[BufferItem], data_map: dict
    ) -> Tuple[List[BufferItem], dict]:
        """Restrict the selection pool so that valid items make up at least the target share.

        ``diversity_valid_ratio`` of ``0`` keeps every candidate, ``1`` keeps only valid ones, and
        an intermediate value keeps all valid candidates plus the highest-scoring invalid ones up
        to the point where the valid share would drop below the target.
        """
        ratio = self.diversity_valid_ratio
        if ratio <= 0:
            return candidates, data_map
        valid = [it for it in candidates if it.valid]
        if ratio >= 1.0:
            filtered = valid
            filtered_data = {id(it): data_map.get(id(it), {}) for it in filtered}
            return filtered, filtered_data

        if not valid:
            return candidates, data_map

        invalid = [it for it in candidates if not it.valid]
        # Enforce valid share >= ratio, i.e. max_invalid <= valid_count * (1 - r) / r.
        max_invalid = int(math.floor(len(valid) * (1.0 - ratio) / max(ratio, 1e-12)))
        if max_invalid >= len(invalid):
            filtered = valid + invalid
        else:
            invalid_sorted = sorted(
                invalid,
                key=lambda it: (it.static_score, it.reward),
                reverse=True,
            )
            filtered = valid + invalid_sorted[:max_invalid]

        filtered_data = {id(it): data_map.get(id(it), {}) for it in filtered}
        return filtered, filtered_data

    def _update_buffer(
        self, st: Dict[str, Any], new_items: List[Tuple[BufferItem, Dict[str, Any]]]
    ) -> None:
        """Merge ``new_items`` into ``st`` and re-select the buffer under the set function."""
        candidates = list(st["items"])
        data_map = dict(st["data"])
        existing_map = {it.canonical_text: it for it in candidates}

        for bi, data in new_items:
            canon = bi.canonical_text
            if canon in existing_map:
                existing_item = existing_map[canon]
                if bi.reward > existing_item.reward:
                    existing_item.reward = bi.reward
                    existing_item.static_score = (self.weight_rew * bi.reward) + (
                        self.weight_val if bi.valid else 0.0
                    )
                    existing_item.seq_len = bi.seq_len
                    existing_item.gen_len = bi.gen_len
                    data_map[id(existing_item)] = data
            else:
                candidates.append(bi)
                existing_map[canon] = bi
                data_map[id(bi)] = data

        uniq: Dict[str, BufferItem] = {}
        for it in candidates:
            c = it.canonical_text
            if c not in uniq or it.reward > uniq[c].reward:
                uniq[c] = it
        candidates = list(uniq.values())

        candidates, data_map = self._filter_for_diversity(candidates, data_map)

        if len(candidates) <= self.buffer_size:
            st["items"] = candidates[: self.buffer_size]
        else:
            st["items"] = self._select_greedy(candidates, self.buffer_size)
        st["data"] = {id(it): data_map.get(id(it), {}) for it in st["items"]}

    def _select_greedy(
        self, candidates: List[BufferItem], K: int, chunk_size: int = 4096
    ) -> List[BufferItem]:
        """Greedily select up to ``K`` candidates maximising the submodular objective.

        Args:
            candidates: Pool to select from.
            K: Cardinality constraint.
            chunk_size: Number of remaining candidates whose similarities are refreshed per call
                into the similarity backend.

        Returns:
            The selected items, in selection order. Selection stops early once no remaining
            candidate has a positive marginal gain.
        """
        if K <= 0 or not candidates:
            return []

        n = len(candidates)
        k_eff = min(K, n)

        sf = self._get_set_function()
        state = sf.begin(self, candidates)

        pool = list(range(n))
        pos = list(range(n))
        selected: List[BufferItem] = []

        def remove_from_pool(i: int) -> None:
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
