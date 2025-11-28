from __future__ import annotations

import gzip
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Utilities for building and querying MCMC n-gram priors.

# This module hosts a lightweight implementation of the Bayesian n-gram MCMC
# used in the tokens_mcmc.ipynb notebook, along with helpers to pack the
# posterior into GPU tensors and to build coarse-grained token mappings.


__all__ = [
    "BayesianNGramMCMC",
    "pack_q_mcmc_to_device",
    "load_q_mcmc",
    "CoarseTokenIndexer",
]


class BayesianNGramMCMC:
    """
    Bayesian n-gram with interpolated backoff and MCMC over (phi_h, lambda_h, z_assignments).

    Supports bigram/trigram (n=2/3) with a BOS token used for left padding. The posterior
    predictive q_MCMC can be retrieved via :meth:`posterior_q`.
    """

    def __init__(
        self,
        n: int = 2,
        alpha: float = 0.5,
        alpha0: float = 0.5,
        a_beta: float = 2.0,
        b_beta: float = 2.0,
        bos_id: int = -1,
        seed: int = 0,
    ):
        assert n in (2, 3), "Only bigram (n=2) or trigram (n=3) supported."
        self.n = n
        self.alpha = alpha
        self.alpha0 = alpha0
        self.a_beta = a_beta
        self.b_beta = b_beta
        self.bos_id = bos_id
        self.rng = np.random.default_rng(seed)

        self.vocab_: list[int] = []
        self.vsz_: int = 0

        self.contexts_: set = set()
        self.contexts_lower_: set = set()

        self.phi_: dict[tuple[int, ...], np.ndarray] = {}
        self.lam_: dict[tuple[int, ...], float] = {}
        self.counts_: dict[tuple[int, ...], Counter] = {}

        self.phi_root_: np.ndarray | None = None
        self.counts_root_: Counter = Counter()

        self.z_assign_: dict[tuple[int, int], int] = {}
        self.data_index_: list[tuple[int, int, tuple[int, ...], int]] = []

    def _extract_vocab(self, sequences: list[list[int]]):
        vocab = set()
        for seq in sequences:
            vocab.update(seq)
        if self.bos_id in vocab:
            raise ValueError("bos_id must be unique and not appear in actual tokens")
        self.vocab_ = sorted(vocab)
        self.vsz_ = len(self.vocab_)
        self._tok2idx = {t: i for i, t in enumerate(self.vocab_)}

    def _bos_pad(self, seq: list[int], k: int) -> list[int]:
        return [self.bos_id] * k + seq

    def _build_data_index(self, sequences: list[list[int]]):
        self.data_index_.clear()
        self.contexts_.clear()
        self.contexts_lower_.clear()

        k = self.n - 1
        for i, seq in enumerate(sequences):
            s = self._bos_pad(seq, k)
            for t in range(k, len(s)):
                ctx = tuple(s[t - k : t])
                w = s[t]
                self.data_index_.append((i, t, ctx, w))
                self.contexts_.add(ctx)
                if self.n == 3:
                    self.contexts_lower_.add(ctx[1:])

        if self.n == 3:
            self.contexts_lower_.add(())

    def _init_params(self):
        self.phi_root_ = self.rng.dirichlet([self.alpha0 / self.vsz_] * self.vsz_)
        self.counts_root_.clear()

        self.phi_.clear()
        self.lam_.clear()
        self.counts_.clear()

        for h in self.contexts_:
            self.phi_[h] = self.rng.dirichlet([self.alpha / self.vsz_] * self.vsz_)
            self.lam_[h] = self.rng.beta(self.a_beta, self.b_beta)
            self.counts_[h] = Counter()

        if self.n == 3:
            for h2 in self.contexts_lower_:
                if h2 not in self.phi_:
                    self.phi_[h2] = self.rng.dirichlet([self.alpha / self.vsz_] * self.vsz_)
                    self.lam_[h2] = self.rng.beta(self.a_beta, self.b_beta)
                    self.counts_[h2] = Counter()

        self.z_assign_.clear()
        for seq_idx, t, h, w in self.data_index_:
            z = 0 if random.random() < 0.5 else 1
            self.z_assign_[(seq_idx, t)] = z
            if z == 0 and w != self.bos_id:
                self.counts_[h][w] += 1

    def _phi_prob(self, h: tuple[int, ...], w: int) -> float:
        if len(h) == 0:
            return self.phi_root_[self._tok2idx[w]]
        return self.phi_[h][self._tok2idx[w]]

    def _p_backoff(self, h: tuple[int, ...], w: int) -> float:
        if len(h) == 0:
            return self.phi_root_[self._tok2idx[w]]
        lam = self.lam_[h]
        ph = self.phi_[h][self._tok2idx[w]]
        pbo = self._p_backoff(h[1:], w)
        return (1 - lam) * ph + lam * pbo

    def _p_full(self, h: tuple[int, ...], w: int) -> float:
        if len(h) == 0:
            return self.phi_root_[self._tok2idx[w]]
        lam = self.lam_[h]
        ph = self.phi_[h][self._tok2idx[w]]
        pbo = self._p_backoff(h[1:], w)
        return (1 - lam) * ph + lam * pbo

    def _sample_phi_all(self):
        self.phi_root_ = self.rng.dirichlet([self.alpha0 / self.vsz_] * self.vsz_)

        for h, cnt in self.counts_.items():
            alphas = np.full(self.vsz_, self.alpha / self.vsz_, dtype=np.float64)
            for tok, c in cnt.items():
                alphas[self._tok2idx[tok]] += c
            self.phi_[h] = self.rng.dirichlet(alphas)

    def _sample_lambda_all(self):
        B = defaultdict(int)
        D = defaultdict(int)
        for seq_idx, t, h, w in self.data_index_:
            z = self.z_assign_[(seq_idx, t)]
            if len(h) == 0:
                continue
            if z == 1:
                B[h] += 1
            else:
                D[h] += 1
        for h in self.phi_.keys():
            if len(h) == 0:
                continue
            a_post = max(self.a_beta + B[h], 1e-6)
            b_post = max(self.b_beta + D[h], 1e-6)
            self.lam_[h] = self.rng.beta(a_post, b_post)

    def _resample_z_all(self):
        for seq_idx, t, h, w in self.data_index_:
            prev_z = self.z_assign_[(seq_idx, t)]
            if prev_z == 0:
                self.counts_[h][w] -= 1
                if self.counts_[h][w] <= 0:
                    del self.counts_[h][w]

            if len(h) == 0:
                self.z_assign_[(seq_idx, t)] = 0
                continue

            lam = self.lam_[h]
            phi_hw = self._phi_prob(h, w)
            p_back = self._p_backoff(h[1:], w)

            p0 = (1 - lam) * max(phi_hw, 1e-12)
            p1 = lam * max(p_back, 1e-12)
            s = p0 + p1
            p0_ = 0.5 if s == 0.0 else p0 / s

            z = 0 if random.random() < p0_ else 1
            self.z_assign_[(seq_idx, t)] = z
            if z == 0:
                self.counts_[h][w] += 1

    def fit(
        self,
        sequences: list[list[int]],
        iters: int = 2000,
        burnin: int = 500,
        thin: int = 5,
        verbose: bool = True,
    ):
        self._extract_vocab(sequences)
        self._build_data_index(sequences)
        self._init_params()

        accum = defaultdict(lambda: np.zeros(self.vsz_, dtype=np.float64))
        samples_kept = 0

        for it in range(1, iters + 1):
            self._sample_phi_all()
            self._sample_lambda_all()
            self._resample_z_all()

            if it > burnin and ((it - burnin) % thin == 0):
                accum[()][...] += self.phi_root_
                for h in self.phi_.keys():
                    if len(h) == 0:
                        accum[h][...] += self.phi_root_
                    else:
                        lam = self.lam_[h]
                        probs = np.zeros(self.vsz_, dtype=np.float64)
                        for i_tok, tok in enumerate(self.vocab_):
                            ph = self.phi_[h][i_tok]
                            pbo = self._p_backoff(h[1:], tok)
                            probs[i_tok] = (1 - lam) * ph + lam * pbo
                        accum[h][...] += probs
                samples_kept += 1

            if verbose and (it % max(50, iters // 20) == 0):
                print(f"[MCMC] iter {it}/{iters}  kept={samples_kept}")

        if samples_kept == 0:
            raise RuntimeError("No posterior samples kept. Increase iters or reduce burnin/thin.")

        self.post_q_: dict[tuple[int, ...], dict[int, float]] = {}
        for h, vec in accum.items():
            avg = vec / samples_kept
            s = float(avg.sum())
            if s <= 0:
                avg = np.full_like(avg, 1.0 / self.vsz_)
            else:
                avg = avg / s
            self.post_q_[h] = {tok: float(avg[self._tok2idx[tok]]) for tok in self.vocab_}
        return self

    def posterior_q(self) -> dict[tuple[int, ...], dict[int, float]]:
        return self.post_q_

    def p_cond(self, context: tuple[int, ...], token: int) -> float:
        if hasattr(self, "post_q_") and context in self.post_q_:
            return self.post_q_[context].get(token, 0.0)
        h = context
        while len(h) > 0:
            h = h[1:]
            if h in getattr(self, "post_q_", {}):
                return self.post_q_[h].get(token, 0.0)
        if hasattr(self, "post_q_") and () in self.post_q_:
            return self.post_q_[()].get(token, 0.0)
        return 1.0 / max(1, self.vsz_)


def pack_q_mcmc_to_device(
    q_mcmc: dict[tuple[int, ...], dict[int, float]],
    vocab_size: int,
    device: torch.device,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float32,
) -> dict[tuple[int, ...], torch.Tensor]:
    """Pack q_MCMC dict-of-dicts into device tensors for fast lookup."""
    out: dict[tuple[int, ...], torch.Tensor] = {}
    for h, mp in q_mcmc.items():
        q = torch.full((vocab_size,), eps, device=device, dtype=dtype)
        for tok_id, prob in mp.items():
            if 0 <= tok_id < vocab_size:
                q[tok_id] = max(float(prob), eps)
        q = q / q.sum().clamp_min(eps)
        out[h] = q
    if () not in out:
        out[()] = torch.full((vocab_size,), 1.0 / vocab_size, device=device, dtype=dtype)
    return out


def _get_q_vec_with_backoff(
    q_packed: dict[tuple[int, ...], torch.Tensor],
    context: tuple[int, ...],
) -> torch.Tensor:
    """Backoff lookup for q_packed with progressively shorter histories."""
    h = context
    while True:
        if h in q_packed:
            return q_packed[h]
        if len(h) == 0:
            return q_packed[()]
        h = h[1:]


def load_q_mcmc(path: str) -> dict[tuple[int, ...], dict[int, float]]:
    """Load q_MCMC dictionary from a (possibly gzipped) pickle file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "post_q" in obj:
        return obj["post_q"]
    return obj


class CoarseTokenIndexer:
    """Utility to map fine tokens to coarse ids and build probability projections."""

    def __init__(
        self,
        token_to_coarse: dict[int, int] | None,
        vocab_size: int | None,
        default_id: int | None = None,
    ) -> None:
        self.token_to_coarse = (
            None
            if token_to_coarse is None
            else {int(k): int(v) for k, v in token_to_coarse.items()}
        )
        self.vocab_size = vocab_size
        self.default_id = default_id
        self._lookup_cache: dict[torch.device, torch.Tensor] = {}
        self._proj_cache: dict[torch.device, torch.Tensor] = {}

        if self.token_to_coarse is None:
            self.coarse_vocab_size = vocab_size
        else:
            max_idx = max(self.token_to_coarse.values()) if self.token_to_coarse else -1
            if self.default_id is None:
                self.default_id = max_idx + 1
            self.coarse_vocab_size = max(max_idx, self.default_id) + 1

    def ensure_vocab_size(self, vocab_size: int) -> None:
        """Set vocab_size lazily and clear caches if it was missing."""
        if self.vocab_size is None:
            self.vocab_size = vocab_size
            self._lookup_cache.clear()
            self._proj_cache.clear()

    def _build_lookup(self, device: torch.device) -> torch.Tensor:
        if device in self._lookup_cache:
            return self._lookup_cache[device]
        if self.vocab_size is None:
            raise ValueError("vocab_size must be provided to build coarse lookup.")
        lookup = torch.full(
            (self.vocab_size,),
            int(self.default_id) if self.default_id is not None else 0,
            device=device,
            dtype=torch.long,
        )
        if self.token_to_coarse is None:
            lookup = torch.arange(self.vocab_size, device=device, dtype=torch.long)
        else:
            for tok, cid in self.token_to_coarse.items():
                if 0 <= tok < self.vocab_size:
                    lookup[tok] = cid
        self._lookup_cache[device] = lookup
        return lookup

    def coarse_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map fine-token tensor to coarse ids (same shape)."""
        lookup = self._build_lookup(tokens.device)
        return lookup[tokens]

    def projection(self, device: torch.device) -> torch.Tensor:
        """Return [V, C] matrix to aggregate fine token probs into coarse probs."""
        if device in self._proj_cache:
            return self._proj_cache[device]
        if self.vocab_size is None:
            raise ValueError("vocab_size must be provided to build coarse projection.")
        lookup = self._build_lookup(device)
        proj = torch.zeros(
            self.vocab_size,
            self.coarse_vocab_size,
            device=device,
            dtype=torch.float32,
        )
        proj[torch.arange(self.vocab_size, device=device), lookup] = 1.0
        self._proj_cache[device] = proj
        return proj
