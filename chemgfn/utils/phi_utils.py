import csv
import math
import random
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor


# token-wise prefix value estimation
def batch_prefix_value_kgram(
    sentences: torch.Tensor,  # (B,T) 生成部分 tokens（不含 prompt）
    y: torch.Tensor,  # (B,) 终局 0/1
    k: int = 6,
    alpha: float = 1.0,
    min_count: int = 2,
    backoff: bool = True,
) -> torch.Tensor:
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


def build_prefix_potential(
    pv: torch.Tensor,  # (B,T) in (0,1)
    ref_log_pf: torch.Tensor,  # (B,T) 每步 token logprob（来自 score_fast 返回）
    non_term_mask: torch.Tensor,  # (B,T) True 表示还没终止的位置（你可以用 score_fast 里的 mask）
    counts: torch.Tensor | None = None,  # (B,T) decayed counts for confidence
    eta: float = 1.0,
    clamp: float = 2.0,
    tau_conf: float = 20.0,
) -> torch.Tensor:
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


# ============== phi_tok -> phi_state and dphi ==============


def phi_tok_to_phi_state_and_dphi(
    phi_tok: Tensor,  # (B,T)
    L: int,  # reward_mixed.shape[1] = T+1
    active_before: Tensor,  # (B,T)
    *,
    anchor_start: float = 0.0,
    anchor_end: float = 0.0,
) -> tuple[Tensor, Tensor]:
    B, T = phi_tok.shape
    assert L == T + 1, f"L must be T+1, got L={L}, T={T}"

    phi_state = torch.zeros((B, L), device=phi_tok.device, dtype=phi_tok.dtype)
    phi_state[:, 1:] = phi_tok
    phi_state[:, 0] = float(anchor_start)
    phi_state[:, -1] = float(anchor_end)

    dphi = phi_state[:, 1:] - phi_state[:, :-1]  # (B,T)
    dphi = dphi * active_before.to(dphi.dtype)
    return phi_state, dphi


@torch.no_grad()
def apply_phi_shaping(
    reward_mixed: Tensor,  # (B, L_state)
    phi_tok: Tensor,  # (B, T_tok) where T_tok = L_state - 1
    active_before: Tensor,  # (B, T_tok) bool
    phi_weight: float,
    mode: Literal["differential", "tokenwise"] = "differential",
    anchor_start: float = 0.0,
    anchor_end: float = 0.0,
    dphi_clip: float | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    将 token-wise φ 映射到 state-wise φ，并把 shaping 应用到 reward_mixed。

    - differential:  reward[:,1:] += w * Δφ   (Δφ = φ_{t+1}-φ_t, masked by active_before)
    - tokenwise:     reward[:,1:] += w * φ_tok (masked by active_before)

    返回:
      reward_out: (B, L_state)
      phi_state:  (B, L_state)
      dphi:       (B, T_tok)  (对 token transition 的 φ 差分，已 mask，且可 clip)
    """
    assert reward_mixed.ndim == 2 and phi_tok.ndim == 2
    B, L_state = reward_mixed.shape
    B2, T_tok = phi_tok.shape
    assert B == B2
    assert L_state == T_tok + 1, f"L_state={L_state} must equal T_tok+1={T_tok+1}"
    assert active_before.shape == (B, T_tok)

    # (B, L_state), (B, T_tok)
    phi_state, dphi = phi_tok_to_phi_state_and_dphi(
        L=reward_mixed.shape[1],
        phi_tok=phi_tok,
        active_before=active_before,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
    )

    if dphi_clip is not None:
        c = float(dphi_clip)
        dphi = dphi.clamp(-c, c)

    reward_out = reward_mixed.clone()

    if mode == "differential":
        reward_out[:, 1:] = reward_out[:, 1:] + float(phi_weight) * dphi

    elif mode == "tokenwise":
        reward_out[:, 1:] = reward_out[:, 1:] + float(phi_weight) * (
            phi_tok * active_before.to(phi_tok.dtype)
        )

    return reward_out, phi_state, dphi


class PrefixValueMemory:
    """
    Cross-batch exp-decayed (EMA-like) Beta-Binomial PV estimator with position buckets (方案A).

    Keeps exp-decayed counts (N,S) for keys:
        key = (pos_bucket, k, tuple(tokens[s:t+1]))
    where pos_bucket = min(t // pos_bucket_size, pos_bucket_cap).

    - update(): uses (sentences, y, non_term_mask) to update stats
    - query_pv(): returns pv(B,T) and decayed counts using backoff from kmax..1
    """

    def __init__(
        self,
        kmax: int = 4,
        alpha: float = 1.0,
        gamma: float = 0.999,  # per-step decay factor
        min_count: float = 50.0,
        max_keys: int = 500_000,
        prune_every: int = 2000,
        prune_threshold: float = 1.0,
        tau_conf: float = 20.0,
        pos_bucket_size: int = 1,
        pos_bucket_cap: Optional[int] = None,
    ):
        self.kmax = int(kmax)
        assert self.kmax >= 1
        self.alpha = float(alpha)
        assert self.alpha > 0.0
        self.gamma = float(gamma)
        assert 0.0 < self.gamma <= 1.0
        self.min_count = float(min_count)
        assert self.min_count >= 0.0

        self.max_keys = int(max_keys)
        self.prune_every = int(prune_every)
        self.prune_threshold = float(prune_threshold)
        self.tau_conf = float(tau_conf)

        self.pos_bucket_size = int(pos_bucket_size)
        assert self.pos_bucket_size >= 1
        self.pos_bucket_cap = None if (pos_bucket_cap is None) else int(pos_bucket_cap)

        # stats[(pos_bucket, k, tokens_tuple)] = [N, S, last_step]
        self.stats: Dict[Tuple[int, int, Tuple[int, ...]], List[float]] = {}

        # global base rate stats
        self.global_N = 0.0
        self.global_S = 0.0
        self.global_last_step = 0

        # current time-step (must be non-decreasing)
        self.step = 0

    # ---------- step / decay ----------
    def _bucket(self, t: int) -> int:
        b = int(t) // self.pos_bucket_size
        if self.pos_bucket_cap is not None:
            b = min(b, self.pos_bucket_cap)
        return b

    def set_step(self, step: int) -> None:
        """Update current step (non-decreasing)."""
        step = int(step)
        if step < self.step:
            raise ValueError(f"step must be non-decreasing: got {step} < {self.step}")
        if step > self.step:
            self.step = step
        self._decay_global()

    def _decay_global(self) -> None:
        delta = self.step - int(self.global_last_step)
        if delta > 0 and self.gamma < 1.0:
            factor = self.gamma**delta
            self.global_N *= factor
            self.global_S *= factor
        self.global_last_step = int(self.step)

    def _decay_key_inplace(self, key: Tuple[int, int, Tuple[int, ...]]) -> Tuple[float, float]:
        """Decay key stats to current step (in-place), return (N,S)."""
        N, S, last = self.stats[key]
        last_i = int(last)
        delta = self.step - last_i
        if delta > 0 and self.gamma < 1.0:
            factor = self.gamma**delta
            N *= factor
            S *= factor
        # write back
        self.stats[key] = [float(N), float(S), float(self.step)]
        return float(N), float(S)

    # ---------- public ----------
    def get_base_rate(self) -> float:
        """Global Beta-Binomial posterior mean with smoothing."""
        self._decay_global()
        if self.global_N > 0.0:
            p0 = (self.global_S + self.alpha) / (self.global_N + 2.0 * self.alpha)
        else:
            p0 = 0.5
        return float(min(max(p0, 1e-4), 1.0 - 1e-4))

    @torch.no_grad()
    def update(
        self,
        sentences: torch.Tensor,  # (B,T) int
        y: torch.Tensor,  # (B,) float {0,1}
        non_term_mask: torch.Tensor,  # (B,T) bool
        *,
        step: Optional[int] = None,
    ) -> None:
        """
        Update memory with a batch.

        IMPORTANT usage:
          - either call memory.set_step(step) outside each training step,
            or pass step=... here.
        """
        if step is not None:
            self.set_step(step)
        else:
            # be robust even if caller forgets set_step
            self._decay_global()

        B, T = sentences.shape
        y = y.to(torch.float32).view(B)

        toks_list = sentences.detach().to("cpu").tolist()
        mask_list = non_term_mask.detach().to("cpu").tolist()
        y_list = y.detach().to("cpu").tolist()

        batch_N: Dict[Tuple[int, int, Tuple[int, ...]], float] = defaultdict(float)
        batch_S: Dict[Tuple[int, int, Tuple[int, ...]], float] = defaultdict(float)

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

                bt = self._bucket(t)
                # collect k-grams ending at t
                for kk in range(1, self.kmax + 1):
                    s = t - kk + 1
                    if s < 0:
                        s = 0
                    key = (bt, kk, tuple(toks[s : t + 1]))
                    batch_N[key] += 1.0
                    batch_S[key] += yi

        # update global (already decayed to current step)
        self.global_N += n_global
        self.global_S += s_global
        self.global_last_step = int(self.step)

        # update per-key
        for key, n in batch_N.items():
            s = batch_S[key]
            if key in self.stats:
                N_old, S_old, _last = self.stats[key]
                # decay old to current step then add new
                self._decay_key_inplace(key)
                N, S, _ = self.stats[key]
                self.stats[key] = [float(N + n), float(S + s), float(self.step)]
            else:
                self.stats[key] = [float(n), float(s), float(self.step)]

        if self.prune_every > 0 and (self.step % self.prune_every) == 0:
            self._prune()

    def _prune(self) -> None:
        """Prune tiny keys (after decaying to current step), and cap by max_keys (keep most recent)."""
        if not self.stats:
            return

        to_del: List[Tuple[int, int, Tuple[int, ...]]] = []
        for k in list(self.stats.keys()):
            # decay in-place first
            self._decay_key_inplace(k)
            N, S, last = self.stats[k]
            if float(N) < self.prune_threshold:
                to_del.append(k)

        for k in to_del:
            self.stats.pop(k, None)

        if len(self.stats) > self.max_keys:
            # keep most recent keys by last_step
            items = sorted(self.stats.items(), key=lambda kv: kv[1][2], reverse=True)
            self.stats = dict(items[: self.max_keys])

    @torch.no_grad()
    def query_pv(
        self,
        sentences: torch.Tensor,  # (B,T)
        non_term_mask: torch.Tensor,  # (B,T) bool
        *,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          pv:     (B,T) float32 in (0,1)
          counts: (B,T) float32, chosen N for that position (after backoff)
        """
        if step is not None:
            self.set_step(step)
        else:
            self._decay_global()

        B, T = sentences.shape
        toks_list = sentences.detach().to("cpu").tolist()
        mask_list = non_term_mask.detach().to("cpu").tolist()

        p0 = self.get_base_rate()

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

                bt = self._bucket(t)

                chosen_p: Optional[float] = None
                chosen_n = 0.0

                # backoff: kmax -> 1
                for kk in range(self.kmax, 0, -1):
                    s = t - kk + 1
                    if s < 0:
                        s = 0
                    key = (bt, kk, tuple(toks[s : t + 1]))
                    if key not in self.stats:
                        continue

                    N, S = self._decay_key_inplace(key)
                    # accept if enough evidence; always allow kk==1 as last resort
                    if (N >= self.min_count) or (kk == 1):
                        chosen_p = (S + self.alpha) / (N + 2.0 * self.alpha)
                        chosen_n = float(N)
                        break

                if chosen_p is None:
                    chosen_p = p0
                    chosen_n = 0.0

                chosen_p = float(min(max(chosen_p, 1e-4), 1.0 - 1e-4))
                pv[i, t] = chosen_p
                counts[i, t] = float(chosen_n)

        return pv, counts

    # ---------- report utils ----------
    @staticmethod
    def _bernoulli_entropy(p: float, eps: float = 1e-12) -> float:
        p = min(max(float(p), eps), 1.0 - eps)
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    @staticmethod
    def _logit(p: float, eps: float = 1e-12) -> float:
        p = min(max(float(p), eps), 1.0 - eps)
        return math.log(p) - math.log(1.0 - p)

    @torch.no_grad()
    def report(
        self,
        *,
        max_keys_sample: int = 20000,
        sample_seed: int = 0,
        phi_eta: float = 1.0,
        phi_clamp: float = 2.0,
        tau_conf: Optional[float] = None,
        step_scale: float = 1.0,
        pv_sat_lo: float = 0.05,
        pv_sat_hi: float = 0.95,
        csv_kgram_path: Optional[str] = None,
        csv_prefix_path: Optional[str] = None,
        probe_tokens: Optional[torch.Tensor] = None,  # (B,T)
        probe_active_before: Optional[torch.Tensor] = None,  # (B,T) bool
        probe_ref_log_pf: Optional[torch.Tensor] = None,  # (B,T)
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Returns dict of scalars (wandb-friendly).
        Optionally writes:
          - csv_kgram_path: per-k aggregated memory stats
          - csv_prefix_path: per-position stats from probe batch
        """
        if step is not None:
            self.set_step(step)
        else:
            self._decay_global()

        if tau_conf is None:
            tau_conf = float(self.tau_conf)
        else:
            tau_conf = float(tau_conf)

        p0 = self.get_base_rate()

        num_keys = len(self.stats)
        if num_keys == 0:
            return {
                "pv_mem/num_keys": 0.0,
                "pv_mem/base_rate": float(p0),
                "pv_mem/global_N": float(self.global_N),
                "pv_mem/global_S": float(self.global_S),
            }

        rng = random.Random(int(sample_seed) + int(self.step))
        keys = list(self.stats.keys())
        if max_keys_sample is not None and len(keys) > int(max_keys_sample):
            keys_sample = rng.sample(keys, k=int(max_keys_sample))
        else:
            keys_sample = keys

        agg: Dict[int, Dict[str, float]] = {
            k: {
                "n_keys": 0.0,
                "sumN": 0.0,
                "sumN2": 0.0,
                "sum_p": 0.0,
                "sum_p2": 0.0,
                "sum_ent": 0.0,
                "sat_cnt": 0.0,
                "sum_conf": 0.0,
                "sum_phi_abs": 0.0,
                "sum_phi2": 0.0,
                "sum_age": 0.0,
            }
            for k in range(1, self.kmax + 1)
        }

        N_list: List[float] = []

        for key in keys_sample:
            # key = (pos_bucket, k, tok_tuple)
            if len(key) != 3:
                continue
            pos_bucket, kk, _tok_tuple = key
            kk = int(kk)
            if kk < 1 or kk > self.kmax:
                continue

            # age must be computed BEFORE decay overwrites last_step to current
            old_last_step = float(self.stats[key][2])
            N, S = self._decay_key_inplace(key)  # in-place sets last_step=self.step
            N = float(N)
            S = float(S)
            if N <= 0.0:
                continue

            p = float((S + self.alpha) / (N + 2.0 * self.alpha))
            p = min(max(p, 1e-6), 1.0 - 1e-6)

            ent = self._bernoulli_entropy(p)
            sat = 1.0 if (p < pv_sat_lo or p > pv_sat_hi) else 0.0

            conf = N / (N + float(tau_conf))
            logit = self._logit(p)
            logit = max(-float(phi_clamp), min(float(phi_clamp), logit))

            # memory-side φ magnitude proxy
            phi = float(phi_eta) * float(step_scale) * logit * conf
            phi_abs = abs(phi)

            age = max(0.0, float(self.step) - old_last_step)

            a = agg[kk]
            a["n_keys"] += 1.0
            a["sumN"] += N
            a["sumN2"] += N * N
            a["sum_p"] += p
            a["sum_p2"] += p * p
            a["sum_ent"] += ent
            a["sat_cnt"] += sat
            a["sum_conf"] += conf
            a["sum_phi_abs"] += phi_abs
            a["sum_phi2"] += phi * phi
            a["sum_age"] += age

            N_list.append(N)

        if len(N_list) > 0:
            sumN = float(sum(N_list))
            sumN2 = float(sum([x * x for x in N_list]))
            eff_keys = (sumN * sumN) / max(1e-12, sumN2)
            N_sorted = sorted(N_list, reverse=True)
            topk = max(1, int(len(N_sorted) * 0.01))
            top_mass = float(sum(N_sorted[:topk]) / max(1e-12, sumN))
        else:
            eff_keys = 0.0
            top_mass = 0.0

        out: Dict[str, float] = {
            "pv_mem/num_keys": float(num_keys),
            "pv_mem/sample_keys": float(len(keys_sample)),
            "pv_mem/base_rate": float(p0),
            "pv_mem/global_N": float(self.global_N),
            "pv_mem/global_S": float(self.global_S),
            "pv_mem/effective_keys_est": float(eff_keys),
            "pv_mem/top1pct_mass_est": float(top_mass),
        }

        for kk in range(1, self.kmax + 1):
            a = agg[kk]
            n = max(1.0, a["n_keys"])
            meanN = a["sumN"] / n
            meanP = a["sum_p"] / n
            varP = max(0.0, a["sum_p2"] / n - meanP * meanP)
            meanEnt = a["sum_ent"] / n
            satRatio = a["sat_cnt"] / n
            meanConf = a["sum_conf"] / n
            meanPhiAbs = a["sum_phi_abs"] / n
            phi2 = a["sum_phi2"] / n
            meanAge = a["sum_age"] / n

            out[f"pv_mem/k{kk}/n_keys_est"] = float(a["n_keys"])
            out[f"pv_mem/k{kk}/meanN_est"] = float(meanN)
            out[f"pv_mem/k{kk}/meanP_est"] = float(meanP)
            out[f"pv_mem/k{kk}/varP_est"] = float(varP)
            out[f"pv_mem/k{kk}/entropy_est"] = float(meanEnt)
            out[f"pv_mem/k{kk}/sat_ratio_est"] = float(satRatio)
            out[f"pv_mem/k{kk}/conf_est"] = float(meanConf)
            out[f"pv_mem/k{kk}/phi_abs_est"] = float(meanPhiAbs)
            out[f"pv_mem/k{kk}/phi_2nd_moment_est"] = float(phi2)
            out[f"pv_mem/k{kk}/age_est"] = float(meanAge)

        if csv_kgram_path is not None:
            with open(csv_kgram_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "k",
                        "n_keys_est",
                        "meanN_est",
                        "meanP_est",
                        "varP_est",
                        "entropy_est",
                        "sat_ratio_est",
                        "conf_est",
                        "phi_abs_est",
                        "phi_2nd_moment_est",
                        "age_est",
                    ]
                )
                for kk in range(1, self.kmax + 1):
                    w.writerow(
                        [
                            kk,
                            out[f"pv_mem/k{kk}/n_keys_est"],
                            out[f"pv_mem/k{kk}/meanN_est"],
                            out[f"pv_mem/k{kk}/meanP_est"],
                            out[f"pv_mem/k{kk}/varP_est"],
                            out[f"pv_mem/k{kk}/entropy_est"],
                            out[f"pv_mem/k{kk}/sat_ratio_est"],
                            out[f"pv_mem/k{kk}/conf_est"],
                            out[f"pv_mem/k{kk}/phi_abs_est"],
                            out[f"pv_mem/k{kk}/phi_2nd_moment_est"],
                            out[f"pv_mem/k{kk}/age_est"],
                        ]
                    )

        # ---- probes (optional) ----
        if (probe_tokens is not None) and (probe_active_before is not None):
            pv_t, counts_t = self.query_pv(probe_tokens, probe_active_before)

            if probe_ref_log_pf is None:
                ref_log_pf = pv_t.new_full(pv_t.shape, float(step_scale))
            else:
                ref_log_pf = probe_ref_log_pf

            phi_tok = build_prefix_potential(
                pv=pv_t,
                ref_log_pf=ref_log_pf,
                non_term_mask=probe_active_before,
                counts=counts_t,
                eta=float(phi_eta),
                clamp=float(phi_clamp),
                tau_conf=float(tau_conf),
            )

            phi_state, dphi = phi_tok_to_phi_state_and_dphi(
                L=pv_t.shape[1] + 1,
                phi_tok=phi_tok,
                active_before=probe_active_before,
                anchor_start=0.0,
                anchor_end=0.0,
            )

            mask = probe_active_before.to(pv_t.dtype)
            p = pv_t.clamp(1e-6, 1.0 - 1e-6)
            ent = -(p * p.log() + (1.0 - p) * (1.0 - p).log())

            def masked_mean_per_t(x: torch.Tensor) -> List[float]:
                num = (x * mask).sum(dim=0)
                den = mask.sum(dim=0).clamp_min(1.0)
                return (num / den).detach().cpu().tolist()

            pv_mean = masked_mean_per_t(pv_t)
            ent_mean = masked_mean_per_t(ent)
            counts_mean = masked_mean_per_t(counts_t)
            phi_abs_mean = masked_mean_per_t(phi_tok.abs())
            dphi_abs_mean = masked_mean_per_t(dphi.abs())

            if csv_prefix_path is not None:
                with open(csv_prefix_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "t",
                            "pv_mean",
                            "pv_entropy_mean",
                            "counts_mean",
                            "phi_abs_mean",
                            "dphi_abs_mean",
                        ]
                    )
                    T = pv_t.shape[1]
                    for t in range(T):
                        w.writerow(
                            [
                                t,
                                pv_mean[t],
                                ent_mean[t],
                                counts_mean[t],
                                phi_abs_mean[t],
                                dphi_abs_mean[t],
                            ]
                        )

            ent_all = (ent * mask).sum() / mask.sum().clamp_min(1.0)
            sat = ((p < pv_sat_lo) | (p > pv_sat_hi)).to(mask.dtype)
            sat_ratio = (sat * mask).sum() / mask.sum().clamp_min(1.0)

            out.update(
                {
                    "pv_probe/entropy_mean": float(ent_all.item()),
                    "pv_probe/sat_ratio": float(sat_ratio.item()),
                    "pv_probe/phi_abs_mean": float(
                        (phi_tok.abs() * mask).sum().div(mask.sum().clamp_min(1.0)).item()
                    ),
                    "pv_probe/dphi_abs_mean": float(
                        (dphi.abs() * mask).sum().div(mask.sum().clamp_min(1.0)).item()
                    ),
                    "pv_probe/counts_mean": float(
                        (counts_t * mask).sum().div(mask.sum().clamp_min(1.0)).item()
                    ),
                }
            )

        return out


def compute_active_before(gen_tokens: torch.Tensor, eos: int) -> torch.Tensor:
    """
    gen_tokens: (B, T_tok) prompt 后 tokens
    active_before[:, t] = Π_{j < t} 1[token_j != eos]
    注意：采样 eos 的那一步 t 仍然 active_before=True（因为 eos 是“这一步采出来的”）
    """
    B, T = gen_tokens.shape
    active_before = torch.ones((B, T), device=gen_tokens.device, dtype=torch.bool)
    if T > 1:
        alive_after = (gen_tokens != eos).to(torch.long).cumprod(dim=1).to(torch.bool)  # Π_{j<=t}
        active_before[:, 1:] = alive_after[:, :-1]  # Π_{j< t}
    return active_before


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8):
    mask_f = mask.to(x.dtype)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(eps)
    return num / den


def _masked_var_across_batch(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    x, mask: (B, T)
    返回 per_step_var: (T,)
    """
    m = _masked_mean(x, mask, dim=0, eps=eps)
    m2 = _masked_mean(x * x, mask, dim=0, eps=eps)
    v = (m2 - m * m).clamp_min(0.0)
    return v


def compute_prefix_diagnostics(
    pv: torch.Tensor,  # (B, T_tok) in (0,1)
    phi_state: torch.Tensor,  # (B, L_state)
    active_before: torch.Tensor,  # (B, T_tok) True=该 token 动作发生时 still active（包含 EOS 那一步）
    pv_sat_lo: float = 0.05,
    pv_sat_hi: float = 0.95,
    eps: float = 1e-8,
):
    """
    产出：
      - 标量：phi_var_mean, dphi_abs_mean, d2phi_abs_mean, pv_entropy_mean, pv_sat_ratio, pv_logit_abs_mean
      - 向量：phi_var_per_step, dphi_abs_per_step, d2phi_abs_per_step, pv_entropy_per_step
    """
    with torch.no_grad():
        B, T_tok = pv.shape
        L_state = phi_state.shape[1]
        assert L_state == T_tok + 1, f"expect L_state=T_tok+1, got {L_state} vs {T_tok}"

        # --- masks 对齐 ---
        # state_active_mask: (B, L_state)  state 0..L_state-2 用 active_before，最后 state 无效
        state_active_mask = torch.cat(
            [active_before, active_before.new_zeros((B, 1))], dim=1
        )  # (B, L_state)

        # --- Var(phi) per state-step (across batch) ---
        phi_var_per_step = _masked_var_across_batch(
            phi_state, state_active_mask, eps=eps
        )  # (L_state,)
        phi_counts = state_active_mask.sum(dim=0)  # (L_state,)
        phi_var_mean = phi_var_per_step[phi_counts > 0].mean()

        # --- Δphi on transitions (token steps) ---
        dphi = phi_state[:, 1:] - phi_state[:, :-1]  # (B, T_tok)
        dphi_abs_per_step = _masked_mean(dphi.abs(), active_before, dim=0, eps=eps)  # (T_tok,)
        dphi_counts = active_before.sum(dim=0)
        dphi_abs_mean = dphi_abs_per_step[dphi_counts > 0].mean()

        # --- Δ^2 phi (second difference) ---
        if T_tok >= 2:
            d2phi = dphi[:, 1:] - dphi[:, :-1]  # (B, T_tok-1)
            mask2 = active_before[:, 1:] & active_before[:, :-1]  # (B, T_tok-1)
            d2phi_abs_per_step = _masked_mean(d2phi.abs(), mask2, dim=0, eps=eps)  # (T_tok-1,)
            d2_counts = mask2.sum(dim=0)
            d2phi_abs_mean = d2phi_abs_per_step[d2_counts > 0].mean()
        else:
            d2phi_abs_per_step = pv.new_zeros((0,))
            d2phi_abs_mean = pv.new_zeros(())

        # --- pv entropy / saturation ---
        p = pv.clamp(1e-6, 1.0 - 1e-6)
        pv_entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())  # Bernoulli entropy, (B,T_tok)
        pv_entropy_per_step = _masked_mean(pv_entropy, active_before, dim=0, eps=eps)  # (T_tok,)
        pv_entropy_mean = pv_entropy_per_step[dphi_counts > 0].mean()

        sat = (p < pv_sat_lo) | (p > pv_sat_hi)
        pv_sat_ratio = (sat & active_before).sum().to(pv.dtype) / active_before.sum().clamp_min(
            1
        ).to(pv.dtype)

        pv_logit = p.log() - (1.0 - p).log()  # logit(p)
        pv_logit_abs_mean = _masked_mean(pv_logit.abs(), active_before, dim=None, eps=eps)

        # 你也可以额外看信号强度（和 grammar penalty 对比）
        phi_abs_mean = _masked_mean(phi_state.abs(), state_active_mask, dim=None, eps=eps)

        return {
            # scalars (直接 log)
            "phi_var_mean": phi_var_mean.detach(),
            "phi_abs_mean": phi_abs_mean.detach(),
            "dphi_abs_mean": dphi_abs_mean.detach(),
            "d2phi_abs_mean": d2phi_abs_mean.detach(),
            "pv_entropy_mean": pv_entropy_mean.detach(),
            "pv_sat_ratio": pv_sat_ratio.detach(),
            "pv_logit_abs_mean": pv_logit_abs_mean.detach(),
        }
