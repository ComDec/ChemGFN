from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def _to_bytes_prefix(tokens: torch.Tensor, m: int) -> bytes:
    """Return prefix key of length m as bytes (int32) for grouping."""
    if m <= 0:
        return b""
    arr = tokens[:m].to(dtype=torch.int32).cpu().numpy()
    return arr.tobytes()


def grouped_weighted_var(keys: list[bytes], values: torch.Tensor) -> dict[str, float]:
    """Compute weighted group variance with structure diagnostics.

    Args:
        keys: list of prefix keys (bytes).
        values: 1D tensor of targets aligned with keys.

    Returns:
        Dict with cond_var, num_groups, singleton_mass, max_group_mass, finite_rate,
        count, and quantiles (min,p1,p50,p99,max).
    """
    if values.numel() == 0 or len(keys) == 0:
        return {}

    vals = values.detach().cpu()
    finite_mask = torch.isfinite(vals)
    vals = vals[finite_mask]
    keys = [k for k, ok in zip(keys, finite_mask.tolist()) if ok]
    if vals.numel() == 0:
        return {
            "cond_var": 0.0,
            "num_groups": 0.0,
            "singleton_mass": 0.0,
            "max_group_mass": 0.0,
            "finite_rate": 0.0,
            "count": 0.0,
        }

    vals_list = vals.tolist()
    groups: dict[bytes, tuple[int, float, float]] = {}
    for k, v in zip(keys, vals_list):
        n, s1, s2 = groups.get(k, (0, 0.0, 0.0))
        groups[k] = (n + 1, s1 + v, s2 + v * v)

    total = float(len(vals_list))
    var_weighted = 0.0
    singleton = 0
    max_group = 0
    for n, s1, s2 in groups.values():
        mean = s1 / float(n)
        var_g = s2 / float(n) - mean * mean
        var_weighted += (float(n) / total) * var_g
        if n == 1:
            singleton += 1
        if n > max_group:
            max_group = n

    v_sorted = torch.sort(vals)[0]

    def q(p: float) -> float:
        idx = min(len(v_sorted) - 1, max(0, int(round(p * (len(v_sorted) - 1)))))
        return float(v_sorted[idx].item())

    return {
        "cond_var": float(var_weighted),
        "num_groups": float(len(groups)),
        "singleton_mass": float(singleton / total),
        "max_group_mass": float(max_group / total) if total > 0 else 0.0,
        "finite_rate": float(vals.numel() / float(len(keys))),
        "count": total,
        "t_min": float(v_sorted[0].item()),
        "t_p1": q(0.01),
        "t_p50": q(0.50),
        "t_p99": q(0.99),
        "t_max": float(v_sorted[-1].item()),
    }


def compute_tau_from_tokens(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    if tokens is None or tokens.numel() == 0:
        return torch.zeros(
            (tokens.shape[0] if tokens is not None else 0,),
            dtype=torch.long,
            device=tokens.device if tokens is not None else "cpu",
        )
    max_len = tokens.shape[1]
    eos_mask = tokens == eos_id
    idxs = torch.arange(max_len, device=tokens.device).unsqueeze(0).expand_as(tokens)
    first_eos = torch.where(eos_mask, idxs, torch.full_like(idxs, max_len))
    tau = first_eos.min(dim=1).values
    tau = torch.clamp(tau, max=max_len - 1)
    return tau.to(dtype=torch.long)


def _pf_prefix_cum(log_pf_steps: torch.Tensor) -> torch.Tensor:
    if log_pf_steps is None or log_pf_steps.numel() == 0:
        return torch.zeros_like(log_pf_steps)
    return log_pf_steps.cumsum(dim=1)


def compute_tb_targets(
    tokens: torch.Tensor,
    log_pf_steps: torch.Tensor,
    log_pterm: torch.Tensor,
    log_r: torch.Tensor,
    tau: torch.Tensor,
    m_values: Sequence[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute TB Ym targets and prefix keys for each m."""
    if any(x is None for x in (tokens, log_pf_steps, log_pterm, log_r, tau)):
        return {}
    pf_cum = _pf_prefix_cum(log_pf_steps)
    N, Lm1 = log_pf_steps.shape
    results: dict[int, dict[str, torch.Tensor]] = {}
    log_pterm_tau = log_pterm.gather(1, tau.view(N, 1)).squeeze(1)
    log_r_tau = log_r.gather(1, tau.view(N, 1)).squeeze(1)
    pf_tau = pf_cum.gather(1, (tau - 1).clamp(min=0, max=pf_cum.shape[1] - 1).view(N, 1)).squeeze(
        1
    )
    pf_tau = pf_tau * (tau > 0).to(pf_tau.dtype)

    for m in m_values:
        if m < 0 or m > Lm1:
            continue
        eligible = tau > m
        if eligible.sum().item() < 1:
            continue
        if m == 0:
            pf_m = torch.zeros_like(pf_tau)
        else:
            pf_m = pf_cum[:, m - 1]
        Ym = log_r_tau - log_pterm_tau - (pf_tau - pf_m)
        Ym = Ym[eligible]
        if Ym.numel() == 0:
            continue
        keys = [_to_bytes_prefix(t, m) for t in tokens[eligible]]
        results[m] = {"keys": keys, "targets": Ym}
    return results


def _reconstruct_ref_logP(ref_log_pf: torch.Tensor, ref_log_pterm: torch.Tensor) -> torch.Tensor:
    B, T = ref_log_pf.shape
    _, L = ref_log_pterm.shape
    assert L == T + 1
    prefix = ref_log_pf.cumsum(dim=1)
    ref_logP = ref_log_pterm.clone()
    ref_logP[:, 1:] = ref_logP[:, 1:] + prefix
    return ref_logP


def compute_raptb_targets(
    tokens: torch.Tensor,
    log_pf: torch.Tensor,
    log_pterm: torch.Tensor,
    log_r: torch.Tensor,
    tau: torch.Tensor,
    valid_end: torch.Tensor,
    m_values: Sequence[int],
    *,
    gamma: float,
    k_min: int,
    extra_absorb_eps: float,
    soft_beta: float,
    soft_rho: float,
    target_mode: str,
    mix_weight: float,
    ref_log_pf: torch.Tensor | None,
    ref_log_pterm: torch.Tensor | None,
    ref_scale: float = 1.0,
    max_prefix_len: int | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute RapTB Yeff targets and masks for each m."""
    if any(x is None for x in (tokens, log_pf, log_pterm, log_r, tau, valid_end)):
        return {}

    B, L = log_pf.shape
    kmin = max(1, min(int(k_min), L - 1))
    if max_prefix_len is None:
        K = None
    else:
        K = max(1, min(int(max_prefix_len), L - 1))

    pos = torch.arange(L, device=log_pf.device).view(1, L)
    if K is None:
        h = tau
    else:
        h = torch.minimum(tau, tau.new_full(tau.shape, K))

    within_h = pos <= h.view(B, 1)
    within_tau = pos <= tau.view(B, 1)

    steps = log_pf[:, :-1]
    pf_cum = _pf_prefix_cum(steps)

    log_pterm_tau = log_pterm.gather(1, tau.view(B, 1)).squeeze(1)
    log_r_tau = log_r.gather(1, tau.view(B, 1)).squeeze(1)
    pf_tau = pf_cum.gather(1, (tau - 1).clamp(min=0, max=pf_cum.shape[1] - 1).view(B, 1)).squeeze(
        1
    )
    pf_tau = pf_tau * (tau > 0).to(pf_tau.dtype)

    if (ref_log_pf is not None) and (ref_log_pterm is not None):
        ref_logP = _reconstruct_ref_logP(ref_log_pf, ref_log_pterm)
        u = log_r - float(ref_scale) * ref_logP
    else:
        u = log_r

    valid_future = (pos <= h.view(B, 1)) & (pos <= tau.view(B, 1))
    u_det = u.detach()
    u_max = _suffix_future_max(u_det, valid_future)
    u_soft = _suffix_future_soft(u_det, valid_future, beta=soft_beta, rho=soft_rho)
    if target_mode == "future_max":
        u_target = u_max
    elif target_mode == "future_soft":
        u_target = u_soft
    elif target_mode == "mix":
        mw = float(mix_weight)
        u_target = mw * u_max + (1.0 - mw) * u_soft
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    u_target = torch.where(valid_future, u_target, torch.zeros_like(u_target))

    results: dict[int, dict[str, torch.Tensor]] = {}
    for m in m_values:
        if m <= 0 or m >= L:
            continue
        # eligibility like loss (positions are 1..L-1; here m is that index)
        m_idx = m  # position index in 0..L-1
        if m_idx >= steps.shape[1] + 1:
            continue
        m_base = within_tau[:, m_idx] & within_h[:, m_idx] & valid_end[:, m_idx - 1]
        after_kmin = m_idx >= kmin
        if not after_kmin:
            continue
        if K is None:
            before_horizon = within_h[:, m_idx]  # [B] bool
            can_absorb_seq = torch.ones_like(before_horizon, dtype=torch.bool)
        else:
            before_horizon = torch.full_like(tau, m_idx < K, dtype=torch.bool)  # [B] bool
            can_absorb_seq = tau >= K  # [B] bool
        u_m = u[:, m_idx]
        u_target_m = u_target[:, m_idx]
        exp = (h - m_idx).clamp_min(0).to(log_pf.dtype)
        alpha = (gamma**exp) * within_h[:, m_idx].to(log_pf.dtype)
        apply_absorb = m_base & (u_m.abs() <= extra_absorb_eps) & before_horizon & can_absorb_seq
        # TB Ym
        pf_m = pf_cum[:, m_idx - 1]
        Ym = log_r_tau - log_pterm_tau - (pf_tau - pf_m)
        alpha_eff = alpha * apply_absorb.to(log_pf.dtype)
        Yeff = Ym - alpha_eff * (u_m - u_target_m)
        # effective-only
        eff_mask = apply_absorb & torch.isfinite(Yeff) & (tau > m_idx)
        if eff_mask.any():
            keys_eff = [_to_bytes_prefix(t, m_idx) for t in tokens[eff_mask]]
            results.setdefault(m, {})["keys_eff"] = keys_eff
            results[m]["targets_eff"] = Yeff[eff_mask]
            results[m]["apply_rate"] = float(apply_absorb.float().mean().item())
        # hybrid: use Yeff where apply_absorb else Ym
        hyb = torch.where(apply_absorb, Yeff, Ym)
        hyb_mask = (tau > m_idx) & torch.isfinite(hyb)
        if hyb_mask.any():
            keys_hyb = [_to_bytes_prefix(t, m_idx) for t in tokens[hyb_mask]]
            results.setdefault(m, {})["keys_hyb"] = keys_hyb
            results[m]["targets_hyb"] = hyb[hyb_mask]
            results[m]["apply_rate"] = float(apply_absorb.float().mean().item())
    return results


def _suffix_future_max(u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    u_mask = torch.where(valid, u, torch.full_like(u, 0))
    rev = torch.flip(u_mask, dims=[1])
    rev_max = torch.cummax(rev, dim=1).values
    return torch.flip(rev_max, dims=[1])


def _suffix_future_soft(
    u: torch.Tensor, valid: torch.Tensor, beta: float, rho: float
) -> torch.Tensor:
    B, L = u.shape
    u_mask = torch.where(valid, u, torch.full_like(u, -torch.inf))
    out = torch.empty_like(u_mask)
    b = float(beta)
    step_pen = b * float(rho)
    Z = b * u_mask[:, -1]
    out[:, -1] = Z / b
    for t in range(L - 2, -1, -1):
        Z = torch.logaddexp(b * u_mask[:, t], Z - step_pen)
        out[:, t] = Z / b
    return out


def compute_subtb_targets_delta(
    tokens: torch.Tensor,
    log_pf: torch.Tensor,
    log_pterm: torch.Tensor,
    log_r: torch.Tensor,
    m_values: Sequence[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Approximate SubTB-aligned target using local delta at position m."""
    if any(x is None for x in (tokens, log_pf, log_pterm, log_r)):
        return {}
    B, L = log_pf.shape
    if L < 2:
        return {}
    delta = log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:] - log_r[:, 1:] - log_pterm[:, :-1]
    results: dict[int, dict[str, torch.Tensor]] = {}
    for m in m_values:
        if m < 0 or m >= delta.shape[1]:
            continue
        vals = delta[:, m]
        mask = torch.isfinite(vals)
        if mask.sum().item() == 0:
            continue
        keys = [
            _to_bytes_prefix(t, m + 1) for t in tokens[mask]
        ]  # prefix length m+1 corresponds to state after step m
        results[m] = {"keys": keys, "targets": vals[mask]}
    return results
