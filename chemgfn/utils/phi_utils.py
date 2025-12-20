import torch


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
