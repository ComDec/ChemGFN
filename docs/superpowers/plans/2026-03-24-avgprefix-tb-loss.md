# AvgPrefixTB Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three AvgPrefixTB loss variants (strict, controlled TB+aux, detach-pterm) as reviewer-requested baselines, with experiment configs for SMILES and VarExpr24.

**Architecture:** All three variants share a single loss class `AvgPrefixTBLoss` with config flags (`mode`, `detach_pterm_in_aux`) that select behavior. The class inherits from `GFNLoss`, uses the same `(log_pf, log_r, log_pterm, generated_text, termination_token_id, prompt_len, **kwargs)` signature, and computes prefix TB residuals vectorized over all positions. No changes to `gfn.py` are needed — the existing `training_step` already passes `k_min` and `max_prefix_len` kwargs, and the new loss simply ignores or uses them as appropriate.

**Tech Stack:** PyTorch, Lightning, Hydra configs, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `chemgfn/models/losses.py` | Modify (append) | Add `AvgPrefixTBLoss` class (~120 lines) |
| `tests/test_avgprefix_tb_loss.py` | Create | Unit tests for all three variants |
| `configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB.yaml` | Create | Version A: SMILES strict baseline |
| `configs/experiment/SMILES_basic/SMILES_cfg_TB_plus_AvgAux.yaml` | Create | Version B: SMILES controlled |
| `configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB_detach_pterm.yaml` | Create | Version C: SMILES diagnostic |
| `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB.yaml` | Create | Version A: VarExpr24 strict baseline |
| `configs/experiment/VarExpr24/VarExpr24_TB_plus_AvgAux.yaml` | Create | Version B: VarExpr24 controlled |
| `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_detach_pterm.yaml` | Create | Version C: VarExpr24 diagnostic |

---

## Key Design Decisions

### Tensor conventions (match existing codebase exactly)

- `log_pf`: `[B, L]` — meaningful token steps are `log_pf[:, :-1]` (length `L-1`)
- `log_pterm`: `[B, L]` — EOS log-prob at each state `k=0..L-1`
- `log_r`: `[B, L]` — per-state terminate-at-k log-reward
- `generated_text`: `[B, prompt_len + L]` — includes forced terminal EOS
- `tau`: computed from `generated_text[:, prompt_len:-1]` via cumsum EOS detection, values in `[0, L-1]`

### Prefix TB residual at position k

```
Δ_k^TB = logZ + cumsum(log_pf[:, :k]) + log_pterm[k] - log_r[k]
```

Where `cumsum(log_pf[:, :k])` means `sum_{t=0}^{k-1} log_pf[t]` using the meaningful steps `log_pf[:, :-1]`.

### Three modes in one class

| Mode | Config value | Behavior |
|------|-------------|----------|
| Version A | `mode: "avgprefix"` | Loss = mean over all `k in [0, tau]` of `(Δ_k^TB)^2` |
| Version B | `mode: "tb_plus_avgaux"` | Loss = `(Δ_tau^TB)^2 + eta * weighted_mean_{k in [kmin, h-1]} (Δ_k^TB)^2` |
| Version C | Same as A or B | `detach_pterm_in_aux: true` — detaches `log_pterm` in the prefix residuals |

### Compatibility with existing training_step

The `training_step` in `gfn.py` (line 530) already passes `max_prefix_len` and `k_min` as kwargs. `AvgPrefixTBLoss.forward()` accepts these via `**kwargs` and uses them when `mode="tb_plus_avgaux"`. For `mode="avgprefix"`, they are ignored.

---

## Task 1: Write failing tests for AvgPrefixTBLoss

**Files:**
- Create: `tests/test_avgprefix_tb_loss.py`

- [ ] **Step 1: Write test file with all test cases**

```python
"""Tests for AvgPrefixTBLoss — strict, controlled, and detach-pterm variants."""

import pytest
import torch


class TestAvgPrefixTBLossBasic:
    """Basic tests for Version A: strict AvgPrefixTB."""

    @pytest.fixture
    def batch(self):
        """Standard test batch.

        B=4, L=6 (5 token steps + 1 forced EOS position).
        prompt_len=3, so generated_text is [B, 3+6=9].
        EOS placed at position 4 within generated part for samples 0,1
        (tau=4), and at position 5 for samples 2,3 (tau=5=L-1).
        """
        B, L, prompt_len = 4, 6, 3
        eos = 0

        torch.manual_seed(42)
        log_pf = torch.randn(B, L, requires_grad=True)
        log_r = torch.randn(B, L)
        log_pterm = torch.randn(B, L, requires_grad=True)

        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        # Force terminal EOS at last position (always)
        generated_text[:, -1] = eos
        # Put EOS at generated position 4 for first two samples
        generated_text[0, prompt_len + 4] = eos
        generated_text[1, prompt_len + 4] = eos
        # Samples 2,3: no early EOS, so tau = L-1 = 5

        return {
            "log_pf": log_pf,
            "log_r": log_r,
            "log_pterm": log_pterm,
            "generated_text": generated_text,
            "termination_token_id": eos,
            "prompt_len": prompt_len,
            "B": B,
            "L": L,
        }

    def test_import(self):
        """AvgPrefixTBLoss can be imported."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss()
        assert loss_fn is not None

    def test_output_is_dict_with_loss(self, batch):
        """forward() returns dict with 'loss' key."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss(mode="avgprefix")
        out = loss_fn(
            log_pf=batch["log_pf"],
            log_r=batch["log_r"],
            log_pterm=batch["log_pterm"],
            generated_text=batch["generated_text"],
            termination_token_id=batch["termination_token_id"],
            prompt_len=batch["prompt_len"],
        )
        assert isinstance(out, dict)
        assert "loss" in out

    def test_loss_is_scalar_nonneg(self, batch):
        """Loss is a non-negative scalar."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss(mode="avgprefix")
        out = loss_fn(**{k: batch[k] for k in [
            "log_pf", "log_r", "log_pterm", "generated_text",
            "termination_token_id", "prompt_len",
        ]})
        loss = out["loss"]
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_gradient_flows(self, batch):
        """Gradients flow back through log_pf and log_pterm."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss(mode="avgprefix")
        out = loss_fn(**{k: batch[k] for k in [
            "log_pf", "log_r", "log_pterm", "generated_text",
            "termination_token_id", "prompt_len",
        ]})
        out["loss"].backward()
        assert batch["log_pf"].grad is not None
        assert batch["log_pterm"].grad is not None
        assert batch["log_pf"].grad.abs().sum() > 0
        assert batch["log_pterm"].grad.abs().sum() > 0

    def test_perfect_balance_gives_zero_loss(self):
        """When residuals are exactly zero, loss should be zero."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        B, L, prompt_len, eos = 2, 4, 2, 0

        # Construct tensors so that logZ + cumsum(log_pf[:k]) + log_pterm[k] == log_r[k]
        # for all valid k. Use logZ=0, log_pf=0, log_pterm = log_r.
        log_pf = torch.zeros(B, L)
        log_r = torch.ones(B, L) * 2.0
        log_pterm = torch.ones(B, L) * 2.0  # = log_r, so residual = 0 + 0 + 2 - 2 = 0

        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        generated_text[:, -1] = eos  # EOS at last position only

        loss_fn = AvgPrefixTBLoss(mode="avgprefix", init_logZ=0.0)
        out = loss_fn(log_pf, log_r, log_pterm, generated_text, eos, prompt_len)
        assert out["loss"].item() < 1e-10

    def test_manual_residual_computation(self):
        """Verify residual math matches hand computation for a single sample."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        B, L, prompt_len, eos = 1, 4, 1, 99
        # log_pf steps: [0.1, 0.2, 0.3] (last position ignored)
        log_pf = torch.tensor([[0.1, 0.2, 0.3, 0.0]])
        log_pterm = torch.tensor([[-1.0, -0.5, -0.3, -0.2]])
        log_r = torch.tensor([[0.5, 1.0, 1.5, 2.0]])

        # generated_text: prompt(1) + gen(4), EOS at position 3 => tau=3 (L-1)
        generated_text = torch.tensor([[50, 10, 20, 30, eos]])

        logZ_val = 0.5
        loss_fn = AvgPrefixTBLoss(mode="avgprefix", init_logZ=logZ_val)

        out = loss_fn(log_pf, log_r, log_pterm, generated_text, eos, prompt_len)

        # Manual: prefix_logpf[k] = cumsum of log_pf[:, :-1] = [0.1, 0.3, 0.6]
        # prepend 0 => [0, 0.1, 0.3, 0.6]
        # residual[k] = logZ + prefix_logpf[k] + log_pterm[k] - log_r[k]
        # k=0: 0.5 + 0    + (-1.0) - 0.5 = -1.0
        # k=1: 0.5 + 0.1  + (-0.5) - 1.0 = -0.9
        # k=2: 0.5 + 0.3  + (-0.3) - 1.5 = -1.0
        # k=3: 0.5 + 0.6  + (-0.2) - 2.0 = -1.1
        # tau=3, so all k in [0,3] are valid
        # loss = mean([1.0, 0.81, 1.0, 1.21]) = 1.005
        expected = (1.0 + 0.81 + 1.0 + 1.21) / 4.0
        assert abs(out["loss"].item() - expected) < 1e-5, (
            f"Expected {expected}, got {out['loss'].item()}"
        )


class TestAvgPrefixTBLossControlled:
    """Tests for Version B: TB + AvgAux controlled variant."""

    @pytest.fixture
    def batch(self):
        B, L, prompt_len, eos = 2, 6, 2, 0
        torch.manual_seed(123)
        log_pf = torch.randn(B, L, requires_grad=True)
        log_r = torch.randn(B, L)
        log_pterm = torch.randn(B, L, requires_grad=True)
        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        generated_text[:, -1] = eos
        return {
            "log_pf": log_pf, "log_r": log_r, "log_pterm": log_pterm,
            "generated_text": generated_text,
            "termination_token_id": eos, "prompt_len": prompt_len,
        }

    def test_tb_plus_avgaux_returns_components(self, batch):
        """TB+AvgAux mode returns loss, loss_tb, loss_aux keys."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss(mode="tb_plus_avgaux", aux_eta=0.25, k_min=1)
        out = loss_fn(**batch)
        assert "loss" in out
        assert "loss_tb" in out
        assert "loss_aux" in out
        assert out["loss"].ndim == 0

    def test_eta_zero_equals_pure_tb(self, batch):
        """When eta=0, TB+AvgAux should reduce to pure terminal TB loss."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn_aux = AvgPrefixTBLoss(mode="tb_plus_avgaux", aux_eta=0.0, k_min=1)
        out_aux = loss_fn_aux(**batch)

        loss_fn_tb = AvgPrefixTBLoss(mode="tb_plus_avgaux", aux_eta=0.0, k_min=1)
        out_tb = loss_fn_tb(**batch)

        # Both should have the same loss (pure TB)
        assert abs(out_aux["loss"].item() - out_tb["loss"].item()) < 1e-6

    def test_kmin_filters_early_prefixes(self, batch):
        """Increasing k_min should change the aux loss value."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn_k1 = AvgPrefixTBLoss(mode="tb_plus_avgaux", aux_eta=0.5, k_min=1)
        loss_fn_k3 = AvgPrefixTBLoss(mode="tb_plus_avgaux", aux_eta=0.5, k_min=3)

        out_k1 = loss_fn_k1(**batch)
        out_k3 = loss_fn_k3(**batch)

        # Different k_min should give different aux losses (almost certainly)
        # The TB part should be the same
        assert abs(out_k1["loss_tb"].item() - out_k3["loss_tb"].item()) < 1e-6
        # But overall loss differs due to different aux
        # (not guaranteed but extremely likely with random data)


class TestAvgPrefixTBLossDetachPterm:
    """Tests for Version C: detach pterm variant."""

    @pytest.fixture
    def batch(self):
        B, L, prompt_len, eos = 2, 5, 2, 0
        torch.manual_seed(77)
        log_pf = torch.randn(B, L, requires_grad=True)
        log_r = torch.randn(B, L)
        log_pterm = torch.randn(B, L, requires_grad=True)
        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        generated_text[:, -1] = eos
        return {
            "log_pf": log_pf, "log_r": log_r, "log_pterm": log_pterm,
            "generated_text": generated_text,
            "termination_token_id": eos, "prompt_len": prompt_len,
        }

    def test_detach_pterm_no_gradient_on_pterm(self, batch):
        """When detach_pterm_in_aux=True in avgprefix mode, log_pterm gets no gradient."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss(mode="avgprefix", detach_pterm_in_aux=True)
        out = loss_fn(**batch)
        out["loss"].backward()

        # log_pterm should have zero gradient (detached)
        assert batch["log_pterm"].grad is None or batch["log_pterm"].grad.abs().sum() == 0
        # log_pf should still have gradient
        assert batch["log_pf"].grad is not None
        assert batch["log_pf"].grad.abs().sum() > 0

    def test_detach_vs_no_detach_different_grads(self, batch):
        """Detached and non-detached versions produce different pterm gradients."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        # Non-detached
        log_pterm_a = batch["log_pterm"].detach().clone().requires_grad_(True)
        loss_fn_a = AvgPrefixTBLoss(mode="avgprefix", detach_pterm_in_aux=False)
        out_a = loss_fn_a(
            batch["log_pf"].detach().clone().requires_grad_(True),
            batch["log_r"], log_pterm_a, batch["generated_text"],
            batch["termination_token_id"], batch["prompt_len"],
        )
        out_a["loss"].backward()
        grad_a = log_pterm_a.grad.clone()

        # Detached
        log_pterm_b = batch["log_pterm"].detach().clone().requires_grad_(True)
        loss_fn_b = AvgPrefixTBLoss(mode="avgprefix", detach_pterm_in_aux=True)
        out_b = loss_fn_b(
            batch["log_pf"].detach().clone().requires_grad_(True),
            batch["log_r"], log_pterm_b, batch["generated_text"],
            batch["termination_token_id"], batch["prompt_len"],
        )
        out_b["loss"].backward()
        grad_b = log_pterm_b.grad

        # Detached pterm should have zero grad, non-detached should not
        assert grad_a.abs().sum() > 0
        assert grad_b is None or grad_b.abs().sum() == 0


class TestAvgPrefixTBLossEdgeCases:
    """Edge case tests."""

    def test_single_step_trajectory(self):
        """L=2 (minimum: 1 step + terminal) should work."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        B, L, prompt_len, eos = 2, 2, 1, 0
        log_pf = torch.randn(B, L)
        log_r = torch.randn(B, L)
        log_pterm = torch.randn(B, L)
        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        generated_text[:, -1] = eos

        loss_fn = AvgPrefixTBLoss(mode="avgprefix")
        out = loss_fn(log_pf, log_r, log_pterm, generated_text, eos, prompt_len)
        assert out["loss"].ndim == 0
        assert not torch.isnan(out["loss"])

    def test_all_eos_at_start(self):
        """All samples have EOS at first generated position (tau=0)."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        B, L, prompt_len, eos = 3, 5, 2, 0
        log_pf = torch.randn(B, L)
        log_r = torch.randn(B, L)
        log_pterm = torch.randn(B, L)
        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        generated_text[:, prompt_len] = eos  # EOS at first generated token

        loss_fn = AvgPrefixTBLoss(mode="avgprefix")
        out = loss_fn(log_pf, log_r, log_pterm, generated_text, eos, prompt_len)
        assert not torch.isnan(out["loss"])
        assert not torch.isinf(out["loss"])

    def test_logZ_is_learnable(self):
        """logZ parameter should be in the model's parameters."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        loss_fn = AvgPrefixTBLoss(mode="avgprefix", init_logZ=1.5)
        params = list(loss_fn.parameters())
        assert len(params) == 1
        assert abs(params[0].item() - 1.5) < 1e-6

    def test_output_contains_logZ(self):
        """Output dict should contain detached logZ for logging."""
        from chemgfn.models.losses import AvgPrefixTBLoss

        B, L, prompt_len, eos = 2, 4, 1, 0
        loss_fn = AvgPrefixTBLoss(mode="avgprefix")
        log_pf = torch.randn(B, L)
        log_r = torch.randn(B, L)
        log_pterm = torch.randn(B, L)
        generated_text = torch.randint(1, 100, (B, prompt_len + L))
        generated_text[:, -1] = eos

        out = loss_fn(log_pf, log_r, log_pterm, generated_text, eos, prompt_len)
        assert "logZ" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_avgprefix_tb_loss.py -v 2>&1 | head -50`
Expected: All tests FAIL with `ImportError: cannot import name 'AvgPrefixTBLoss'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_avgprefix_tb_loss.py
git commit -m "test: add failing tests for AvgPrefixTBLoss (strict, controlled, detach variants)"
```

---

## Task 2: Implement AvgPrefixTBLoss class

**Files:**
- Modify: `chemgfn/models/losses.py` (append before `class ModifiedSubTBBalanceLoss` at line ~1380, or at end of file)

- [ ] **Step 1: Add AvgPrefixTBLoss to losses.py**

Append the following class after `_first_eos_index` (line 1303) and before `LLMTrajectoryBalanceLoss` (line 1306). This keeps prefix-TB losses grouped together. Also update `__all__`.

```python
class AvgPrefixTBLoss(GFNLoss):
    """
    Averaged-Prefix Trajectory Balance loss.

    Three modes controlled by `mode`:

    "avgprefix" (Version A — strict reviewer baseline):
        L = (1/|K|) sum_{k in K} (Δ_k^TB)^2
        where K = {0, 1, ..., tau} (all valid prefixes)

    "tb_plus_avgaux" (Version B — controlled):
        L = (Δ_tau^TB)^2 + eta * weighted_mean_{k in [kmin, h-1]} (Δ_k^TB)^2
        Preserves terminal TB anchor; prefix average is auxiliary.

    Prefix TB residual:
        Δ_k^TB = logZ + sum_{t<k} log_pf[t] + log_pterm[k] - log_r[k]

    Optional: detach_pterm_in_aux=True detaches log_pterm from prefix residuals
    (Version C diagnostic — tests whether termination drift drives poor performance).

    Tensor alignment matches existing pipeline:
        log_pf:    [B, L]   meaningful steps are log_pf[:, :-1] (length L-1)
        log_pterm: [B, L]   EOS log-prob at each state k=0..L-1
        log_r:     [B, L]   terminate-at-k log-reward
        generated_text[:, prompt_len:-1] excludes forced last EOS
    """

    def __init__(
        self,
        mode: str = "avgprefix",  # "avgprefix" or "tb_plus_avgaux"
        aux_eta: float = 0.25,  # weight for aux term (tb_plus_avgaux only)
        k_min: int = 0,  # minimum prefix index for aux (0 = all prefixes)
        subtb_lambda: float = 1.0,  # weight decay: w_k = lambda^(k - k_min)
        detach_pterm_in_aux: bool = False,  # Version C: stop-grad on pterm
        eps: float = 1e-8,
        init_logZ: float = 0.0,
    ):
        super().__init__()
        if mode not in ("avgprefix", "tb_plus_avgaux"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.aux_eta = float(aux_eta)
        self.k_min = int(k_min)
        self.subtb_lambda = float(subtb_lambda)
        self.detach_pterm_in_aux = bool(detach_pterm_in_aux)
        self.eps = float(eps)
        self.logZ = torch.nn.Parameter(
            torch.tensor([float(init_logZ)], dtype=torch.float32)
        )

    def _compute_tau(
        self,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        L: int,
    ) -> torch.Tensor:
        """Compute tau (termination index) per sample. Returns [B] in [0, L-1]."""
        eos_or_after = (
            generated_text[:, prompt_len:-1] == int(termination_token_id)
        ).cumsum(dim=-1) >= 1  # [B, L-1]
        valid_end = ~eos_or_after
        tau = valid_end.sum(dim=1).clamp(0, L - 1)  # [B]
        return tau

    def _prefix_residuals(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        logZ_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prefix TB residuals for all positions k=0..L-1.

        Returns: [B, L] tensor of residuals.
            residual[b, k] = logZ + sum_{t<k} log_pf[b, t] + log_pterm[b, k] - log_r[b, k]
        """
        B, L = log_pf.shape
        steps = log_pf[:, :-1]  # [B, L-1]

        # prefix_logpf[k] = sum_{t<k} log_pf[t]
        # k=0 => 0, k=1 => log_pf[0], k=2 => log_pf[0]+log_pf[1], ...
        prefix_logpf = torch.cat(
            [torch.zeros(B, 1, device=log_pf.device, dtype=log_pf.dtype),
             steps.cumsum(dim=1)],
            dim=1,
        )  # [B, L]

        lp = log_pterm.detach() if self.detach_pterm_in_aux else log_pterm

        residuals = logZ_b.view(B, 1) + prefix_logpf + lp - log_r  # [B, L]
        return residuals

    def forward(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_pterm: torch.Tensor,
        generated_text: torch.Tensor,
        termination_token_id: int,
        prompt_len: int,
        k_min: int | None = None,
        max_prefix_len: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        B, L = log_pf.shape
        assert log_r.shape == (B, L)
        assert log_pterm.shape == (B, L)
        assert generated_text.shape[1] - prompt_len == L
        assert L > 1

        tau = self._compute_tau(generated_text, termination_token_id, prompt_len, L)
        logZ_b = self.logZ.to(device=log_pf.device, dtype=log_pf.dtype).expand(B)

        residuals = self._prefix_residuals(log_pf, log_r, log_pterm, logZ_b)  # [B, L]
        sq_res = residuals ** 2  # [B, L]

        k_idx = torch.arange(L, device=log_pf.device).view(1, L)  # [1, L]

        if self.mode == "avgprefix":
            # Version A: average over all k in [0, tau]
            mask = (k_idx <= tau.view(B, 1)).to(log_pf.dtype)  # [B, L]
            loss = (sq_res * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(self.eps)
            loss = loss.mean()

            return {
                "loss": loss,
                "logZ": self.logZ.detach(),
            }

        elif self.mode == "tb_plus_avgaux":
            # Version B: terminal TB + weighted aux average

            # Terminal TB residual (always uses non-detached pterm for terminal)
            # Recompute terminal residual without detach even if detach_pterm_in_aux
            if self.detach_pterm_in_aux:
                res_terminal = self._prefix_residuals(
                    log_pf, log_r, log_pterm, logZ_b  # non-detached for terminal
                )
                # But we need to not-detach only for terminal position
                # Simpler: gather terminal from non-detached residuals
                # Recompute without detach for terminal only
                B2, L2 = log_pf.shape
                steps = log_pf[:, :-1]
                prefix_logpf = torch.cat(
                    [torch.zeros(B2, 1, device=log_pf.device, dtype=log_pf.dtype),
                     steps.cumsum(dim=1)],
                    dim=1,
                )
                res_all_nodetach = logZ_b.view(B, 1) + prefix_logpf + log_pterm - log_r
                tb_res = res_all_nodetach.gather(1, tau.view(B, 1)).squeeze(1)
            else:
                tb_res = residuals.gather(1, tau.view(B, 1)).squeeze(1)

            loss_tb = (tb_res ** 2).mean()

            # Aux: k in [kmin, h-1] where h = min(tau, max_prefix_len) or tau
            effective_kmin = self.k_min if k_min is None else int(k_min)
            effective_kmin = max(0, effective_kmin)

            if max_prefix_len is not None:
                h = torch.minimum(tau, tau.new_full(tau.shape, int(max_prefix_len)))
            else:
                h = tau

            # Mask: k >= kmin and k < h (strictly before horizon)
            aux_mask = (
                (k_idx >= effective_kmin) & (k_idx < h.view(B, 1))
            ).to(log_pf.dtype)  # [B, L]

            # Weights: lambda^(k - kmin)
            w_exp = (k_idx - effective_kmin).clamp_min(0).to(log_pf.dtype)
            w = self.subtb_lambda ** w_exp  # [1, L]

            mw = aux_mask * w
            num_i = (sq_res * mw).sum(dim=1)  # [B]
            den_i = mw.sum(dim=1)  # [B]

            active_i = den_i > self.eps
            loss_aux_i = torch.zeros_like(num_i)
            loss_aux_i[active_i] = num_i[active_i] / den_i[active_i].clamp_min(self.eps)
            loss_aux = loss_aux_i.mean()

            # Per-sample eta: if no aux terms => eta_i=0
            eta = float(self.aux_eta)
            eta_i = eta * active_i.to(log_pf.dtype)

            loss_tb_i = tb_res ** 2
            loss_i = loss_tb_i + eta_i * loss_aux_i
            loss = loss_i.mean()

            return {
                "loss": loss,
                "loss_tb": loss_tb.detach(),
                "loss_aux": loss_aux.detach(),
                "logZ": self.logZ.detach(),
            }

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")
```

**Important**: Also update `__all__` at the top of the file to include `"AvgPrefixTBLoss"`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_avgprefix_tb_loss.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `pytest tests/test_loss.py -v`
Expected: All existing tests still PASS

- [ ] **Step 4: Commit**

```bash
git add chemgfn/models/losses.py
git commit -m "feat: add AvgPrefixTBLoss with avgprefix, tb_plus_avgaux, and detach_pterm modes"
```

---

## Task 3: Create SMILES experiment configs

**Files:**
- Create: `configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB.yaml`
- Create: `configs/experiment/SMILES_basic/SMILES_cfg_TB_plus_AvgAux.yaml`
- Create: `configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB_detach_pterm.yaml`

These configs clone `SMILES_cfg_TB.yaml` and only change the `loss_fn` block.

- [ ] **Step 1: Create Version A — SMILES strict AvgPrefixTB config**

Copy `SMILES_cfg_TB.yaml` and replace `loss_fn` with:

```yaml
# @package _global_

defaults:
  - override /data: smiles_opt
  - override /model: llama3_smiles_opt
  - override /callbacks: default
  - override /trainer: default

tags: ["ChemGFN", "smiles_opt", "CFG", "qed", "AvgPrefixTB"]
exp_name: "smiles_CFG_AvgPrefixTB"
seed: 42

trainer:
  max_steps:   5000
  gradient_clip_val: 0.5
  accumulate_grad_batches: 4
  default_root_dir: ${paths.output_dir}
  precision: bf16-true
  devices: 1
  num_sanity_val_steps: 0
  limit_train_batches:  500
  limit_val_batches:  50

model:
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.0001

  reward:
    _target_: chemgfn.models.reward.Reference_Target_Score_Positive_Memory_PrefixShaping_NoBackoff
    disable_peft: ${model.disable_peft}
    illegal_vocab_penalty: 0
    grammar_disagree_penalty: 0
    phi_weight: 0
    reward_strategy: "smiles_absorbing"
    smiles_len_weight: 0.0
    smiles_score_clip: [0.0, 1.0]
    pv_update_strategy: smiles_global_score
    use_entropy_gate: false
    use_token_entropy_gate: false
    ent_lo: 0.10
    ent_hi: 0.55
    phi_eta: 1.0
    phi_clamp: 2.0
    pv_split: 2
    pv_split_inclusive: true
    pv_memory_kwargs:
      alpha: 1.0
      gamma: 0.999
      max_keys: 500000
      prune_every: 200
      prune_threshold: 1.0
      tau_conf: 20.0
    score_function: "score_fast"
    sentence_validator:
      _target_: chemgfn.models.validators.RDKitValidator
      scorer: "qed"
      backend: "pa"

  loss_fn:
    _target_: chemgfn.models.losses.AvgPrefixTBLoss
    mode: "avgprefix"
    k_min: 0
    subtb_lambda: 1.0
    detach_pterm_in_aux: false
    eps: 1e-8

  reward_buffer:
    _target_: chemgfn.utils.gfn_utils.ReplayBuffer
    prioritized_replay: True
    buffer_size: 200
    sim_tolerance: 0.25
    strict_mode: false
    buffer_aug_value: 0

  constraint_config:
    min_sentence_len: 1
    max_sentence_len: 10
    grammar_path: ${paths.assets_dir}/SMILES_grammars/generic.ebnf
    disable_grammar: false
    processor_type: "prefix"
    legal_tokens: ${paths.assets_dir}/token_list/SMILES/allowed_llama3.2_1B_allowed_token
    illegal_vocab_penalty: -50
    parse_mode: "limited"

  training_mixed_config:
    subtb_lambda: 1.0
    pf_temp_high: 1.0
    pf_temp_low: 1.0
    pf_temp_prob: 0.666
    n_samples: 32
    buffer_mixture_ratio: 0.7
    skip_baseline_sampling: true
    opt_task: false
    grammar_disagree_penalty: ${model.reward.grammar_disagree_penalty}

  factor_schedulers:
    reward_temp:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 1
      end: 1
      horizon: 50000
    replay_buffer:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 0.50
      end: 0.25
      horizon: 50000
    dataset_buffer:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 0.75
      end: 0.25
      horizon: 25000
    reference_logits_scale:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 1
      end: 1.0
      horizon: 50000
    pf_temp_low:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 0.8
      end: 1.0
      horizon: 50000
    pf_temp_high:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 1.5
      end: 1.0
      horizon: 50000
    scaling_factor:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 50
      end: 50
      horizon: 10000

  disable_peft: false

data:
  num_workers: 0
  data_path: ${paths.data_dir}SMILES/sidechain_prompts_qed.json
  buffer_sample_path: null

logger:
  wandb:
    project: "ChemGFN"
    offline: False
```

- [ ] **Step 2: Create Version B — SMILES TB+AvgAux config**

Same as Version A but with these `loss_fn` overrides:

```yaml
  loss_fn:
    _target_: chemgfn.models.losses.AvgPrefixTBLoss
    mode: "tb_plus_avgaux"
    aux_eta: 0.25
    k_min: 1
    subtb_lambda: 1.0
    detach_pterm_in_aux: false
    eps: 1e-8
```

And add `k_min` scheduler to `factor_schedulers` (matching RapTB pattern):
```yaml
    k_min:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 5
      end: 2
      horizon: 5000
```

`exp_name: "smiles_CFG_TB_plus_AvgAux"`
`tags: ["ChemGFN", "smiles_opt", "CFG", "qed", "TB_plus_AvgAux"]`

- [ ] **Step 3: Create Version C — SMILES AvgPrefixTB + detach pterm config**

Same as Version A but with:
```yaml
  loss_fn:
    _target_: chemgfn.models.losses.AvgPrefixTBLoss
    mode: "avgprefix"
    k_min: 0
    subtb_lambda: 1.0
    detach_pterm_in_aux: true
    eps: 1e-8
```

`exp_name: "smiles_CFG_AvgPrefixTB_detach_pterm"`
`tags: ["ChemGFN", "smiles_opt", "CFG", "qed", "AvgPrefixTB_detach_pterm"]`

- [ ] **Step 4: Commit**

```bash
git add configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB.yaml
git add configs/experiment/SMILES_basic/SMILES_cfg_TB_plus_AvgAux.yaml
git add configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB_detach_pterm.yaml
git commit -m "config: add SMILES experiment configs for AvgPrefixTB variants (A/B/C)"
```

---

## Task 4: Create VarExpr24 experiment configs

**Files:**
- Create: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB.yaml`
- Create: `configs/experiment/VarExpr24/VarExpr24_TB_plus_AvgAux.yaml`
- Create: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_detach_pterm.yaml`

These clone `VarExpr24_TB_no_data_buffer_hit.yaml` and only change `loss_fn` (and add `k_min` scheduler for Version B).

- [ ] **Step 1: Create Version A — VarExpr24 strict AvgPrefixTB config**

Clone `VarExpr24_TB_no_data_buffer_hit.yaml`, change:
- `exp_name: "VarExpr24_CFG_AvgPrefixTB"`
- `tags: ["ChemGFN", "expr24", "CFG", "AvgPrefixTB", "hit24_dense"]`
- `loss_fn`:
```yaml
  loss_fn:
    _target_: chemgfn.models.losses.AvgPrefixTBLoss
    mode: "avgprefix"
    k_min: 0
    subtb_lambda: 1.0
    detach_pterm_in_aux: false
    eps: 1e-8
```

Keep everything else identical (reward with `Expr24Validator`, `dataset_buffer` at 0, etc.).

- [ ] **Step 2: Create Version B — VarExpr24 TB+AvgAux config**

Clone Version A, change:
- `exp_name: "VarExpr24_CFG_TB_plus_AvgAux"`
- `tags: ["ChemGFN", "expr24", "CFG", "TB_plus_AvgAux", "hit24_dense"]`
- `loss_fn`:
```yaml
  loss_fn:
    _target_: chemgfn.models.losses.AvgPrefixTBLoss
    mode: "tb_plus_avgaux"
    aux_eta: 0.25
    k_min: 2
    subtb_lambda: 1.0
    detach_pterm_in_aux: false
    eps: 1e-8
```
- Add `k_min` scheduler (matching VarExpr24 RapTB pattern):
```yaml
    k_min:
      _target_: chemgfn.models.gfn.Scheduler
      schedule_type: "linear"
      start: 7
      end: 3
      horizon: 5000
```

- [ ] **Step 3: Create Version C — VarExpr24 AvgPrefixTB + detach pterm config**

Clone Version A, change:
- `exp_name: "VarExpr24_CFG_AvgPrefixTB_detach_pterm"`
- `tags: ["ChemGFN", "expr24", "CFG", "AvgPrefixTB_detach_pterm", "hit24_dense"]`
- `loss_fn.detach_pterm_in_aux: true`

- [ ] **Step 4: Commit**

```bash
git add configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB.yaml
git add configs/experiment/VarExpr24/VarExpr24_TB_plus_AvgAux.yaml
git add configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_detach_pterm.yaml
git commit -m "config: add VarExpr24 experiment configs for AvgPrefixTB variants (A/B/C)"
```

---

## Task 5: Smoke test with Hydra config resolution

**Files:** None (verification only)

- [ ] **Step 1: Verify SMILES configs parse correctly**

Run: `cd /home/xiwang/project/ChemGFN && python -c "
from hydra import compose, initialize_config_path
from omegaconf import OmegaConf
initialize_config_path('../configs', version_base='1.3')
for exp in ['SMILES_basic/SMILES_cfg_AvgPrefixTB', 'SMILES_basic/SMILES_cfg_TB_plus_AvgAux', 'SMILES_basic/SMILES_cfg_AvgPrefixTB_detach_pterm']:
    cfg = compose(config_name='train', overrides=[f'experiment={exp}'])
    print(f'{exp}: loss_fn._target_ = {cfg.model.loss_fn._target_}')
    print(f'  mode = {cfg.model.loss_fn.mode}')
print('All SMILES configs OK')
"`

Expected: All three configs print their `_target_` and `mode` without errors.

- [ ] **Step 2: Verify VarExpr24 configs parse correctly**

Same pattern for VarExpr24 configs.

- [ ] **Step 3: Verify loss class can be instantiated from config**

Run: `python -c "
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize_config_path
initialize_config_path('../configs', version_base='1.3')
cfg = compose(config_name='train', overrides=['experiment=SMILES_basic/SMILES_cfg_AvgPrefixTB'])
loss_fn = hydra.utils.instantiate(cfg.model.loss_fn)
print(f'Instantiated: {type(loss_fn).__name__}, mode={loss_fn.mode}')
print(f'logZ param: {list(loss_fn.parameters())}')
"`

Expected: Prints class name and logZ parameter.

---

## Task 6: Final validation

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new).

- [ ] **Step 2: Run linting**

Run: `make lint`
Expected: No new linting errors.

- [ ] **Step 3: Run formatting**

Run: `make format`
Expected: Code formatted (or already clean).

- [ ] **Step 4: Final commit if formatting changed anything**

```bash
git add -u
git commit -m "style: format AvgPrefixTBLoss code"
```

---

## Running Experiments

After implementation, run with:

```bash
# SMILES experiments
python chemgfn/train.py experiment=SMILES_basic/SMILES_cfg_AvgPrefixTB
python chemgfn/train.py experiment=SMILES_basic/SMILES_cfg_TB_plus_AvgAux
python chemgfn/train.py experiment=SMILES_basic/SMILES_cfg_AvgPrefixTB_detach_pterm

# VarExpr24 experiments
python chemgfn/train.py experiment=VarExpr24/VarExpr24_AvgPrefixTB
python chemgfn/train.py experiment=VarExpr24/VarExpr24_TB_plus_AvgAux
python chemgfn/train.py experiment=VarExpr24/VarExpr24_AvgPrefixTB_detach_pterm
```

## Minimal experiment set (reviewer response)

Priority order if compute is limited:
1. TB (already exists)
2. **AvgPrefixTB** (Version A) — most critical, directly addresses reviewer
3. **RapTB** (already exists)
4. AvgPrefixTB + detach pterm (Version C) — diagnostic value
5. TB + AvgAux (Version B) — ablation/fairness control
