# Sweep Round 1 Results — β × ρ Grid

**Task**: Expr24 RP replay
**Config**: n_samples=64, grad_accum=1, max_steps=5000, seed=42
**Base**: VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune
**Source**: wandb project `ChemGFN-rebuttal`, group `rebuttal_sweep_beta_rho`
**Date**: 2026-03-25

## Status

- 8/9 runs completed (β=5, ρ=0.5 missing — was in batch 2, not launched)
- All reached step=5000, epoch=19
- Test-phase evaluation NOT completed (eval.py test dataloader was 50000 batches — config bug)
- Only **validation metrics** available, NOT Table 3 metrics (NormCov, KL, JS)

## Known Issues

1. **Test eval too slow**: `eval.py` inherits `total_size=100000` from training config → 50000 test batches. Need to override with smaller test size (~6400 samples = 100 batches at bsz=64).
2. **Missing run**: β=5, ρ=0.5 (9th run) not started.
3. **Val metrics ≠ paper metrics**: val/diversity is entropy-based, NOT NormCov (oracle coverage). Cannot directly compare with Table 3.

## Results: Validation Metrics

### Accuracy (val/acc) — All near-perfect

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| **ρ=0.0** | 0.999 | **1.000** | **1.000** |
| **ρ=0.1** | **1.000** | 0.998 | 0.996 |
| **ρ=0.5** | 0.994 | **1.000** | — |

### Diversity (val/diversity) — Token entropy

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| **ρ=0.0** | 0.782 | 0.820 | **0.882** |
| **ρ=0.1** | **0.896** | 0.749 | 0.815 |
| **ρ=0.5** | **0.900** | 0.777 | — |

### Termination Calibration (log_pterm - log_pterm_ref)

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| **ρ=0.0** | **-7.19** | -11.30 | -7.61 |
| **ρ=0.1** | -8.60 | -9.44 | -8.21 |
| **ρ=0.5** | -9.06 | -8.88 | — |

Closer to 0 = better calibration. β=3,ρ=0 has the worst termination drift (-11.3).

### Validation Loss

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| **ρ=0.0** | 32.3 | **126.8** | 30.4 |
| **ρ=0.1** | 31.2 | 48.0 | 31.5 |
| **ρ=0.5** | 39.0 | 39.2 | — |

β=3,ρ=0 has anomalously high loss (126.8) — correlates with worst termination drift.

### Mean Sequence Length (val/sentence_len)

All runs: 8.88–8.94 (max=9). No length collapse.

### Prefix Collapse (prefix_top1_auc)

All runs: 0.61–0.65. Relatively uniform — no severe prefix collapse in any config.

## Preliminary Interpretation (for rebuttal)

1. **No catastrophic failure**: All 8 (β,ρ) configs achieve val/acc ≥ 0.994. The method is NOT a fragile sharp optimum.
2. **Diversity varies but stays healthy**: Range 0.75–0.90. No config collapses to zero diversity.
3. **β=3,ρ=0 is the weakest**: High loss (126.8), worst termination drift (-11.3), moderate diversity (0.82). This makes sense — ρ=0 means no distance penalty on distant suffix evidence, causing instability in soft backup.
4. **Paper default (β=3,ρ=0.5)** performs well: acc=1.000, div=0.777, good termination calibration (-8.88).
5. **Best diversity**: β=1,ρ=0.5 (0.900) and β=1,ρ=0.1 (0.896) — lower temperature smooths more.

## What's Still Needed

- [ ] Complete test-phase evaluation with proper metrics (NormCov, KL, JS, Unique_valid, log_pterm(τ))
- [ ] Run β=5, ρ=0.5
- [ ] Run η sweep (3 runs) and k_min sweep (3 runs) — can use 1750 steps
- [ ] Fix eval config: override test dataloader to ~100 batches instead of 50000

## Config Fix for Next Round

```yaml
# In eval or +test mode, override:
trainer.limit_test_batches: 100  # 100 × 64 = 6400 samples
# Or equivalently in CLI:
+trainer.limit_test_batches=100
```
