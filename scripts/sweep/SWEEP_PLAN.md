# RapTB Hyperparameter Sweep Plan — Rebuttal Robustness Experiments

## Existing Experiment Inventory (from wandb comdec/ChemGFN)

### What we already have (paper defaults, full 5000 steps)

| Run Name | Task | β | ρ | η | k_min | val/acc | val/div |
|---|---|---|---|---|---|---|---|
| VarExpr24_CFG_RapTB_...tune_v0 | Expr24 RP | 3 | 0.5 | 0.25 | 7→3 | 0.990 | 1.200 |
| VarExpr24_CFG_RapTB_...tune_oracle | Expr24 Oracle | 3 | 0.5 | 0.25 | 7→3 | 0.926 | 1.637 |
| VarExpr24_CFG_RapTB_...tune_subM_div | Expr24 RP+SubM | 3 | 0.5 | 0.25 | 7→3 | 0.998 | 1.506 |
| smiles_RapTB_v2_kmin_5_to_2_mix_fix | SMILES | 5 | 0.1 | 0.25 | 5→2 | 0.999 | 1.841 |
| smiles_RapTB_v2_...max_only | SMILES | - | - | 0.25 | 5→2 | 0.998 | 1.707 |
| smiles_RapTB_v2_...soft_only | SMILES | - | - | 0.25 | 5→2 | 0.999 | 1.523 |

### What does NOT exist (sweep gap)

- NO runs with alternative (β, ρ) combinations on the final codebase
- NO systematic η sweep
- NO k_min ablation on the final paper code version (early kmin_0/2/3 runs used different reward/code)
- Configs were NOT logged to wandb (empty config dict), only summary metrics available

**Conclusion: All 15 screening + 18 full runs must be done fresh.**

---

## Hyperparameter → Config Key Mapping

| Paper Symbol | Config Key | Expr24 Default | SMILES Default |
|---|---|---|---|
| β (soft backup temperature) | `model.loss_fn.soft_beta` | 3.0 | 5.0 |
| ρ (step penalty discount) | `model.loss_fn.soft_rho` | 0.5 | 0.1 |
| η (aux loss weight) | `model.loss_fn.aux_weight` | 0.25 | 0.25 |
| k_min schedule | `model.factor_schedulers.k_min.{start,end,horizon}` | 7→3/5000 | 5→2/5000 |
| γ (absorption discount) | `model.loss_fn.gamma` | 0.99 | 0.99 |
| α (mix_weight) | `model.loss_fn.mix_weight` | 0.8 | 0.5 |

---

## Sweep Execution Plan

### Phase 1: Screening on Expr24 + RP Replay

All screening uses:
- **Base experiment**: `VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune`
- **Budget**: 1750 steps (~35% of 5000), 1 seed (42)
- **Why Expr24 RP**: fast, sparse reward, termination drift visible early, reviewer focus

#### Sweep 1: β × ρ Joint Grid (9 runs)
**Script**: `run_sweep1_beta_rho.sh`

| | ρ=0 | ρ=0.1 | ρ=0.5 |
|---|---|---|---|
| **β=1** | new | new | new |
| **β=3** | new | new | **paper default** |
| **β=5** | new | new | new |

- β=3, ρ=0.5 is the paper default — still re-run at 1750 steps for fair screening comparison
- ρ=0 tests "no suffix step penalty" (reviewer concern: is penalization necessary?)
- β=1 tests "flat softmax" (reviewer concern: is temperature shaping necessary?)

**GPU time**: 9 runs × ~25 min each ≈ 4h serial / <1h on 8 GPUs

#### Sweep 2: η (aux_weight) Sweep (3 runs)
**Script**: `run_sweep2_eta.sh`

After Sweep 1: plug in best (β*, ρ*), then sweep:
- η ∈ {0.1, 0.25, 0.5}
- η=0.25 is paper default; η=0.1 = weaker aux; η=0.5 = stronger aux

**GPU time**: 3 runs × ~25 min ≈ <15 min on 3 GPUs

#### Sweep 3: k_min Schedule Ablation (3 runs)
**Script**: `run_sweep3_kmin.sh`

After Sweep 1+2: plug in best (β*, ρ*, η*), then test:

| Variant | k_min start | k_min end | Schedule |
|---|---|---|---|
| Fixed low | 3 | 3 | none |
| Paper default | 7 | 3 | linear/5000 |
| Fixed high | 7 | 7 | none |

Answers: "Is schedule necessary? Are early short prefixes too noisy?"

**GPU time**: 3 runs × ~25 min ≈ <15 min on 3 GPUs

#### Phase 1 Total: 15 screening runs, <1.5h wall-clock on 8 GPUs

---

### Phase 2: Full Validation (top-2 configs × 3 seeds × 3 settings)
**Script**: `run_full_validation.sh`

After screening, take lexicographic top-2 configs and run:
1. **Expr24 + RP replay** — 2 configs × 3 seeds = 6 runs
2. **Expr24 + Oracle** — 2 configs × 3 seeds = 6 runs
3. **SMILES main** — 2 configs × 3 seeds = 6 runs

**GPU time**: 18 runs × ~70 min each ≈ 21h serial / ~3h on 8 GPUs

---

## Lexicographic Selection Criteria

### Expr24
1. **Filter**: Acc ≥ 0.99
2. **Rank**: NormCov descending (higher = better coverage)
3. **Tiebreak**: |log p_term(τ)| ascending (closer to 0 = better calibration)
4. **Final**: KL/JS ascending

### SMILES
1. **Filter**: validity ≥ 0.90, Acc not collapsed
2. **Rank**: FPDiv / MacroFP descending
3. **Tiebreak**: Score descending
4. **Filter**: exclude extreme Len (too short or too long)

---

## Rebuttal Figure Plan

### Figure 1 (highest priority): β × ρ Heatmap on Expr24
- 3×3 grid, dual panels: NormCov (color) + log p_term(τ) (annotation)
- Shows: absorbed soft backup is NOT a fragile sharp optimum
- Shows: termination calibration stable across the grid

### Figure 2 (if space): k_min Schedule Bar Chart on Expr24
- 3 bars: fixed-low / schedule / fixed-high
- Metrics: NormCov, Acc, log p_term(τ)
- Shows: schedule is beneficial but not critical

---

## Execution Checklist

```
[ ] Phase 1 — Screening
    [ ] Run sweep 1: bash scripts/sweep/run_sweep1_beta_rho.sh
    [ ] Analyze: python scripts/sweep/analyze_sweep.py --wandb-group sweep1_beta_rho
    [ ] Select best (β*, ρ*), update run_sweep2_eta.sh
    [ ] Run sweep 2: bash scripts/sweep/run_sweep2_eta.sh
    [ ] Select best η*, update run_sweep3_kmin.sh
    [ ] Run sweep 3: bash scripts/sweep/run_sweep3_kmin.sh
    [ ] Final selection: top-2 configs

[ ] Phase 2 — Full Validation
    [ ] Update run_full_validation.sh with top-2 config params
    [ ] Run full validation: bash scripts/sweep/run_full_validation.sh
    [ ] Collect eval metrics

[ ] Figures
    [ ] Generate heatmap: python scripts/sweep/analyze_sweep.py --sweep beta_rho
    [ ] Generate barplot: python scripts/sweep/analyze_sweep.py --sweep kmin
```

## Total Compute Budget

| Phase | Runs | Steps/run | GPU-hours (est., 1x RTX 8000) |
|---|---|---|---|
| Sweep 1 | 9 | 1750 | ~3.75h |
| Sweep 2 | 3 | 1750 | ~1.25h |
| Sweep 3 | 3 | 1750 | ~1.25h |
| Full validation | 18 | 5000 | ~21h |
| **Total** | **33** | — | **~27h serial / ~4h on 8 GPUs** |
