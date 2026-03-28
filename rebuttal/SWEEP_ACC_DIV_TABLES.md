# Sweep Acc / Diversity Tables (for cA3o-C2)

## Expr24 β × ρ Sweep (Acc / Diversity)

**Config**: seed=42, N=6400 eval samples
**Old sweep**: n_samples=64, accum=1 (effective bsz=64), 5000 steps
**Paper**: n_samples=32, accum=4 (effective bsz=128), 5000 steps, 3 seeds avg

### Old Sweep Results (bsz=64) — diversity ~30% lower than paper

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| **ρ=0.0** | 0.998 / 0.789 | 1.000 / 0.822 | 0.998 / 0.891 |
| **ρ=0.1** | 1.000 / 0.893 | 0.999 / 0.749 | 0.996 / 0.807 |
| **ρ=0.5** | 0.994 / 0.894 | 0.999 / 0.780 | 1.000 / 0.838 |

**Paper reference** (β=3, ρ=0.5): Acc=0.991, Diversity=1.208

### New Sweep Results (bsz=128, paper config) — TODO

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| **ρ=0.0** | — | — | — |
| **ρ=0.1** | — | — | — |
| **ρ=0.5** | — | — | **ref: 0.991 / 1.208** |

---

## Expr24 η Sweep (Acc / Diversity)

### Old (bsz=64, 1750 steps)

| η=0.1 | η=0.25 | η=0.5 |
|---|---|---|
| 1.000 / — | 1.000 / — | 0.998 / — |

(Diversity not available from old eval — will be in new runs)

### New (bsz=128, 5000 steps) — TODO

---

## Expr24 k_min Ablation (Acc / Diversity)

### Old (bsz=64, 1750 steps)

| Fixed k=3 | Schedule 7→3 | Fixed k=7 |
|---|---|---|
| 0.990 / — | 1.000 / — | 1.000 / — |

### New (bsz=128, 5000 steps) — TODO

---

## SMILES β × ρ Sweep (Acc / Diversity)

**Config**: seed=42
**Old sweep**: n_samples=32, accum=4 (effective bsz=128 — same as paper!), ~2500 steps (50%)
**Paper**: n_samples=32, accum=4 (effective bsz=128), 5000 steps, 3 seeds avg

### Old Sweep Results (2500 steps, 50% training)

| | β=1 | β=5 | β=10 |
|---|---|---|---|
| **ρ=0.0** | 0.996 / 2.30 | 0.992 / 2.06 | 0.993 / 2.06 |
| **ρ=0.1** | 0.998 / 2.18 | 0.987 / 2.31 | *running* |
| **ρ=0.5** | 0.989 / 2.15 | 0.985 / 2.03 | *running* |

**Paper reference** (β=5, ρ=0.1): Acc=0.996, Diversity=2.461

**Note**: SMILES sweep already uses paper bsz (n_samples=32, accum=4). Diversity gap is purely from 50% steps. Full 5000 steps should close the gap.

### New Sweep Results (5000 steps, full training) — TODO

| | β=1 | β=5 | β=10 |
|---|---|---|---|
| **ρ=0.0** | — | — | — |
| **ρ=0.1** | — | — | — |
| **ρ=0.5** | — | — | — |

**Paper reference** (β=5, ρ=0.1): Acc=0.996, Diversity=2.461
