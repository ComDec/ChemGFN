# Sweep Results: AC-Perspective Analysis

## Verdict: PARTIALLY rebuttal-ready. Needs 2 targeted fixes before submission.

---

## What's STRONG (can use directly in rebuttal)

### 1. Accuracy robustness — EXCELLENT
All 9 β×ρ configs achieve Acc ≥ 0.994. This is the single strongest rebuttal point.

**Reviewer will read this as**: "The method doesn't break regardless of hyperparameter choice."

### 2. Termination calibration — EXCELLENT
log_pterm(τ) ranges from -0.04 to -0.25 across all configs. Compare to SubTB's -79.6 (Table 4).

**Reviewer will read this as**: "RapTB's termination fix is robust, not a fragile artifact of one hyperparameter setting."

### 3. η sweep direction — GOOD
η=0.1→0.25→0.5 shows monotonic improvement in NormCov (0.008→0.012→0.014) and KL/JS. This tells a clean mechanistic story: stronger auxiliary weight = more prefix supervision = better distributional alignment.

### 4. k_min ablation — GOOD
Fixed-low (k=3) is clearly worst: Acc drops to 0.990, NormCov drops to 0.006, log_pterm worsens to -0.38. This validates the design rationale that early short prefixes are noisy and should be excluded initially.

---

## What's WEAK (reviewer will attack)

### 1. NormCov is uniformly low — CRITICAL PROBLEM

| Setting | NormCov | CovCount/Oracle |
|---------|---------|-----------------|
| Paper RapTB (RP, 3 seeds, 5000 steps) | **0.039** | — |
| Paper TB (RP, 3 seeds, 5000 steps) | 0.001 | — |
| Best sweep config (β=5,ρ=0.5) | 0.014 | 35/2520 |
| Most sweep configs | 0.006 | 14/2520 |

The sweep NormCov is **3-6x lower** than the paper's own RapTB number. A reviewer will immediately ask: "Why is your new experiment worse than your reported result?"

**Root causes** (all legitimate but need to be explicitly stated):
- Single seed (42) vs paper's 3 seeds (42, 123, 2024) averaged
- Round 2 runs at 1750 steps (not 5000)
- n_samples=64 with grad_accum=1 (effective bsz=64) vs paper's 32×4=128
- Stochastic variation in a sparse-reward task with RP replay

**Risk**: If not explained, this undermines the entire sweep's credibility. Reviewer may think "the method got worse when you tried to reproduce it."

### 2. Mixed training budgets — METHODOLOGICAL FLAW

β×ρ round 1 runs: 5000 steps
β=5,ρ=0.5 + η sweep + k_min ablation: 1750 steps

You cannot put these in the same table without noting this. The β=5,ρ=0.5 cell is not directly comparable to the other 8 β×ρ cells.

### 3. Single seed — NO ERROR BARS

Reviewer cA3o explicitly has Confidence 3 and is statistically aware. A single-seed sweep with no CI will be questioned. The paper reports "mean ± 95% CI over 3 seeds."

### 4. β×ρ NormCov heatmap is nearly flat

| | β=1 | β=3 | β=5 |
|---|---|---|---|
| ρ=0.0 | 0.006 | 0.006 | 0.006 |
| ρ=0.1 | 0.010 | 0.007 | 0.006 |
| ρ=0.5 | 0.006 | 0.007 | 0.014 |

This is almost no variation. A skeptical reviewer reads this as "NormCov is insensitive to β,ρ because all configs are equally mediocre under single-seed RP." The Acc heatmap being nearly flat is GOOD (robustness), but NormCov being flat is AMBIGUOUS.

---

## Recommendations

### MUST DO (before submitting rebuttal)

**1. Re-run paper default (β=3,ρ=0.5) at 5000 steps with 3 seeds to get a fair anchor point**

This single experiment (~3h on 3 GPUs) gives you:
- A direct comparison between your sweep config and the paper's reported number
- Error bars for the anchor point
- If NormCov lands near 0.039 (paper value), it validates the sweep methodology
- If not, you know there's a systematic difference from the config change (bsz/grad_accum)

Without this anchor, the reviewer has no way to calibrate your sweep numbers against Table 3.

**2. In the rebuttal, present Acc and log_pterm as the PRIMARY metrics, not NormCov**

The argument is: "RapTB maintains near-perfect accuracy and stable termination calibration across all 9 (β,ρ) configs. Coverage (NormCov) varies modestly but stays above the TB baseline (0.001) in all cases."

Don't lead with NormCov — it's your weakest metric here.

### SHOULD DO (strengthens significantly)

**3. Run the β×ρ grid at uniform 5000 steps (or uniform 1750 steps)**

Currently 8 cells at 5000, 1 at 1750. Either:
- (Preferred) Re-run β=5,ρ=0.5 at 5000 steps to match round 1, OR
- Report only the 8 complete cells and note "β=5,ρ=0.5 omitted due to compute"

Mixing budgets in one heatmap is a methodological error that a sharp reviewer will catch.

### NICE TO HAVE (if time permits)

**4. One SMILES sweep point to show cross-task consistency**

Run paper default (β=5,ρ=0.1 for SMILES) at 1750 steps with the new config to confirm the Acc/diversity holds. This addresses the "is it only Expr24?" concern without full SMILES grid.

---

## How to Present in Rebuttal (draft language)

### For cA3o-C2 (hyperparameter sensitivity):

> We conduct a (β,ρ) sensitivity study on Expr24 under RP replay. Key findings:
>
> **Accuracy is uniformly robust**: All 9 (β,ρ) configs achieve Acc ≥ 0.994, with no catastrophic failure at any grid point. This confirms the method is not a fragile sharp optimum.
>
> **Termination calibration is stable**: log p_term(τ) ranges from -0.04 to -0.25 across all configs, compared to SubTB's -79.6 (Table 4). RapTB's termination fix is robust across the grid.
>
> **η (aux weight) shows clear monotonic benefit**: Increasing η from 0.1 to 0.5 consistently improves KL/JS and NormCov, confirming the auxiliary branch provides genuine optimization signal.
>
> **k_min schedule is beneficial**: Fixed k_min=3 (all prefixes from the start) hurts accuracy (0.990) and calibration (-0.38), while schedule (7→3) and fixed-high (k=7) perform similarly. This validates the design rationale that early short prefixes carry noise.
>
> [Refer to Figure X: β×ρ heatmap in revised PDF]

### DO NOT say:

- "NormCov is stable across the grid" (it's flat because it's low everywhere)
- "All configs match paper performance" (they don't — absolute NormCov is lower)
- "These results replicate Table 3" (they don't — different config, single seed)

### DO say:

- "Accuracy and termination calibration are robust"
- "Coverage varies modestly" (honest hedge)
- "The relative ordering is consistent with the paper's design rationale"

---

## Summary: What to run next

| Priority | Experiment | Purpose | Time |
|----------|-----------|---------|------|
| **P0** | Paper default (β=3,ρ=0.5) × 3 seeds × 5000 steps | Anchor point for sweep calibration | ~3h |
| **P0** | β=5,ρ=0.5 at 5000 steps (match round 1) | Complete the grid fairly | ~1h |
| P1 | AvgPrefixTB (in progress) | QHmk-C6 | already running |
| P2 | One SMILES point | Cross-task consistency | ~1h |
