# Rebuttal to Reviewer QHmk — Rich Version (v2.1)

We thank the reviewer for the careful and constructive feedback. We agree that our original framing under-contextualized the broader RL literature.

---

## W1: RL Contextualization

The target distribution in our setting can indeed be written in the KL-regularized / MaxEnt RL form, and LLM post-training with GFlowNet objectives should be positioned within that view rather than as "in contrast to RL" [Tiapkin et al., AISTATS 2024; Deleu et al., UAI 2024]. TBA further demonstrates this by starting from the KL-regularized RL objective and using TB to learn the corresponding posterior; Hu et al. frame LLM+GFlowNet as amortized posterior sampling.

Our contribution is therefore narrower: **within off-policy TB-family training for this posterior, we study how to improve prefix-level credit assignment on terminable prefix trees.** This is an objective-design contribution that applies regardless of whether one views the problem through the GFlowNet or entropy-regularized RL lens.

We will revise the introduction to remove the "in contrast to RL" framing and properly contextualize our work within the broader KL-regularized RL / MaxEnt framework.

---

## W2: PPO/GRPO Baselines

We added PPO and GRPO reference baselines on Expr24 under the same model/LoRA/training budget.

**Expr24 (6400 samples, verified results)**:

| Method | Acc | Valid/6400 | Unique | Length pattern |
|--------|-----|-----------|--------|---------------|
| GRPO | 0.002 | 12.3±1.2 | 1 | 99.9% at L_max |
| PPO | 0.003 | ~20 | 1 | collapsed; CUDA crash at step 250 |
| TB (paper) | 1.000 | 6400 | 5.3 | concentrated at L=9 |
| RapTB (paper) | 0.991 | 6343 | 246.7 | diverse |

GRPO yields only 12/6400 valid samples (Acc ≈ 0.002) with near-complete length collapse to L_max, producing a single unique valid expression. PPO collapses even faster — entropy reaches 0 within 50 steps, KL diverges to -87, and training crashes with CUDA NaN at step ~250.

We report these cautiously as **reference baselines** rather than fully tuned head-to-head competitors. The main apples-to-apples comparison in this paper remains within the TB family, because the goal is to isolate **objective-level mechanisms** rather than compare entire post-training stacks. These results are consistent with Hu et al. (2024), who also show that reward-maximizing RL produces valid-but-skewed or spurious behavior compared to GFlowNet objectives on distributional tasks.

[OPTIONAL — if SMILES PPO/GRPO logs are confirmed, add:]
On SMILES (dense QED reward), GRPO achieves QED=0.661 but entropy=0.98 (vs RapTB+SubM's 2.726 — only 36% diversity); PPO collapses to entropy=0.00. The diversity gap persists even on dense reward, confirming the issue is objective design, not reward sparsity.

---

## W3: TBA Baseline

We agree that TBA [Bartoldson et al., NeurIPS 2025] is important related work and that our discussion was insufficient.

TBA is a **highly relevant system-level neighbor** — it uses sequence-level VarGrad TB inside an asynchronous distributed system with global replay, reward/recency sampling, and optional IS / reference-reset variants. Its key contribution is scalability (410M-7B, 4-50× speedup over PPO/DPO).

RapTB is orthogonal: it changes the **objective inside the learner** by adding rooted absorbed prefix supervision and detaching the auxiliary termination gradients. It does not replace TBA's surrounding system. We will revise the paper to:
1. Discuss TBA explicitly in related work
2. Position RapTB as a drop-in objective for TBA-style pipelines
3. Note that TBA explicitly acknowledges TB's gradient variance as a limitation and suggests "learning partial energy functions" as future work — RapTB's absorbed suffix backups address exactly this gap

We do not want to claim an apples-to-apples TBA experiment without a clean reimplementation in our constrained prefix-tree setting. This is a limitation we acknowledge.

---

## W4/W5: Mathematical Explanation and Terminology

**RapTB = TB + auxiliary regularizer** (Eq. 9):

$$\mathcal{L}_{\text{RapTB}} = \mathbb{E}_{\xi \sim P_F^\theta}\left[\underbrace{\Delta^{\text{TB}}(\xi)^2}_{\text{TB anchor (retains } Z\text{)}} + \eta \underbrace{\mathcal{L}_{\text{aux}}(\xi)}_{\text{prefix credit}}\right]$$

**Design rationale**:

1. **The learnable Z is NOT removed.** It remains in the terminal TB anchor. The rooted residual (Eq. 4) cancels the shared Z **only in the auxiliary branch**, so that every prefix update is not forced to redundantly reoptimize the same global scalar.

2. **Absorbed suffix targets are conservative hindsight backups over observed suffix task-only signals** — not formal lower bounds on state flow. Specifically:
   - u_k^max: best observed downstream task-only outcome (conservative backup)
   - u_k^soft: smooth aggregation of multiple suffix signals with distance-discounted log-sum-exp
   - u_k^tgt = α·u_max + (1-α)·u_soft: empirical bias-variance trade-off (Table 6 confirms neither endpoint alone is optimal)

3. **Stop-gradient on the termination head** in the auxiliary branch (Eq. 27) directly prevents the termination drift diagnosed in Appendix C.6. The mechanism: in terminable prefix trees, arbitrary-start windows (as in SubTB) impose **heterogeneous boundary conditions** on the shared termination head. The model can reduce many window residuals simultaneously by **globally shifting stop logits** rather than improving token-level transitions. This is exactly the drift we observe in Table 4 (SubTB log p_term = -79.638) and Table 5 (SubTB Δ log p_term = -28.32).

**Terminology**: We will define "termination drift" at first mention in Section 1 and forward-reference Appendix C.6 for the formal analysis.

---

## Q1: Global Optimum Guarantee

**No**: minimizing the full composite RapTB objective with finite auxiliary weight η does not by itself guarantee exact sampling from the target distribution. The exact reward-proportional fixed point remains tied to the terminal TB term alone. The auxiliary term is a **variance-reducing regularizer** that can bias the optimum while improving optimization in practice.

We will revise the wording to remove any implication that RapTB itself introduces a new exact fixed-point theorem.

In principle, annealing η→0 would recover pure TB, but **we do not claim or verify this here**.

Empirically, the regularization is beneficial: RapTB achieves better distributional fidelity than TB on all metrics (Table 3: JS 0.147 vs 0.339 under RP; 0.048 vs 0.049 under SubM).

---

## Q2: Averaged-Prefix TB Baseline

We implement exactly the baseline the reviewer suggests: averaging (Δ_k^TB)² over all k ∈ {0,...,τ} with a learnable log Z.

**SMILES (L_max=10, 3200×3 samples)**:

| Method | Acc | QED↑ | Entropy↑ | FPDiv↑ | Avg Len |
|--------|-----|------|----------|--------|---------|
| TB | 0.998 | 0.717 | 2.503 | 0.807 | 3.06 |
| AvgPrefixTB | 1.000 | 0.661 | 0.665 | 0.649 | 2.89 |
| RapTB | 0.996 | 0.740 | 2.448 | 0.860 | 6.14 |
| RapTB+SubM | 0.988 | **0.844** | **2.726** | **0.898** | 7.44 |

**Expr24 (RP replay, 6400 samples)**:

| Method | Acc | Unique↑ | NormCov↑ | JS↓ | log p_term | Avg Len |
|--------|-----|---------|----------|-----|------------|---------|
| TB | 1.000 | 5.3 | 0.001 | 0.339 | -0.001 | 8.98 |
| SubTB | 0.229 | 324.7 | 0.051 | 0.109 | -79.638 | 8.09 |
| AvgPrefixTB | 0.998 | 142 | 0.016 | 0.213 | -0.560 | 5.74 |
| RapTB | 0.991 | 246.7 | 0.039 | 0.147 | -0.065 | 8.99 |

**Analysis**:

AvgPrefixTB is **viable and stronger than plain TB** on Expr24 diversity (142 vs 5.3 unique), but it remains **materially different from RapTB**.

On SMILES, AvgPrefixTB achieves the **worst** QED (0.661), diversity (0.665), and FPDiv (0.649) among all methods — worse even than vanilla TB. Avg Len 2.89 with 54% of samples at L=1-2: the model finds a shortcut where short sequences trivially satisfy all prefix constraints.

On Expr24, AvgPrefixTB shows the same short-sequence collapse: Avg Len 5.74 vs TB's 8.98 and RapTB's 8.99. Its termination calibration (log p_term = -0.560) is notably worse than RapTB's (-0.065), consistent with partial termination suppression from conflicting prefix constraints.

**The gain is not merely "more prefix supervision"** but specifically:
1. **Rooted Z-cancellation** in the auxiliary branch (removes O(τ) redundant Z optimization pressure)
2. **Suffix-absorbed targets** that reduce early-prefix conditional variance
3. **Stop-gradient on the auxiliary termination head** that prevents drift

These results support our claim that each of RapTB's design choices addresses a specific failure mode that simple prefix averaging does not solve.

---

## Presentation

We agree about terminology. We will:
- Define "termination drift" at first mention (Section 1)
- Move the RapTB derivation intuition earlier in the main text
- Expand the mathematical motivation in the body
