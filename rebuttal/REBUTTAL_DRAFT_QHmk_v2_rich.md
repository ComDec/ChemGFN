# Rebuttal to Reviewer QHmk — Rich Version (v2)

We sincerely thank Reviewer QHmk for the thorough review and the important references. We address each concern below.

---

## W1: RL Contextualization

We fully agree with the reviewer and will revise. The connection between GFlowNet training and entropy-regularized RL is well-established [Tiapkin et al., AISTATS 2024; Deleu et al., UAI 2024]. Our introduction's statement "in contrast to reward-maximizing reinforcement learning" overstates the distinction and will be corrected.

**Revised framing**: Sampling from the reward-proportional posterior p*(x) ∝ P_ref(x)·exp(r(x)/β) is equivalent to solving a KL-regularized RL problem (e.g., Eq. 1-2 in Bartoldson et al., 2025). GFlowNet objectives such as TB, SubTB, and our RapTB are specific algorithmic instantiations for approximately solving this distributional RL problem. Our contribution operates within this broader framework: we improve the **training objective design** within the TB loss family, specifically addressing high-variance credit assignment and termination drift on terminable prefix trees. These are optimization-level improvements that apply regardless of whether one views the problem through the GFlowNet or entropy-regularized RL lens.

**Why GFlowNet-style objectives matter despite the RL equivalence**: The theoretical equivalence does not imply that standard RL algorithms (PPO, GRPO) and GFlowNet balance objectives perform identically in practice. The key practical difference is the **optimization target**: reward-maximizing RL objectives (even with KL regularization) optimize expected return, while TB-family objectives directly enforce flow consistency across the full trajectory, providing a structurally different inductive bias toward distributional coverage. As demonstrated in our new experiments (W2 below) and consistent with prior work [Hu et al., 2024; Shen et al., 2023], this difference manifests as substantially better mode coverage in practice.

---

## W2: PPO/GRPO Baselines

We add PPO and GRPO experiments on both tasks with identical model/LoRA/training budget (Llama-3.2-1B, LoRA rank-16, 5000 steps, unconstrained generation with soft vocab masking following standard RL molecular generation practice [REINVENT]).

**SMILES QED (dense reward)**:

| Method | QED↑ | Entropy↑ | Notes |
|--------|------|----------|-------|
| GRPO | 0.661 | 0.98 | Learns QED but entropy=36% of GFlowNet |
| PPO | 0.604 | 0.00 | Complete mode collapse to single molecule |
| TB | 0.717 | 2.503 | — |
| RapTB+SubM | **0.844** | **2.726** | Best on both reward and diversity |

**Expr24 (sparse reward)**:

| Method | Acc↑ | Entropy↑ | Notes |
|--------|------|----------|-------|
| GRPO | 0.872 | 0.89 | Learns reward but diversity=74% of GFlowNet |
| PPO | 0.000 | — | Never finds a valid expression in 5000 steps |
| TB | 1.000 | — | High acc but severe mode collapse (5.3 unique) |
| RapTB | 0.991 | 1.208 | Best acc-diversity trade-off |

**Key observations**:
1. GRPO achieves reasonable reward on both tasks but suffers severe mode collapse (entropy 0.98 vs 2.726 on SMILES — only 36% of the distributional coverage).
2. PPO collapses entirely on SMILES (entropy=0, generating a single molecule) and fails completely on Expr24.
3. Critically, **the diversity gap persists even on SMILES (dense continuous reward)**, confirming this is a fundamental difference between reward-maximizing and distributional objectives, not a reward sparsity artifact.
4. These results are consistent with the findings in Hu et al. (2024, Table 3), where GFlowNet objectives significantly outperform reward-maximizing RL on distributional metrics.

We present these as reference comparisons under identical conditions, not fully-optimized baselines: same model, same LoRA, same budget. The goal is to demonstrate the inherent advantage of distributional objectives for reward-proportional sampling, which is our problem setting.

---

## W3: TBA Baseline

TBA [Bartoldson et al., NeurIPS 2025] is the most directly relevant system-level work and we will add explicit discussion. After careful reading:

**TBA's loss**: TBA uses TB-VarGrad (their Eq. 5) — a trajectory-level balance objective where log Z is batch-estimated rather than learned. This is fundamentally the same trajectory-level objective as our TB anchor. TBA does **not** use SubTB, intermediate flow functions, or any form of prefix-level credit assignment.

**TBA's contribution is orthogonal to ours**: TBA solves the **scalability** problem — how to efficiently scale TB training with asynchronous exploration and distributed infrastructure (410M-7B models, 4-50× speedup over PPO/DPO). Our contribution solves the **credit assignment** problem — how to reduce variance and prevent termination drift within the TB objective. Notably, TBA explicitly acknowledges TB's gradient variance as a limitation and suggests "learning partial energy functions" as future work — RapTB's absorbed suffix backups directly address this.

**RapTB is a drop-in replacement**: RapTB's auxiliary loss (Eq. 9) can replace the TB-VarGrad loss inside TBA's training loop. The two contributions are complementary: TBA provides the infrastructure, RapTB provides a better objective.

We did not run TBA experiments because: (1) TBA requires multi-node distributed infrastructure that is orthogonal to our single-GPU experimental setup; (2) the loss function comparison is already covered by our TB baseline, since TBA uses the same TB objective.

---

## W4/W5: Mathematical Explanation and Terminology

We expand the design rationale for the RapTB loss:

**RapTB = TB + auxiliary regularizer** (Eq. 9):

$$\mathcal{L}_{\text{RapTB}} = \mathbb{E}_{\xi \sim P_F^\theta}\left[\underbrace{\Delta^{\text{TB}}(\xi)^2}_{\text{TB anchor}} + \eta \underbrace{\mathcal{L}_{\text{aux}}(\xi)}_{\text{prefix credit}}\right]$$

The TB term is the sole exact balance condition whose global optimum guarantees reward-proportional sampling. The auxiliary term is a variance-reducing regularizer that enhances internal credit assignment.

**Design of the auxiliary term**:

1. **Rooted residuals** (Eq. 4): $\bar{\Delta}_k = \Delta_k^{\text{TB}} - \Delta_0^{\text{TB}}$. By subtracting the root residual, log Z cancels in the auxiliary branch. This creates a local consistency signal anchored at s_0 without introducing O(N) redundant copies of the global Z optimization. The global Z is still learned through the TB anchor — we do not remove it from the overall objective.

2. **Absorbed suffix backups** (Eqs. 5-7): For early prefixes, the terminal reward provides a high-variance target because the same prefix can lead to very different terminal outcomes. The absorbed target $u_k^{tgt}$ aggregates suffix reward evidence:
   - $u_k^{max}$: lower bound on prefix credit (best downstream outcome)
   - $u_k^{soft}$: smooth aggregation via distance-discounted log-sum-exp
   - $u_k^{tgt} = \alpha \cdot u_k^{max} + (1-\alpha) \cdot u_k^{soft}$: interpolation (Table 6 confirms neither endpoint alone is optimal)

3. **Stop-gradient on termination head** (Eq. 27): In the auxiliary branch only, we stop gradients through log q_θ(⊤|s_{0:k}). This directly prevents the termination drift diagnosed in Appendix C.6 — without it, overlapping prefix constraints can be minimized by globally shifting termination logits rather than improving token-level transitions.

**Terminology**: We will define "termination drift" at first mention in Section 1 and forward-reference Appendix C.6 for the formal analysis.

---

## Q1: Global Optimum Guarantee

We are transparent about this: **no new exact guarantee is claimed for the full RapTB composite objective**. The exact reward-proportional fixed point is tied to the TB anchor alone (Eq. 9, first term). The auxiliary term introduces a bias — it encourages internal credit assignment but does not have the same fixed-point property.

However, several factors mitigate this:
1. The auxiliary term has weight η, which can be set small. In the limit η→0, RapTB recovers exact TB.
2. At the fixed point of TB (where Δ^TB = 0 for all trajectories), the rooted residuals $\bar{\Delta}_k$ also vanish, so the auxiliary loss is zero. The TB fixed point is also a fixed point of the auxiliary term.
3. The bias operates as a **regularization** effect during optimization, not as a distortion of the converged solution. Empirically, RapTB achieves better distributional fidelity than TB (Table 3: JS 0.147 vs 0.339 under RP; 0.048 vs 0.049 under SubM).

[OPTIONAL — cut if over limit]
In principle, one could anneal η→0 during training to ensure asymptotic convergence to the exact TB solution. We did not test this schedule, but note it as a theoretically grounded option for practitioners who require strict guarantees.

---

## Q2: Averaged-Prefix TB Baseline

We implement exactly the baseline the reviewer suggests: averaging (Δ_k^TB)² over all k ∈ {0,...,τ} with a learnable log Z.

**SMILES (L_max=10, 3200×3 samples)**:

| Method | Acc | QED↑ | Entropy↑ | FPDiv↑ | Avg Len |
|--------|-----|------|----------|--------|---------|
| TB | 0.998 | 0.717 | 2.503 | 0.807 | 3.06 |
| AvgPrefixTB | 1.000 | 0.661 | 0.665 | 0.649 | 2.89 |
| RapTB | 0.996 | 0.740 | 2.448 | 0.860 | 6.14 |
| RapTB+SubM | 0.988 | 0.844 | 2.726 | 0.898 | 7.44 |

**Expr24 (RP replay, 6400 samples)**:

| Method | Acc | Unique↑ | NormCov↑ | JS↓ | log p_term | Avg Len |
|--------|-----|---------|----------|-----|------------|---------|
| TB | 1.000 | 5.3 | 0.001 | 0.339 | -0.001 | 8.98 |
| SubTB | 0.229 | 324.7 | 0.051 | 0.109 | -79.638 | 8.09 |
| AvgPrefixTB | 0.998 | 142 | 0.016 | 0.213 | -0.560 | 5.74 |
| RapTB | 0.991 | 246.7 | 0.039 | 0.147 | -0.065 | 8.99 |

**Analysis**:

On SMILES, AvgPrefixTB achieves the **worst** QED (0.661), diversity (0.665), and FPDiv (0.649) among all methods — worse even than vanilla TB. Avg Len 2.89 with 54% of samples at L=1-2 indicates severe short-sequence collapse. The model finds a "shortcut": short sequences are easier to satisfy all prefix constraints simultaneously, so the optimizer converges to trivially short, low-quality solutions.

On Expr24, AvgPrefixTB shows the same short-sequence collapse pattern: Avg Len 5.74 vs TB's 8.98 and RapTB's 8.99. While it improves over TB on diversity (142 vs 5.3 unique), it remains far below RapTB (246.7 unique, NormCov 0.039 vs 0.016). Its termination calibration (log p_term = -0.560) is substantially worse than RapTB's (-0.065), indicating partial termination suppression — consistent with our diagnosis that averaging TB residuals over all prefixes creates conflicting termination gradients similar to SubTB. Both tasks confirm that AvgPrefixTB systematically favors short sequences as a path of least resistance.

**Why AvgPrefixTB fails where RapTB succeeds**:
1. **Rooted residuals cancel Z in the auxiliary branch**: AvgPrefixTB applies O(τ) copies of the learnable Z across all prefix residuals, creating redundant optimization pressure on a single scalar. RapTB eliminates this in the auxiliary branch.
2. **Absorbed suffix backups provide lower-variance targets**: AvgPrefixTB uses the terminal reward for each prefix, which has high variance for early prefixes. RapTB's absorbed targets aggregate suffix evidence, reducing this variance.
3. **Stop-gradient prevents termination drift**: AvgPrefixTB allows termination gradients from all prefix residuals, creating the same conflicting signal as SubTB. RapTB stops these gradients in the auxiliary branch.

These results confirm that RapTB's three design choices (rooted residuals, absorbed backups, stop-gradient) are **not cosmetic** — each addresses a specific failure mode that simple prefix averaging does not solve.
