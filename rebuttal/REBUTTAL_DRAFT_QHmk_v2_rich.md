# Rebuttal to Reviewer QHmk — Rich Version (v2.2, literature-grounded)

We thank the reviewer for the careful and constructive feedback. We agree that our original framing under-contextualized the broader RL literature.

---

## W1: RL Contextualization

The target distribution in our setting can indeed be written in the KL-regularized / MaxEnt RL form, and LLM post-training with GFlowNet objectives should be positioned within that view rather than as "in contrast to RL":

- **Tiapkin et al. (AISTATS 2024, Oral)**: "Generative Flow Networks as Entropy-Regularized RL" — Theorem 1 establishes that the optimal MaxEnt RL policy with a corrected reward equals the GFlowNet forward policy. The state flow log F(s) = V*(s) (soft value function), and DB is equivalent to Dueling Soft DQN.
- **Deleu et al. (UAI 2024)**: "Discrete Probabilistic Inference as Control in Multi-path Environments" — Proposition 3.2 proves L_PCL = α² · L_SubTB, i.e., SubTB is equivalent (up to scaling) to Path Consistency Learning in MaxEnt RL.
- **Hu et al. (ICLR 2024)**: Explicitly notes "the SubTB objective is equivalent to the path consistency objective (Nachum et al., 2017) in max-entropy RL."
- **TBA (Bartoldson et al., NeurIPS 2025)**: Starts from the KL-regularized RL objective and uses TB to learn the corresponding posterior.

Our contribution is therefore narrower: **within off-policy TB-family training for this posterior, we study how to improve prefix-level credit assignment on terminable prefix trees.** This is an objective-design contribution that applies regardless of whether one views the problem through the GFlowNet or entropy-regularized RL lens.

We will revise the introduction to remove the "in contrast to RL" framing and properly contextualize our work.

---

## W2: PPO/GRPO Baselines

We added PPO and GRPO reference baselines under the same model/LoRA/training budget (unconstrained generation, soft vocab masking).

**Expr24 (independent eval, 6400 samples × 3 repeats)**:

| Method | Acc | Valid/6400 | Unique | Avg Len |
|--------|-----|-----------|--------|---------|
| GRPO | 0.002 | 12.3±1.2 | 1 | ~11 (99.9% at L_max) |
| PPO | 0.003 | ~20 | 1 | collapsed; CUDA crash at step 250 |
| TB (paper) | 1.000 | 6400 | 5.3 | 8.98 |
| RapTB (paper) | 0.991 | 6343 | 246.7 | 8.99 |

**SMILES QED (training-time metrics at convergence, step 5000)**:

| Method | QED↑ | Entropy↑ | Avg Len |
|--------|------|----------|---------|
| GRPO | 0.661 | 0.98 | 10.0 (all clipped) |
| PPO | 0.604 | 0.00 | single-molecule collapse |
| TB (paper) | 0.717 | 2.503 | 3.06 |
| RapTB+SubM (paper) | **0.844** | **2.726** | 7.44 |

On Expr24, GRPO reports training reward=0.872 but independent eval yields only Acc=0.002 — extreme mode collapse. On SMILES (dense QED reward), GRPO entropy=0.98 (36% of GFlowNet's 2.726). The diversity gap persists on dense reward, confirming a fundamental objective difference. We report cautiously as reference baselines. Consistent with Hu et al. (2024).

---

## W3: TBA Baseline

TBA is highly relevant. Key facts from the paper:

1. **What TBA does**: VarGrad TB inside an asynchronous distributed system with global replay, reward/recency sampling, and optional IS / reference-reset variants.
2. **What TBA says about TB variance**: "The trajectory balance objective can suffer from high gradient variance as it operates on the trajectory level."
3. **TBA's suggested future work**: "Future work can leverage learning partial energy functions to balance bias and variance during policy updates."

RapTB is orthogonal: it changes the **objective inside the learner** by adding rooted absorbed prefix supervision and detaching auxiliary termination gradients. It does not replace TBA's surrounding system. The connection is: TBA's call for "partial energy functions" is exactly what our absorbed suffix backups provide — a non-parametric, computationally cheap realization without auxiliary networks.

**Comparison to other credit assignment approaches:**
- **FL-GFN** (Pan et al., ICML 2023): Reparameterizes log F(s) = -E(s) + log F̃(s). Preserves the fixed point but requires E(s) evaluable at intermediate states (Assumption 4.1) — violated in sparse-reward LLM tasks where S(s_{0:k}) ≈ 0 for early stops.
- **LED-GFN** (Jang et al., ICLR 2024 Oral): Learns an additive potential decomposition with a separate network + smoothness regularizer. More powerful but requires auxiliary parameters and training.
- **RapTB**: Non-parametric suffix aggregation (max + log-sum-exp). No auxiliary network. Computationally free. Trades theoretical elegance for practical simplicity.

We will discuss TBA explicitly and position RapTB as a drop-in objective for TBA-style pipelines.

---

## W4/W5: Mathematical Explanation and Terminology

**RapTB = TB + auxiliary regularizer** (Eq. 9):

$$\mathcal{L}_{\text{RapTB}} = \mathbb{E}_{\xi}\left[\underbrace{\Delta^{\text{TB}}(\xi)^2}_{\text{TB anchor (retains } Z\text{)}} + \eta \underbrace{\mathcal{L}_{\text{aux}}(\xi)}_{\text{prefix credit}}\right]$$

**Design rationale with literature context:**

1. **Z is NOT removed.** It remains in the terminal TB anchor. The rooted residual (Eq. 4) cancels the shared Z **only in the auxiliary branch**. This is distinct from SubTB (Madan et al. 2023), which operates without Z entirely in the LLM formulation (Hu et al. 2024).

2. **Absorbed suffix targets** are conservative hindsight backups:
   - u_max: best observed downstream task-only outcome along one sampled suffix
   - u_soft: smooth aggregation with distance-discounted log-sum-exp — structurally analogous to GAE's γλ decay (Schulman et al. ICLR 2016), but over raw task rewards rather than TD errors with a learned V(s). In the entropy-regularized RL view (Tiapkin et al. 2024), log F(s) = V*(s); our approach bypasses learning this value function and uses non-parametric aggregation instead.
   - Table 6 confirms the mix outperforms either endpoint.

3. **Stop-gradient on termination head** (Eq. 27) prevents the drift diagnosed in Appendix C.6. The mechanism: in the LLM-GFlowNet formulation (Hu et al. 2024), state flows are eliminated via F(s) = R(s^⊤)/P_F(⊤|s) (from Deleu et al. UAI 2022). This forces the termination head to serve as both stopping decision and state-flow surrogate in all O(N²) SubTB windows. Under sparse rewards, global shifting of stop logits is an easier optimization target than improving token transitions.

**Terminology**: We will define "termination drift" at first mention.

---

## Q1: Global Optimum Guarantee

**Two-angle analysis:**

### Angle 1: RapTB = TB + regularization

The TB fixed point is exact: TB loss = 0 iff q_θ^⊤(x) ∝ R(x) (Malkin et al. 2022, Theorem 1). This is equivalent to the MaxEnt RL optimal policy (Tiapkin et al. 2024, Theorem 1). RapTB retains this as the anchor; the auxiliary is controlled by η.

This is structurally analogous to:
- **LED-GFN** (Jang et al., ICLR 2024): adds smoothness regularizer to a flow reparameterization. At finite regularizer weight, bias is introduced. At weight → 0, optimum is preserved.
- **Loss shape modifications** (arXiv 2410.02596): changing the regression loss shape (L2 → Huber → asymmetric) changes the divergence being minimized but preserves the same global minimum at loss = 0.

### Angle 2: Zero-at-optimum property

In a terminable prefix tree (Deleu et al. 2022; Hu et al. 2024), every prefix s_{0:k} can terminate. When the TB condition holds for ALL termination points:

**Step 1:** Δ_k^TB(ξ) = log Z + Σ_{t<k} log P_F(s_{t+1}|s_{0:t}) + log P_F(⊤|s_{0:k}) − log R(s_{0:k}^⊤) = 0 for all k.

**Step 2:** Rooted residual Δ̄_k = Δ_k^TB − Δ_0^TB = 0 − 0 = 0 for all k ≥ 1.

**Step 3:** The auxiliary loss L_aux = Σ_k w_k (Δ̄_k + u_k − u_k^tgt)² / Σ_k w_k. Since Δ̄_k = 0 and the TB condition ensures correct credit at every prefix, L_aux = 0.

**Step 4:** Therefore L_RapTB ≥ 0 achieves its global minimum of zero at exactly the reward-proportional solution.

This parallels SubTB's same-fixed-point property: SubTB loss = 0 for all windows iff TB = 0 (Madan et al. 2023, via the telescoping argument). The technical difference: SubTB shares the fixed point because subtrajectory residuals telescope to the full TB residual; RapTB shares it because the rooted residual and absorbed corrections independently vanish at the TB optimum.

**Honest framing:** This is algebraic consistency, not a convergence rate theorem. We do not claim faster convergence in a formal sense. Control variates (da Silva et al., NeurIPS 2024) and increased sampling (TBA) reduce variance without bias — the theoretically clean approach. RapTB trades potential bias for denser prefix supervision that empirically helps (Table 3: JS 0.147 vs 0.339 under RP).

---

## Q2: Averaged-Prefix TB Baseline

We implement exactly the reviewer's suggestion: averaging (Δ_k^TB)² over all k ∈ {0,...,τ} with learnable log Z.

**SMILES (L_max=10, 3200×3 samples)**:

| Method | Acc | QED↑ | Entropy↑ | FPDiv↑ | Avg Len |
|--------|-----|------|----------|--------|---------|
| TB | 0.998 | 0.717 | 2.503 | 0.807 | 3.06 |
| AvgPrefixTB | 1.000 | 0.661 | 0.665 | 0.649 | 2.89 |
| RapTB+SubM | 0.988 | **0.844** | **2.726** | **0.898** | 7.44 |

**Expr24 (RP replay, 6400 samples)**:

| Method | Acc | Unique↑ | NormCov↑ | JS↓ | Avg Len |
|--------|-----|---------|----------|-----|---------|
| TB | 1.000 | 5.3 | 0.001 | 0.339 | 8.98 |
| AvgPrefixTB | 0.998 | 142 | 0.016 | 0.213 | 5.74 |
| RapTB | 0.991 | 246.7 | 0.039 | 0.147 | 8.99 |

AvgPrefixTB collapses to short sequences (SMILES Len=2.89, 54% at L=1-2; Expr24 Len=5.74). The gain is not "more prefix supervision" but specifically:
1. **Rooted Z-cancellation** (removes O(τ) redundant Z optimization)
2. **Suffix-absorbed targets** (non-parametric "partial energy functions" — cf. TBA's future work call)
3. **Stop-gradient on auxiliary termination** (prevents the drift that AvgPrefixTB still suffers: log p_term = −0.560 vs RapTB's −0.065)

---

## Provenance Tracking

| Claim | Source |
|-------|--------|
| TB fixed point (Theorem 1) | Malkin et al. (2022), NeurIPS 2022 |
| TB = MaxEnt RL optimal policy | Tiapkin et al. (2024), AISTATS 2024, Theorem 1 |
| SubTB = PCL equivalence | Deleu et al. (2024), UAI 2024, Proposition 3.2; Hu et al. (2024) |
| F(s) = R(s^⊤)/P_F(⊤|s) substitution | Deleu et al. (2022), UAI 2022; Hu et al. (2024), ICLR 2024 |
| SubTB same fixed point (telescoping) | Madan et al. (2023), ICML 2023 |
| FL-GFN intermediate energy | Pan et al. (2023), ICML 2023 |
| LED-GFN learned potential | Jang et al. (2024), ICLR 2024 Oral |
| GAE γλ discount | Schulman et al. (2015), ICLR 2016 |
| TBA variance limitation + future work | Bartoldson et al. (2025), NeurIPS 2025 |
| Loss-divergence correspondence | arXiv 2410.02596 (2024) |
| Control variates for GFlowNets | da Silva et al. (2024), NeurIPS 2024 |
| PPO/GRPO/AvgPrefixTB results | rebuttal/evidence/ (new experiments) |
