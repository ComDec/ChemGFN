# Rebuttal to Reviewer JxzD — Rich Version (v3.1, literature-grounded)

> PASTE_READY_JxzD_v3.txt: within 5000 limit. This rich version contains extended analysis marked `[EXTENDED]`.

---

We thank the reviewer for the thoughtful questions and for highlighting the value of the ablations and the replay design.

## On convergence / global optimality (W1 + W5)

**Two-angle analysis:**

*Angle 1 (structure):* RapTB = TB + weighted auxiliary regularizer (Eq. 9). The TB fixed point (Malkin et al. 2022, Theorem 1) — equivalent to the MaxEnt RL optimal policy (Tiapkin et al. AISTATS 2024, Theorem 1) — is the sole exact balance condition. The auxiliary is controlled by η.

*Angle 2 (zero-at-optimum):* In a terminable prefix tree, when the TB condition holds for ALL termination points (Δ_k^TB = 0 for all k), then: (a) rooted residuals Δ̄_k = Δ_k^TB − Δ_0^TB = 0; (b) absorbed corrections vanish; (c) L_aux = 0. The reward-proportional distribution is simultaneously a global minimum of both terms. This parallels SubTB's same-fixed-point property (Madan et al. ICML 2023, via the telescoping argument).

**[EXTENDED — fixed-point literature context]**

| Method | Type | Same Fixed Point as TB? | Reference |
|--------|------|------------------------|-----------|
| SubTB | Different loss, same conditions | **Yes** (telescoping) | Madan et al. ICML 2023 |
| FL-GFN | Flow reparameterization | **Yes** (Prop 4.2) | Pan et al. ICML 2023 |
| LED-GFN | Reparameterization + regularizer | **No** (finite reg. weight) | Jang et al. ICLR 2024 |
| Loss shape change | Different regression loss | **Yes** (loss=0 same) | arXiv 2410.02596 |
| Control variates | Gradient estimation | **Yes** (unbiased) | da Silva et al. NeurIPS 2024 |
| TBA | Sampling + control variate | **Yes** | Bartoldson et al. NeurIPS 2025 |
| **RapTB** | TB + auxiliary loss | **Yes** (zero-at-optimum) | This paper |

RapTB is unique: unlike LED-GFN whose smoothness regularizer can have L_reg > 0 at the TB optimum, RapTB's auxiliary is algebraically guaranteed to be zero at the TB fixed point.

This is algebraic consistency, not a convergence theorem. We emphasize this distinction.

## Q1: Why do SubTB windows include termination probabilities?

**The chain of derivations:**

1. **Original SubTB** (Madan et al. ICML 2023): Uses **explicit learned state flow** F(s; θ) as a separate network head (Table 1 of Madan et al.). The subtrajectory balance condition (Eq. 8): F(s_i) · Π P_F / (F(s_j) · Π P_B).

2. **Tree structure → P_B = 1** (Malkin et al. 2022, Remark after Eq. 16): "In the case of auto-regressive generation, G is a directed tree... P_B is trivially P_B = 1."

3. **Terminable prefix tree → F(s) substitution** (Deleu et al. UAI 2022; Hu et al. ICLR 2024): Since every prefix can terminate, the flow identity F(s) · P_F(⊤|s) = R(s^⊤) gives F(s) = R(s^⊤)/P_F(⊤|s). Hu et al. state: "since each state is a valid terminable state, we can incorporate the modification to account for this from Deleu et al. (2022). Specifically, note that at convergence we have R(s_n^⊤) = F(s_n)P_F(⊤|s_n). Using this, we can simply substitute F(s_n) = R(s_n^⊤)/P_F(⊤|s_n)."

4. **Consequence for SubTB windows:** Substituting F(s) = R(s^⊤)/P_F(⊤|s) into the SubTB residual yields:

```
Δ_{i→j}^SubTB = Σ_{k=i}^{j-1} log P_F(s_{k+1}|s_{0:k})
               + [log q(⊤|s_{0:j}) − log q(⊤|s_{0:i})]    ← boundary terms
               + [log R(s_{0:i}^⊤) − log R(s_{0:j}^⊤)]
```

The key insight: in the original SubTB, the learned F(s; θ) network absorbed the "baseline" role. In the LLM formulation, this role is forced onto the termination probability — a single shared output of the LLM. All O(N²) windows exert gradient pressure on the same parameters.

5. **Termination drift mechanism:** Under sparse rewards, the optimizer finds a shortcut: globally shifting stop logits reduces many residuals simultaneously. Evidence: SubTB log p_term = −79.638 (Table 4), vs RapTB's −0.065.

**[EXTENDED — the SubTB-PCL equivalence]** Hu et al. (2024) and Deleu et al. (2024, Proposition 3.2) establish that SubTB is equivalent (up to scaling) to Path Consistency Learning in MaxEnt RL. This means the termination drift is not a GFlowNet-specific pathology but a structural issue with PCL-type objectives on terminable prefix trees with shared parameterization.

We have **not** run a separate explicit-state-flow parameterization; we note it is a plausible alternative.

## W2: Why absorbed suffix backups? — The u_max analysis

### What u_max is NOT

u_max is **not** a formal lower bound on the true prefix flow F(s_{0:k}) = Σ_{x ∈ X(s_{0:k})} R(x).

Three reasons:
1. **Single trajectory vs. all paths.** True flow sums over exponentially many terminals.
2. **Log-space max vs. log-sum-exp.** Even along one path, the true contribution is logsumexp of log rewards.
3. **Task-only component.** u_max captures only λS(·), not the full reward including κ log P_ref.

### What u_max actually is

A **conservative hindsight target** for the task-only stop-reward component: the best task-only signal observed along one sampled suffix. "Conservative" means: max is the tightest operator guaranteed to match or exceed any individual suffix observation without extrapolating.

### u_soft: smoothed long-range credit

u_soft aggregates all suffix signals via tempered log-sum-exp with distance decay ρ(j−k). Structurally analogous to GAE's γλ discount (Schulman et al. ICLR 2016), but over raw task rewards rather than TD errors with a learned V(s).

**[EXTENDED — literature comparison]**

| Method | Mechanism | Relation to RapTB | Key Difference |
|--------|-----------|-------------------|----------------|
| **GAE** (Schulman 2015) | Σ (γλ)^l δ_{t+l}^V | u_soft's ρ ↔ GAE's γλ | GAE uses learned V(s); RapTB uses observed rewards |
| **FL-GFN** (Pan ICML 2023) | log F(s) = −E(s) + log F̃(s) | Both provide intermediate credit | FL-GFN requires E(s) evaluable (Assumption 4.1); fails on sparse tasks |
| **LED-GFN** (Jang ICLR 2024) | Learned additive potential φ_θ | Both decompose terminal credit | LED-GFN needs auxiliary network + training |
| **RUDDER** (Arjona-Medina NeurIPS 2019) | Learned return predictor → contribution analysis | Both redistribute terminal reward | RUDDER needs auxiliary LSTM; guarantees conservation |
| **HCA** (Harutyunyan NeurIPS 2019) | P(a_t|s_t, G) via Bayes' rule | Both use hindsight | HCA modifies policy gradient; RapTB modifies regression target |
| **TBA future work** (Bartoldson NeurIPS 2025) | Calls for "partial energy functions" | RapTB = non-parametric realization | TBA does not implement this |

**No prior GFlowNet paper** proposes suffix-based backup targets in the u_max/u_soft form.

**In the entropy-regularized RL view** (Tiapkin et al. 2024): log F(s) = V*(s), the soft value function. FL-GFN reparameterizes this via known intermediate energy. LED-GFN learns it with auxiliary potentials. RapTB bypasses learning V*(s) entirely and uses non-parametric suffix aggregation as a cheap, biased-but-lower-variance substitute.

## W3: Hyperparameters / robustness

We swept (β, ρ, η, k_min) on both Expr24 and SMILES (18 configs total); γ and K fixed/shared.

- **Expr24**: Acc ≥ 0.983 across all 9 (β,ρ) configs. log_pterm ∈ [−0.25, −0.04]
- **SMILES**: 8/9 Acc ≥ 0.991, FPDiv 0.849–0.883 (paper: 0.860)
- **η sweep**: Monotonic diversity improvement
- **k_min**: Fixed-low clearly worst

Performance varies as quality/diversity tradeoff, not catastrophic failure.

## Q2: Prefix survival

Surv(k) = n_k/n_valid. Joint interpretation with entropy and top-1. Figure 3: TB exhibits collapse; RapTB achieves genuine branching.

## Q3: Longer sequence tasks

### AMP (20–50 amino acids) — NEW

| Method | Perf↑ | Div↑ | Nov↑ | Avg Len | Steps |
|--------|-------|------|------|---------|-------|
| RapTB+SubM | 0.916 | **16.92** | **15.77** | 25.6 | **3K** |
| RapTB | 0.919 | 8.83 | 14.44 | 22.4 | 5K |
| TB | 0.927 | 7.39 | 10.65 | 17.4 | 10K |
| SubTB† | 0.897 | 21.37 | 28.68 | 49.3 | 9K |

† SubTB: length collapse to max. RapTB+SubM surpasses Jain et al.'s non-AL GFlowNet on all metrics.

### 3B scale-up — NEW

SubTB Acc=0.313 at 3B; RapTB+SubM best (0.996/0.856/0.937). Failure is structural.

### L_max=15 stress test (existing)

Table 2: RapTB+SubM best long-horizon coverage (Frac(11+)=0.701, lowest Top1=0.071).

## Q4: Why fine-tune an LLM?

Our setting is specifically LLM-GFlowNet post-training (Hu et al. 2024). The failure modes are structural to the terminable prefix tree — the 3B scale-up confirms they worsen with scale.

---

## Provenance Tracking

| Claim | Source |
|-------|--------|
| TB fixed point (Theorem 1) | Malkin et al. (2022), NeurIPS 2022 |
| TB = MaxEnt RL optimal policy | Tiapkin et al. (2024), AISTATS 2024, Theorem 1 |
| P_B = 1 in trees | Malkin et al. (2022), Remark after Eq. 16 |
| F(s) = R(s^⊤)/P_F(⊤|s) | Deleu et al. (2022), UAI 2022; Bengio et al. (2023), JMLR |
| Hu et al. SubTB formulation + quote | Hu et al. (2024), ICLR 2024 |
| SubTB = PCL equivalence | Deleu et al. (2024), UAI 2024, Prop 3.2 |
| SubTB same fixed point (telescoping) | Madan et al. (2023), ICML 2023 |
| FL-GFN Assumption 4.1 | Pan et al. (2023), ICML 2023 |
| LED-GFN learned potential | Jang et al. (2024), ICLR 2024 |
| GAE γλ discount | Schulman et al. (2015), ICLR 2016 |
| RUDDER return decomposition | Arjona-Medina et al. (2019), NeurIPS 2019 |
| HCA hindsight credit | Harutyunyan et al. (2019), NeurIPS 2019 |
| TBA variance limitation + future work | Bartoldson et al. (2025), NeurIPS 2025 |
| Loss-divergence correspondence | arXiv 2410.02596 (2024) |
| Control variates for GFlowNets | da Silva et al. (2024), NeurIPS 2024 |
| AMP/3B/sweep results | rebuttal/evidence/ |
| Jain et al. non-AL baseline | Jain et al. (2022), ICML 2022 |

## Version History

- v1: Original draft (3529 chars)
- v2: Variance reduction framing, AMP context (4878 chars)
- v3: Feedback corrections, u_max analysis, AMP justification (4996 chars)
- v3.1: Full literature grounding — TB/SubTB/FL-GFN/LED-GFN/GAE/RUDDER/HCA/TBA citations, zero-at-optimum derivation, SubTB-PCL equivalence, fixed-point comparison table
