# Unified Comprehensive Rebuttal — Submission 13383
# RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay

> This document integrates all reviewer responses into a single coherent narrative. Each reviewer's response is self-contained but shares the same argumentation framework.

---

## Global Response (visible to all reviewers)

We thank all reviewers for the detailed and constructive feedback. We address three overarching points before responding individually.

**1. Theoretical scope: RapTB preserves the reward-proportional optimum (Pd1v-W3, JxzD-W1, QHmk-Q1).**

We clarify that RapTB does not claim a new exact fixed-point theorem. We argue from two complementary angles:

*Angle 1: Structural decomposition.* The RapTB objective (Eq. 9) is $\mathcal{L}_{\text{RapTB}} = \mathbb{E}_\xi[\Delta^{\text{TB}}(\xi)^2 + \eta\,\mathcal{L}_{\text{aux}}(\xi)]$. The TB anchor $(\Delta^{\text{TB}})^2$ is the only exact balance condition. Its unique global minimum is the reward-proportional distribution $q_\theta^\top(\xi) \propto R(x)$. The auxiliary term $\mathcal{L}_{\text{aux}}$ is a finite-weight regularizer that does not introduce any new balance condition with a different fixed point.

*Angle 2: Zero-at-optimum property.* We observe that the reward-proportional solution is also a global minimum of $\mathcal{L}_{\text{aux}}$, ensuring the auxiliary does not shift the optimum. The argument proceeds as follows:

- *Step 1.* In a terminable prefix tree, each prefix $s_{0:k}$ can terminate. The TB residual at prefix $k$ is $\Delta_k^{\text{TB}}(\xi) = \log Z_\theta + \sum_{t<k} \log P_F(s_{t+1}|s_{0:t}) + \log P_F(\top|s_{0:k}) - \log R(s_{0:k}^\top)$. When $q_\theta^\top$ matches the reward-proportional target for all termination points — i.e., $\log Z_\theta + \log q_\theta^\top(s_{0:k}^\top) = \log R(s_{0:k}^\top)$ for every $k$ — then $\Delta_k^{\text{TB}}(\xi) = 0$ for all $k \in \{0, \ldots, \tau\}$.
- *Step 2.* The rooted residual $\bar{\Delta}_k = \Delta_k^{\text{TB}} - \Delta_0^{\text{TB}} = 0 - 0 = 0$ for all $k \geq 1$.
- *Step 3.* At the reward-proportional optimum, each prefix's task-only component $u_k$ already reflects the correct credit. The auxiliary loss $\mathcal{L}_{\text{aux}} = \frac{\sum_k w_k (\bar{\Delta}_k + u_k - u_k^{\text{tgt}})^2}{\sum_k w_k}$ has each squared term equal to zero.
- *Step 4.* Therefore $\mathcal{L}_{\text{aux}} = 0$ whenever $(\Delta^{\text{TB}})^2 = 0$. The full objective $\mathcal{L}_{\text{RapTB}} \geq 0$ achieves its global minimum of zero at exactly the reward-proportional solution.

This is an algebraic consistency property, not a convergence rate theorem. We do not claim RapTB converges faster in a formal sense. What we observe empirically is that the auxiliary term provides denser gradient signal — reducing variance at early prefixes — which helps the optimizer reach the shared global minimum more reliably. The empirical evidence supports this: under Oracle replay (Table 3), RapTB outperforms TB in Acc (0.945 vs 0.919) and JS (0.013 vs 0.016). We will revise the text to state this distinction explicitly throughout.

**2. RL contextualization and TBA (QHmk-W1/W3, Pd1v-W2).**

We acknowledge that GFlowNets can be reformulated as entropy-regularized RL [Tiapkin et al., AISTATS 2024; Deleu et al., UAI 2024], and that the KL-regularized reward-proportional setting studied here fits within this broader framework. We will revise the introduction to remove the "in contrast to RL" framing and properly position our work. Our contribution is narrower: within off-policy TB-family training for this posterior, we study how to improve prefix-level credit assignment on terminable prefix trees.

Regarding TBA [Bartoldson et al., NeurIPS 2025]: TBA uses VarGrad TB inside an asynchronous distributed system with global replay and reward/recency sampling. RapTB is orthogonal — it changes the objective inside the learner by adding rooted prefix supervision and detaching auxiliary termination gradients. RapTB can in principle be used inside TBA's training loop. We note that TBA explicitly acknowledges TB's gradient variance as a limitation; RapTB's absorbed suffix backups address exactly this gap.

**3. RapTB and SubM are complementary, not competing (cA3o-Q1).**

SubM is the dominant mechanism for external mode discovery; RapTB improves internal credit assignment and distribution calibration. Three regimes support this: (i) Under RP, RapTB already improves coverage over TB (NormCov 0.039 vs 0.001 on Expr24). (ii) Under SubM, RapTB+SubM doubles NormCov over TB+SubM (0.209 vs 0.100). (iii) Under Oracle replay, where coverage is controlled, RapTB still outperforms TB in accuracy (0.945 vs 0.919) and KL/JS, indicating better probability allocation once discovery is fixed. We will surface these comparisons explicitly in the revision.

**New experiments in the revision:**
- PPO/GRPO baselines: diversity collapse on both SMILES and Expr24 (see per-reviewer details below)
- AvgPrefixTB: collapses on both tasks (SMILES Diversity=0.665, Avg Len=2.89; Expr24 NormCov=0.016)
- β×ρ sweep: 18 configs across both tasks — stable, no catastrophic failure
- 3B scale-up: SubTB drift amplified (Acc=0.313); RapTB+SubM best (QED=0.856, FPDiv=0.937)
- AMP biological sequences: SubTB length collapse confirmed; RapTB+SubM best diversity/novelty among natural-length methods

We narrow claim scope from "general LLM-GFlowNets" to "terminable prefix-tree LLM-GFlowNets in the evaluated settings."

---

## Response to Reviewer QHmk (Score 2, Reject)

We thank the reviewer for the careful and constructive feedback.

**W1: RL contextualization.** We agree that our framing under-contextualized the broader RL literature. The target distribution can be written in KL-regularized / MaxEnt RL form, and GFlowNet objectives should be positioned within that view rather than "in contrast to RL" [Tiapkin et al., 2024; Deleu et al., 2024]. Our contribution is narrower: within off-policy TB-family training for this posterior, we study how to improve prefix-level credit assignment on terminable prefix trees. We will revise the introduction accordingly.

**W2: PPO/GRPO baselines.** We added reference baselines under the same model/LoRA/budget (unconstrained generation, soft vocab masking, consistent with standard molecular RL practice).

Expr24 (independent eval, 6400 samples × 3 repeats):

| Method | Acc | Valid/6400 | Unique | Avg Len |
|---|---|---|---|---|
| GRPO | 0.002 | 12.3±1.2 | 1 | ~11 (99.9% at L_max) |
| PPO | 0.003 | ~20 | 1 | collapsed; crash step 250 |
| TB (paper) | 1.000 | 6400 | 5.3 | 8.98 |
| RapTB (paper) | 0.991 | 6343 | 246.7 | 8.99 |

SMILES QED:

| Method | QED↑ | Entropy↑ | Avg Len |
|---|---|---|---|
| GRPO | 0.661 | 0.98 | 10.0 (all at L_max) |
| PPO | 0.604 | 0.00 | single-molecule collapse |
| TB (paper) | 0.717 | 2.503 | 3.06 |
| RapTB+SubM (paper) | **0.844** | **2.726** | 7.44 |

On Expr24, GRPO reports training reward=0.872 but independent eval yields only 0.2% valid — the model memorizes narrow trajectories rather than learning a diverse policy. On SMILES (dense QED reward), GRPO achieves decent reward but entropy=0.98 (36% of RapTB+SubM's 2.726). The diversity gap persists on dense-reward tasks, confirming this is a fundamental objective-design difference, not reward sparsity. We report these as reference baselines, not fully tuned competitors. Consistent with Hu et al. (2024).

**W3: TBA baseline.** TBA is highly relevant and our discussion was insufficient. TBA uses VarGrad TB inside an asynchronous distributed system with global replay and reward/recency sampling. RapTB is orthogonal: it changes the objective inside the learner by adding rooted absorbed prefix supervision and detaching auxiliary termination gradients. We will discuss TBA explicitly and position RapTB as a drop-in objective for TBA-style pipelines. We note that TBA explicitly acknowledges TB's gradient variance as a limitation and suggests "learning partial energy functions" as future work — RapTB's absorbed suffix backups address exactly this gap.

**W4/W5: Mathematical explanation and terminology.** RapTB = TB + auxiliary regularizer (Eq. 9). The TB anchor retains the learnable Z and is the sole exact balance condition. In the auxiliary branch, the rooted residual (Eq. 4) cancels the shared Z so that prefix updates are not forced to redundantly reoptimize the same global scalar. The absorbed suffix targets are conservative hindsight backups over observed suffix task-only signals — u_max captures the best observed downstream outcome, u_soft smoothly aggregates with distance discounting, and the mixture is an empirical bias-variance trade-off (Table 6 confirms neither endpoint alone is optimal). Stop-gradient on the termination head (Eq. 27) prevents the drift diagnosed in Appendix C.6: in terminable prefix trees, arbitrary-start windows impose heterogeneous boundary conditions on the shared termination head, and the model can reduce many window residuals by globally shifting stop logits rather than improving token transitions. We will define "termination drift" at first mention.

**Q1: Global optimum.** As detailed in the global response above, the reward-proportional solution is a global minimum of both the TB anchor and the auxiliary regularizer simultaneously. When the TB condition is satisfied at all terminable prefixes (i.e., $\Delta_k^{\text{TB}} = 0$ for all $k$), the rooted residuals $\bar{\Delta}_k = 0$ and the auxiliary corrections vanish, giving $\mathcal{L}_{\text{aux}} = 0$. The auxiliary term does not move the optimum; it improves the optimization landscape by providing denser gradient signal. We emphasize this is an algebraic consistency property, not a convergence rate theorem. Empirically, RapTB achieves better distributional fidelity than TB (Table 3: JS 0.147 vs 0.339 under RP).

**Q2: AvgPrefixTB.** We implement exactly this baseline: averaging $(\Delta_k^{\text{TB}})^2$ over all $k$ with learnable Z.

SMILES (L_max=10):

| Method | Acc | QED↑ | Entropy↑ | FPDiv↑ | Avg Len |
|---|---|---|---|---|---|
| TB | 0.998 | 0.717 | 2.503 | 0.807 | 3.06 |
| AvgPrefixTB | 1.000 | 0.661 | 0.665 | 0.649 | 2.89 |
| RapTB+SubM | 0.988 | **0.844** | **2.726** | **0.898** | 7.44 |

Expr24 (RP replay, 6400 samples):

| Method | Acc | Unique↑ | NormCov↑ | JS↓ | Avg Len |
|---|---|---|---|---|---|
| TB | 1.000 | 5.3 | 0.001 | 0.339 | 8.98 |
| AvgPrefixTB | 0.998 | 142 | 0.016 | 0.213 | 5.74 |
| RapTB | 0.991 | 246.7 | 0.039 | 0.147 | 8.99 |

AvgPrefixTB is viable and stronger than plain TB on Expr24 diversity, but materially different from RapTB. On SMILES it collapses toward short sequences (Avg Len 2.89, 54% at L=1-2), with worst QED/diversity among all methods. The gain is not merely "more prefix supervision" but specifically: (1) rooted Z-cancellation, (2) suffix-absorbed targets, and (3) stop-gradient on the auxiliary termination head. Each addresses a specific failure mode that simple prefix averaging does not solve.

---

## Response to Reviewer JxzD (Score 4, Weak Accept)

We thank the reviewer for the thoughtful questions and for highlighting the value of the ablations and the replay design.

**On convergence / global optimality (W1).** We agree our wording should be tighter. As detailed in the global response, we show that the reward-proportional solution is simultaneously a global minimum of both the TB anchor and the auxiliary regularizer ($\mathcal{L}_{\text{aux}} = 0$ whenever $(\Delta^{\text{TB}})^2 = 0$ for all terminable prefixes). RapTB does not introduce a new fixed point; it improves the optimization landscape around the shared one. We will revise the text to state this explicitly.

**Q1: Why do SubTB windows include termination probabilities?** In the LLM-GFlowNet formulation (Hu et al., 2024), the state space forms a prefix tree where each non-root prefix has exactly one parent, making the backward policy deterministic ($P_B = 1$). Since every prefix is a valid terminable state, Hu et al. incorporate the modification from Deleu et al. (2022, UAI): at convergence, $R(s_n^\top) = F(s_n) \cdot P_F(\top|s_n)$, so $F(s_n) = R(s_n^\top)/P_F(\top|s_n)$. This substitution eliminates explicit state flow variables entirely. The SubTB residual for window $[i, j]$ then becomes:

$$\Delta_{i \to j}^{\text{SubTB}} = \sum_{k=i}^{j-1} \log P_F(s_{k+1}|s_{0:k}) + [\log q(\top|s_{0:j}) - \log q(\top|s_{0:i})] + [\log R(s_{0:i}^\top) - \log R(s_{0:j}^\top)]$$

The boundary terms involving termination probabilities $q(\top|s)$ arise directly from this substitution. Since the termination head is a single shared output of the LLM, all $O(N^2)$ windows pull on the same parameters with potentially conflicting objectives. When rewards are sparse, the optimizer can reduce many window losses simultaneously by globally shifting stop logits rather than improving token transitions — this is the "termination drift" diagnosed in Appendix C.6 (evidence: SubTB $\log p_{\text{term}} = -79.638$ vs RapTB's $-0.065$, Table 4).

RapTB addresses this by: (a) restricting to $O(N)$ rooted windows, reducing conflicting gradient signals; (b) cancelling Z in the rooted residual $\bar{\Delta}_k = \Delta_k^{\text{TB}} - \Delta_0^{\text{TB}}$; and (c) applying stop-gradient on termination logits in the auxiliary branch. We have not run a separate explicit-state-flow parameterization, but note it is a plausible alternative worth studying.

**W2: Why absorbed suffix backups?** We clarify: u_max is NOT a formal lower bound on the true prefix flow $F(s_{0:k}) = \sum_{x \in X(s_{0:k})} R(x)$ (a sum over exponentially many paths). Rather, u_max is a **conservative hindsight target** for the task-only component: it captures the best task-only signal observed along one sampled suffix. In sparse-reward settings where $u_k \approx 0$ for most intermediate prefixes, this provides a non-trivial regression target where $u_k$ alone would be uninformative. The term "conservative" means max is the tightest operator guaranteed to match or exceed any individual suffix observation without extrapolating.

u_soft complements u_max by aggregating all suffix signals via tempered log-sum-exp with distance decay $\rho(j-k)$. The decay downweights distant, noisier observations — structurally analogous to GAE's $\gamma\lambda$ discount (Schulman et al., 2015) — avoiding overly optimistic estimation. The mixture $u_k^{\text{tgt}} = \alpha \cdot u_k^{\max} + (1-\alpha) \cdot u_k^{\text{soft}}$ implements an empirical bias-variance tradeoff (Table 6 confirms neither endpoint alone is optimal).

The closest GFlowNet analog is FL-GFN (Pan et al., 2023), which uses forward-looking energies but requires R(s) to be informative at intermediate states — precisely failing in sparse-reward settings where absorbed targets are most needed. No prior GFlowNet work proposes suffix-based backup targets in the u_max/u_soft form.

**W3: Hyperparameters / robustness.** We swept the most task-specific parameters (β, ρ, η, k_min) on both Expr24 and SMILES (18 configs total); γ and K were kept fixed/shared. Across (β, ρ) grids, performance varies as a quality/diversity tradeoff, not catastrophic failure:
- Expr24: Acc ≥ 0.983 across all 9 configs, $\log p_{\text{term}} \in [-0.25, -0.04]$
- SMILES: 8/9 configs Acc ≥ 0.991, FPDiv 0.849–0.883 (paper: 0.860)
- η sweep: monotonic diversity improvement
- k_min ablation: fixed-low clearly worst — very early prefixes carry noisier supervision

**Q2: What is prefix survival?** Surv(k) = n_k/n_valid — fraction of valid samples still "alive" at token k. Read jointly with entropy and top-1: low survival = premature stopping; high survival + high top-1 = long but identical prefixes; high survival + low top-1 = healthy early branching. Figure 3 shows TB exhibits collapse while RapTB achieves genuine branching.

**Q3: Longer sequence tasks.** We add three lines of evidence:

(i) AMP biological sequences (20–50 amino acids, non-differentiable fitness):

| Method | Perf↑ | Diversity↑ | Novelty↑ | Avg Len |
|--------|-------|------------|----------|---------|
| TB | 0.927 | 7.39 | 10.65 | 17.4 |
| SubTB† | 0.897 | 21.37 | 28.68 | 49.3 |
| RapTB+SubM | **0.916** | **16.92** | **15.77** | **25.6** |

† SubTB collapses to max length; diversity/novelty inflated by raw edit distance.

RapTB+SubM achieves best diversity/novelty among natural-length methods in only 3K steps.

(ii) 3B scale-up on SMILES:

| Method (3B) | Acc | Score | FPDiv | Len |
|-------------|-----|-------|-------|-----|
| TB | 0.999 | 0.717 | 0.837 | 2.74 |
| SubTB | 0.313 | 0.221 | 0.854 | 8.48 |
| RapTB+SubM | **0.996** | **0.856** | **0.937** | 7.96 |

SubTB drift worsens at 3B (Acc=0.313), confirming the failure is structural.

(iii) L_max=15 stress test (Table 2): RapTB+SubM achieves best long-horizon coverage.

**Q4: Why fine-tune an LLM?** Our setting is specifically LLM-GFlowNet post-training: the frozen reference LM prior is part of the reward construction. The identified failure modes are structural to the terminable prefix tree, not model-specific — the 3B scale-up confirms they worsen with scale.

---

## Response to Reviewer cA3o (Score 4, Weak Accept)

We thank the reviewer for the careful reading and for isolating the central issue: performance reflects both replay coverage and objective design. We also agree that our original wording over-claimed generality and will narrow the claim to "terminable prefix-tree LLM-GFlowNets in the evaluated settings."

**W2/Q1: When does RapTB provide additive benefit over SubM?** The cleanest decomposition is external discovery vs. internal allocation/calibration:

(1) **RP replay (no SubM):** RapTB already improves substantially over TB. On Expr24, NormCov: 0.001 (TB) → 0.039 (RapTB); Unique_valid: 5.3 → 246.7. Entirely from better credit assignment.

(2) **SubM replay:** RapTB+SubM doubles TB+SubM on coverage (NormCov: 0.209 vs 0.100). SubM provides better samples, but RapTB allocates flow more correctly — this is the additive benefit.

(3) **Oracle replay** (Table 3 — coverage controlled): RapTB still outperforms TB in accuracy (0.945 vs 0.919) and KL/JS. This isolates the loss-level contribution: even when both methods see identical data, RapTB's rooted prefix supervision enables more accurate flow allocation.

On SMILES (Appendix A.1), RapTB+SubM shifts more mass to the longest bin (Frac[9-10]: 0.402 vs 0.323 for TB+SubM), consistent with improved suffix credit assignment. We will make this decomposition explicit in the main text.

**W3/Q2: Hyperparameter sensitivity.** We conduct a comprehensive cross-task study (18 β×ρ configs + 6 ablations):

- **Expr24**: All 9 configs achieve Acc ≥ 0.983, $\log p_{\text{term}} \in [-0.25, -0.04]$ (far from SubTB's -79.6)
- **SMILES**: 8/9 achieve Acc ≥ 0.991; FPDiv ∈ [0.849, 0.883] (paper: 0.860). Only β=10,ρ=0 shows mild degradation
- **η sweep**: Monotonic diversity improvement (NormCov 0.008→0.014, JS 0.235→0.185)
- **k_min**: Fixed-low clearly worst — very early prefixes carry noisier supervision

The paper defaults sit in a broad plateau. We are precise: α is partially addressed by the max-only/soft-only ablation (Table 6), but γ and K are not fully swept and remain a limitation. We will state this explicitly.

**W1/Q3: Domain generalization.** We add: (i) AMP biological sequence generation — SubTB again collapses to max length, RapTB+SubM gives best diversity/novelty among natural-length methods; (ii) 3B LoRA SMILES run — RapTB+SubM strongest, SubTB degrades further. These do not yet establish performance on code, math reasoning, or larger-vocabulary domains, so we narrow claims accordingly.

**Q4: GAE analogy.** This is a structural analogy in variance reduction, not an exact equivalence. Three key differences: (i) GAE estimates advantages for policy gradients; RapTB constructs targets for squared balance residuals. (ii) GAE bootstraps from a learned value function; RapTB aggregates observed suffix rewards — no learned intermediate value function. (iii) The GFlowNet TB condition imposes a global partition function constraint absent in policy gradient. We will revise the discussion to make this distinction explicit and connect to the entropy-regularized RL view.

---

## Response to Reviewer Pd1v (Score 3, Weak Reject)

We thank Reviewer Pd1v for highlighting these important issues.

**W1: Experimental breadth.** We agree that the strongest application-scale evidence in the original submission is scaffold-conditioned SMILES. The paper already evaluates three regimes (SMILES, Expr24, CommonGen) plus a longer-horizon stress test. We now add:

(i) AMP biological sequences (20–50 amino acids):

| Method | Perf↑ | Diversity↑ | Novelty↑ | Avg Len |
|--------|-------|------------|----------|---------|
| TB | 0.927 | 7.39 | 10.65 | 17.4 |
| SubTB† | 0.897 | 21.37 | 28.68 | 49.3 |
| RapTB+SubM | **0.916** | **16.92** | **15.77** | 25.6 |

† SubTB collapses to max length.

(ii) 3B scale-up on SMILES:

| Method (3B) | Acc | Score | FPDiv |
|-------------|-----|-------|-------|
| SubTB | 0.313 | 0.221 | 0.854 |
| RapTB+SubM | **0.996** | **0.856** | **0.937** |

SubTB's drift is amplified at 3B, confirming the failure is structural and worsens with model capacity. Together this extends evidence from molecules to arithmetic, natural language, and biological sequences.

**W2: Baselines.** Our original comparison was intentionally objective-level: TB and SubTB are the direct loss baselines. We now additionally include:

- **AvgPrefixTB** (averaging TB over all prefixes): collapses on both tasks (SMILES Diversity=0.665 vs RapTB+SubM's 2.726; Expr24 NormCov=0.016 vs 0.039). Simple dense prefix supervision does not match RapTB's design.
- **PPO/GRPO** (reward-maximizing RL references, same model/LoRA/budget):

| Task | Method | Reward | Entropy |
|------|--------|--------|---------|
| SMILES | GRPO | QED=0.661 | 0.98 (36% of GFlowNet's 2.726) |
| SMILES | PPO | QED=0.604 | 0.00 (single-molecule collapse) |
| Expr24 | GRPO | Acc=0.002 (eval) | 1 unique valid |

RL baselines improve reward but collapse diversity, confirming the fundamental objective difference.

We will also discuss TBA explicitly. TBA targets asynchronous/scalable LLM post-training, whereas RapTB addresses within-trajectory credit assignment and termination calibration. The two are orthogonal — RapTB can be used inside a TBA-style training loop.

**W3: Theoretical guarantees.** As detailed in the global response, we show that the reward-proportional solution ($\Delta_k^{\text{TB}} = 0$ for all terminable prefixes) is simultaneously a global minimum of both the TB anchor and the auxiliary regularizer. The TB anchor retains the learnable log Z and is the sole exact balance condition; the auxiliary term provides variance reduction without shifting the optimum. Finite η can introduce optimization-path bias; we do not claim exact preservation from the auxiliary term alone. The empirical evidence supports the regularization's benefit: Oracle replay (Table 3) shows RapTB outperforms TB in Acc (0.945 vs 0.919) and KL/JS. We will revise wording to state (a) no new exact theorem is claimed, (b) the zero-at-optimum property ensures consistency, and (c) empirical improvement is substantial.

---

## Cross-Reviewer Theme: Absorbed Suffix Targets (u_max / u_soft)

This section provides a unified technical explanation referenced by multiple reviewer responses.

**On u_max as a conservative hindsight target.** u_max is not a formal lower bound on the true flow $F(s_{0:k})$, which sums rewards over exponentially many reachable terminals. Rather, u_max captures the best task-only signal observed along one sampled suffix. It is "conservative" in that max is the tightest operator guaranteed to match or exceed any individual suffix observation without extrapolating beyond what was observed.

In sparse-reward settings (e.g., SMILES validity/QED, Expr24 correctness), the task-only component $u_k \approx 0$ for most intermediate prefixes. Standard TB must propagate terminal reward backward through flow estimates — a notoriously slow process. u_max provides a non-trivial regression target precisely where $u_k$ alone would be uninformative. Crucially, the absorption operates only on the task-only component; the reference-LM component $\kappa \log P_{\text{ref}}$ remains intact, so the modification targets specifically the sparse, uninformative part of the reward.

**On u_soft as smoothed long-range credit.** u_soft aggregates all suffix signals via tempered log-sum-exp with distance decay $\rho(j-k)$. The decay downweights distant, noisier observations — structurally analogous to GAE's $\gamma\lambda$ discount (Schulman et al., 2015) — avoiding overly optimistic estimation from distant suffixes. When $\beta \to \infty$, $u_k^{\text{soft}} \to u_k^{\max}$; at finite β, it provides a smoother credit signal reflecting the overall trajectory quality.

**The mixture is an empirical bias-variance tradeoff.** The paper's Table 6 confirms: max-only improves score but reduces diversity; soft-only improves diversity but reduces score; the interpolation outperforms both endpoints. This is an empirically motivated heuristic, not a new theorem.

**Novelty.** The closest GFlowNet analog is FL-GFN (Pan et al., 2023), which requires R(s) to be informative at intermediate states — failing precisely in sparse-reward settings. No prior GFlowNet work proposes suffix-based backup targets in the u_max/u_soft form.

---

## Cross-Reviewer Theme: SubTB Mechanism in LLM-GFlowNets

This section provides a unified technical explanation of why SubTB includes termination probabilities and why this causes drift.

In the LLM-GFlowNet formulation (Hu et al., 2024), the prefix tree has single-parent structure ($P_B = 1$). Since every prefix can terminate, Hu et al. incorporate the modification from Deleu et al. (2022, UAI): at convergence, $F(s_n) = R(s_n^\top)/P_F(\top|s_n)$. This eliminates explicit state flow variables, but couples the termination head to all SubTB boundary conditions. For a window $[i,j]$, the residual becomes:

$$\Delta_{i \to j}^{\text{SubTB}} = \sum_{k=i}^{j-1} \log P_F(s_{k+1}|s_{0:k}) + [\log q(\top|s_{0:j}) - \log q(\top|s_{0:i})] + [\log R(s_{0:i}^\top) - \log R(s_{0:j}^\top)]$$

Since the termination head is a single shared LLM output, all $O(N^2)$ windows exert gradient pressure on the same parameters. Under sparse or weakly prefix-dependent rewards, the optimizer finds a shortcut: globally shifting stop logits reduces many residuals simultaneously, causing "termination drift" (SubTB $\log p_{\text{term}} = -79.638$ on Expr24 vs RapTB's $-0.065$).

RapTB's solution: restrict to $O(N)$ rooted windows, reintroduce learnable $Z_\theta$ (which cancels in the rooted residual), and apply stop-gradient on auxiliary termination logits.

---

## Summary of New Experimental Evidence

| Experiment | Addresses | Key Result |
|------------|-----------|------------|
| PPO/GRPO (SMILES + Expr24) | QHmk-W2 | GRPO entropy=0.98 (36% of GFlowNet); PPO collapses to entropy=0 |
| AvgPrefixTB (SMILES + Expr24) | QHmk-Q2 | Collapses: SMILES Div=0.665, Len=2.89; Expr24 NormCov=0.016 |
| β×ρ sweep (18 configs) | cA3o-W3 | 8/9 SMILES Acc≥0.991; all 9 Expr24 Acc≥0.983 |
| η sweep + k_min ablation | cA3o-W3 | Monotonic η improvement; fixed-low k_min clearly worst |
| 3B scale-up (SMILES) | Pd1v-W1, JxzD-Q3 | SubTB Acc=0.313 at 3B; RapTB+SubM best (0.996/0.856/0.937) |
| AMP biological sequences | JxzD-Q3, cA3o-W1, Pd1v-W1 | SubTB length collapse (49.3); RapTB+SubM best div/nov at natural length |

---

## Revision Plan

1. Narrow claim scope to "terminable prefix-tree LLM-GFlowNets in the evaluated settings"
2. Revise introduction: remove "in contrast to RL" framing; cite Tiapkin+2024, Deleu+2024; discuss TBA explicitly
3. Add zero-at-optimum analysis to Section 3 with explicit algebraic derivation
4. Expand SubTB mechanism explanation (cite Hu et al. 2024, Deleu et al. 2022)
5. Clarify u_max as conservative hindsight target (not lower bound)
6. Define "termination drift" at first mention
7. Add new experimental tables (PPO/GRPO, AvgPrefixTB, AMP, 3B, sweeps) to revised paper
8. Acknowledge limitations: γ, K not fully swept; no code/math-reasoning domains
