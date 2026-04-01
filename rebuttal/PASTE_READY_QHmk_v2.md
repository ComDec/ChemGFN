We thank the reviewer for the thorough and constructive feedback. We address all concerns below, grouping weaknesses and questions thematically.

---

**W1: RL contextualization.** We agree that our framing insufficiently contextualized the broader RL literature. The GFlowNet target distribution admits a KL-regularized / MaxEnt RL formulation [1, 2]. SubTB is equivalent to Path Consistency Learning [4; 2, Prop. 3.2], and TBA departs from the KL-regularized RL objective [3]. Our contribution is narrower and compatible with either lens: within off-policy TB-family training, we improve prefix-level credit assignment on terminable prefix trees. This is an objective-design contribution — it applies whether one views the problem as GFlowNet training or entropy-regularized RL. We will revise the introduction to remove language that positions GFlowNets "in contrast to RL" and instead present them as a specialized off-policy entropy-regularized RL regime where the balance conditions provide structured loss functions.

---

**W2+W3: PPO/GRPO and TBA baselines.** We added PPO/GRPO under the same model, LoRA rank, and compute budget (unconstrained generation, soft vocabulary masking).

*SMILES QED (training-time metrics at convergence, step 5000):*

| Method | QED ↑ | Entropy ↑ | Avg Len | Note |
|---|---|---|---|---|
| GRPO | 0.661 | 0.98 | 10.0 | 36% of GFN entropy |
| PPO | 0.604 | 0.00 | — | Single-molecule collapse |
| TB (paper) | 0.717 | 2.503 | 3.06 | |
| RapTB (paper) | 0.740 | 2.448 | 6.14 | |
| RapTB+SubM (paper) | **0.844** | **2.726** | 7.44 | |

*Expr24 (independent eval, 6400 samples × 3 repeats):*

| Method | Acc | Valid / 6400 | Unique | Avg Len |
|---|---|---|---|---|
| GRPO | 0.002 | 12.3 ± 1.2 | 1 | ~11 (99.9% at L_max) |
| PPO | 0.003 | ~20 | 1 | collapsed |
| TB (paper) | 1.000 | 6400 | 5.3 | 8.98 |
| RapTB (paper) | 0.991 | 6343 | 246.7 | 8.99 |

On Expr24, GRPO reports training reward = 0.872 but independent evaluation yields only 0.2% valid — the model memorizes narrow trajectories rather than learning a distributional policy. On SMILES, where the reward landscape is dense, GRPO still achieves entropy = 0.98 (36% of GFlowNet's 2.726), confirming the diversity gap is not merely a sparse-reward artifact. We report these as reference baselines rather than fully tuned competitors, consistent with Hu et al. [4].

*TBA [3]:* TBA is a system-level pipeline (asynchronous generation, VarGrad TB objective, global replay) targeting scalable LLM post-training. RapTB is orthogonal: it changes the objective inside the learner. TBA explicitly acknowledges TB's gradient-variance limitation — "The trajectory balance objective can suffer from high gradient variance as it operates on the trajectory level" — and suggests "learning partial energy functions to balance bias and variance" as future work. RapTB's absorbed suffix backups are a non-parametric, computationally cheap realization of this idea, without requiring auxiliary networks (cf. FL-GFN [7] which needs intermediate energy evaluation, and LED-GFN [8] which learns an additive potential decomposition). RapTB can serve as a drop-in objective replacement inside TBA-style pipelines. We do not claim an apples-to-apples comparison without a clean reimplementation of TBA's full system.

---

**W4+W5+Q1: Mathematical explanation of RapTB and global optimum.**

We appreciate the request for a deeper walkthrough. We explain each design choice below, then address the global optimum question.

*Overall structure.* RapTB decomposes as $\mathcal{L}_{\text{RapTB}} = \mathcal{L}_{\text{TB}} + \eta \cdot \mathcal{L}_{\text{aux}}$ (Eq. 9). The TB term is the standard trajectory balance loss with a learnable $\log Z$; the auxiliary term provides per-prefix credit assignment. We explain the three key design choices in order.

**(1) Rooted residuals and the role of $Z$.**
The credit bounds (Eqs. 6–8) are **not** introduced to remove $Z$. The learnable $\log Z$ remains in the TB anchor $\mathcal{L}_{\text{TB}}$, which is the sole exact balance condition. In the auxiliary branch, the rooted residual $\bar{\Delta}_k = \Delta_k^{\text{TB}} - \Delta_0^{\text{TB}}$ cancels the shared $Z$ so that per-prefix gradient updates do not redundantly reoptimize a global scalar. Without this cancellation, every prefix position would push $Z$ in potentially conflicting directions, adding noise. This is why Eqs. 6–8 take the rooted form they do — it is a variance-reduction design, not a $Z$-removal.

**(2) Absorbed suffix targets.**
In a terminable prefix tree, the task-only reward $u_k = \log R(s_{0:k}^\top)$ at most prefixes is near zero under sparse rewards (e.g., only complete valid expressions score nonzero in Expr24). This makes $u_k$ alone uninformative as a regression target for prefix $k$. We introduce absorbed suffix targets that look ahead along the sampled trajectory to provide a non-trivial signal:

| Target | Definition | Role |
|---|---|---|
| $u_k^{\max}$ | $\max_{j \in [k,h]} u_j$ | Conservative: best observed downstream signal |
| $u_k^{\text{soft}}$ | $\frac{1}{\beta}\log \sum_{j=k}^{h} \exp(\beta \cdot u_j - \beta \cdot \rho \cdot (j-k))$ | Smoothed: distance-discounted log-sum-exp |
| $u_k^{\text{tgt}}$ | $\alpha \cdot u_k^{\max} + (1-\alpha) \cdot u_k^{\text{soft}}$ | Interpolated: Table 6 confirms mix outperforms either endpoint |

$u_k^{\max}$ is a conservative hindsight target — the best observed task-only signal along a sampled suffix. $u_k^{\text{soft}}$ is structurally analogous to GAE's $\gamma\lambda$ discount (Schulman et al. 2015, [9]) but over raw rewards, not TD errors with a learned $V(s)$. The closest GFlowNet analogues are FL-GFN (Pan et al. [7]), which requires intermediate energy $E(s)$ evaluable at every state (Assumption 4.1 — fails under sparse rewards), and LED-GFN (Jang et al. [8]), which requires an auxiliary network. TBA [3] explicitly calls for "learning partial energy functions" as future work; our absorbed targets are a non-parametric realization requiring no auxiliary model.

**(3) Stop-gradient on termination in auxiliary.**
Eq. 27 applies stop-gradient to the termination log-probability $\log P_F(\top | s_{0:k})$ inside $\mathcal{L}_{\text{aux}}$. Without it, the auxiliary branch pushes the termination head toward values that minimize auxiliary loss but violate the TB balance condition — we call this "termination drift" (formally: the phenomenon where the termination log-probability departs from calibrated values under auxiliary-driven gradients). Appendix C.6 diagnoses this failure mode: SubTB exhibits $\log p_{\text{term}}(\tau) \approx -79.6$ (Table 4), whereas RapTB with stop-gradient maintains $\log p_{\text{term}} \in [-0.25, -0.04]$. We will define this term at first mention in the revision.

*Global optimum — does $\mathcal{L}_{\text{RapTB}} = 0$ imply sampling from the target distribution?*

We address this from two angles. We do not claim a new fixed-point theorem; we show the auxiliary does not shift the known TB fixed point.

*Angle 1 (regularizer structure):* The TB fixed point — established by Malkin et al. [5, Theorem 1] and equivalent to the MaxEnt RL optimal policy (Tiapkin et al. [1, Theorem 1]) — is the sole exact balance condition. $\mathcal{L}_{\text{aux}}$ is a regularizer controlled by $\eta$, structurally analogous to how LED-GFN [8] adds a smoothness regularizer to a flow reparameterization while preserving the same fixed point.

*Angle 2 (zero-at-optimum):* The TB residual at prefix $k$ is:

$$\Delta_k^{\text{TB}}(\xi) = \log Z_\theta + \sum_{t<k} \log P_F(s_{t+1}|s_{0:t}) + \log P_F(\top|s_{0:k}) - \log R(s_{0:k}^\top)$$

When $\Delta_k^{\text{TB}} = 0$ for all $k$ in the terminable prefix tree:
- Rooted residuals: $\bar{\Delta}_k = \Delta_k^{\text{TB}} - \Delta_0^{\text{TB}} = 0$
- All correction terms vanish: $u_k = u_k^{\text{tgt}}$ (the policy already assigns correct probability at every prefix)
- Therefore: $\mathcal{L}_{\text{aux}} = \sum_k w_k \cdot (\bar{\Delta}_k + u_k - u_k^{\text{tgt}})^2 / \sum_k w_k = 0$
- And: $\mathcal{L}_{\text{RapTB}} = (\Delta^{\text{TB}})^2 + \eta \cdot 0 = 0$

The reward-proportional distribution is a global minimum of both terms simultaneously. The auxiliary does not shift the optimum; it provides denser gradient signal toward it. This parallels SubTB's same-fixed-point property via telescoping (Madan et al. [6]). We emphasize this is algebraic consistency — showing the auxiliary vanishes at the TB fixed point — not a convergence theorem. Empirically, the Oracle replay experiment (Table 3) supports this under controlled coverage: RapTB achieves Acc 0.945 vs. TB 0.919, with better distributional fidelity (JS 0.013 vs. 0.016). Under RP replay, the gap is larger: JS 0.147 vs. 0.339. We will surface the zero-at-optimum property explicitly in the revised paper alongside narrowed claim scope.

---

**Q2: AvgPrefixTB baseline.** We implement exactly the reviewer's suggestion: average (Δ_k^TB)² over all k ∈ {0, …, h} with the original learnable Z retained.

*SMILES (L_max = 10):*

| Method | Acc | QED ↑ | Entropy ↑ | FPDiv ↑ | Avg Len |
|---|---|---|---|---|---|
| TB | 0.998 | 0.717 | 2.503 | 0.807 | 3.06 |
| SubTB | 0.328 | 0.755 | 2.127 | 0.836 | 8.35 |
| AvgPrefixTB | 1.000 | 0.661 | 0.665 | 0.649 | 2.89 |
| RapTB | 0.996 | 0.740 | 2.448 | 0.860 | 6.14 |
| RapTB+SubM | 0.988 | **0.844** | **2.726** | **0.898** | 7.44 |

*Expr24 (RP replay, 6400 samples):*

| Method | Acc | Unique ↑ | NormCov ↑ | JS ↓ | Avg Len |
|---|---|---|---|---|---|
| TB | 1.000 | 5.3 | 0.001 | 0.339 | 8.98 |
| SubTB | 0.229 | 324.7 | 0.051 | 0.109 | 8.09 |
| AvgPrefixTB | 0.998 | 142 | 0.016 | 0.213 | 5.74 |
| RapTB | 0.991 | 246.7 | 0.039 | 0.147 | 8.99 |

AvgPrefixTB collapses to short sequences (SMILES Avg Len = 2.89 with 54% of mass at L = 1–2; Expr24 Avg Len = 5.74 vs. ~9.0 for TB and RapTB). It improves over TB on Expr24 diversity (142 unique vs. 5.3), confirming that multi-prefix supervision has value — but it falls far short of RapTB (246.7 unique, NormCov 0.039 vs. 0.016). The gap traces to three specific designs absent in AvgPrefixTB: (1) rooted Z-cancellation so auxiliary gradients do not redundantly update the partition function, (2) suffix-absorbed targets providing non-parametric "partial energy" signals (cf. TBA's suggested future work [3]), and (3) stop-gradient on the auxiliary termination head preventing drift. AvgPrefixTB produces meaningfully different — and worse — results.

---

**References**

[1] Tiapkin et al. "Generative Flow Networks as Entropy-Regularized RL." AISTATS 2024.
[2] Deleu et al. "Discrete Probabilistic Inference as Control in Multi-path Environments." UAI 2024.
[3] Bartoldson et al. "Trajectory Balance with Asynchrony." NeurIPS 2025.
[4] Hu et al. "Amortizing Intractable Inference in Large Language Models." ICLR 2024.
[5] Malkin et al. "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS 2022.
[6] Madan et al. "Learning GFlowNets from Partial Episodes." ICML 2023.
[7] Pan et al. "Better Training of GFlowNets with Local Credit." ICML 2023.
[8] Jang et al. "Learning Energy Decompositions for Partial Inference in GFlowNets." ICLR 2024.
[9] Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." ICLR 2016.
