We thank the reviewer for the careful and constructive feedback. We address each concern below.

**W1: RL contextualization.** We agree that our framing under-contextualized the broader RL literature. The target distribution can be written in KL-regularized / MaxEnt RL form, and GFlowNet objectives should be positioned within that view rather than "in contrast to RL" [Tiapkin et al., 2024; Deleu et al., 2024]. Our contribution is narrower: within off-policy TB-family training for this posterior, we study how to improve prefix-level credit assignment on terminable prefix trees. We will revise the introduction accordingly.

**W2: PPO/GRPO baselines.** We added reference baselines under the same model/LoRA/budget (unconstrained, soft vocab masking).

Expr24 (independent eval, 6400 samples × 3 repeats):

| Method | Acc | Valid/6400 | Unique | Avg Len |
|---|---|---|---|---|
| GRPO | 0.002 | 12.3±1.2 | 1 | ~11 (99.9% at L_max) |
| PPO | 0.003 | ~20 | 1 | collapsed; crash step 250 |
| TB (paper) | 1.000 | 6400 | 5.3 | 8.98 |
| RapTB (paper) | 0.991 | 6343 | 246.7 | 8.99 |

SMILES QED (training-time metrics at convergence, step 5000):

| Method | QED↑ | Entropy↑ | Avg Len |
|---|---|---|---|
| GRPO | 0.661 | 0.98 | 10.0 (all clipped) |
| TB (paper) | 0.717 | 2.503 | 3.06 |
| RapTB+SubM (paper) | **0.844** | **2.726** | 7.44 |

On Expr24, GRPO reports training reward=0.872 but independent eval yields only 0.2% valid — the model memorizes narrow trajectories rather than learning a diverse policy. On SMILES, GRPO achieves decent QED but entropy=0.98 (36% of GFlowNet's 2.726). We report these as reference baselines, not fully tuned competitors. Consistent with Hu et al. (2024).

**W3: TBA baseline.** TBA is highly relevant and our discussion was insufficient. TBA uses VarGrad TB inside an asynchronous distributed system with global replay and reward/recency sampling. RapTB is orthogonal: it changes the objective inside the learner by adding rooted prefix supervision and detaching auxiliary termination gradients. We will discuss TBA explicitly and position RapTB as a drop-in objective for TBA-style pipelines. We do not claim an apples-to-apples comparison without a clean reimplementation in our prefix-tree setting.

**W4/W5: Mathematical explanation.** RapTB = TB + auxiliary regularizer (Eq. 9). The TB anchor retains the learnable Z and is the sole exact balance condition. In the auxiliary branch, the rooted residual (Eq. 4) cancels the shared Z so that prefix updates are not forced to redundantly reoptimize the same global scalar. The absorbed suffix targets are conservative hindsight backups over observed suffix task-only signals — u_max is the best observed downstream outcome, u_soft smoothly aggregates multiple suffix signals with distance discounting, and the mixture is an empirical bias-variance trade-off (Table 6 confirms neither endpoint alone is optimal). Stop-gradient on the termination head in the auxiliary branch (Eq. 27) prevents the drift diagnosed in Appendix C.6: in terminable prefix trees, arbitrary-start windows impose heterogeneous boundary conditions on the shared termination head, and the model can reduce many window residuals simultaneously by globally shifting stop logits rather than improving token-level transitions. We will define "termination drift" at first mention.

**Q1: Global optimum.** No: minimizing the full RapTB composite with finite η does not by itself guarantee exact sampling from the target. The exact reward-proportional fixed point remains tied to the terminal TB term. The auxiliary term is a variance-reducing regularizer that can bias the optimum while improving optimization in practice. In principle, annealing η→0 would recover pure TB, but we do not claim or verify this here. Empirically, RapTB achieves better distributional fidelity than TB (Table 3: JS 0.147 vs 0.339).

**Q2: AvgPrefixTB.** We implement exactly this: averaging (Δ_k^TB)² over all k with learnable Z.

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

AvgPrefixTB is viable and stronger than plain TB on Expr24 diversity, but materially different from RapTB. On SMILES it collapses toward short sequences (Avg Len 2.89, 54% at L=1-2), with worst QED/diversity among all methods. On Expr24 it similarly shortens (Avg Len 5.74 vs ~9.0). The gain is not merely "more prefix supervision" but specifically rooted Z-cancellation, suffix-absorbed targets, and stop-gradient on the auxiliary termination head.
