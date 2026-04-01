We thank Reviewer Pd1v for the thorough evaluation. We address each weakness below with new experimental evidence and clarified theoretical analysis.

**W1: Experimental validation — small LLM, narrow benchmarks.**

We acknowledge that scaffold-conditioned SMILES is our strongest application-scale result. However, the original submission already spans three distinct tasks (SMILES, Expr24, CommonGen) plus a longer-horizon L_max=15 stress test. We now add two further evaluations:

**(1) AMP biological sequence generation** (Jain et al. 2022 [8]), a standard GFlowNet benchmark with 20-amino-acid vocabulary and non-differentiable fitness predictor:

| Method | Perf | Diversity | Novelty | Avg Len |
|--------|------|-----------|---------|---------|
| TB | 0.927 | 7.39 | 10.65 | 17.4 |
| SubTB | 0.897 | 21.37 | 28.68 | 49.3 |
| RapTB | 0.919 | 8.83 | 14.44 | 22.4 |
| RapTB+SubM | **0.916** | **16.92** | **15.77** | **25.6** |

SubTB collapses to maximum length; its diversity/novelty are inflated by raw edit distance over unnaturally long sequences. RapTB+SubM achieves the best diversity/novelty among natural-length methods within 3K steps.

**(2) Scale-up from Llama-3.2-1B to 3B** on SMILES:

| Method (3B) | Acc | Score | FPDiv | Len |
|-------------|-----|-------|-------|-----|
| TB | 0.999 | 0.717 | 0.837 | 2.74 |
| SubTB | 0.313 | 0.221 | 0.854 | 8.48 |
| RapTB | 0.984 | 0.732 | 0.864 | 6.86 |
| RapTB+SubM | **0.996** | **0.856** | **0.937** | **7.96** |

SubTB drift is amplified at 3B (Acc drops to 0.313, with 64% of samples at max length), confirming the failure mode is structural and worsens with model capacity. Together with the original results, our evidence now spans molecules, arithmetic expressions, natural language, biological sequences, and two model scales.

**W2: Baseline methods — only TB and SubTB compared.**

Our original comparison was objective-level by design: RapTB modifies the TB-family loss for terminable prefix-tree LLM-GFlowNets (Hu et al. 2024 [4]), making TB and SubTB the direct loss baselines. We also compared RP and PRT as replay baselines. We now add three further comparisons:

**(1) AvgPrefixTB** (averaging TB loss uniformly over all prefixes):

| Task | Method | Key Metrics | Avg Len |
|------|--------|-------------|---------|
| SMILES | TB | Div=2.503, QED=0.717 | 3.06 |
| SMILES | AvgPrefixTB | Div=0.665, QED=0.661 | 2.89 |
| SMILES | RapTB | Div=2.448, QED=0.740 | 6.14 |
| SMILES | RapTB+SubM | Div=2.726, QED=0.844 | 7.44 |
| Expr24 | TB | NormCov=0.001, Unique=5.3 | 8.98 |
| Expr24 | AvgPrefixTB | NormCov=0.016, Unique=142 | 5.74 |
| Expr24 | RapTB | NormCov=0.039, Unique=246.7 | 8.99 |

Simple dense prefix supervision collapses on both tasks, confirming that RapTB's absorbed-suffix backup design is essential, not merely the act of applying loss at every prefix.

**(2) PPO and GRPO** (reward-maximizing RL, same model/LoRA/budget, unconstrained generation with soft vocab masking):

*SMILES QED:*

| Method | Acc | QED ↑ | Entropy ↑ | Avg Len |
|--------|-----|-------|-----------|---------|
| GRPO | 0.997 | 0.661 | 0.98 | 10.0 |
| PPO | 1.000 | 0.604 | 0.00 | — |
| TB (paper) | 0.998 | 0.717 | 2.503 | 3.06 |
| RapTB (paper) | 0.996 | 0.740 | 2.448 | 6.14 |
| RapTB+SubM (paper) | 0.988 | **0.844** | **2.726** | 7.44 |

*Expr24 (independent eval, 6400 samples × 3 repeats):*

| Method | Acc | Valid/6400 | Unique ↑ | Avg Len |
|--------|-----|-----------|----------|---------|
| GRPO | 0.002 | 12.3±1.2 | 1 | ~11 |
| PPO | 0.003 | ~20 | 1 | collapsed |
| TB (paper) | 1.000 | 6400 | 5.3 | 8.98 |
| RapTB (paper) | 0.991 | 6343 | 246.7 | 8.99 |

GRPO achieves decent QED on SMILES but entropy=0.98 (36% of GFlowNet's 2.726); on Expr24 training reward reaches 0.872 but independent eval yields only 0.2% valid. The diversity gap persists on dense-reward tasks, confirming a fundamental objective difference rather than a sparse-reward artifact.

**(3) TBA** (Bartoldson et al. NeurIPS 2025 [5]) is a system-level contribution (asynchronous VarGrad TB + global replay buffer), while RapTB is an objective-level contribution (within-trajectory credit assignment). TBA itself acknowledges TB's variance limitation and calls for "learning partial energy functions" as future work -- RapTB's absorbed-suffix backups realize this non-parametrically. The two are orthogonal and composable: RapTB can serve as the training objective inside a TBA-style pipeline.

**W3: No theoretical guarantees for preserving reward-proportional terminal distribution.**

We agree the paper does not claim a new exact fixed-point theorem. Our analysis proceeds from two angles:

*Angle 1 (regularizer structure):* RapTB decomposes as L_RapTB = L_TB + eta * L_aux. The TB fixed point (Malkin et al. 2022, Theorem 1 [1]) is equivalent to the MaxEnt RL optimal policy (Tiapkin et al. AISTATS 2024, Theorem 1 [2]). The auxiliary term acts as a variance-reducing regularizer. This is structurally analogous to how LED-GFN (Jang et al. ICLR 2024 [7]) adds a learned energy decomposition and FL-GFN (Pan et al. ICML 2023 [6]) introduces intermediate energies -- both preserve the TB fixed point while improving credit assignment.

*Angle 2 (zero-at-optimum):* In a terminable prefix tree, when the TB residual Delta_k^TB = 0 for all k:
- Rooted residuals: Delta_bar_k = Delta_k^TB - Delta_0^TB = 0
- Auxiliary corrections vanish: L_aux = sum_k w_k * (0 + u_k - u_k^tgt)^2 / sum_k w_k = 0
- Therefore L_RapTB = (Delta^TB)^2 + eta * 0 = 0

The reward-proportional distribution is a global minimum of both terms simultaneously. This parallels SubTB's same-fixed-point property established via telescoping (Madan et al. ICML 2023 [3]). We emphasize this is algebraic consistency (the correct solution is a zero of the loss), not a convergence rate theorem.

Empirically, the Oracle replay experiment (Table 3) supports this under controlled coverage: RapTB achieves Acc 0.945 vs TB 0.919, with better KL and JS divergence to the target distribution. We will surface the zero-at-optimum property explicitly in the revised paper alongside narrowed claim scope.

**References**

[1] Malkin et al. "Trajectory Balance: Improved Credit Assignment in GFlowNets." NeurIPS 2022.
[2] Tiapkin et al. "GFlowNets as Entropy-Regularized RL." AISTATS 2024.
[3] Madan et al. "Learning GFlowNets from Partial Episodes." ICML 2023.
[4] Hu et al. "Amortizing Intractable Inference in LLMs." ICLR 2024.
[5] Bartoldson et al. "TBA: Trajectory Balance with Asynchrony." NeurIPS 2025.
[6] Pan et al. "FL-GFN: Better Training with Local Credit." ICML 2023.
[7] Jang et al. "LED-GFN: Learning Energy Decompositions." ICLR 2024.
[8] Jain et al. "Biological Sequence Design with GFlowNets." ICML 2022.
