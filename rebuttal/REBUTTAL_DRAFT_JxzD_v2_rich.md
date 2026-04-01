# Rebuttal to Reviewer JxzD — Rich Version (v2)

> This version contains extended explanations marked with `[OPTIONAL]` that exceed the 5000-char venue limit. The PASTE_READY version is the submission version.

---

We thank Reviewer JxzD for the thoughtful evaluation. We are glad the ablations and submodular replay were found valuable.

## W1: Convergence analysis

The terminal TB term (Eq. 9) retains the exact reward-proportional fixed point: when the TB residual is zero for all terminals, q(x) ∝ R(x). This is the **only exact balance condition** in RapTB. The auxiliary branch (Eq. 8) is a **variance-reducing regularizer** that improves prefix credit assignment but can introduce bias for finite η. We do not claim a new fixed-point theorem for the composite objective.

Key structural point: TB guarantees global log Z; the auxiliary operates **without** log Z (the rooted residual cancels it via Eq. 5), so it cannot learn an incorrect global partition function. The auxiliary provides dense prefix-level supervision purely through relative comparisons anchored at s_0. Empirically, the bias introduced by finite η is small relative to the variance reduction benefit (Tables 1, 3, 6).

**[OPTIONAL — cut if over limit]** To be concrete about the bias-benefit tradeoff: on Expr24 under Oracle replay (Table 3), where exploration noise is controlled, RapTB still improves KL (0.045 vs TB's 0.140) and JS (0.010 vs 0.040), confirming the auxiliary term improves internal credit allocation beyond what exploration alone provides. The TB anchor ensures the global balance is maintained even as the auxiliary introduces some bias.

## W2: Absorbed suffix backup motivation

The core problem is **conditional variance**: when training early prefixes, the learning signal comes from the full trajectory's terminal reward. The same prefix can lead to very different suffixes with very different rewards, creating high conditional variance for early-prefix updates (Appendix C.5, Eq. 32 — the bias-variance decomposition).

**SubTB's approach and its failure in the LLM setting.** SubTB addresses variance most directly by learning flow at every sub-trajectory boundary. However, in the LLM prefix-tree formulation (Hu et al., 2024), flow is implicitly parameterized as:

```
F(s) = R(s^T) / q(T|s)
```

There is no separate flow head — the termination probability `q(T|s)` implicitly encodes log-flow. Each SubTB window imposes boundary conditions on the shared termination logits, creating O(N²) conflicting constraints. Because each termination logit participates in O(N²) windows simultaneously, the optimizer can reduce the aggregate squared loss by globally shifting the termination head rather than improving token-level transitions. This is **termination drift**.

Evidence:
- Table 4: SubTB log p_term = −79.638 under RP, −86.415 under Oracle
- Table 5: SubTB Δlog p_term = −28.32, saturating length at 20.00 on CommonGen
- RootSubTBLogZ ablation (Table 4): restricting to rooted windows + reintroducing Z_θ restores accuracy to ~100%

**[OPTIONAL]** This diagnosis aligns with Reviewer cA3o's assessment: "the mechanistic diagnosis of SubTB's failure on terminable prefix trees is definitely their strongest technical contribution." The ablation directly validates the structural cause.

**RapTB's approach.** RapTB = TB + auxiliary regularizer. The TB term preserves the exact global balance. The auxiliary:

1. **Rooted residual** (Eq. 5): Cancels shared Z in the auxiliary branch, providing O(N) prefix-local supervision without arbitrary-start boundary heterogeneity. This is NOT "removing the learnable scalar" — Z stays in the terminal TB. It only disappears from the auxiliary through algebraic cancellation.

2. **Absorbed suffix targets** (Eqs. 6–7):
   - **u_max**: Conservative hindsight backup — the best task-only signal observed from position k onward along the sampled trajectory. NOT a formal lower bound on true prefix flow (which would require knowing the complete state flow). It is a conservative backup of the observed suffix task-only component.
   - **u_soft**: Smooth log-sum-exp aggregation with distance discounting (ρ), providing longer-range credit propagation. Acts as a softer, more informative signal than u_max alone.
   - **α interpolation**: Empirical bias-variance tradeoff. u_max has low variance but can be overly conservative (biased toward the single best suffix point). u_soft is smoother but noisier. Table 6 confirms mixed > either endpoint alone.

3. **Stop-gradient on p_term**: Prevents the auxiliary from shifting termination behavior. Without this, sequences collapse to Len 3.40 (Table 6).

**[OPTIONAL — important nuance]** We acknowledge that RapTB introduces some bias relative to pure TB. The absorbed targets are computed from the *sampled* trajectory's suffix, not from the true expected suffix reward over all possible continuations. The max/soft combination is an empirically motivated heuristic, not a new theorem. The key design principle is: TB provides the global anchor, the auxiliary provides dense variance-reduced prefix signals, and the stop-gradient prevents the auxiliary from corrupting termination behavior.

## W3: Hyperparameter robustness

We conducted cross-task sweeps: 9 (β, ρ) configurations on Expr24 and 9 on SMILES (18 total).

**Expr24 results:** Acc ≥ 0.994 across all 9 configs. log_pterm(τ) ∈ [−0.25, −0.04] — all well-calibrated.

**SMILES results:** 8/9 configs achieve Acc ≥ 0.991, FPDiv 0.849–0.883 (paper: 0.860). Only (β=10, ρ=0) shows mild degradation (0.968), which is the extreme no-distance-discount regime.

**Parameter roles and sensitivity:**
- **k_min** (minimum prefix depth for auxiliary supervision): Smaller k_min emphasizes shorter prefixes. This creates a "shortcut path" — the model can allocate flow to high-reward short sequences more easily, which improves quality but reduces diversity. k_min=7 (fixed) is worst for diversity (0.85); the linear schedule (5→2) used in the paper provides a good balance.
- **η** (auxiliary weight): Monotonic diversity increase from 0.98 to 1.15 across the sweep. Higher η strengthens prefix credit assignment.
- **β, ρ** (soft backup temperature, distance discount): Mainly affect the bias-variance tradeoff of the absorbed target. Robust across 18 configs.

Sensitivity table will be added to the revision.

## Q1: SubTB termination coupling

In the terminable prefix-tree specialization, SubTB eliminates explicit state flow variables. The SubTB residual for window [i, j] (Eq. 33) takes the form:

```
Δ_{i→j} = Σ log P_F(transitions) + [log q(T|s_j) - log q(T|s_i)] + [log R(s_i^T) - log R(s_j^T)]
```

The boundary difference `log q(T|s_j) - log q(T|s_i)` is the key problematic term. Because windows overlap extensively and all share a single termination head, the optimizer faces O(N²) heterogeneous boundary conditions on the same logits. It can reduce aggregate squared loss by globally shifting the termination head — this is more efficient (from the optimizer's perspective) than improving N² individual token-level transitions.

Our RootSubTBLogZ ablation (Table 4) directly validates this diagnosis: restricting SubTB to rooted windows and reintroducing learnable Z_θ restores accuracy to ~100% on Expr24. This confirms the failure is **structural** — caused by the interaction between flow-free parameterization and overlapping arbitrary-start windows — not by any deficiency in SubTB's principle itself. SubTB with proper flow parameterization would likely avoid this issue.

**[OPTIONAL]** Learning explicit state flow values (a separate value head, as in the original SubTB formulation for non-LLM GFlowNets) could reduce termination head pressure. This would decouple the flow estimate from the termination probability, removing the O(N²) coupling. However, it changes the parameterization substantially and introduces additional training complexity (the value head must be trained jointly). We note this as a promising direction for future work.

## Q2: Prefix survival

**Surv(k) = n_k / n_valid** (Appendix B.3), where n_k counts valid samples with length ≥ k.

**Intuitive explanation:** "What fraction of valid generated sequences are still being built at position k, rather than having already stopped?" A method that terminates most sequences early (e.g., at length 2–3) will show rapid survival decay. A method that maintains sequences to longer lengths shows higher survival.

**Why this metric matters:** Standard terminal diversity metrics (token entropy, fingerprint diversity) measure diversity among completed sequences but **cannot detect prefix collapse**. Prefix collapse is the failure mode where diverse-looking terminals share near-identical early prefixes and only branch late — the tree is narrow at the trunk and fans out only at the leaves.

**Joint interpretation:**
- High Surv + high Top1 = sequences reach depth k but share few prefix patterns (shallow branching — bad)
- High Surv + low Top1 + high PefEnt = genuine early branching with diverse prefixes (good)
- Low Surv = sequences terminate too early (length bias)

Figure 3 shows TB has rapid survival decay with increasing Top1 (prefix collapse), while RapTB maintains higher survival with lower Top1 — genuine diverse prefix branching rather than superficial terminal diversity.

## Q3: Longer sequences and biological sequences

We provide three lines of new evidence:

### (i) AMP generation (20–50 amino acids)

We evaluate on the antimicrobial peptide design task from Jain et al. (2022), using a non-differentiable MLP fitness predictor (ProtTrans embeddings). Sequences are 20–50 amino acids (D0 mean = 22.3 AA).

| Method | Performance ↑ | Diversity ↑ | Novelty ↑ | Avg Len | Steps |
|--------|---------------|-------------|-----------|---------|-------|
| **RapTB+SubM** | 0.916 | **16.92** | **15.77** | 25.6 | **3K** |
| RapTB | 0.919 | 8.83 | 14.44 | 22.4 | 5K |
| TB | 0.927 | 7.39 | 10.65 | 17.4 | 10K |
| SubTB † | 0.897 | 21.37 | 28.68 | 49.3 | 9K |
| Jain et al. GFN (no AL) | 0.868 | 11.32 | 15.72 | ~22 | 10K |
| Jain et al. GFN-AL | 0.932 | 22.34 | 28.44 | ~22 | 10K×10 rounds |

† SubTB exhibits length collapse (all sequences at max length 50), confirming termination drift generalizes to biological sequences. Its inflated diversity/novelty reflects longer sequences, not compositional diversity.

**Key findings:**
- Without active learning, RapTB+SubM surpasses Jain et al.'s non-AL GFlowNet baseline on all three metrics (Perf: 0.916 vs 0.868, Div: 16.92 vs 11.32, Nov: 15.77 vs 15.72).
- RapTB+SubM achieves competitive performance with their 10-round AL pipeline (0.932) using only single-round training in 3K steps.
- SubTB's termination drift is not SMILES-specific — it reproduces on amino acid sequences with a completely different vocabulary and reward function.

**[OPTIONAL — honest limitation]** RapTB+SubM does not match the full AL pipeline's diversity (16.92 vs 22.34) or novelty (15.77 vs 28.44). The AL pipeline uses 10 rounds of proxy retraining with oracle access, which is orthogonal to objective-level improvements. Combining RapTB with AL is a natural extension.

### (ii) 3B scale-up (SMILES)

| Method | Acc | Score | Entropy | FPDiv | Len |
|--------|-----|-------|---------|-------|-----|
| TB (3B) | 0.999 | 0.717 | 1.905 | 0.837 | 2.74 |
| SubTB (3B) | 0.313 | 0.221 | 2.090 | 0.854 | 8.48 |
| RapTB (3B) | 0.984 | 0.732 | 2.252 | 0.864 | 6.86 |
| RapTB+SubM (3B) | 0.996 | 0.856 | 2.447 | 0.937 | 7.96 |

SubTB's drift **worsens** at 3B (Acc 0.313, 64% at max length). RapTB+SubM achieves the best tradeoff.

### (iii) SMILES L_max=15

Table 2 shows RapTB+SubM achieves the best long-horizon coverage (Frac(11+) = 0.701) and lowest prefix concentration (Top1 = 0.071).

## Q4: Why fine-tune LLM?

The frozen reference LM prior P_ref is part of the reward definition (Eq. 17), anchoring generation toward well-formed outputs — it is integral to the problem formulation, not an incidental choice. LLMs provide a strong sequence model initialization for SMILES.

**Honest assessment:** We note that the current experiments primarily demonstrate the benefit of a **well-calibrated sequence prior**. They do not directly show that the LLM leverages chemistry-specific knowledge from pretraining — the Llama-3.2-1B was not pretrained on chemistry papers or SMILES data specifically. The performance gains come from the structured sequence modeling capability and the P_ref regularization, not from domain knowledge.

**[OPTIONAL — future direction]** We believe the LLM-GFlowNet framework becomes more compelling with chemistry-pretrained models (e.g., models trained on chemical literature + SMILES), which could provide richer molecular context than diffusion or graph-based alternatives. This is a future direction that the current framework naturally supports.

Training a model from scratch would answer a different research question. We expect the identified failure modes (prefix collapse, termination drift) to persist, as they are structural to the prefix tree + SubTB objective, not model-specific. The 3B scale-up results (Q3-ii) support this — the failures worsen with scale rather than disappearing.

---

## Provenance Tracking

| Claim | Source |
|-------|--------|
| TB fixed point (Eq. 9) | Paper Proposition 1 |
| SubTB flow-free formulation | Hu et al. (2024), Appendix A.2 |
| SubTB log p_term = −79.638 | Paper Table 4 |
| RootSubTBLogZ ~100% accuracy | Paper Table 4 |
| AMP results | `rebuttal/evidence/amp_results.md` (user_confirmed_result) |
| 3B scale-up results | `rebuttal/evidence/` (user_confirmed_result) |
| Hyperparameter sweep (18 configs) | `rebuttal/evidence/SWEEP_RESULTS_COMPLETE.md`, `SMILES_SWEEP_RESULTS.md` (user_confirmed_result) |
| Jain et al. non-AL baseline | Jain et al. (2022) Table 1 |
| Table 6 ablation results | Paper Table 6 |

## Character Count

- PASTE_READY_JxzD_v2.txt: ~4878 chars (within 5000 limit)
- This rich version: extended with [OPTIONAL] sections
