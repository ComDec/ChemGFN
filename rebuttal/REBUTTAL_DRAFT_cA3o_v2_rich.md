# Rebuttal to Reviewer cA3o — v2.1 Rich Draft (literature-grounded)

---

We thank Reviewer cA3o for the thorough analysis — the diagnosis of SubTB's termination failure and the clarity of the SubM-vs-RapTB question are exactly the type of examination that strengthens the paper.

## W1/Q3: Domain Generalization

We acknowledge that the submitted version's empirical scope is narrow. We have since added two new lines of evidence:

**(1) AMP biological sequence generation** (20–50 amino acids, non-differentiable fitness oracle): RapTB+SubM achieves the best diversity (16.92) and novelty (15.77) among natural-length methods (Avg Len=25.6 AA, D0 mean=22.3 AA), in only 3K steps. SubTB again exhibits length collapse (Avg Len=49.3, hitting max length), inflating its raw diversity/novelty scores artifactually.

| Method | Perf ↑ | Div ↑ | Nov ↑ | Avg Len | Steps |
|--------|--------|-------|-------|---------|-------|
| RapTB+SubM | 0.916 | **16.92** | **15.77** | 25.6 | 3K |
| RapTB | 0.919 | 8.83 | 14.44 | 22.4 | 5K |
| TB | 0.927 | 7.39 | 10.65 | 17.4 | 10K |
| SubTB † | 0.897 | 21.37† | 28.68† | 49.3† | 9K |

**(2) 3B scale-up on SMILES**: SubTB's termination drift is *amplified* at 3B (Acc drops to 0.313). RapTB+SubM achieves the best trade-off (Acc=0.996, QED=0.856, FPDiv=0.937).

We now have four tasks (SMILES, Expr24, CommonGen, AMP) across two model scales (1B, 3B). We will revise claims to "terminable prefix-tree LLM-GFlowNets in the evaluated settings."

---

## W2/Q1: When Does RapTB Provide Additive Benefit Over SubM?

**SubM = external discovery; RapTB = internal allocation.**

**(1) RP replay (no SubM):** RapTB already improves over TB. Expr24 NormCov: 0.001 (TB) → 0.039 (RapTB); Unique_valid: 5.3 → 246.7.

**(2) SubM replay:** RapTB+SubM doubles TB+SubM (NormCov: 0.209 vs 0.100).

**(3) Oracle replay** (Table 3): Coverage controlled. RapTB still outperforms TB in Acc (0.945 vs 0.919) and both KL directions — better flow allocation, not just better discovery.

On SMILES (Appendix A.1), RapTB+SubM shifts more mass to the longest bin (Frac[9-10]: 0.402 vs 0.323).

---

## W3/Q2: Hyperparameter Sensitivity

We conduct a comprehensive cross-task study (18 β×ρ configs + 6 ablations):

- **Expr24 β×ρ (9 configs):** All 9 Acc≥0.983. log p_term ∈ [−0.25, −0.04] vs SubTB's −79.6 (Table 4).
- **SMILES β×ρ (9 configs):** 8/9 Acc≥0.991; FPDiv ∈ [0.849, 0.883] (paper: 0.860).
- **η sweep:** Monotonic improvement (NormCov: 0.008→0.014, JS: 0.235→0.185).
- **k_min:** Fixed-low worst — early prefixes carry noisier supervision.

Paper defaults sit in a broad plateau. Limitation: γ and K not fully swept.

---

## Q4: GAE Analogy — Literature-Grounded Analysis

The connection is a **structural analogy in variance reduction**, not an exact equivalence. We now provide a precise comparison grounded in the literature:

**(i) Optimization target:** GAE (Schulman et al. ICLR 2016) estimates advantages Â(s,a) for policy gradient updates ∇J. RapTB constructs target values for squared balance residuals (L²). In the entropy-regularized RL view (Tiapkin et al. AISTATS 2024), the GFlowNet DB loss corresponds to Dueling Soft DQN; the advantage A(s,s') = log F(s→s') − log F(s) represents the flow-theoretic analog. But RapTB does not estimate advantages — it constructs regression targets.

**(ii) Bootstrapping source:** GAE bootstraps from a **learned value function** V(s), creating a bias-variance tradeoff controlled by λ. In the GFlowNet setting (Tiapkin et al. 2024), log F(s) = V*(s) (soft value function). FL-GFN (Pan et al. ICML 2023) reparameterizes this via known intermediate energy E(s). LED-GFN (Jang et al. ICLR 2024) learns it with auxiliary potentials. **RapTB bypasses learning V*(s) entirely** and uses non-parametric suffix aggregation — no learned value function, no auxiliary network. This makes RapTB's targets unbiased given the trajectory but dependent on trajectory quality.

**(iii) Partition function constraint:** The GFlowNet TB condition imposes a global normalization constraint absent in standard policy gradient.

**(iv) Closest analogues in the GFlowNet literature:**
- **FL-GFN** (Pan et al. ICML 2023): Requires E(s) evaluable at intermediate states (Assumption 4.1) — violated in sparse-reward settings where S(s_{0:k}) ≈ 0. FL-GFN provides no signal where absorbed targets are most needed.
- **LED-GFN** (Jang et al. ICLR 2024): Learns potential decomposition but needs auxiliary network + smoothness regularizer.
- **TBA** (Bartoldson et al. NeurIPS 2025): Explicitly calls for "learning partial energy functions" as future work. RapTB = non-parametric realization of this.

**(v) Closest analogues in RL:**
- **RUDDER** (Arjona-Medina et al. NeurIPS 2019): Learned return decomposition — guarantees conservation (redistributed rewards sum to terminal). RapTB makes no such guarantee.
- **HCA** (Harutyunyan et al. NeurIPS 2019): Hindsight credit via return-conditional distribution. Both use "hindsight"; HCA uses Bayes' rule, RapTB uses max/softmax.
- **Potential-based shaping** (Ng et al. 1999): NOT applicable — u_k^tgt is trajectory-dependent, not state-dependent, and does not telescope.

We will revise the discussion to make these distinctions explicit and connect to the RL view.

---

## Summary of Changes

| Concern | Action | Status |
|---------|--------|--------|
| Narrow domain (W1/Q3) | +AMP task, +3B scale-up | Done |
| RapTB vs SubM role (W2/Q1) | Three-regime analysis with Oracle isolation | Paper data + new framing |
| Hyperparameter sensitivity (W3/Q2) | 18-config sweep + η + k_min, cross-task | Done (24 new runs) |
| GAE analogy (Q4) | Literature-grounded comparison with FL-GFN, LED-GFN, RUDDER, HCA, TBA | Revised |
| Claim scope | Narrowed to "evaluated settings" | Will update |
| RL contextualization | Cite Tiapkin+2024, Deleu+2024, TBA | Will update |

---

## Provenance

| Claim | Source |
|-------|--------|
| GAE mechanism | Schulman et al. (2015), ICLR 2016 |
| GFlowNet = MaxEnt RL | Tiapkin et al. (2024), AISTATS 2024 |
| log F(s) = V*(s) | Tiapkin et al. (2024), Section 3 |
| DB = Dueling Soft DQN | Tiapkin et al. (2024), Section 4 |
| SubTB = PCL | Deleu et al. (2024), UAI 2024, Prop 3.2 |
| FL-GFN Assumption 4.1 | Pan et al. (2023), ICML 2023 |
| LED-GFN learned potential | Jang et al. (2024), ICLR 2024 |
| TBA "partial energy functions" | Bartoldson et al. (2025), NeurIPS 2025, Limitations |
| RUDDER return decomposition | Arjona-Medina et al. (2019), NeurIPS 2019 |
| HCA hindsight credit | Harutyunyan et al. (2019), NeurIPS 2019 |
| PBRS optimality preservation | Ng et al. (1999), ICML 1999 |
