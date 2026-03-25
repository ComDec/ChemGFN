# Rebuttal Strategy Plan — Submission 13383

## Overall Assessment

**Current scores**: 4, 4, 3, 2 (two weak accept, one weak reject, one reject)
**Path to acceptance**: Flip QHmk (2→4) or Pd1v (3→4). Both are achievable with correct framing + one strong new result.
**This is NOT a dead paper.** The strongest contribution is the structural diagnosis of SubTB failure + minimal fix.

## Global Narrative Corrections

Three mandatory reframes in the rebuttal opener:

1. **TB exact, RapTB aux regularizer** — Do NOT let anyone read "RapTB has exact reward-proportional guarantee"
2. **SubM = external coverage, RapTB = internal assignment** — They are complementary, not competing
3. **Strongest contribution = structural diagnosis + minimal fix** — Not "universal new objective"

## Character Budget (TBD — waiting for venue limit)

Tentative allocation:
- Opener (global themes): 10-15%
- Per-reviewer responses: 75-80% (QHmk gets the most)
- Closing (meta-reviewer summary): 5-10%

## Per-Reviewer Strategy

### QHmk (Reject → target: Weak Accept) — HIGHEST PRIORITY

This reviewer is the most dangerous but also most convertible. Core issue is **framing**, not fundamental flaws.

| Issue | Response | Evidence |
|---|---|---|
| QHmk-C1: RL context | **Concede**: GFlowNets ⊂ MaxEnt/KL-regularized RL. Reframe as "reward-proportional posterior sampling within broader RL family" | Cite Tiapkin+ 2024, Deleu+ 2024 |
| QHmk-C2: PPO/GRPO | **Half-concede**: Valuable as reward-maximization references. Primary comparison is TB-family to isolate objective-level mechanism | Structural argument |
| QHmk-C3: TBA | **Acknowledge + orthogonality**: TBA = system pipeline; RapTB = objective modification. Complementary. | Bartoldson+ 2025 abstract |
| QHmk-C4: Math clarity | **Explain**: Z stays in terminal TB. Rooted residual cancels shared Z in aux only. Absorbed target = variance reduction | Eq. 9, Table 6 |
| QHmk-C5: Global optimum | **Directly confirm**: No new exact theorem. TB anchor = exact fixed point. Aux = regularizer | Eq. 9 |
| QHmk-C6: AvgPrefixTB | **NEW EXPERIMENT**: Run AvgPrefixTB + detach_pterm variants on Expr24 (RP + Oracle) | Code ready, configs ready |

### cA3o (Weak Accept — reinforce) — SECOND PRIORITY

This reviewer is the most technically engaged. Respond in their framing.

| Issue | Response | Evidence |
|---|---|---|
| cA3o-C1: RapTB vs SubM | **Three-regime evidence**: RP (RapTB > TB), SubM (RapTB+SubM > TB+SubM), Oracle (RapTB > TB with coverage controlled) | Tables 3, 4; Appendix A.1 |
| cA3o-C2: Hyperparam sensitivity | **NEW EXPERIMENT**: β×ρ heatmap + η + k_min ablation | Sweep scripts ready |
| cA3o-C3: Domain generalization | **Concede partially**: 3 tasks + L_max=15. Failure modes structural. Narrow claims. | GT-4 |
| cA3o-C4: GAE analogy | **Clarify**: Bias-variance analogy, not exact equivalence. Different objective geometry. | Structural argument |

### JxzD (Weak Accept — reinforce) — THIRD PRIORITY

This reviewer wants mechanism understanding. Answer directly.

| Issue | Response | Evidence |
|---|---|---|
| JxzD-C1: SubTB termination | **Explain**: Appendix C.6 + RootSubTBLogZ ablation (Table 4). State flow = reasonable conjecture, not tested | Paper evidence |
| JxzD-C2: Prefix survival | **Define**: Surv(k) = n_k/n_valid (Appendix B.3) | Paper evidence |
| JxzD-C3: AMP/GFP | **Honest**: Not available. L_max=15 is current longest. Natural next step. | Future work |
| JxzD-C4: Why LLM | **Clarify**: Problem formulation is LLM post-training. Reference prior is part of setup. | Paper design |
| JxzD-C5/C6: Theory + motivation | **Combine with GT-1**: TB exact, aux = regularizer. Absorbed backup reduces early-prefix variance | Eq. 9, Appendix C.5 |

### Pd1v (Weak Reject → target: borderline) — FOURTH PRIORITY

This reviewer's concerns overlap with others. Partially addressed by QHmk + cA3o responses.

| Issue | Response | Evidence |
|---|---|---|
| Pd1v-C1: Narrow benchmarks | **Concede scope + show breadth**: 3 tasks + L_max=15. Narrow claims. | GT-4 |
| Pd1v-C2: Weak baselines | **Same as QHmk-C2/C3**: TB/SubTB = objective-level. TBA = orthogonal system. | GT-2 |
| Pd1v-C3: No theory | **Same as QHmk-C5**: TB exact, aux = regularizer | GT-1 |

## Evidence Sprint Plan

### Already implemented (code + configs ready):

1. **AvgPrefixTB baseline** (QHmk-C6)
   - `VarExpr24_AvgPrefixTB.yaml` — strict reviewer baseline
   - `VarExpr24_AvgPrefixTB_detach_pterm.yaml` — detach pterm variant
   - `VarExpr24_AvgPrefixTB_oracle.yaml` — oracle replay
   - `VarExpr24_AvgPrefixTB_detach_pterm_oracle.yaml` — oracle + detach
   - `VarExpr24_TB_plus_AvgAux.yaml` — controlled version
   - SMILES variants also available

2. **Hyperparameter sweep** (cA3o-C2)
   - `scripts/sweep/run_sweep1_beta_rho.sh` — 9-point β×ρ grid
   - `scripts/sweep/run_sweep2_eta.sh` — 3-point η sweep
   - `scripts/sweep/run_sweep3_kmin.sh` — 3-point k_min ablation
   - `scripts/sweep/run_full_validation.sh` — top-2 full validation
   - `scripts/sweep/analyze_sweep.py` — analysis + heatmap generation

### Needs implementation:

3. **SynCodon-Oracle** (cA3o-C3, JxzD-C3, Pd1v-C1) — OPTIONAL, only if time allows

## Execution Priority

1. **Day 1**: Run AvgPrefixTB experiments (Expr24 RP + Oracle, 4 variants × 3 seeds)
2. **Day 1-2**: Run sweep Phase 1 screening (15 runs, ~1.5h on 8 GPUs)
3. **Day 2-3**: Analyze sweep results, run Phase 2 full validation (18 runs)
4. **Day 3-5**: Draft rebuttal with narrative corrections + new evidence
5. **Day 4-5**: SynCodon-Oracle if compute allows (OPTIONAL)
6. **Day 6**: Stress test via Codex MCP
7. **Day 7**: Finalize PASTE_READY.txt

## Blocked Items

- [ ] **Character limit**: User must confirm ICML 2026 rebuttal character limit before drafting
- [ ] **AvgPrefixTB results**: Must run experiments before QHmk-C6 can be answered with evidence
- [ ] **Sweep results**: Must run screening before cA3o-C2 can be answered with evidence

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| AvgPrefixTB matches RapTB | Medium | Medium | Reframe: "dense prefix supervision is the key insight; absorbed target adds robustness" |
| AvgPrefixTB clearly worse | High | Positive | Validates RapTB's design choices |
| Sweep shows fragile optima | Low | High | Show broad stable region, acknowledge outliers |
| QHmk immovable regardless | Medium | High | Focus on convincing AC/SAC via cA3o+JxzD reinforcement |
