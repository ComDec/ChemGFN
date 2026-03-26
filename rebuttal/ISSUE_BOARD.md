# Issue Board — Submission 13383

## Global Themes

1. **GT-1: Theoretical framing** — RapTB is NOT a new exact fixed-point theorem; TB anchor remains exact, RapTB aux is variance-reducing regularizer
2. **GT-2: RL contextualization** — GFlowNets sit within MaxEnt/KL-regularized RL; TBA is orthogonal system-level work
3. **GT-3: RapTB vs SubM complementarity** — RapTB = internal credit assignment; SubM = external replay coverage
4. **GT-4: Scope revision** — Claims narrowed from "general LLM-GFlowNets" to "terminable prefix-tree LLM-GFlowNets in evaluated settings"

---

## Reviewer Pd1v (Score 3, Weak Reject, Confidence 2)

| ID | Raw Anchor | Type | Severity | Response Mode | Status |
|---|---|---|---|---|---|
| Pd1v-C1 | "experimental validation not strong enough...small LLM, narrow benchmarks" | empirical_support | major | narrow_concession + grounded_evidence | open |
| Pd1v-C2 | "baseline methods not sufficient, only TB and SubTB" | baseline_comparison | major | nearest_work_delta + grounded_evidence | open |
| Pd1v-C3 | "no theoretical guarantees...reward-proportional terminal distribution" | theorem_rigor | major | direct_clarification | open |

### Pd1v-C1: Narrow benchmarks, small model
- **Theme**: GT-4
- **Evidence**: Paper has 3 tasks (SMILES, Expr24, CommonGen) + L_max=15 stress test. Failure modes observed across all three.
- **Concession**: Strongest application-scale evidence is SMILES. Will revise claim scope.
- **New evidence needed**: AvgPrefixTB baseline results, potentially SynCodon task

### Pd1v-C2: Insufficient baselines
- **Theme**: GT-2
- **Evidence**: TB/SubTB are direct objective-level baselines. TBA is orthogonal (system-level).
- **Response**: Acknowledge TBA importance, clarify orthogonality, note RapTB can plug into TBA pipeline.

### Pd1v-C3: No theoretical guarantee
- **Theme**: GT-1
- **Evidence**: Eq. (9) already states TB anchor is exact. Oracle replay results (Table 3) show RapTB improves allocation, not just discovery.
- **Response**: Explicitly confirm no new exact theorem. TB anchor = exact fixed point. Aux = variance reducer.

---

## Reviewer cA3o (Score 4, Weak Accept, Confidence 3)

| ID | Raw Anchor | Type | Severity | Response Mode | Status |
|---|---|---|---|---|---|
| cA3o-C1 | "TB+SubM outperforms RapTB alone on coverage...when does RapTB provide additive benefit?" | empirical_support | critical | grounded_evidence | open |
| cA3o-C2 | "seven task-specific hyperparameters...sensitivity unknown" | complexity | major | grounded_evidence + future_work_boundary | **answered** |
| cA3o-C3 | "empirical support rests almost entirely on one domain" | empirical_support | major | narrow_concession + grounded_evidence | open |
| cA3o-C4 | "structurally analogous to GAE...equivalent to particular GAE estimator?" | assumptions | minor | direct_clarification | open |

### cA3o-C1: RapTB vs SubM additive benefit
- **Theme**: GT-3
- **Three regimes**: (1) Under RP: RapTB > TB on coverage. (2) Under SubM: RapTB+SubM > TB+SubM (NormCov 0.209 vs 0.100). (3) Under Oracle: RapTB > TB on accuracy/KL/JS (coverage controlled).
- **Key insight**: SubM dominates external discovery; RapTB dominates internal allocation. Appendix A.1 ceiling effect on SMILES.

### cA3o-C2: Hyperparameter sensitivity — **ANSWERED**
- **Theme**: (none — direct experimental response)
- **Existing defense**: Table 6 max/soft ablation; eta/gamma shared across tasks.
- **New evidence (completed)**: β×ρ sweep (9 configs, 5000 steps), η sweep (3 runs), k_min ablation (3 runs). See `SWEEP_RESULTS_FINAL.md`.
- **Key result**: Acc ≥ 0.994 across all 9 (β,ρ) configs. log_pterm(τ) ∈ [-0.25, -0.04] everywhere. η shows monotonic improvement. Fixed-low k_min is worst, validating schedule design.

### cA3o-C3: Domain generalization
- **Theme**: GT-4
- **Existing**: SMILES + Expr24 + CommonGen + L_max=15. Failure modes structural, not domain-specific.
- **New evidence potential**: SynCodon-Oracle biology benchmark.

### cA3o-C4: GAE analogy
- **Response**: Analogy in bias-variance sense, not exact equivalence. GAE = bootstrapped multi-step in policy gradient; RapTB = absorbed suffix in squared rooted balance residual with terminal TB anchor. Different fixed-point structure.

---

## Reviewer JxzD (Score 4, Weak Accept, Confidence 4)

| ID | Raw Anchor | Type | Severity | Response Mode | Status |
|---|---|---|---|---|---|
| JxzD-C1 | "Why termination probabilities in each subtrajectory? Tested with state flow values?" | assumptions | major | direct_clarification | open |
| JxzD-C2 | "explain prefix survival metric" | clarity | minor | direct_clarification | open |
| JxzD-C3 | "longer sequence tasks (AMP/GFP)?" | empirical_support | major | future_work_boundary | open |
| JxzD-C4 | "why finetune LLM, not train small model from scratch?" | assumptions | minor | direct_clarification | open |
| JxzD-C5 | "lack of convergence/global optimal analysis" | theorem_rigor | major | direct_clarification | open |
| JxzD-C6 | "absorbed suffix backups need clearer motivation" | clarity | major | direct_clarification | open |

### JxzD-C1: SubTB termination coupling
- **Evidence**: Appendix C.6 — SubTB residual boundary differences cause O(N^2) termination head coupling. RootSubTBLogZ ablation (Table 4) confirms: rooted + Z_theta restores accuracy.
- **State flow**: Reasonable conjecture but not tested. Would change parameterization.

### JxzD-C2: Prefix survival
- **Definition**: Surv(k) = n_k / n_valid (Appendix B.3). Interpret jointly with PefEnt and Top1.

### JxzD-C3: Longer sequences (AMP/GFP)
- **Honest response**: Current longest is SMILES L_max=15. AMP/GFP is natural next step but not available.

### JxzD-C4: Why LLM fine-tuning
- **Response**: Problem formulation is LLM-GFlowNet post-training. Frozen reference LM prior is part of the setup, not incidental. Training from scratch answers a different question.

### JxzD-C5: Global optimal analysis
- **Theme**: GT-1
- **Response**: TB anchor has exact fixed point. Auxiliary term is regularizer. See GT-1.

### JxzD-C6: Absorbed suffix backup motivation
- **Response**: Terminal-only rewards give high variance for early prefixes. Absorbed target backs up observed suffix rewards to create lower-variance proxy credit. Max component = lower bound; soft component = smooth aggregate. Mixed via alpha for score-diversity balance (Table 6).

---

## Reviewer QHmk (Score 2, Reject, Confidence 4)

| ID | Raw Anchor | Type | Severity | Response Mode | Status |
|---|---|---|---|---|---|
| QHmk-C1 | "fails to put itself in broader context of RL literature...GFlowNet = entropy-regularized RL" | novelty | critical | narrow_concession + direct_clarification | open |
| QHmk-C2 | "add PPO and GRPO as baselines" | baseline_comparison | major | grounded_evidence | **answered** |
| QHmk-C3 | "TBA is most direct competitor...should be added" | baseline_comparison | critical | nearest_work_delta | open |
| QHmk-C4 | "RapTB loss section should be expanded...mathematical derivation" | clarity | major | direct_clarification | open |
| QHmk-C5 | "does reaching global optimum guarantee target distribution?" | theorem_rigor | critical | direct_clarification | open |
| QHmk-C6 | "simpler baseline: averaging TB loss over all prefixes" | baseline_comparison | critical | grounded_evidence | open |
| QHmk-C7 | "terminology used before definition (termination drift)" | clarity | minor | direct_clarification | open |

### QHmk-C1: RL contextualization
- **Theme**: GT-2
- **Response**: Acknowledge GFlowNets within MaxEnt/KL-regularized RL family. Revise "in contrast to reward-maximizing RL" framing. Cite Tiapkin+ (AISTATS 2024), Deleu+ (UAI 2024).

### QHmk-C2: PPO/GRPO baselines — **ANSWERED**
- **New evidence**: GRPO on Expr24: Acc=0.002 (12/6400 valid), 99.9% length collapse to L_max, NormCov=0, prefix_top1_auc=0.977 (near-total prefix collapse). Confirms reward-maximizing RL fails at distributional sampling.
- **Response**: GRPO result directly demonstrates reward maximization ≠ reward-proportional sampling. Primary comparison remains TB-family for objective-level mechanism isolation.

### QHmk-C3: TBA baseline
- **Theme**: GT-2
- **Response**: TBA is important related work. It's a system-level pipeline (async search + replay + TB training). Our contribution is orthogonal: objective-level (rooted/absorbed + termination calibration). RapTB can plug into TBA.

### QHmk-C4: Mathematical explanation
- **Response**: Z not removed — stays in terminal TB. Rooted residual cancels shared Z in auxiliary branch. Absorbed target reduces early-prefix variance. Table 6 validates design.

### QHmk-C5: Global optimum guarantee
- **Theme**: GT-1
- **Response**: No new exact guarantee claimed. Terminal TB = exact balance condition. Auxiliary = variance-reducing regularizer.

### QHmk-C6: Averaged-prefix TB baseline
- **Theme**: (direct experimental response)
- **Status**: AvgPrefixTBLoss ALREADY IMPLEMENTED in losses.py. Configs ready for Expr24 (RP + Oracle) and SMILES. Includes detach_pterm variant.
- **Evidence needed**: Run experiments and report results.

### QHmk-C7: Terminology timing
- **Response**: Will define "termination drift" at first mention and forward-reference the formal analysis.

---

## Summary Statistics

| Severity | Count |
|---|---|
| Critical | 5 (QHmk-C1, QHmk-C3, QHmk-C5, QHmk-C6, cA3o-C1) |
| Major | 10 |
| Minor | 3 |

| Response Mode | Count |
|---|---|
| direct_clarification | 8 |
| grounded_evidence | 5 |
| narrow_concession | 4 |
| nearest_work_delta | 2 |
| future_work_boundary | 2 |

| Status | Count |
|---|---|
| **answered** | 2 (cA3o-C2, QHmk-C2) |
| open | 14 |
| needs_user_input | 1 (QHmk-C6 — AvgPrefixTB results pending) |
| deferred | 1 (JxzD-C3 — AMP/GFP future work) |
| open | 18 |
| answered | 0 |
| needs_user_input | 0 |
