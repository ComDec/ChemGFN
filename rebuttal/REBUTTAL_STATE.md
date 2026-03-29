# Rebuttal State

- **Paper**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training
- **Venue**: ICML 2026
- **Submission ID**: 13383
- **Current Phase**: Phase 7 COMPLETE — ready to submit
- **Response Mode**: TEXT_ONLY
- **Character Limit**: 5000 per response, multiple submissions + revised PDF allowed

## Reviewers

| ID | Score | Stance | Key Concern |
|---|---|---|---|
| Pd1v | 3 (Weak Reject) | negative | Narrow benchmarks, weak baselines, no theory |
| cA3o | 4 (Weak Accept) | swing-positive | SubM vs RapTB role, hyperparameter sensitivity, generalization |
| JxzD | 4 (Weak Accept) | swing-positive | SubTB termination mechanism, longer sequences, LLM motivation |
| QHmk | 2 (Reject) | negative | RL contextualization, missing baselines (PPO/GRPO/TBA), simpler baseline |

## Evidence Status — ALL EXPERIMENTS COMPLETE

| Experiment | Answers | Status | Results |
|---|---|---|---|
| β×ρ sweep Expr24 (9 configs, bsz=128, 5000 steps) | cA3o-C2 | **DONE** | Acc 0.983-1.000, Div 0.77-1.02 |
| β×ρ sweep SMILES (9 configs, bsz=128, 5000 steps) | cA3o-C2 | **DONE** | Acc 0.968-0.999, FPDiv 0.849-0.883 |
| η sweep (3 runs, bsz=128, 5000 steps) | cA3o-C2 | **DONE** | Monotonic: Div 0.98→1.15 |
| k_min ablation (3 runs, bsz=128) | cA3o-C2 | **DONE** | Fixed-low worst (0.85) |
| PPO baseline (Expr24) | QHmk-C2 | **DONE** | 20/6400 valid, 1 unique, crash |
| GRPO baseline (Expr24) | QHmk-C2 | **DONE** | 12/6400 valid, 1 unique |
| AvgPrefixTB (Expr24 + SMILES) | QHmk-C6 | **DONE** | Collapse on both tasks |
| 3B scale-up (SMILES, 4 methods) | Pd1v-C1, JxzD-C3 | **DONE** | RapTB+SubM best |
| AMP biological sequence (4 methods) | JxzD-C3, cA3o-C3, Pd1v-C1 | **DONE** | RapTB+SubM best diversity (16.92), SubTB length collapse |

## Final Drafts

| File | Chars | Status |
|---|---|---|
| `PASTE_READY_global.txt` | 3,550 | ✅ Ready |
| `PASTE_READY_QHmk.txt` | 4,475 | ✅ Ready |
| `PASTE_READY_cA3o.txt` | 3,667 | ✅ Ready |
| `PASTE_READY_JxzD.txt` | 3,529 | ✅ Ready |
| `PASTE_READY_Pd1v.txt` | 4,969 | ✅ Ready |

## Workflow Progress

- [x] Phase 0: Initialize
- [x] Phase 1: Validate inputs, normalize reviews
- [x] Phase 2: Atomize concerns → ISSUE_BOARD.md (18 issues)
- [x] Phase 3: Strategy plan → STRATEGY_PLAN.md
- [x] Phase 3.5: Evidence sprint — all experiments complete
- [x] Phase 4: Draft rebuttal — all 5 drafts written with real data
- [x] Phase 4.5: Comprehensive analysis → REBUTTAL_ANALYSIS_CN.md
- [x] Phase 5: Safety validation — all 6 checks PASS
- [x] Phase 7: Finalize — PASTE_READY versions + REBUTTAL_FINAL_SUMMARY.md
- [ ] Submit to OpenReview/CMT (user action)
