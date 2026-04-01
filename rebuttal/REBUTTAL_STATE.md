# Rebuttal State

- **Paper**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training
- **Venue**: ICML 2026
- **Submission ID**: 13383
- **Current Phase**: Phase 7.1 — Literature-grounded revision complete
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

## Final Drafts — v3 Literature-Grounded

| File | Status | Key Updates in v3 |
|---|---|---|
| `PASTE_READY_global.txt` | ✅ Ready | +zero-at-optimum, +TBA "partial energy" quote, +SubTB=PCL |
| `PASTE_READY_QHmk_v2.md` | ✅ Ready | +two-angle optimality, +TBA future work quote, +FL-GFN/LED-GFN comparison |
| `PASTE_READY_cA3o_v3.txt` | ✅ Ready | +GAE/FL-GFN/LED-GFN/TBA literature, +Tiapkin V*(s) connection |
| `PASTE_READY_JxzD_v3.txt` | ✅ Ready | +Deleu/Hu derivation chain, +SubTB-PCL equivalence, +FL-GFN/LED-GFN |
| `PASTE_READY_Pd1v_v3.md` | ✅ Ready | +two-angle optimality, +TBA positioning, +SubTB fixed-point parallel |
| `REBUTTAL_DRAFT_QHmk_v2_rich.md` | ✅ Rich | Full literature: 15+ citations with provenance |
| `REBUTTAL_DRAFT_JxzD_v3_rich.md` | ✅ Rich | Full literature: fixed-point comparison table, SubTB mechanism chain |
| `REBUTTAL_DRAFT_cA3o_v2_rich.md` | ✅ Rich | Full literature: GAE/FL-GFN/LED-GFN/RUDDER/HCA/PBRS/TBA |
| `REBUTTAL_SUMMARY_CN.md` | ✅ Ready | v2: 完整文献引用表，三核心论点文献支撑 |

## Key Literature Citations Added

| Citation | Used For |
|---|---|
| Malkin et al. (2022) NeurIPS — TB Theorem 1 | Fixed-point exactness |
| Tiapkin et al. (2024) AISTATS Oral — Theorem 1 | GFlowNet = MaxEnt RL |
| Deleu et al. (2024) UAI — Prop 3.2 | SubTB = PCL equivalence |
| Deleu et al. (2022) UAI | F(s) = R(s^⊤)/P_F(⊤|s) |
| Hu et al. (2024) ICLR Oral | LLM-GFlowNet SubTB formulation |
| Madan et al. (2023) ICML | SubTB telescoping, same fixed point |
| Pan et al. (2023) ICML — FL-GFN | Intermediate energy, Assumption 4.1 |
| Jang et al. (2024) ICLR Oral — LED-GFN | Learned potential, regularizer |
| Bartoldson et al. (2025) NeurIPS — TBA | TB variance + "partial energy" future work |
| Schulman et al. (2015) ICLR 2016 — GAE | γλ structural analogy |
| Arjona-Medina et al. (2019) NeurIPS — RUDDER | Return decomposition comparison |
| Harutyunyan et al. (2019) NeurIPS — HCA | Hindsight credit comparison |
| da Silva et al. (2024) NeurIPS | Control variates (unbiased) |

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
- [x] Phase 7.1: Literature-grounded revision — 15+ citations, 3 core arguments strengthened
- [ ] Submit to OpenReview/CMT (user action)
