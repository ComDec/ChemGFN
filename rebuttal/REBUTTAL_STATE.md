# Rebuttal State

- **Paper**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training
- **Venue**: ICML 2026
- **Submission ID**: 13383
- **Current Phase**: Phase 4→5 (Drafting complete, pending AMP results and safety validation)
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
| β×ρ sweep Expr24 (9 configs, 5000 steps) | cA3o-C2 | **DONE** | `SWEEP_RESULTS_FINAL.md` |
| β×ρ sweep SMILES (7/9 configs, ~2500 steps) | cA3o-C2 | **DONE** | `SMILES_SWEEP_RESULTS.md` |
| η sweep (3 runs, 1750 steps) | cA3o-C2 | **DONE** | `SWEEP_RESULTS_FINAL.md` |
| k_min ablation (3 runs, 1750 steps) | cA3o-C2 | **DONE** | `SWEEP_RESULTS_FINAL.md` |
| Paper-exact anchor (3 seeds, 5000 steps) | Sweep calibration | **DONE** | `ALL_EXPERIMENTS_STATUS.md` |
| PPO baseline (Expr24) | QHmk-C2 | **DONE** | Crash at step 50, Acc=0.003 |
| GRPO baseline (Expr24) | QHmk-C2 | **DONE** | Acc=0.002, length collapse |
| AvgPrefixTB (Expr24 RP+SubM+Oracle, SMILES) | QHmk-C6 | **DONE** | `avgprefix_tb_results.md` |
| 3B scale-up (SMILES, 4 methods) | Pd1v-C1, JxzD-Q3 | **DONE** | `ALL_EXPERIMENTS_STATUS.md` |
| AMP task (4 methods, 10K steps) | JxzD-C3, cA3o-C3, Pd1v-C1 | **IN PROGRESS** | `amp_results.md` (RapTB+SubM ~50%) |

## Rebuttal Drafts

| File | Reviewer | Chars | Placeholders |
|---|---|---|---|
| `PASTE_READY_global.txt` | All | ~3000 | **None** |
| `PASTE_READY_QHmk.txt` | QHmk | ~4500 | **None** |
| `PASTE_READY_cA3o.txt` | cA3o | ~4500 | **None** |
| `PASTE_READY_JxzD.txt` | JxzD | ~5400 | **None** (may need trim) |
| `PASTE_READY_Pd1v.txt` | Pd1v | ~5000 | **None** |

## Workflow Progress

- [x] Phase 0: Initialize
- [x] Phase 1: Validate inputs, normalize reviews
- [x] Phase 2: Atomize concerns → ISSUE_BOARD.md (18 issues)
- [x] Phase 3: Strategy plan → STRATEGY_PLAN.md
- [x] Phase 3.5: Evidence sprint — core experiments complete, AMP in progress
- [x] Phase 4: Draft rebuttal — all 5 drafts written with real data
- [~] Phase 4.5: Comprehensive analysis — `REBUTTAL_ANALYSIS_CN.md` written
- [ ] Phase 5: Safety validation (coverage, provenance, commitment, tone, consistency, limit)
- [ ] Phase 6: Stress test (Codex MCP)
- [ ] Phase 7: Finalize (two versions: PASTE_READY + rich draft)
