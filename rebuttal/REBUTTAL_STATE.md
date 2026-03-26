# Rebuttal State

- **Paper**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training
- **Venue**: ICML 2026
- **Submission ID**: 13383
- **Current Phase**: Phase 3.5 (Evidence Sprint — experiments in progress)
- **Response Mode**: TEXT_ONLY
- **Character Limit**: 5000 per response, multiple submissions + revised PDF allowed

## Reviewers

| ID | Score | Stance | Key Concern |
|---|---|---|---|
| Pd1v | 3 (Weak Reject) | negative | Narrow benchmarks, weak baselines, no theory |
| cA3o | 4 (Weak Accept) | swing-positive | SubM vs RapTB role, hyperparameter sensitivity, generalization |
| JxzD | 4 (Weak Accept) | swing-positive | SubTB termination mechanism, longer sequences, LLM motivation |
| QHmk | 2 (Reject) | negative | RL contextualization, missing baselines (PPO/GRPO/TBA), simpler baseline |

## Evidence Status

### Completed Experiments

| Experiment | Answers | Status | Results File |
|---|---|---|---|
| β×ρ sweep (9 cells, 5000 steps) | cA3o-C2 | **DONE** | `SWEEP_RESULTS_FINAL.md` |
| η sweep (3 runs, 1750 steps) | cA3o-C2 | **DONE** | `SWEEP_RESULTS_FINAL.md` |
| k_min ablation (3 runs, 1750 steps) | cA3o-C2 | **DONE** | `SWEEP_RESULTS_FINAL.md` |
| PPO/GRPO baselines (Expr24) | QHmk-C2 | **DONE** (external) | RL eval completed |

### In Progress

| Experiment | Answers | Status | ETA |
|---|---|---|---|
| Paper-exact anchor (3 seeds, 5000 steps) | Sweep calibration | **55% trained** | ~2h |
| AvgPrefixTB baseline (Expr24 RP+Oracle) | QHmk-C6 | **User running** | TBD |

### Not Started

| Experiment | Answers | Priority | Decision |
|---|---|---|---|
| SMILES sweep point | Cross-task robustness | P2 | Optional |
| SynCodon-Oracle biology task | cA3o-C3, JxzD-C3 | P3 | Future work |

## Rebuttal Drafts

| File | Reviewer | Chars | Status |
|---|---|---|---|
| `PASTE_READY_global.txt` | All | 2664 | Has sweep placeholders |
| `PASTE_READY_QHmk.txt` | QHmk | 3769 | Has AvgPrefixTB placeholder |
| `PASTE_READY_cA3o.txt` | cA3o | 3619 | **Ready to update with sweep results** |
| `PASTE_READY_JxzD.txt` | JxzD | 4620 | Complete (no experiments needed) |
| `PASTE_READY_Pd1v.txt` | Pd1v | 3675 | Complete |

## Workflow Progress

- [x] Phase 0: Initialize
- [x] Phase 1: Validate inputs, normalize reviews
- [x] Phase 2: Atomize concerns → ISSUE_BOARD.md
- [x] Phase 3: Strategy plan → STRATEGY_PLAN.md
- [~] Phase 3.5: Evidence sprint
  - [x] Sweep: β×ρ grid (9/9 complete with Table 3 metrics)
  - [x] Sweep: η sweep (3/3 complete)
  - [x] Sweep: k_min ablation (3/3 complete)
  - [ ] Anchor: paper-exact config 3 seeds (training ~55%)
  - [ ] AvgPrefixTB baseline (user running)
- [ ] Phase 4: Draft rebuttal (update cA3o draft with sweep results)
- [ ] Phase 5: Safety validation
- [ ] Phase 6: Stress test
- [ ] Phase 7: Finalize
