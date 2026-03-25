# Rebuttal State

- **Paper**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training
- **Venue**: ICML 2026
- **Submission ID**: 13383
- **Current Phase**: Phase 3 (Strategy Plan complete, awaiting character limit for drafting)
- **Response Mode**: TEXT_ONLY
- **Character Limit**: TBD (user to confirm)

## Reviewers

| ID | Score | Stance | Key Concern |
|---|---|---|---|
| Pd1v | 3 (Weak Reject) | negative | Narrow benchmarks, weak baselines, no theory |
| cA3o | 4 (Weak Accept) | swing-positive | SubM vs RapTB role, hyperparameter sensitivity, generalization |
| JxzD | 4 (Weak Accept) | swing-positive | SubTB termination mechanism, longer sequences, LLM motivation |
| QHmk | 2 (Reject) | negative | RL contextualization, missing baselines (PPO/GRPO/TBA), simpler baseline |

## Status

- [x] Phase 0: Initialize
- [x] Phase 1: Validate inputs, normalize reviews
- [x] Phase 2: Atomize concerns → ISSUE_BOARD.md
- [x] Phase 3: Strategy plan → STRATEGY_PLAN.md
- [ ] Phase 3.5: Evidence sprint (AvgPrefixTB baseline ready in code, sweep scripts ready)
- [ ] Phase 4: Draft rebuttal (blocked on character limit)
- [ ] Phase 5: Safety validation
- [ ] Phase 6: Stress test
- [ ] Phase 7: Finalize
