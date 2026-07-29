# ChemGFN Tests

This document summarizes the test suite and how to run it.

## Scope

| File | Covers |
| --- | --- |
| `test_losses.py` | The GFlowNet objectives in `chemgfn/models/losses.py` (TB, SubTB, RootSubTBLogZ, RapTB, AvgPrefixTB). |
| `test_reward.py` | Reward construction in `chemgfn/models/reward.py`, including `score_fast` and the three reward mixers. |
| `test_validators.py` | The task validators in `chemgfn/models/validators.py` (Expr24, RDKit, CommonGen). |
| `test_amp.py` | The AMP oracle in `chemgfn/models/amp_oracle.py` and `AMPValidator`. |
| `test_replay_buffer.py` | `ReplayBuffer` and `ReplayBufferSubmodular` in `chemgfn/utils/replay_buffer.py`. |
| `test_gfn_utils.py` | Rollout and bookkeeping helpers in `chemgfn/utils/gfn_utils.py`. |
| `test_sequence_metrics.py` | Levenshtein diversity, novelty and top-k selection in `chemgfn/utils/sequence_metrics.py`. |

## Run Tests

Run the full suite:
```bash
pytest tests/ -v
```

Run a single file:
```bash
pytest tests/test_losses.py -v
```

Run a single class or test:
```bash
pytest tests/test_losses.py::TestRapTBLoss -v
pytest tests/test_losses.py::TestRapTBLoss::test_zero_aux_weight_recovers_terminal_tb -v
```

## Coverage

```bash
pytest tests/ --cov=chemgfn --cov-report=term --cov-report=html
```

## Markers

Declared in `pyproject.toml` and enforced with `--strict-markers`:

| Marker | Meaning |
| --- | --- |
| `slow` | Long-running tests. |
| `gpu` | Requires a CUDA device. |
| `gated_model` | Requires access to a gated Hugging Face model. |
| `requires_model` | Downloads pretrained weights from the Hugging Face Hub. |

Deselect a group with, for example:

```bash
pytest tests/ -v -m "not requires_model"
```

## Notes

- The suite runs on CPU and needs neither a GPU nor a Weights & Biases account.
- Tests that need a tokenizer or a language model obtain it through the helpers in
  `conftest.py`, which turn a missing network connection, a cold cache or a gated repository
  into a skip rather than a failure. The suite therefore stays green offline, with a larger
  number of skips.
- Install the test dependencies with `pip install -e ".[dev]"`.
