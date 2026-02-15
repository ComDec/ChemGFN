# ChemGFN Tests

This document summarizes the test suite and how to run it.

## Scope

The tests cover:

- GFlowNet utilities in `chemgfn/utils/gfn_utils.py`
- Reward computation and validators in `chemgfn/models/reward.py`
- Loss functions in `chemgfn/models/losses.py`
- Training flow integration in `chemgfn/train.py`
- Cosine restart scheduling in `chemgfn/models/gfn.py`

## Run Tests

Run the full suite:
```bash
pytest tests/ -v
```

Run a single file:
```bash
pytest tests/test_gfn_utils.py -v
```

Run a single class or test:
```bash
pytest tests/test_loss.py::TestModifiedSubTBLoss -v
pytest tests/test_loss.py::TestGradientFlow::test_gradients_exist -v
```

## Coverage

```bash
pytest tests/ --cov=chemgfn --cov-report=term --cov-report=html
```

## GPU and Slow Tests

Some tests require a GPU and are marked with `@RunIf(min_gpus=1)`.

```bash
pytest tests/ -v -m "not gpu"
```

## Notes

- Some tests use Hugging Face tokenizers and may download artifacts on first run.
- Install test dependencies as needed: `pytest`, `pytest-cov`, and `pytest-xdist`.
