# ChemGFN Test Suite Documentation

This document describes the complete test suite for the ChemGFN project, including test coverage, execution methods, and design principles.

## 📋 Test File Overview

### 1. `test_gfn_utils.py` - GFlowNet Utility Functions Tests
Tests core utility functions in `chemgfn/utils/gfn_utils.py`:

**Test Coverage:**
- ✅ Token mask preparation (`prepare_token_mask`)
- ✅ Diversity calculation (`calculate_diversity`)
- ✅ Generation with termination probabilities (`generate_and_return_termination_logprob`)
- ✅ Modified SubTB loss function (`modified_subtb_loss`)
- ✅ Termination value extraction (`get_termination_vals`)
- ✅ Replay buffer (`ReplayBuffer`)
- ✅ LoRA adapter switching (`lora_to_base`, `base_to_lora`)

**Test Classes:**
```python
TestPrepareTokenMask         # Token mask preparation
TestCalculateDiversity       # Diversity metrics
TestModifiedSubTBLoss       # SubTB loss function
TestGetTerminationVals      # Termination value extraction
TestReplayBuffer            # Replay buffer operations
TestLoRAUtilities           # LoRA utilities
TestGenerateAndReturnTerminationLogProb  # Generation function
```

### 2. `test_reward.py` - Reward Computation Tests
Tests reward models and validators in `chemgfn/models/reward.py`:

**Test Coverage:**
- ✅ Utility functions (penalty application, tensor processing, etc.)
- ✅ Fast scoring function (`score_fast`)
- ✅ Sentence validator base class (`SentenceValidator`)
- ✅ Frozen model rewards (`FrozenModelSentenceGivenPrompt`)
- ✅ Mixed reward model (`Reference_Target_Score_Positive_Mixed_Invalid_Mask`)
- ✅ Base model context manager (`use_base_model`)

**Test Classes:**
```python
TestUtilityFunctions         # Utility functions
TestScoreFast               # Fast scoring
TestSentenceValidator       # Validators
TestFrozenModelSentenceGivenPrompt  # Frozen model rewards
TestReferenceTargetScorePositiveMixedInvalidMask  # Mixed rewards
TestUseBaseModel            # LoRA switching
TestRewardIntegration       # End-to-end integration
```

### 3. `test_loss.py` - Loss Function Detailed Tests
Comprehensive tests focused on `modified_subtb_loss` function:

**Test Coverage:**
- ✅ Basic loss computation (scalar output, non-negativity, no NaN/Inf)
- ✅ SubTB Lambda parameter behavior
- ✅ Balance parameter (token coverage balancing)
- ✅ Early termination handling
- ✅ Gradient flow and backpropagation
- ✅ Numerical stability (large/small/mixed/zero values)
- ✅ Edge cases (minimum sequence length, single batch, long sequences)

**Test Classes:**
```python
TestBasicLossComputation     # Basic computation
TestSubTBLambda             # Lambda parameter
TestBalanceParameter        # Balance parameter
TestEarlyTermination        # Early termination
TestGradientFlow            # Gradient flow
TestNumericalStability      # Numerical stability
TestEdgeCases               # Edge cases
```

### 4. `test_training_flow.py` - Training Flow Integration Tests
Tests complete training pipeline:

**Test Coverage:**
- ✅ Training step (`training_step`)
- ✅ Forward pass (`forward`)
- ✅ Loss computation and backpropagation
- ✅ Replay buffer integration
- ✅ Metric logging
- ✅ Temperature scheduling
- ✅ Scaling factor scheduling
- ✅ Buffer sampling probability scheduling
- ✅ Validation step
- ✅ Model configuration handling
- ✅ Error handling

**Test Classes:**
```python
TestTrainingStep            # Training step
TestForwardPass            # Forward pass
TestLossComputation        # Loss computation
TestReplayBufferIntegration  # Buffer integration
TestMetricLogging          # Metric logging
TestTemperatureScheduling  # Temperature scheduling
TestScalingFactorSchedule  # Scaling factor scheduling
TestBufferSamplingSchedule  # Buffer sampling scheduling
TestValidationStep         # Validation step
TestModelConfiguration     # Model configuration
TestErrorHandling          # Error handling
TestIntegrationScenarios   # Integration scenarios
```

## 🚀 Running Tests

### Run All Tests
```bash
# From project root
pytest tests/ -v

# Or use make command
make test
```

### Run Specific Test File
```bash
# Test GFN utility functions
pytest tests/test_gfn_utils.py -v

# Test reward module
pytest tests/test_reward.py -v

# Test loss function
pytest tests/test_loss.py -v

# Test training flow
pytest tests/test_training_flow.py -v
```

### Run Specific Test Class
```bash
# Test SubTB loss
pytest tests/test_loss.py::TestModifiedSubTBLoss -v

# Test replay buffer
pytest tests/test_gfn_utils.py::TestReplayBuffer -v
```

### Run Specific Test Method
```bash
# Test loss gradient flow
pytest tests/test_loss.py::TestGradientFlow::test_gradients_exist -v
```

### With Coverage Report
```bash
# Generate coverage report
pytest tests/ --cov=chemgfn --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

### Parallel Execution (Faster)
```bash
# Use pytest-xdist for parallel execution
pytest tests/ -n auto -v
```

### Run Only Fast Tests (Skip Slow Ones)
```bash
pytest tests/ -v -m "not slow"
```

## 📊 Test Coverage Statistics

| Module | Test File | Test Classes | Test Methods | Coverage Scope |
|------|---------|---------|-----------|---------|
| gfn_utils.py | test_gfn_utils.py | 8 | 40+ | Core functions, buffers |
| reward.py | test_reward.py | 7 | 35+ | Reward computation, validators |
| loss (SubTB) | test_loss.py | 7 | 45+ | Loss function comprehensive |
| training flow | test_training_flow.py | 12 | 30+ | Training pipeline integration |

## 🎯 Test Design Principles

### 1. **Unit Tests First**
- Each function has independent tests
- Use mocks to isolate dependencies
- Test normal paths and edge cases

### 2. **Numerical Stability**
- Test large, small, zero value inputs
- Check for NaN and Inf
- Verify gradient computation

### 3. **Parameterized Testing**
- Test multiple batch sizes
- Test multiple sequence lengths
- Test different parameter combinations

### 4. **Integration Tests**
- End-to-end flow testing
- Inter-module interaction testing
- Real scenario simulation

### 5. **Gradient Checking**
- Verify gradients exist
- Verify gradients have no NaN
- Test second-order gradients (optional)

## 🔍 Debugging Failed Tests

### View Detailed Output
```bash
pytest tests/test_gfn_utils.py -v -s
```

### Enter Debugger on Failure
```bash
pytest tests/test_gfn_utils.py --pdb
```

### Run Only Last Failed Tests
```bash
pytest --lf
```

### Stop at First Failure
```bash
pytest -x
```

## 📝 Adding New Tests

### 1. Choose Appropriate Test File
- Utility functions → `test_gfn_utils.py`
- Reward related → `test_reward.py`
- Loss function → `test_loss.py`
- Training flow → `test_training_flow.py`

### 2. Create Test Class
```python
class TestMyNewFeature:
    """Test description."""

    @pytest.fixture
    def setup_data(self):
        """Create test data."""
        return ...

    def test_basic_functionality(self, setup_data):
        """Test basic behavior."""
        result = my_function(setup_data)
        assert result is not None

    def test_edge_case(self, setup_data):
        """Test edge case."""
        ...
```

### 3. Test Checklist
For each new feature, ensure you test:
- ✅ Normal input cases
- ✅ Boundary values (min, max)
- ✅ Invalid inputs (if applicable)
- ✅ Return type and shape
- ✅ Numerical correctness
- ✅ Gradient flow (if gradients needed)
- ✅ Numerical stability
- ✅ Integration with other components

## 🔧 Dependencies and Environment

### Test Dependencies
```bash
pip install pytest pytest-cov pytest-xdist
pip install transformers torch
```

### GPU Tests
Some tests require GPU (marked with `@RunIf(min_gpus=1)`):
```bash
# Run GPU tests
pytest tests/ -v -m gpu

# Skip GPU tests
pytest tests/ -v -m "not gpu"
```

## 📈 Continuous Integration

### GitHub Actions Configuration Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=chemgfn
```

## 🎓 Best Practices

1. **Keep Tests Independent**: Each test should run independently
2. **Use Fixtures**: Share test data using pytest fixtures
3. **Clear Test Names**: Test names should describe what is being tested
4. **Appropriate Assertions**: Use specific assertion messages
5. **Test Coverage**: Target 80%+ code coverage
6. **Fast Execution**: Unit tests should be fast (<1 second)
7. **Run Regularly**: Run tests before each commit
8. **Maintain Tests**: Update tests when code changes

## 🐛 Common Issues

### Q: Tests fail with module not found error
```bash
# Ensure package is installed
pip install -e .
```

### Q: GPU tests fail on CPU machine
```bash
# Skip GPU tests
pytest tests/ -v -m "not gpu"
```

### Q: Tests run slowly
```bash
# Run in parallel
pytest tests/ -n auto

# Run only fast tests
pytest tests/ -m "not slow"
```

### Q: How to debug a single test
```bash
# Use pdb debugger
pytest tests/test_loss.py::TestGradientFlow::test_gradients_exist --pdb -s
```

## 📚 Reference Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/notes/testing.html)
- [Transformers Testing Guide](https://huggingface.co/docs/transformers/testing)

## 📧 Contact

If you have test-related questions:
1. Check this documentation
2. Check comments in test code
3. Submit an Issue to the project repository

---

**Last Updated**: November 3, 2025
**Version**: 1.0
