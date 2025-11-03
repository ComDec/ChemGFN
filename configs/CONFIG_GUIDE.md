# Configuration Guide

## Configuration File Structure Overview

This project uses Hydra for configuration management, with a hierarchical structure for easy reuse and override.

### Directory Structure

```
configs/
├── model/              # Model configurations (all model-related parameters)
│   ├── _template.yaml  # Configuration template (shows all available parameters)
│   ├── llama3_expr24.yaml
│   └── llama3_smiles_opt.yaml
├── experiment/         # Experiment configurations (override specific parameters)
│   ├── debug.yaml
│   └── debug_expr24.yaml
├── data/              # Data configurations
├── trainer/           # Trainer configurations
└── ...
```

## Important Variable Descriptions

### ✅ Unified Variable Names

The following variables have been refactored for clearer naming:

| Variable Name | Old Variable Name | Description | Default Value |
|---------|---------|------|--------|
| `illegal_vocab_penalty` | `naughty_vocab_alpha`, `invalid_vocab_alpha` | Penalty value for illegal vocabulary tokens | -50 |
| `grammar_disagree_penalty` | `disagree_alpha` | Penalty value for grammar-inconsistent tokens | -99 |

### Configuration Locations

1. **`illegal_vocab_penalty`** is configured in two places:
   - `model.constraint_config.illegal_vocab_penalty`: Used for generation-time constraints
   - `model.reward.illegal_vocab_penalty`: Used for reward computation

2. **`grammar_disagree_penalty`** is configured in one place:
   - `model.reward.grammar_disagree_penalty`: Used in reward computation to penalize grammar violations

## Configuration File Details

### 1. Model Configuration (`configs/model/*.yaml`)

Model configuration contains all model-related parameters and is the core of the configuration.

#### Key Configuration Sections

```yaml
# 1. LoRA Configuration
lora_config:
  r: 16                    # LoRA rank (controls parameter count)
  lora_alpha: 16           # LoRA scaling factor
  lora_dropout: 0.1        # Dropout probability

# 2. Base Model
net_config:
  pretrained_model_name_or_path: "meta-llama/Llama-3.2-1B"

# 3. Reward Function Configuration
reward:
  _target_: chemgfn.models.reward.Reference_Target_Score_Positive_Mixed_Invalid_Mask
  invalid_start_ratio: 0.2           # Invalid penalty multiplier at sequence start
  invalid_end_ratio: 1.2             # Invalid penalty multiplier at sequence end
  illegal_vocab_penalty: -50         # Illegal vocabulary penalty
  grammar_disagree_penalty: -99      # Grammar violation penalty

  sentence_validator:                # Sequence validator
    _target_: chemgfn.models.reward.RDKitValidator
    scorer: "logP"                   # Optimization target: logP, QED, SA, etc.
    backend: "pa"                    # Backend: rdkit or pa

# 4. Reward Scheduling Configuration
reward_config:
  reward_temp_start: 2.0             # Starting reward temperature
  reward_temp_end: 0.8               # Ending reward temperature
  reward_temp_horizon: 50000         # Steps to reach ending temperature

  scaling_factor_start: 50           # Starting scaling factor
  scaling_factor_end: 100            # Ending scaling factor
  scaling_factor_horizon: 5000       # Steps to reach ending value

# 5. Generation Constraint Configuration
constraint_config:
  min_sentence_len: 2                # Minimum generation length
  max_sentence_len: 10               # Maximum generation length
  grammar_path: ${paths.assets_dir}/SMILES_grammars/generic.ebnf
  apply_grammar: true                # Whether to apply grammar constraints
  processor_type: "prefix"           # Grammar processor type
  legal_tokens: ${paths.assets_dir}/token_list/SMILES/...
  illegal_vocab_penalty: -50        # Illegal vocabulary penalty (used during generation)
  parse_mode: "limited"              # Parse mode

# 6. Training Configuration
training_mixed_config:
  subtb_lambda: 1.0                  # SubTB lambda (length decay weight)
  pf_temp_high: 2.0                  # Forward policy high temperature
  pf_temp_low: 0.5                   # Forward policy low temperature
  pf_temp_prob: 0.666                # Probability of using temperature sampling

  use_buffer_prob: 0.25              # Probability of using replay buffer
  n_samples: 8                       # Number of sequences generated per batch

  use_buffer_sample_start_prob: 0.8  # Buffer sample mixing starting probability
  use_buffer_sample_end_prob: 0.2    # Buffer sample mixing ending probability
  buffer_sample_steps: 20000         # Steps to reach ending probability
  buffer_mixture_ratio: 0.5          # Buffer sample replacement ratio

  skip_baseline_sampling: true       # Skip baseline sampling

  balance_start: 0.0                 # Token-level SubTB starting balance factor
  balance_end: 1.0                   # Token-level SubTB ending balance factor
  balance_horizon: 50000             # Steps to reach ending balance

  opt_task: false                    # Whether this is an optimization task (sidechain)
```

### 2. Experiment Configuration (`configs/experiment/*.yaml`)

Experiment configuration overrides specific parameters in model configuration to avoid duplication.

#### Best Practices

```yaml
# ❌ Wrong: Redefining all parameters
model:
  reward:
    invalid_start_ratio: 0.2
    invalid_end_ratio: 1.2
    illegal_vocab_penalty: -80
    grammar_disagree_penalty: -120
    sentence_validator: ...

# ✅ Correct: Only override parameters that need to change
model:
  reward:
    illegal_vocab_penalty: -80
    grammar_disagree_penalty: -120
```

### 3. Parameter Validation Checklist

After modifying configuration, ensure:

#### ✅ Parameters to Check

1. **Penalty Parameter Consistency**
   ```yaml
   # These two values should be coordinated (constraint_config's value typically more negative)
   model.constraint_config.illegal_vocab_penalty: -50
   model.reward.illegal_vocab_penalty: -80
   ```

2. **Sequence Length Sanity**
   ```yaml
   # min < max
   constraint_config.min_sentence_len: 2
   constraint_config.max_sentence_len: 10
   ```

3. **Scheduler Horizon Sanity**
   ```yaml
   # horizon should be < total training steps
   reward_config.reward_temp_horizon: 50000
   # total steps ≈ max_epochs * batches_per_epoch
   ```

4. **Buffer Configuration Sanity**
   ```yaml
   # sim_tolerance range [0, 1]
   reward_buffer.sim_tolerance: 0.25  # ✅
   reward_buffer.sim_tolerance: 2.0   # ❌ Too large

   # buffer_mixture_ratio range [0, 1]
   training_mixed_config.buffer_mixture_ratio: 0.5  # ✅
   ```

5. **Temperature Parameter Sanity**
   ```yaml
   # Temperature should be > 0
   reward_config.reward_temp_start: 2.0   # ✅
   reward_config.reward_temp_end: 0.8     # ✅
   reward_config.reward_temp_end: 0.0     # ❌ Will cause numerical issues
   ```

## Common Configuration Patterns

### Pattern 1: Quick Debugging

```yaml
trainer:
  max_epochs: 2
  limit_train_batches: 0.1
  limit_val_batches: 0.1

model:
  training_mixed_config:
    n_samples: 4  # Reduce samples to speed up
```

### Pattern 2: High Exploration Training

```yaml
# High temperature, high exploration
model:
  reward_config:
    reward_temp_start: 3.0
    reward_temp_end: 1.5

  training_mixed_config:
    pf_temp_high: 2.5
    pf_temp_low: 1.0
```

### Pattern 3: Exploitation Training (Using Buffer)

```yaml
# Heavily use high-quality samples from replay buffer
model:
  training_mixed_config:
    use_buffer_prob: 0.5
    buffer_mixture_ratio: 0.7
    use_buffer_sample_start_prob: 0.9
```

## Configuration Validation Script

Create a simple validation script:

```python
# validate_config.py
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def validate(cfg: DictConfig):
    print("="*60)
    print("Configuration Validation")
    print("="*60)

    # Check penalty values
    if hasattr(cfg.model, 'constraint_config') and hasattr(cfg.model, 'reward'):
        constraint_penalty = cfg.model.constraint_config.get('illegal_vocab_penalty', None)
        reward_penalty = cfg.model.reward.get('illegal_vocab_penalty', None)
        if constraint_penalty and reward_penalty:
            print(f"✓ Constraint penalty: {constraint_penalty}")
            print(f"✓ Reward penalty: {reward_penalty}")

    # Check sequence lengths
    if hasattr(cfg.model, 'constraint_config'):
        min_len = cfg.model.constraint_config.get('min_sentence_len')
        max_len = cfg.model.constraint_config.get('max_sentence_len')
        if min_len and max_len:
            assert min_len < max_len, "min_sentence_len must be < max_sentence_len"
            print(f"✓ Sequence length: [{min_len}, {max_len}]")

    # Check temperatures
    if hasattr(cfg.model, 'reward_config'):
        temp_start = cfg.model.reward_config.get('reward_temp_start')
        temp_end = cfg.model.reward_config.get('reward_temp_end')
        if temp_start and temp_end:
            assert temp_start > 0 and temp_end > 0, "Temperatures must be > 0"
            print(f"✓ Reward temperature: {temp_start} → {temp_end}")

    print("\n✓ All validations passed!")


if __name__ == "__main__":
    validate()
```

Usage:

```bash
# Validate default configuration
python validate_config.py

# Validate specific experiment configuration
python validate_config.py experiment=debug

# Show full configuration
python validate_config.py --cfg job
```

## Migration Guide: Old Config → New Config

If you have old configuration files, use the following mapping to update:

| Old Parameter | New Parameter | Location |
|-------|--------|------|
| `naughty_vocab_alpha` | `illegal_vocab_penalty` | `constraint_config` or `reward` |
| `invalid_vocab_alpha` | `illegal_vocab_penalty` | `constraint_config` or `reward` |
| `vocab_naughty_mask` | `vocab_invalid_mask` | Function parameters |
| `disagree_alpha` | `grammar_disagree_penalty` | `reward` |

## Troubleshooting

### Issue 1: `KeyError: 'illegal_vocab_penalty'`

**Cause**: Old configuration files used `naughty_vocab_alpha` or `invalid_vocab_alpha`

**Solution**:
```yaml
# Old configuration
constraint_config:
  naughty_vocab_alpha: -50
  # OR
  invalid_vocab_alpha: -50

# New configuration
constraint_config:
  illegal_vocab_penalty: -50
```

### Issue 2: Reward Computation Produces NaN

**Possible causes**:
- Temperature set to 0
- Penalty value too small (e.g. -1000)

**Solution**:
```yaml
# Ensure temperature > 0
reward_config:
  reward_temp_end: 0.8  # ✅ Don't set to 0

# Ensure penalty in reasonable range
reward:
  illegal_vocab_penalty: -80    # ✅ Reasonable
  illegal_vocab_penalty: -1000  # ❌ Too small
```

### Issue 3: Low GPU Utilization

**Check**: Buffer and sampling configuration

```yaml
# Increase batch size
training_mixed_config:
  n_samples: 16  # Increase from 8 to 16

# Adjust number of workers
data:
  num_workers: 16  # Increase data loading parallelism
```

## Summary

✅ **Configuration Best Practices**:
1. Use `_template.yaml` as reference
2. Define complete configuration in `model/*.yaml`
3. Override only necessary parameters in `experiment/*.yaml`
4. Use validation script to check configuration
5. Consistently use new parameter names (`illegal_vocab_penalty`, `grammar_disagree_penalty`)

📝 **Remember**:
- `illegal_vocab_penalty`: Controls illegal vocabulary penalty (negative value, typically -50 to -100)
- `grammar_disagree_penalty`: Controls grammar violation penalty (negative value, typically -80 to -120)
- Grammar penalty is typically more negative (stricter) than illegal vocab penalty

🚀 **Quick Start**:
```bash
# 1. View template to understand all parameters
cat configs/model/_template.yaml

# 2. Run debug experiment
python chemgfn/train.py experiment=debug

# 3. Validate configuration
python validate_config.py experiment=debug
```
