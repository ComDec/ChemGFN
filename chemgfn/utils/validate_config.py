#!/usr/bin/env python
"""
Configuration Validation Script

Validates Hydra configurations to ensure all parameters are correct and consistent.
Run this before training to catch configuration errors early.

Usage:
    python validate_config.py                    # Validate default config
    python validate_config.py experiment=SMILES_basic/SMILES_cfg_TB   # Validate a specific experiment
    python validate_config.py --cfg job          # Show full composed config
"""

import sys

import hydra
from omegaconf import DictConfig, OmegaConf


def check_penalty_values(cfg: DictConfig) -> list[str]:
    """Check that penalty values are reasonable."""
    issues = []

    # Check illegal_vocab_penalty in constraint_config
    constraint_penalty = cfg.model.constraint_config.get("illegal_vocab_penalty", -50)
    if constraint_penalty > 0:
        issues.append(
            f"WARN: constraint_config.illegal_vocab_penalty ({constraint_penalty}) should be negative"
        )
    if constraint_penalty < -200:
        issues.append(
            f"WARN: constraint_config.illegal_vocab_penalty ({constraint_penalty}) is very negative (< -200)"
        )

    # Check illegal_vocab_penalty in reward
    reward_penalty = cfg.model.reward.get("illegal_vocab_penalty", -50)
    if reward_penalty > 0:
        issues.append(f"WARN: reward.illegal_vocab_penalty ({reward_penalty}) should be negative")
    if reward_penalty < -200:
        issues.append(
            f"WARN: reward.illegal_vocab_penalty ({reward_penalty}) is very negative (< -200)"
        )

    # Check grammar_disagree_penalty
    disagree_penalty = cfg.model.reward.get("grammar_disagree_penalty", -99)
    if disagree_penalty > 0:
        issues.append(
            f"WARN: reward.grammar_disagree_penalty ({disagree_penalty}) should be negative"
        )

    if not issues:
        print("OK: Penalty values:")
        print(f"  - constraint_config.illegal_vocab_penalty: {constraint_penalty}")
        print(f"  - reward.illegal_vocab_penalty: {reward_penalty}")
        print(f"  - reward.grammar_disagree_penalty: {disagree_penalty}")

    return issues


def check_sequence_lengths(cfg: DictConfig) -> list[str]:
    """Check that sequence length constraints are valid."""
    issues = []

    min_len = cfg.model.constraint_config.min_sentence_len
    max_len = cfg.model.constraint_config.max_sentence_len

    if min_len >= max_len:
        issues.append(
            f"ERROR: min_sentence_len ({min_len}) must be < max_sentence_len ({max_len})"
        )

    if min_len < 0:
        issues.append(f"ERROR: min_sentence_len ({min_len}) must be >= 0")

    if max_len > 100:
        issues.append(f"WARN: max_sentence_len ({max_len}) is very large (> 100)")

    if not issues:
        print(f"\nOK: Sequence lengths: min={min_len}, max={max_len}")

    return issues


def check_temperature_values(cfg: DictConfig) -> list[str]:
    """Check that temperature values are valid."""
    issues = []

    temp_start = cfg.model.reward_config.reward_temp_start
    temp_end = cfg.model.reward_config.reward_temp_end

    if temp_start <= 0:
        issues.append(f"ERROR: reward_temp_start ({temp_start}) must be > 0")
    if temp_end <= 0:
        issues.append(f"ERROR: reward_temp_end ({temp_end}) must be > 0")

    if temp_end > temp_start:
        issues.append(
            f"WARN: reward_temp_end ({temp_end}) > reward_temp_start ({temp_start}) - unusual but allowed"
        )

    # Check forward policy temperatures
    pf_high = cfg.model.training_mixed_config.pf_temp_high
    pf_low = cfg.model.training_mixed_config.pf_temp_low

    if pf_high <= 0 or pf_low <= 0:
        issues.append(f"ERROR: pf_temp values must be > 0 (high={pf_high}, low={pf_low})")

    if pf_low > pf_high:
        issues.append(f"WARN: pf_temp_low ({pf_low}) > pf_temp_high ({pf_high}) - unusual")

    if not issues:
        print("\nOK: Temperatures:")
        print(f"  - Reward: {temp_start} -> {temp_end}")
        print(f"  - Forward policy: {pf_low} to {pf_high}")

    return issues


def check_buffer_config(cfg: DictConfig) -> list[str]:
    """Check buffer-related parameters."""
    issues = []

    buffer_ratio = cfg.model.training_mixed_config.buffer_mixture_ratio
    if not (0 <= buffer_ratio <= 1):
        issues.append(f"ERROR: buffer_mixture_ratio ({buffer_ratio}) must be in [0, 1]")

    sim_tolerance = cfg.model.reward_buffer.sim_tolerance
    if not (0 <= sim_tolerance <= 1):
        issues.append(f"ERROR: sim_tolerance ({sim_tolerance}) must be in [0, 1]")

    buffer_size = cfg.model.reward_buffer.buffer_size
    if buffer_size <= 0:
        issues.append(f"ERROR: buffer_size ({buffer_size}) must be > 0")

    if not issues:
        print("\nOK: Buffer configuration:")
        print(f"  - buffer_size: {buffer_size}")
        print(f"  - sim_tolerance: {sim_tolerance}")
        print(f"  - buffer_mixture_ratio: {buffer_ratio}")

    return issues


def check_horizon_values(cfg: DictConfig) -> list[str]:
    """Check that horizon values are reasonable."""
    issues = []

    max_epochs = cfg.trainer.max_epochs
    limit_train = cfg.trainer.get("limit_train_batches", None)

    # Estimate total steps (rough)
    if isinstance(limit_train, int):
        estimated_steps = max_epochs * limit_train
    else:
        estimated_steps = max_epochs * 1000  # Rough estimate

    horizons = {
        "reward_temp_horizon": cfg.model.reward_config.reward_temp_horizon,
        "scaling_factor_horizon": cfg.model.reward_config.scaling_factor_horizon,
        "buffer_sample_steps": cfg.model.training_mixed_config.buffer_sample_steps,
        "balance_horizon": cfg.model.training_mixed_config.balance_horizon,
    }

    print(f"\nOK: Horizons (estimated total steps: ~{estimated_steps}):")
    for name, value in horizons.items():
        status = "OK" if value <= estimated_steps * 2 else "WARN"
        print(f"  {status} {name}: {value}")
        if value > estimated_steps * 5:
            issues.append(f"WARN: {name} ({value}) >> estimated total steps ({estimated_steps})")

    return issues


def check_deprecated_params(cfg: DictConfig) -> list[str]:
    """Check for deprecated parameter names."""
    issues = []

    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    cfg_str = str(cfg_dict)

    deprecated = {
        "naughty_vocab_alpha": "illegal_vocab_penalty",
        "invalid_vocab_alpha": "illegal_vocab_penalty",
        "vocab_naughty_mask": "vocab_invalid_mask",
        "disagree_alpha": "grammar_disagree_penalty",
    }

    found_deprecated = []
    for old_name, new_name in deprecated.items():
        if old_name in cfg_str:
            found_deprecated.append((old_name, new_name))

    if found_deprecated:
        print("\nWARN: Deprecated parameters found:")
        for old, new in found_deprecated:
            print(f"  - '{old}' -> please use '{new}'")
            issues.append(f"Deprecated parameter: {old}")
    else:
        print("\nOK: No deprecated parameters found")

    return issues


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main validation function."""

    print("\n" + "=" * 70)
    print(" " * 20 + "CONFIGURATION VALIDATION")
    print("=" * 70)

    print(f"\nExperiment: {cfg.get('exp_name', 'default')}")
    print(f"Model: {cfg.model._target_}")
    print(f"Data: {cfg.data._target_}")

    all_issues = []

    # Run all checks
    all_issues.extend(check_penalty_values(cfg))
    all_issues.extend(check_sequence_lengths(cfg))
    all_issues.extend(check_temperature_values(cfg))
    all_issues.extend(check_buffer_config(cfg))
    all_issues.extend(check_horizon_values(cfg))
    all_issues.extend(check_deprecated_params(cfg))

    # Summary
    print("\n" + "=" * 70)
    if not all_issues:
        print("OK: ALL CHECKS PASSED - Configuration is valid!")
        print("=" * 70)
        return 0
    else:
        print(f"WARN: FOUND {len(all_issues)} ISSUE(S):")
        print("=" * 70)
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        print("\n" + "=" * 70)
        print("Please fix the issues above before training.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
