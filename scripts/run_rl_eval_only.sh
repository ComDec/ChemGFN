#!/usr/bin/env bash
# Run eval for trained PPO and GRPO models.
# Usage: bash scripts/run_rl_eval_only.sh [GPU]
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
GPU=${1:-0}
OUTPUT_ROOT=./logs/rl_baselines

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$GPU

echo "=== Evaluating GRPO ==="
$PYTHON scripts/eval_rl_baseline.py \
    --model_path "$OUTPUT_ROOT/grpo_expr24/final" \
    --exp_name VarExpr24_GRPO \
    --n_samples 6400 \
    --batch_size 32 \
    --test_repeats 3 \
    --output_dir "$OUTPUT_ROOT/eval_grpo" \
    --wandb_project ChemGFN

echo "=== Evaluating PPO ==="
$PYTHON scripts/eval_rl_baseline.py \
    --model_path "$OUTPUT_ROOT/ppo_expr24/final" \
    --exp_name VarExpr24_PPO \
    --n_samples 6400 \
    --batch_size 32 \
    --test_repeats 3 \
    --output_dir "$OUTPUT_ROOT/eval_ppo" \
    --wandb_project ChemGFN

echo "=== Done ==="
