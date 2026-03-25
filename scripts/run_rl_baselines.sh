#!/usr/bin/env bash
# Train PPO and GRPO baselines for VarExpr24 and evaluate them.
# Usage: bash scripts/run_rl_baselines.sh [GPU_ID]
set -euo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
GPU=${1:-0}
OUTPUT_ROOT=./logs/rl_baselines
SEED=42

export CUDA_VISIBLE_DEVICES=$GPU
cd "$(dirname "$0")/.."

echo "=== Training GRPO (GPU $GPU) ==="
$PYTHON scripts/train_grpo_expr24.py \
    --output_dir "$OUTPUT_ROOT/grpo_expr24" \
    --max_steps 5000 \
    --seed $SEED \
    --num_generations 32 \
    --beta 0.05 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 9 \
    --wandb_project ChemGFN

echo "=== Training PPO (GPU $GPU) ==="
$PYTHON scripts/train_ppo_expr24.py \
    --output_dir "$OUTPUT_ROOT/ppo_expr24" \
    --max_steps 5000 \
    --seed $SEED \
    --n_samples 32 \
    --ppo_epochs 4 \
    --clip_eps 0.2 \
    --kl_coeff 0.05 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --wandb_project ChemGFN

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

echo "=== All done ==="
