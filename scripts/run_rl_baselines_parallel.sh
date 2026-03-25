#!/usr/bin/env bash
# Train PPO and GRPO in parallel on separate GPUs, then evaluate sequentially.
set -uo pipefail

PYTHON=/data1/xw3763/miniforge3/envs/torch/bin/python
OUTPUT_ROOT=./logs/rl_baselines
SEED=42
GPU_GRPO=${1:-0}
GPU_PPO=${2:-2}

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_ROOT"

echo "$(date): Starting parallel training — GRPO on GPU $GPU_GRPO, PPO on GPU $GPU_PPO"

# === Train GRPO (background) ===
CUDA_VISIBLE_DEVICES=$GPU_GRPO $PYTHON scripts/train_grpo_expr24.py \
    --output_dir "$OUTPUT_ROOT/grpo_expr24" \
    --max_steps 5000 \
    --seed $SEED \
    --num_generations 32 \
    --beta 0.05 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 9 \
    --wandb_project ChemGFN \
    > "$OUTPUT_ROOT/grpo_train.log" 2>&1 &
PID_GRPO=$!
echo "  GRPO PID=$PID_GRPO (log: $OUTPUT_ROOT/grpo_train.log)"

# === Train PPO (background) ===
CUDA_VISIBLE_DEVICES=$GPU_PPO $PYTHON scripts/train_ppo_expr24.py \
    --output_dir "$OUTPUT_ROOT/ppo_expr24" \
    --max_steps 5000 \
    --seed $SEED \
    --n_samples 32 \
    --ppo_epochs 4 \
    --clip_eps 0.2 \
    --kl_coeff 0.05 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --wandb_project ChemGFN \
    > "$OUTPUT_ROOT/ppo_train.log" 2>&1 &
PID_PPO=$!
echo "  PPO  PID=$PID_PPO (log: $OUTPUT_ROOT/ppo_train.log)"

# === Wait for both ===
echo "$(date): Waiting for training to finish..."
FAIL=0
wait $PID_GRPO || { echo "GRPO training FAILED (exit $?)"; FAIL=1; }
echo "$(date): GRPO training done."
wait $PID_PPO || { echo "PPO training FAILED (exit $?)"; FAIL=1; }
echo "$(date): PPO training done."

if [ $FAIL -ne 0 ]; then
    echo "One or more training jobs failed. Check logs. Continuing to eval what we can..."
fi

# === Evaluate (sequential, reuse GPU 0) ===
if [ -d "$OUTPUT_ROOT/grpo_expr24/final" ]; then
    echo "$(date): Evaluating GRPO..."
    CUDA_VISIBLE_DEVICES=$GPU_GRPO $PYTHON scripts/eval_rl_baseline.py \
        --model_path "$OUTPUT_ROOT/grpo_expr24/final" \
        --exp_name VarExpr24_GRPO \
        --n_samples 6400 \
        --batch_size 32 \
        --test_repeats 3 \
        --output_dir "$OUTPUT_ROOT/eval_grpo" \
        --wandb_project ChemGFN \
        > "$OUTPUT_ROOT/grpo_eval.log" 2>&1
    echo "  GRPO eval done (log: $OUTPUT_ROOT/grpo_eval.log)"
fi

if [ -d "$OUTPUT_ROOT/ppo_expr24/final" ]; then
    echo "$(date): Evaluating PPO..."
    CUDA_VISIBLE_DEVICES=$GPU_GRPO $PYTHON scripts/eval_rl_baseline.py \
        --model_path "$OUTPUT_ROOT/ppo_expr24/final" \
        --exp_name VarExpr24_PPO \
        --n_samples 6400 \
        --batch_size 32 \
        --test_repeats 3 \
        --output_dir "$OUTPUT_ROOT/eval_ppo" \
        --wandb_project ChemGFN \
        > "$OUTPUT_ROOT/ppo_eval.log" 2>&1
    echo "  PPO eval done (log: $OUTPUT_ROOT/ppo_eval.log)"
fi

echo "$(date): All done."
