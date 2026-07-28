#!/usr/bin/env bash
#
# Train the four AMP experiments reported in the paper (TB, SubTB, RapTB, RapTB+SubM), one job
# per GPU. The trainer overrides below are the ones the reported AMP numbers were produced with.
#
# Environment variables:
#   GPUS     (optional)  Space-separated CUDA device ids, one per experiment. Default: 0 1 2 3.
#                        Fewer ids than experiments are reused round-robin, which will oversubscribe
#                        the listed devices.
#   LOG_DIR  (optional)  Directory for the nohup logs. Default: logs.
#
# Example:
#   GPUS="4 5 6 7" bash scripts/run_amp_all.sh

set -euo pipefail

read -r -a gpus <<< "${GPUS:-0 1 2 3}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

COMMON="trainer.max_steps=5000 trainer.limit_train_batches=500 trainer.limit_val_batches=50"

experiments=(
  "amp/tb"
  "amp/subtb"
  "amp/raptb"
  "amp/raptb_subm"
)

echo "=== Launching ${#experiments[@]} AMP experiments ==="

idx=0
for experiment in "${experiments[@]}"; do
  log_name="${experiment//\//_}"
  gpu="${gpus[$((idx % ${#gpus[@]}))]}"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup python chemgfn/train.py \
    experiment="${experiment}" \
    ${COMMON} \
    > "${LOG_DIR}/${log_name}.log" 2>&1 &
  echo "${experiment} -> PID $! on CUDA ${gpu} (${LOG_DIR}/${log_name}.log)"
  idx=$((idx + 1))
done

echo
echo "Validation metrics to watch:"
echo "  val/topk_performance  (paper: Performance)"
echo "  val/topk_diversity    (paper: Diversity)"
echo "  val/topk_novelty      (paper: Novelty)"
