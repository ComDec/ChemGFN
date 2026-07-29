#!/usr/bin/env bash
#
# Evaluate every Expr24 experiment reported in the paper.
#
# No model weights ship with this repository, so this script cannot be run as-is: train the
# experiments first with `python chemgfn/train.py experiment=<name>`, then point CKPT_ROOT at
# the directory holding the resulting checkpoints.
#
# Expected layout (one directory per run, named after the experiment's `exp_name`, which is the
# config path with `/` replaced by `_`):
#
#   ${CKPT_ROOT}/expr24_rp_tb/last.ckpt
#   ${CKPT_ROOT}/expr24_rp_subtb/last.ckpt
#   ...
#
# Environment variables:
#   CKPT_ROOT  (required)  Directory containing one subdirectory per run.
#   CKPT_NAME  (optional)  Checkpoint file inside each subdirectory. Default: last.ckpt.
#   GPUS       (optional)  Space-separated CUDA device ids to spread jobs over. Default: 0.
#
# Example:
#   CKPT_ROOT=/path/to/checkpoints GPUS="0 1 2 3" bash scripts/run_eval_expr24_all.sh
#
# The evaluation protocol (200 test batches, 3 independent sampling repeats) is fixed here
# because it is the protocol the reported numbers use; do not change it when comparing to the
# paper.

set -uo pipefail

: "${CKPT_ROOT:?Set CKPT_ROOT to the directory containing your trained checkpoints}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
read -r -a gpus <<< "${GPUS:-0}"

experiments=(
  # standard replay buffer
  "expr24/rp_tb"
  "expr24/rp_subtb"
  "expr24/rp_raptb"
  "expr24/rp_avgprefixtb"
  "expr24/rp_rootsubtblogz"

  # + submodular replay buffer
  "expr24/subm_tb"
  "expr24/subm_subtb"
  "expr24/subm_raptb"

  # oracle dataset buffer
  "expr24/oracle_tb"
  "expr24/oracle_subtb"
  "expr24/oracle_raptb"
  "expr24/oracle_rootsubtblogz"

  # pretrained-reference (PRT) variants
  "expr24/prt_tb"
  "expr24/prt_subtb"
  "expr24/prt_raptb"
)

pids=()
failures=0
idx=0

wait_for_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  pids=()
}

for experiment in "${experiments[@]}"; do
  exp_name="${experiment//\//_}"
  ckpt="${CKPT_ROOT}/${exp_name}/${CKPT_NAME}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping ${experiment}: no checkpoint at ${ckpt}"
    continue
  fi

  gpu="${gpus[$((idx % ${#gpus[@]}))]}"
  echo "Launching ${experiment} on CUDA ${gpu} with ${ckpt}"
  CUDA_VISIBLE_DEVICES="${gpu}" python chemgfn/eval.py \
    experiment="${experiment}" \
    exp_name="${exp_name}" \
    ckpt_path="${ckpt}" \
    +trainer.limit_test_batches=200 \
    test_repeats=3 &
  pids+=("$!")
  idx=$((idx + 1))

  if (( ${#pids[@]} >= ${#gpus[@]} )); then
    wait_for_batch
  fi
done

wait_for_batch

if (( failures > 0 )); then
  echo "All evaluations finished with ${failures} failed job(s)."
  exit 1
fi
echo "All evaluations finished successfully."
