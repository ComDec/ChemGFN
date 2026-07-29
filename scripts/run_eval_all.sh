#!/usr/bin/env bash
#
# Evaluate every SMILES experiment (L_max = 10 and L_max = 15) reported in the paper.
#
# No model weights ship with this repository, so this script cannot be run as-is: train the
# experiments first with `python chemgfn/train.py experiment=<name>`, then point CKPT_ROOT at
# the directory holding the resulting checkpoints.
#
# Expected layout (one directory per experiment, named after the experiment's `exp_name`, which
# is the config path with `/` replaced by `_`):
#
#   ${CKPT_ROOT}/smiles_tb/last.ckpt
#   ${CKPT_ROOT}/smiles_len15_tb/last.ckpt
#   ...
#
# Environment variables:
#   CKPT_ROOT  (required)  Directory containing one subdirectory per experiment.
#   CKPT_NAME  (optional)  Checkpoint file inside each subdirectory. Default: last.ckpt.
#   GPUS       (optional)  Space-separated CUDA device ids to spread jobs over. Default: 0.
#
# Example:
#   CKPT_ROOT=/path/to/checkpoints GPUS="0 1 2 3" bash scripts/run_eval_all.sh
#
# The evaluation protocol (100 test batches, 3 independent sampling repeats) is fixed here
# because it is the protocol the reported numbers use; do not change it when comparing to the
# paper.

set -uo pipefail

: "${CKPT_ROOT:?Set CKPT_ROOT to the directory containing your trained checkpoints}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
read -r -a gpus <<< "${GPUS:-0}"

experiments=(
  # baselines
  "smiles/tb"
  "smiles/subtb"
  "smiles/ablation_tb_no_reference_prior"
  "smiles/avgprefixtb"
  "smiles/tb_subm"
  "smiles/subtb_subm"

  # RapTB and its target-mode ablations
  "smiles/raptb"
  "smiles/raptb_subm"
  "smiles/ablation_raptb_absorb_max_only"
  "smiles/ablation_raptb_absorb_soft_only"

  # L_max = 15
  "smiles_len15/tb"
  "smiles_len15/subtb"
  "smiles_len15/raptb"
  "smiles_len15/raptb_subm"
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
  name="${experiment//\//_}"
  ckpt="${CKPT_ROOT}/${name}/${CKPT_NAME}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping ${experiment}: no checkpoint at ${ckpt}"
    continue
  fi

  gpu="${gpus[$((idx % ${#gpus[@]}))]}"
  echo "Launching ${experiment} on CUDA ${gpu} with ${ckpt}"
  CUDA_VISIBLE_DEVICES="${gpu}" python chemgfn/eval.py \
    experiment="${experiment}" \
    ckpt_path="${ckpt}" \
    +trainer.limit_test_batches=100 \
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
