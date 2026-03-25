#!/usr/bin/env bash
# run all evals distributed across selected CUDA devices; keep going if one fails
set -uo pipefail

# Root directory containing training logs. Override via environment variable:
#   LOGS_ROOT=/your/logs/path bash scripts/run_eval_all.sh
LOGS_ROOT="${LOGS_ROOT:-./logs/train}"

readarray -t cmds <<EOF
# baseline
python chemgfn/eval.py experiment="SMILES_basic/SMILES_cfg_TB" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_TB/train/runs/2025-12-31_05-22-27/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_basic/SMILES_cfg_no_TB" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_TB_no_CFG/train/runs/2025-12-27_22-42-15/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_SubM/SMILES_cfg_TB_subM_replay_add_len_func" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_TB_subM_replay_add_len_func/train/runs/2026-01-06_13-17-11/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_basic/SMILES_cfg_subTB" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_subTB/train/runs/2026-01-03_14-02-50/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_SubM/SMILES_cfg_SubTB_subM_full" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_SubTB_subM_full/train/runs/2026-01-20_13-00-48/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_basic/SMILES_cfg_TB_wo_ref" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_TB_wo_ref/train/runs/2026-01-18_04-46-17/checkpoints/last.ckpt" test_repeats=3

# RapTB
python chemgfn/eval.py experiment="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_RapTB_v2_kmin_5_to_2_mix_fix_softmax_overflow/train/runs/2026-01-10_06-13-37/checkpoints/last_2.47.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix_subM" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_RapTB_v2_kmin_5_to_2_mix_fix_softmax_overflow_subM/train/runs/2026-01-10_06-26-26/checkpoints/epoch_009_diversity_2.6636_best.ckpt" test_repeats=3

# RapTB ablation
python chemgfn/eval.py experiment="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_max_only" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_RapTB_v2_kmin_5_to_2_max_only/train/runs/2026-01-19_05-30-27/checkpoints/epoch_019_diversity_2.3899.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_soft_only" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_RapTB_v2_kmin_5_to_2_soft_only/train/runs/2026-01-19_05-30-58/checkpoints/epoch_019_diversity_2.0664.ckpt" test_repeats=3

# length 15
python chemgfn/eval.py experiment="SMILES_Length/SMILES_cfg_TB_len_15" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_TB_len_15/train/runs/2026-01-07_03-26-51/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_Length/SMILES_cfg_subTB_len_15" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_CFG_subTB_len_15/train/runs/2026-01-12_22-46-25/checkpoints/last.ckpt" test_repeats=3

# Length 15 RapTB
python chemgfn/eval.py experiment="SMILES_Length/SMILES_cfg_RapTB_v2_kmin_12_to_8_mix_fix_len15" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_RapTB_v2_kmin_12_to_8_mix_fix_len15/train/runs/2026-01-12_00-51-08/checkpoints/last_1.91.ckpt" test_repeats=3
python chemgfn/eval.py experiment="SMILES_Length/SMILES_cfg_RapTB_v2_kmin_12_to_8_mix_fix_len15_subM" +trainer.limit_test_batches=100 ckpt_path="${LOGS_ROOT}/smiles_RapTB_v2_kmin_12_to_8_mix_fix_len15_subM/train/runs/2026-01-13_08-57-58/checkpoints/2.25.ckpt" test_repeats=3

EOF

gpus=(4 5 6 7)
per_batch=${#gpus[@]}
pids=()
idx=0
failures=()

launch_batch() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures+=("$pid")
    fi
  done
  pids=()
}

for cmd in "${cmds[@]}"; do
  # skip comments/blank lines from the here-doc
  [[ -z "${cmd// }" || "${cmd}" == \#* ]] && continue
  gpu=${gpus[$((idx % per_batch))]}
  echo "Launching on CUDA ${gpu}: ${cmd}"
  CUDA_VISIBLE_DEVICES="${gpu}" ${cmd} &
  pids+=("$!")
  ((idx++))

  if (( ${#pids[@]} >= per_batch )); then
    launch_batch
  fi
done

launch_batch
if (( ${#failures[@]} )); then
  echo "All evaluations finished with ${#failures[@]} failed job(s): ${failures[*]}"
else
  echo "All evaluations finished successfully."
fi
