#!/usr/bin/env bash
# run all evals distributed across selected CUDA devices; keep going if one fails
set -uo pipefail

# Root directory containing training logs. Override via environment variable:
#   LOGS_ROOT=/your/logs/path bash scripts/run_eval_expr24_all.sh
LOGS_ROOT="${LOGS_ROOT:-./logs/train}"

readarray -t cmds <<EOF
# baseline
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_TB_no_data_buffer_hit" exp_name="VarExpr24_TB" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_TB_hit24_dense/train/runs/2026-01-13_09-08-36/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_SubTB_no_data_buffer_hit" exp_name="VarExpr24_SubTB" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense/train/runs/2026-01-13_09-08-56/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune" exp_name="VarExpr24_RapTB" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_v0/train/runs/2026-01-14_12-26-48/checkpoints/epoch_039_diversity_1.2109.ckpt" test_repeats=3

# SubM
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_TB_no_data_buffer_hit_subM_div_on_valid"  exp_name="VarExpr24_TB_SubM" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_TB_no_data_buffer_hit24_dense_subM/train/runs/2026-01-14_11-33-39/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_subM_div_on_valid"  exp_name="VarExpr24_SubTB_SubM" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_subM_div_on_valid/train/runs/2026-01-14_11-34-19/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_subM_div_on_valid" exp_name="VarExpr24_RapTB_SubM" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_subM_div_on_valid_wo_len_ext_size/train/runs/2026-01-16_13-32-52/checkpoints/epoch_029_diversity_1.5169.ckpt" test_repeats=3

python chemgfn/eval.py experiment="VarExpr24/VarExpr24_TB_no_data_buffer_hit_subM_div_on_valid"  exp_name="VarExpr24_TB_SubM_ext_size" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_TB_no_data_buffer_hit24_dense_subM_ext_size/train/runs/2026-01-24_06-16-38/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_subM_div_on_valid"  exp_name="VarExpr24_SubTB_SubM_ext_size" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_subM_ext_size/train/runs/2026-01-24_06-16-57/checkpoints/last.ckpt" test_repeats=3


# Oracle
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_TB_no_data_buffer_hit_oracle" exp_name="VarExpr24_TB_Oracle" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_TB_hit24_dense_oracle/train/runs/2026-01-13_09-11-34/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_oracle" exp_name="VarExpr24_SubTB_Oracle" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_oracle/train/runs/2026-01-13_09-11-55/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle" exp_name="VarExpr24_RapTB_Oracle" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_oracle/train/runs/2026-01-13_09-30-05/checkpoints/last.ckpt" test_repeats=3


# SubTB variants
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RootSubTBLogZ_no_data_buffer_hit_dense_oracle"  exp_name="VarExpr24_RootSubTBLogZ_Oracle" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RootSubTBLogZ_hit24_dense_oracle/train/runs/2026-01-15_12-28-26/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RootSubTBLogZ_no_data_buffer_hit_dense"  exp_name="VarExpr24_RootSubTBLogZ" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RootSubTBLogZ_hit24_dense/train/runs/2026-01-15_12-27-27/checkpoints/last.ckpt" test_repeats=3

# PRT variants
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_TB_no_data_buffer_hit_PRT" exp_name="VarExpr24_TB_PRT" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_TB_hit24_dense_PRT/train/runs/2026-01-21_07-22-37/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_PRT" exp_name="VarExpr24_SubTB_PRT" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_SubTB_no_data_buffer_hit24_dense_PRT/train/runs/2026-01-21_07-22-42/checkpoints/last.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_PRT" exp_name="VarExpr24_RapTB_PRT" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_v0_PRT/train/runs/2026-01-21_07-22-47/checkpoints/epoch_019_diversity_0.9058.ckpt" test_repeats=3
python chemgfn/eval.py experiment="VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_PRT" exp_name="VarExpr24_RapTB_PRT_last_ckpt" +trainer.limit_test_batches=200 ckpt_path="${LOGS_ROOT}/VarExpr24_CFG_RapTB_kmin_7_to_3_mix_wo_dbuff_hit24_dense_tune_v0_PRT/train/runs/2026-01-21_07-22-47/checkpoints/last.ckpt" test_repeats=3

EOF

gpus=(0 1 2 3 4 5 6 7)
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
