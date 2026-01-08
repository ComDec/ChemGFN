#!/usr/bin/env bash
# run all evals distributed across CUDA 0-5; keep going if one fails
set -uo pipefail

readarray -t cmds <<'EOF'
# baseline
# python eval.py experiment="Expr24_basic/Expr24_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_TB/train/runs/2026-01-03_13-59-49/checkpoints/last.ckpt"
python eval.py experiment="Expr24_basic/Expr24_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_TB/train/runs/2026-01-03_13-59-49/checkpoints/epoch_509_diversity_1.7302.ckpt"

# python eval.py experiment="Expr24_SubM/Expr24_TB_subM" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_TB_subM_replay/train/runs/2026-01-03_14-02-14/checkpoints/last.ckpt"
python eval.py experiment="Expr24_SubM/Expr24_TB_subM" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_TB_subM_replay/train/runs/2026-01-03_14-02-14/checkpoints/epoch_179_diversity_1.7668.ckpt"

# RapTB loss
# python eval.py experiment="Expr24_RapTB/Expr24_RapTB_kmin_0" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_RapTB_kmin_0/train/runs/2026-01-03_14-00-39/checkpoints/last.ckpt"
python eval.py experiment="Expr24_RapTB/Expr24_RapTB_kmin_0" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_RapTB_kmin_0/train/runs/2026-01-03_14-00-39/checkpoints/epoch_569_diversity_1.7442.ckpt"

# python eval.py experiment="Expr24_RapTB/Expr24_RapTB_kmin_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_RapTB_kmin_2/train/runs/2026-01-03_14-01-09/checkpoints/last.ckpt"
python eval.py experiment="Expr24_RapTB/Expr24_RapTB_kmin_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_RapTB_kmin_2/train/runs/2026-01-03_14-01-09/checkpoints/epoch_509_diversity_1.7304.ckpt"

# python eval.py experiment="Expr24_RapTB/Expr24_RapTB_kmin_5_to_2_mix" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_RapTB_kmin_5_to_2_mix/train/runs/2026-01-03_14-01-32/checkpoints/last.ckpt"
python eval.py experiment="Expr24_RapTB/Expr24_RapTB_kmin_5_to_2_mix" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/Expr24_CFG_RapTB_kmin_5_to_2_mix/train/runs/2026-01-03_14-01-32/checkpoints/epoch_509_diversity_1.7205.ckpt"
EOF

gpus=(0 1 2 3 4 5)
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
