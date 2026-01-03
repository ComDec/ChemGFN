#!/usr/bin/env bash
# run all evals distributed across CUDA 0-5; keep going if one fails
set -uo pipefail

readarray -t cmds <<'EOF'
# baseline
python eval.py experiment="SMILES_cfg_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB/train/runs/2025-12-31_05-22-27/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_no_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB_no_CFG/train/runs/2025-12-27_22-42-15/checkpoints/last.ckpt"

# RapTB loss
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_0" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_0/train/runs/2025-12-30_12-37-10/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2/train/runs/2025-12-30_12-38-21/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_3" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_3/train/runs/2025-12-30_12-38-33/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_mix" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_mix/train/runs/2025-12-31_09-16-12/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2/train/runs/2025-12-30_12-38-21/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_3" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_3/train/runs/2025-12-30_12-38-33/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_soft_ab" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_softab/train/runs/2025-12-31_09-15-27/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_2/train/runs/2025-12-31_09-14-58/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5/train/runs/2025-12-31_09-15-09/checkpoints/last.ckpt"

# Replay Buffer
python eval.py experiment="SMILES_cfg_TB_no_replay" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB_no_replay/train/runs/2026-01-02_15-12-12/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_TB_random_select" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB_random_select/train/runs/2026-01-02_12-05-17/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_TB_subM_replay" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB_subM_replay/train/runs/2026-01-02_12-05-06/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_TB_subM_replay_add_len_func" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB_subM_replay_add_len_func/train/runs/2026-01-02_12-05-08/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_no_replay" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_mix_no_replay/train/runs/2026-01-02_15-13-15/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_subM_replay" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_mix_subM_replay/train/runs/2026-01-02_12-05-11/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_subM_replay_add_len_func" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_mix_subM_replay_add_len_func/train/runs/2026-01-02_12-05-14/checkpoints/last.ckpt"
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
