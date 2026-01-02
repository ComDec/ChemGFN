# RapTB_v2_new
python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_0_kmax_2_to_11" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_0_kmax_2_to_11/train/runs/2025-12-30_12-38-47/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_0_soft_ab" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_0_soft_ab/train/runs/2025-12-30_12-38-59/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_0" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_0/train/runs/2025-12-30_12-37-10/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_2_to_5" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_2_to_5/train/runs/2025-12-30_12-38-08/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2/train/runs/2025-12-30_12-38-21/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_3" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_3/train/runs/2025-12-30_12-38-33/checkpoints/last.ckpt"
