# RapTB_v2_new
python eval.py experiment="SMILES_cfg_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB/train/runs/2025-12-31_05-22-27/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_mix" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_mix/train/runs/2025-12-31_09-16-12/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2/train/runs/2025-12-30_12-38-21/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_3" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_3/train/runs/2025-12-30_12-38-33/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5_to_2_soft_ab" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5_to_2_softab/train/runs/2025-12-31_09-15-27/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_2/train/runs/2025-12-31_09-14-58/checkpoints/last.ckpt"

python eval.py experiment="SMILES_cfg_RapTB_v2_kmin_5" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_kmin_5/train/runs/2025-12-31_09-15-09/checkpoints/last.ckpt"
