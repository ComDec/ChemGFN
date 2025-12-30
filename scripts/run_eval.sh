# RapTB_v1
python eval.py experiment="SMILES_cfg_RapTB_v1" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v1/train/runs/2025-12-29_15-00-11/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v1_wo_detach_term" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v1_wo_detach_term/train/runs/2025-12-27_22-40-46/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v1_weight_50" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v1_weight_50/train/runs/2025-12-28_22-50-48/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v1_weight_75" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v1_weight_75/train/runs/2025-12-29_14-55-47/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v1_weight_100" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v1_weight_100/train/runs/2025-12-28_22-48-30/checkpoints/last.ckpt"

# RapTB_v2
python eval.py experiment="SMILES_cfg_RapTB_v2" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2/train/runs/2025-12-29_14-58-03/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_RapTB_v2_weight_50" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_RapTB_v2_weight_50/train/runs/2025-12-29_14-59-19/checkpoints/last.ckpt"

# baseline
python eval.py experiment="SMILES_cfg_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB/train/runs/2025-12-26_11-55-56/checkpoints/last.ckpt"
python eval.py experiment="SMILES_cfg_no_TB" +trainer.limit_test_batches=100 ckpt_path="/data1/xw3763/project/gflow/ChemGFN/logs/train/smiles_CFG_TB_no_CFG/train/runs/2025-12-27_22-42-15/checkpoints/last.ckpt"
