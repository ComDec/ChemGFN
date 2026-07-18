# RapTB (ChemGFN)

To be updated.

ChemGFN is a minimal, camera-ready codebase for training GFlowNets with LLMs on two tasks:

- SMILES optimization with grammar-constrained generation
- VarExpr24 arithmetic generation (variable-length expressions)

## Highlights

- Hydra-based configuration for training and evaluation
- Grammar-constrained sampling for SMILES
- Reproducible evaluation scripts for paper configs

## Repository Layout

- `chemgfn/` core models, data modules, and utilities
- `configs/` Hydra configs for data, models, experiments, and trainers
- `scripts/` batch evaluation helpers
- `tests/` unit and integration tests
- `data/` expected data locations (user provided)

## Requirements

- Python 3.10
- PyTorch 2.0+
- CUDA optional for GPU training and evaluation

## Setup

Conda (recommended):
```bash
conda env create -f environment.yaml
conda activate chemgfn
pip install -e .
```

Pip:
```bash
pip install -r requirements.txt
pip install -e .
```

## Data Layout

Default configs expect the following files:

- SMILES: `data/SMILES/sidechain_prompts_sa.json`
- VarExpr24: `data/24_points/prompts.txt`
- VarExpr24 buffer: `data/24_points/buffer_24_non_zero.pt`

If your data lives elsewhere, update the paths under `configs/data/`.

## Training

SMILES optimization (TB baseline):
```bash
python chemgfn/train.py experiment=SMILES_basic/SMILES_cfg_TB
```

VarExpr24 (TB baseline):
```bash
python chemgfn/train.py experiment=VarExpr24/VarExpr24_TB_no_data_buffer_hit
```

Common overrides:
```bash
python chemgfn/train.py \
  experiment=SMILES_basic/SMILES_cfg_TB \
  trainer.devices=1 \
  trainer.max_steps=5000
```

## Evaluation

Single run:
```bash
python chemgfn/eval.py \
  experiment=SMILES_basic/SMILES_cfg_TB \
  ckpt_path="/path/to/checkpoint.ckpt"
```

Batch evaluation (paper configs):

- `scripts/run_eval_all.sh` for SMILES tasks
- `scripts/run_eval_expr24_all.sh` for VarExpr24 tasks

Update the `ckpt_path` entries in those scripts to match your local checkpoints and adjust
the GPU list if needed.

## Paper Configs

The following configs reproduce the reported results.

SMILES (baseline and ablations):

- `configs/experiment/SMILES_basic/SMILES_cfg_TB.yaml`
- `configs/experiment/SMILES_basic/SMILES_cfg_no_TB.yaml`
- `configs/experiment/SMILES_basic/SMILES_cfg_subTB.yaml`
- `configs/experiment/SMILES_basic/SMILES_cfg_TB_wo_ref.yaml`
- `configs/experiment/SMILES_SubM/SMILES_cfg_TB_subM_replay_add_len_func.yaml`
- `configs/experiment/SMILES_SubM/SMILES_cfg_SubTB_subM_full.yaml`
- `configs/experiment/SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix.yaml`
- `configs/experiment/SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_mix_fix_subM.yaml`
- `configs/experiment/SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_max_only.yaml`
- `configs/experiment/SMILES_RapTB/SMILES_cfg_RapTB_v2_kmin_5_to_2_soft_only.yaml`

SMILES length-15:

- `configs/experiment/SMILES_Length/SMILES_cfg_TB_len_15.yaml`
- `configs/experiment/SMILES_Length/SMILES_cfg_subTB_len_15.yaml`
- `configs/experiment/SMILES_Length/SMILES_cfg_RapTB_v2_kmin_12_to_8_mix_fix_len15.yaml`
- `configs/experiment/SMILES_Length/SMILES_cfg_RapTB_v2_kmin_12_to_8_mix_fix_len15_subM.yaml`

VarExpr24:

- `configs/experiment/VarExpr24/VarExpr24_TB_no_data_buffer_hit.yaml`
- `configs/experiment/VarExpr24/VarExpr24_SubTB_no_data_buffer_hit.yaml`
- `configs/experiment/VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune.yaml`
- `configs/experiment/VarExpr24/VarExpr24_TB_no_data_buffer_hit_subM_div_on_valid.yaml`
- `configs/experiment/VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_subM_div_on_valid.yaml`
- `configs/experiment/VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_subM_div_on_valid.yaml`
- `configs/experiment/VarExpr24/VarExpr24_TB_no_data_buffer_hit_oracle.yaml`
- `configs/experiment/VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_oracle.yaml`
- `configs/experiment/VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_oracle.yaml`
- `configs/experiment/VarExpr24/VarExpr24_RootSubTBLogZ_no_data_buffer_hit_dense.yaml`
- `configs/experiment/VarExpr24/VarExpr24_RootSubTBLogZ_no_data_buffer_hit_dense_oracle.yaml`
- `configs/experiment/VarExpr24/VarExpr24_TB_no_data_buffer_hit_PRT.yaml`
- `configs/experiment/VarExpr24/VarExpr24_SubTB_no_data_buffer_hit_PRT.yaml`
- `configs/experiment/VarExpr24/VarExpr24_RapTB_kmin_7_to_3_mix_wo_dbuff_hit_tune_PRT.yaml`

## Tests

```bash
pytest tests -v
```

See `tests/README_TESTS.md` for more detail.
