# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChemGFN trains GFlowNets with LLMs (Llama-3.2-1B + LoRA) on two generative tasks:
- **SMILES optimization**: Grammar-constrained molecular generation with constraint satisfaction
- **VarExpr24**: Variable-length arithmetic expression generation targeting value 24

## Common Commands

```bash
# Setup
conda env create -f environment.yaml && conda activate chemgfn && pip install -e .

# Training (always specify an experiment config)
python chemgfn/train.py experiment=SMILES_basic/SMILES_cfg_TB
python chemgfn/train.py experiment=VarExpr24/VarExpr24_TB_no_data_buffer_hit

# Evaluation
python chemgfn/eval.py experiment=SMILES_basic/SMILES_cfg_TB ckpt_path=/path/to/ckpt

# Tests
make test                     # or: pytest tests/ -v
pytest tests/test_loss.py -v  # single file
pytest tests/test_loss.py::TestModifiedSubTBLoss::test_loss_is_scalar -v  # single test

# Formatting & linting
make format   # black (line-length 99) + isort (black profile)
make lint     # flake8 (max-line-length 100)
```

**Note:** Pre-commit hooks enforce black at line-length 99. The Makefile `format` target uses 100; prefer the pre-commit config as authoritative.

## Architecture

### Entry Points

- `chemgfn/train.py` — Hydra-based training entry point (`@hydra.main`, config in `configs/train.yaml`)
- `chemgfn/eval.py` — Hydra-based evaluation entry point (`configs/eval.yaml`)

Both use `rootutils.setup_root()` with `.project-root` indicator to set `PROJECT_ROOT` env var and PYTHONPATH.

### Core Data Flow

```
BufferDataModule (data/gfn_datamodule.py)
  → loads prompts (JSON/txt) + optional buffer samples
  → BufferDataPipe tokenizes and yields batches

ChemGFNModule.training_step (models/gfn.py)
  → autoregressive generation with grammar-constrained sampling
  → reward computation via SentenceValidator (RDKit or Expr24)
  → prefix value estimation (phi-shaping via k-gram memory)
  → SubTrajectory Balance loss (ModifiedSubTBLoss)
  → replay buffer update with prioritized sampling
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `models/gfn.py` | `ChemGFNModule` (LightningModule) — wraps training, generation, reward, loss |
| `models/losses.py` | `GFNLoss` ABC → `ModifiedSubTBLoss`, `ModifiedSubTBBalanceLoss`, `LLMTrajectoryBalanceLoss` |
| `models/reward.py` | Reward computation with prefix shaping, reference scaling, invalid masking |
| `models/validators.py` | `RDKitValidator` (SMILES scoring), `Expr24Validator` (expression scoring) |
| `models/components/` | `gpt2_wrapper.py`, `llama3_wrapper.py` — model wrappers |
| `data/gfn_datamodule.py` | `BufferDataModule` + `BufferDataPipe` for prompt loading/batching |
| `utils/replay_buffer.py` | Prioritized replay with similarity-based deduplication |
| `utils/phi_utils.py` | Prefix value k-gram estimation for reward shaping |
| `utils/cfg_grammar.py` | Grammar-constrained logits processing (via transformers_cfg) |
| `utils/schedulers.py` | Linear/cosine/cosine_restart schedules for factor scheduling |
| `utils/cond_var_metrics.py` | TB/SubTB/RapTB target computations |

### Configuration System (Hydra)

All components instantiated via `hydra.utils.instantiate()` with `_target_` strings. Config hierarchy:

```
configs/
├── train.yaml / eval.yaml          # root configs (compose defaults)
├── experiment/                      # override bundles for paper reproducibility
│   ├── SMILES_basic/                # TB, SubTB, no-TB baselines
│   ├── SMILES_SubM/                 # SubM variants
│   ├── SMILES_RapTB/                # RapTB variants
│   ├── SMILES_Length/               # Length-15 variants
│   └── VarExpr24/                   # 14 VarExpr24 configs
├── model/ data/ trainer/ callbacks/ logger/ paths/ extras/ hydra/
└── local/                           # machine-specific (git-ignored)
```

Experiment configs are the primary interface — they override model, trainer, and data params. Training always uses `experiment=<path>`.

### Key Patterns

- **LoRA fine-tuning**: Base Llama model frozen, LoRA adapter (rank-16) on q/k/v/o/gate/down/up projections via `peft`
- **Factor schedulers**: Dict of callable schedulers (`reward_temp`, `replay_buffer`, `dataset_buffer`, `scaling_factor`, `k_max`, `k_min`, etc.) that return float values per training step
- **Grammar constraints**: `transformers_cfg` library masks illegal tokens during generation for SMILES validity
- **Replay buffer mixing**: Training mixes on-policy samples with replay buffer and dataset buffer at configurable ratios

## Dependencies

Python 3.10. Key: PyTorch 2.0+, Lightning 2.x, transformers, peft, transformers_cfg==0.2.6, rdkit, hydra-core 1.3.2, wandb.

## Data Layout

Configs expect: `data/SMILES/sidechain_prompts_sa.json`, `data/24_points/prompts.txt`, `data/24_points/buffer_24_non_zero.pt`. Override via `configs/data/` or CLI.


# 新任务设计原则
1. 每个任务（如果有）可以传入一个限定的词表，你需要额外确认该改词表只会被相应的Tokenizer Tokenize成一个Token，而不是被拆开
2. Reward设计上，额外的外部Reward如RDKiT QED等分数，必须是每一个位置都有reward数值，设计参考成熟的validator，需要额外注意Shape和Index的匹配