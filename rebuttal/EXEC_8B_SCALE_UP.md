# 8B Scale-Up Experiment Execution Plan

> For Reviewer JxzD: "further increasing LLM model size"
> Extends the 1B→3B scale-up to 1B→3B→8B

## Background

- Paper uses Llama-3.2-1B (base) with LoRA rank-16
- 3B rebuttal: SubTB Acc=0.313 at 3B; RapTB+SubM best (0.996/0.856/0.937)
- 8B: uses Llama-3.1-8B (base), same tokenizer (128K vocab), verified token list compatible

## Model Choice: meta-llama/Llama-3.1-8B

- Same LLaMA-3 tokenizer family as 3.2-1B/3B → `allowed_llama3.2_1B_allowed_token` works directly
- Verified: all 224 SMILES tokens map to identical single-token IDs between 1B and 8B
- Base model (not Instruct) → consistent with 1B/3B experiments
- bf16 ≈ 16GB → fits single H100 (80GB) with LoRA comfortably

## Config Changes vs 3B

| Parameter | 3B | 8B | Reason |
|-----------|----|----|--------|
| model | `llama3_3b_smiles_opt` | `llama3_8b_smiles_opt` | Larger backbone |
| `pretrained_model_name_or_path` | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.1-8B` | 8B base model |
| `n_samples` | 32 | 16 | Memory + speed control |
| `accumulate_grad_batches` | 4 | 8 | Keep effective batch = 128 |
| Everything else | Same | Same | Fair comparison |

## Pre-requisites

### Step 0: Download Model (run ONCE before training)

```bash
# Set HF cache to local storage (adjust path for your machine)
export HF_HOME=/path/to/hf_cache

# Download Llama-3.1-8B base model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B', torch_dtype='auto')
print('Done!')
"
```

If access is gated, ensure `huggingface-cli login` is done with a token that has Llama-3.1 access.

### Step 1: Verify Environment

```bash
cd /path/to/ChemGFN
conda activate chemgfn  # or your env name
pip install -e .

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)') for i in range(torch.cuda.device_count())]"
```

## Execution

### Step 2: Launch 4 Experiments in Parallel (tmux)

```bash
# Create a tmux session
tmux new-session -d -s 8b_scaleup

# GPU 0: TB
tmux send-keys -t 8b_scaleup "CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=SMILES_8B/SMILES_8B_cfg_TB" Enter

# GPU 1: SubTB
tmux split-window -t 8b_scaleup
tmux send-keys -t 8b_scaleup "CUDA_VISIBLE_DEVICES=1 python chemgfn/train.py experiment=SMILES_8B/SMILES_8B_cfg_subTB" Enter

# GPU 2: RapTB
tmux split-window -t 8b_scaleup
tmux send-keys -t 8b_scaleup "CUDA_VISIBLE_DEVICES=2 python chemgfn/train.py experiment=SMILES_8B/SMILES_8B_cfg_RapTB" Enter

# GPU 3: RapTB+SubM
tmux split-window -t 8b_scaleup
tmux send-keys -t 8b_scaleup "CUDA_VISIBLE_DEVICES=3 python chemgfn/train.py experiment=SMILES_8B/SMILES_8B_cfg_RapTB_subM" Enter

tmux select-layout -t 8b_scaleup tiled
tmux attach -t 8b_scaleup
```

Or as a simple launch script:

```bash
#!/bin/bash
# run_8b_all.sh - Launch all 4 experiments in parallel
set -e

EXPERIMENTS=(
    "SMILES_8B/SMILES_8B_cfg_TB"
    "SMILES_8B/SMILES_8B_cfg_subTB"
    "SMILES_8B/SMILES_8B_cfg_RapTB"
    "SMILES_8B/SMILES_8B_cfg_RapTB_subM"
)

for i in "${!EXPERIMENTS[@]}"; do
    echo "Launching ${EXPERIMENTS[$i]} on GPU $i"
    tmux new-window -t 8b_scaleup -n "gpu${i}" \
        "CUDA_VISIBLE_DEVICES=${i} python chemgfn/train.py experiment=${EXPERIMENTS[$i]}; bash"
done
```

### Estimated Time

| Hardware | Per-run | Total (4 parallel) |
|----------|---------|-------------------|
| 4x H100 80GB | ~4-6h | ~4-6h |
| 4x RTX 6000 Ada 48GB | ~8-12h | ~8-12h |

## Monitoring

```bash
# Check GPU utilization
watch -n 5 nvidia-smi

# Check WandB for live metrics
# Project: ChemGFN
# Run names: smiles_CFG_TB_8B, smiles_CFG_subTB_8B, smiles_RapTB_v2_kmin_5_to_2_mix_fix_8B, smiles_RapTB_v2_kmin_5_to_2_mix_fix_subM_8B
```

## Expected Results & What to Collect

### Metrics to Extract (from WandB or logs)

| Metric | Key | Description |
|--------|-----|-------------|
| Acc | `val/accuracy` or equivalent | Valid SMILES fraction |
| QED | `val/qed_mean` or equivalent | Drug-likeness score |
| Entropy | `val/entropy` or equivalent | Output diversity |
| FPDiv | `val/fp_diversity` or equivalent | Fingerprint diversity |
| Avg Len | `val/avg_len` or equivalent | Average token length |
| log p_term | `val/log_pterm` or equivalent | Termination probability (SubTB drift indicator) |

### Expected Result Table (for rebuttal)

```
## 8B Scale-Up (Llama-3.1-8B, LoRA rank-16)

| Method      | Acc↑  | QED↑  | Entropy↑ | FPDiv↑ | Avg Len | log p_term |
|-------------|-------|-------|----------|--------|---------|------------|
| TB          |       |       |          |        |         |            |
| SubTB       |       |       |          |        |         |            |
| RapTB       |       |       |          |        |         |            |
| RapTB+SubM  |       |       |          |        |         |            |

Cross-scale comparison:
| Scale | SubTB Acc | RapTB+SubM Acc | Gap |
|-------|-----------|----------------|-----|
| 1B    | 0.996     | 0.996          | 0   |
| 3B    | 0.313     | 0.996          | ↑↑↑ |
| 8B    |           |                |     |
```

### Expected Narrative

If results follow the 3B pattern:
- SubTB failure should **worsen** at 8B (structural termination drift amplified by larger model)
- TB may show mode collapse (length collapse to short sequences)
- RapTB and RapTB+SubM should maintain high Acc/FPDiv
- This confirms: "failure modes are structural to the terminable prefix tree and worsen with scale"

## Troubleshooting

### OOM on single GPU
Reduce `n_samples` from 16 to 8 and increase `accumulate_grad_batches` from 8 to 16:
```bash
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=SMILES_8B/SMILES_8B_cfg_TB \
    model.training_mixed_config.n_samples=8 trainer.accumulate_grad_batches=16
```

### Model download issues
```bash
# Use specific HF token
huggingface-cli login --token YOUR_TOKEN

# Or set env var
export HF_TOKEN=your_token_here
```

### Grammar constraint errors
The grammar processor and token list are tokenizer-independent (verified: 224 tokens have identical IDs across 1B/3B/8B). If issues arise, check `HF_HOME` points to the correct cache.

## File List

```
configs/model/llama3_8b_smiles_opt.yaml          # 8B model config
configs/experiment/SMILES_8B/
├── SMILES_8B_cfg_TB.yaml                         # TB baseline
├── SMILES_8B_cfg_subTB.yaml                      # SubTB baseline
├── SMILES_8B_cfg_RapTB.yaml                      # RapTB (ours)
└── SMILES_8B_cfg_RapTB_subM.yaml                 # RapTB+SubM (ours, best)
```
