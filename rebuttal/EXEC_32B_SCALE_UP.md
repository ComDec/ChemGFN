# 32B Scale-Up Experiment Execution Plan

> For Reviewer JxzD: "further increasing LLM model size"
> Extends the 1B→3B→8B scale-up to 1B→3B→8B→32B (cross-architecture: Llama→Qwen)

## Background

- Paper uses Llama-3.2-1B (base) with LoRA rank-16
- 3B rebuttal: SubTB Acc=0.313 at 3B; RapTB+SubM best (0.996/0.856/0.937)
- 8B: uses Llama-3.1-8B (base), same tokenizer family
- **32B: uses Qwen3-32B (base)** — different model family, demonstrates architecture-agnostic scaling

## Model Choice: Qwen/Qwen3-32B

- 32B parameters (64 layers, hidden=5120, 40 heads, 8 KV heads GQA)
- bf16 ≈ 64GB → fits single H100 NVL 96GB with LoRA
- Frozen reference model: **Qwen/Qwen3-0.6B** (same tokenizer, vocab_size=151936)
- `Qwen2TokenizerFast` — natively supported by `transformers_cfg==0.2.6`

## Tokenizer & Allow List

- Qwen3 tokenizer (vocab=151936) differs from Llama3 tokenizer (vocab=128256)
- Generated `allowed_qwen3_32B_token`: **192 tokens** (all single-token verified)
- Source: tokenized all SMILES from `250k_rndm_zinc_drugs_clean_3.csv`
- Qwen3-0.6B and Qwen3-32B share identical tokenizer: 0 token mismatches verified
- eos_token: `<|im_end|>` (id=151645), bos_token: None

## Memory Estimate (Single H100 NVL 96GB)

| Component | Size |
|-----------|------|
| Qwen3-32B bf16 (training model) | ~64 GB |
| Qwen3-0.6B bf16 (frozen reference) | ~1.2 GB |
| LoRA adapter + Adam optimizer | ~1.5 GB |
| Activations (n_samples=8, seq≤30) | ~3 GB |
| **Total** | **~70 GB < 96 GB ✅** |

## Config Changes vs 8B

| Parameter | 8B | 32B | Reason |
|-----------|----|----|--------|
| model | `llama3_8b_smiles_opt` | `qwen3_32b_smiles_opt` | Larger backbone, different arch |
| `pretrained_model_name_or_path` | `meta-llama/Llama-3.1-8B` | `Qwen/Qwen3-32B` | 32B model |
| `frozen_model_name_or_path` | (same as training) | `Qwen/Qwen3-0.6B` | Small ref model, same tokenizer |
| `n_samples` | 16 | 8 | Memory + speed control |
| `accumulate_grad_batches` | 8 | 16 | Keep effective batch = 128 |
| `legal_tokens` | `allowed_llama3.2_1B_allowed_token` | `allowed_qwen3_32B_token` | Qwen3 tokenizer BPE |
| tokenizer `trust_remote_code` | N/A | `true` | Required for Qwen3 |
| Everything else | Same | Same | Fair comparison |

## Pre-requisites

### Step 0: Download Models (run ONCE before training)

```bash
export HF_HOME=/path/to/hf_cache

# Download Qwen3-32B (main training model)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen3-32B tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-32B', trust_remote_code=True)
print('Downloading Qwen3-32B model...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-32B', torch_dtype='auto', trust_remote_code=True)
print('Done!')
"

# Download Qwen3-0.6B (frozen reference model)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen3-0.6B...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype='auto', trust_remote_code=True)
print('Done!')
"
```

### Step 1: Verify Environment and Tokenizer

```bash
cd /path/to/ChemGFN
conda activate chemgfn
pip install -e .

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)') for i in range(torch.cuda.device_count())]"
```

**CRITICAL: Verify tokenizer compatibility**

```bash
python -c "
from transformers import AutoTokenizer

# Load tokenizers
t32 = AutoTokenizer.from_pretrained('Qwen/Qwen3-32B', trust_remote_code=True)
t06 = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)

# Verify eos_token
assert t32.eos_token == '<|im_end|>', f'WRONG eos: {t32.eos_token}'
assert t32.eos_token_id == 151645, f'WRONG eos_id: {t32.eos_token_id}'
print(f'OK: eos_token={t32.eos_token!r} (id={t32.eos_token_id})')
print(f'OK: bos_token={t32.bos_token!r} (id={t32.bos_token_id})')

# Verify SMILES token list compatibility between 32B and 0.6B
with open('assets/token_list/SMILES/allowed_qwen3_32B_token') as f:
    tokens = [line.rstrip('\n') for line in f.readlines()]

mismatch = 0
for tok in tokens:
    ids_32 = t32.encode(tok, add_special_tokens=False)
    ids_06 = t06.encode(tok, add_special_tokens=False)
    assert len(ids_32) == 1, f'MULTI-TOKEN in 32B: {tok!r} -> {ids_32}'
    assert ids_32 == ids_06, f'MISMATCH: {tok!r} -> 32B={ids_32}, 0.6B={ids_06}'
print(f'OK: all {len(tokens)} SMILES tokens are single tokens and identical between 32B/0.6B')
"
```

**Verify CFG grammar works with Qwen3 tokenizer:**

```bash
cd /path/to/transformers-CFG
python -c "
from transformers import AutoTokenizer
from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer

t = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
with open('/path/to/ChemGFN/assets/SMILES_grammars/generic.ebnf') as f:
    grammar_str = f.read()
recognizer = IncrementalTokenRecognizer(grammar_str, 'root', t)
print('OK: Grammar recognizer built successfully for Qwen3')
"
```

## Execution

### Step 2: Launch 4 Experiments in Parallel (tmux)

```bash
# Create a tmux session
tmux new-session -d -s 32b_scaleup

# GPU 0: TB
tmux send-keys -t 32b_scaleup "CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=SMILES_32B/SMILES_32B_cfg_TB" Enter

# GPU 1: SubTB
tmux split-window -t 32b_scaleup
tmux send-keys -t 32b_scaleup "CUDA_VISIBLE_DEVICES=1 python chemgfn/train.py experiment=SMILES_32B/SMILES_32B_cfg_subTB" Enter

# GPU 2: RapTB
tmux split-window -t 32b_scaleup
tmux send-keys -t 32b_scaleup "CUDA_VISIBLE_DEVICES=2 python chemgfn/train.py experiment=SMILES_32B/SMILES_32B_cfg_RapTB" Enter

# GPU 3: RapTB+SubM
tmux split-window -t 32b_scaleup
tmux send-keys -t 32b_scaleup "CUDA_VISIBLE_DEVICES=3 python chemgfn/train.py experiment=SMILES_32B/SMILES_32B_cfg_RapTB_subM" Enter

tmux select-layout -t 32b_scaleup tiled
tmux attach -t 32b_scaleup
```

Or as a launch script:

```bash
#!/bin/bash
# run_32b_all.sh - Launch all 4 experiments in parallel
set -e

EXPERIMENTS=(
    "SMILES_32B/SMILES_32B_cfg_TB"
    "SMILES_32B/SMILES_32B_cfg_subTB"
    "SMILES_32B/SMILES_32B_cfg_RapTB"
    "SMILES_32B/SMILES_32B_cfg_RapTB_subM"
)

for i in "${!EXPERIMENTS[@]}"; do
    echo "Launching ${EXPERIMENTS[$i]} on GPU $i"
    tmux new-window -t 32b_scaleup -n "gpu${i}" \
        "CUDA_VISIBLE_DEVICES=${i} python chemgfn/train.py experiment=${EXPERIMENTS[$i]}; bash"
done
```

### Estimated Time

| Hardware | Per-run | Total (4 parallel) |
|----------|---------|-------------------|
| 4x H100 NVL 96GB | ~16-24h | ~16-24h |

## Monitoring

```bash
# Check GPU utilization
watch -n 5 nvidia-smi

# Check WandB for live metrics
# Project: ChemGFN
# Run names: smiles_CFG_TB_32B, smiles_CFG_subTB_32B, smiles_RapTB_v2_kmin_5_to_2_mix_fix_32B, smiles_RapTB_v2_kmin_5_to_2_mix_fix_subM_32B
```

## Troubleshooting

### OOM on single GPU
Reduce `n_samples` from 8 to 4 and increase `accumulate_grad_batches` from 16 to 32:
```bash
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=SMILES_32B/SMILES_32B_cfg_TB \
    model.training_mixed_config.n_samples=4 trainer.accumulate_grad_batches=32
```

### Model download issues
```bash
huggingface-cli login --token YOUR_TOKEN
# Qwen3 models are open-weight, no gated access needed
```

### CFG grammar issues
The grammar processor and token list use Qwen3's BPE tokenizer. Verified:
- `Qwen2TokenizerFast` is in `transformers_cfg` SUPPORTED_TOKENIZERS
- `IncrementalTokenRecognizer` builds successfully with Qwen3 tokenizer
- 192 SMILES tokens all single-token in Qwen3

### bos_token_id=None
Fixed in `chemgfn/utils/gfn_utils.py:90` — added None check before masking bos_token.

## Expected Results & What to Collect

### Metrics to Extract (from WandB or logs)

| Metric | Key | Description |
|--------|-----|-------------|
| Acc | `val/accuracy` or equivalent | Valid SMILES fraction |
| QED | `val/qed_mean` or equivalent | Drug-likeness score |
| Entropy | `val/entropy` or equivalent | Output diversity |
| FPDiv | `val/fp_diversity` or equivalent | Fingerprint diversity |
| Avg Len | `val/avg_len` or equivalent | Average token length |
| log p_term | `val/log_pterm` or equivalent | Termination probability |

### Expected Result Table (for rebuttal)

```
## 32B Scale-Up (Qwen3-32B, LoRA rank-16, frozen ref: Qwen3-0.6B)

| Method      | Acc↑  | QED↑  | Entropy↑ | FPDiv↑ | Avg Len | log p_term |
|-------------|-------|-------|----------|--------|---------|------------|
| TB          |       |       |          |        |         |            |
| SubTB       |       |       |          |        |         |            |
| RapTB       |       |       |          |        |         |            |
| RapTB+SubM  |       |       |          |        |         |            |

Cross-scale comparison (cross-architecture):
| Scale | Arch    | SubTB Acc | RapTB+SubM Acc | Gap |
|-------|---------|-----------|----------------|-----|
| 1B    | Llama   | 0.996     | 0.996          | 0   |
| 3B    | Llama   | 0.313     | 0.996          | ↑↑↑ |
| 8B    | Llama   |           |                |     |
| 32B   | Qwen3   |           |                |     |
```

### Expected Narrative

If results follow the 3B pattern:
- SubTB failure should **worsen** at 32B (structural termination drift amplified)
- TB may show mode collapse (length collapse to short sequences)
- RapTB and RapTB+SubM should maintain high Acc/FPDiv
- Cross-architecture scaling: failure modes are **not** architecture-specific
- This confirms: "failure modes are structural to the terminable prefix tree, worsen with scale, and persist across architectures"

## Code Changes Summary

| File | Change | Description |
|------|--------|-------------|
| `chemgfn/utils/gfn_utils.py:90` | Bug fix | Guard `bos_token_id` None check for Qwen3 |
| `chemgfn/models/gfn.py:104` | Feature | Support `frozen_model_name_or_path` for separate ref model |
| `configs/model/qwen3_32b_smiles_opt.yaml` | New | 32B model config with Qwen3-0.6B frozen ref |
| `assets/token_list/SMILES/allowed_qwen3_32B_token` | New | 192 SMILES tokens for Qwen3 tokenizer |
| `configs/experiment/SMILES_32B/*.yaml` | New | 4 experiment configs (TB/SubTB/RapTB/RapTB+SubM) |

## File List

```
assets/token_list/SMILES/allowed_qwen3_32B_token    # Qwen3 SMILES token list (192 tokens)
configs/model/qwen3_32b_smiles_opt.yaml              # 32B model config
configs/experiment/SMILES_32B/
├── SMILES_32B_cfg_TB.yaml                           # TB baseline
├── SMILES_32B_cfg_subTB.yaml                        # SubTB baseline
├── SMILES_32B_cfg_RapTB.yaml                        # RapTB (ours)
└── SMILES_32B_cfg_RapTB_subM.yaml                   # RapTB+SubM (ours, best)
```
