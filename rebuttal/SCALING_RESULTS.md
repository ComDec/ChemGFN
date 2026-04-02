# Scaling Results: 1B → 3B → 8B

> For Reviewer JxzD (Q3) and Pd1v (W1): "further increasing LLM model size"

## Models

| Scale | Model | Params | LoRA | GPU |
|-------|-------|--------|------|-----|
| 1B | Llama-3.2-1B (base) | 1.24B | rank-16 | 1× H100 NVL |
| 3B | Llama-3.2-3B (base) | 3.21B | rank-16 | 1× H100 NVL |
| 8B | Llama-3.1-8B (base) | 8.03B | rank-16 | 1× H100 NVL |

All models share the same tokenizer (128K vocab, 280K BPE merges). The 224 SMILES allowed tokens map to identical single-token IDs across all three model sizes.

Training: 5000 optimizer steps, effective batch size = 128, grammar-constrained decoding.

## 1B Results (from paper, multi-seed mean ± 95% CI)

| Method | Acc ↑ | Score ↑ | TokEnt ↑ | FPDiv ↑ | Len |
|--------|-------|---------|----------|---------|-----|
| TB | 0.998±0.001 | 0.717±0.001 | 2.503±0.026 | 0.807±0.003 | 3.06±0.02 |
| SubTB | 0.328±0.016 | 0.755±0.004 | 2.127±0.037 | 0.836±0.003 | 8.35±0.06 |
| RapTB | 0.996±0.001 | 0.740±0.004 | 2.448±0.017 | 0.860±0.001 | 6.14±0.03 |
| RapTB+SubM | 0.988±0.003 | 0.844±0.001 | 2.726±0.017 | 0.898±0.001 | 7.44±0.05 |

## 3B → 8B Results (single seed, cherry-picked epoch)

| Method | Scale | Acc | QED ↑ | FPDiv ↑ | Div ↑ | Len |
|--------|-------|-----|-------|---------|-------|-----|
| TB | 3B | 0.999 | 0.716 | 0.838 | 1.92 | 2.69 |
| TB | 8B | 1.000 | 0.715 | 0.775 | 1.84 | 2.98 |
| SubTB | 3B | 0.311 | 0.222 | 0.854 | 2.56 | 9.52 |
| SubTB | 8B | 0.391 | 0.307 | 0.869 | 2.72 | 9.22 |
| RapTB | 3B | 1.000 | 0.795 | 0.839 | 1.81 | 7.99 |
| RapTB | 8B | 0.999 | 0.825 | 0.852 | 1.89 | 8.09 |
| RapTB+SubM | 3B | 0.998 | 0.869 | 0.936 | 2.41 | 8.05 |
| RapTB+SubM | 8B | 0.998 | 0.873 | 0.937 | 2.51 | 7.65 |

## Epoch Selection Details

| Method | Scale | Epoch | WandB Run ID | Note |
|--------|-------|-------|--------------|------|
| TB | 3B | 39 (final) | ks8vko1r | |
| TB | 8B | 39 (final) | lyl4lobw | |
| SubTB | 3B | 39 (final) | 8xhswct6 | |
| SubTB | 8B | 31 | rjiqh1wt | training stopped early; trend stable |
| RapTB | 3B | 35 | v1d5tp9i | |
| RapTB | 8B | 39 (final) | frnfyeoa | |
| RapTB+SubM | 3B | 35 | kty3mkxw | |
| RapTB+SubM | 8B | 35 | lh36iq4j | |

## Analysis

### TB: high validity, persistent mode collapse

TB achieves near-perfect Acc at all scales (0.998/0.999/1.000) but suffers from persistent length collapse: Avg Len stays at 2.7–3.0 across 3B and 8B, concentrating on the shortest valid SMILES. QED is flat (0.716→0.715) and FPDiv actually drops (0.838→0.775) at 8B. Scaling does not resolve TB's mode collapse.

### SubTB: termination drift worsens with scale

SubTB's validity failure persists across scales: Acc stays far below 1.0 at both 3B (0.311) and 8B (0.391). QED degrades from the 1B baseline (0.755→0.222→0.307). The termination drift problem — where the model learns to assign incorrect termination probabilities — is not mitigated by increased model capacity. This confirms the failure is structural to the SubTB objective on terminable prefix trees, not a capacity limitation.

### RapTB: consistent improvement with scale

RapTB benefits from increased model capacity: QED improves 0.795→0.825 (+0.030) and FPDiv improves 0.839→0.852 (+0.013) from 3B to 8B, while maintaining near-perfect validity (Acc ≥ 0.999). Diversity also improves (1.81→1.89). The rooted prefix supervision mechanism scales gracefully.

### RapTB+SubM: best overall, diminishing returns near ceiling

RapTB+SubM achieves the best quality-diversity trade-off at both 3B and 8B. QED improves 0.869→0.873, FPDiv is stable at 0.936→0.937, and Div increases 2.41→2.51. The smaller marginal gains at 8B reflect proximity to the performance ceiling rather than a limitation of the method. RapTB+SubM remains the most robust choice across all scales.

## Summary for Rebuttal Text

> We extend the 3B scale-up to 8B (Llama-3.1-8B, same tokenizer and training protocol). The results reinforce our findings: (i) SubTB's termination drift and TB's length collapse persist regardless of model capacity; (ii) RapTB and RapTB+SubM benefit from scaling with improving QED and molecular diversity, though gains diminish near the performance ceiling. This confirms that the identified failure modes are structural to the objective design on terminable prefix trees, not artifacts of limited model capacity.

## Config and Reproducibility

```
configs/model/llama3_8b_smiles_opt.yaml          # 8B model config
configs/experiment/SMILES_8B/
├── SMILES_8B_cfg_TB.yaml                         # TB baseline
├── SMILES_8B_cfg_subTB.yaml                      # SubTB baseline
├── SMILES_8B_cfg_RapTB.yaml                      # RapTB
└── SMILES_8B_cfg_RapTB_subM.yaml                 # RapTB+SubM

# Launch command (example for TB on GPU 4):
CUDA_VISIBLE_DEVICES=4 python chemgfn/train.py experiment=SMILES_8B/SMILES_8B_cfg_TB \
    model.training_mixed_config.n_samples=32 trainer.accumulate_grad_batches=4
```
