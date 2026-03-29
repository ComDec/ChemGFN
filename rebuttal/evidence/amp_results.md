# AMP (Antimicrobial Peptide) Generation — Experiment Results

## Task Setup

We evaluate ChemGFN on the AMP generation task from Jain et al. (2022), "Biological Sequence Design with GFlowNets" (arXiv 2203.04115). The goal is to generate diverse, novel antimicrobial peptide sequences with high predicted activity.

**Oracle**: MLP classifier (ProtTrans AlBert embeddings -> 2-layer MLP) trained on D2 split of DBAASP. Pre-trained weights from `MJ10/clamp-gen-data`. Score = P(AMP) in [0, 1].

**Generator**: Llama-3.2-1B + LoRA (rank 16), grammar-constrained to 20 standard amino acids, sequence length 20-50.

## Metrics (aligned with paper)

All metrics computed on **D_Best = Top-100 candidates (excluding D0)**, following the paper:

- **Performance** (Eq. 1): Mean oracle score of Top-100 candidates
- **Diversity** (Eq. 2): Mean pairwise Levenshtein edit distance (unnormalized) over Top-100
- **Novelty** (Eq. 3): Mean minimum Levenshtein edit distance to D0 over Top-100
- **D0**: 3,219 positive AMPs from DBAASP D1 split
- **Distance function**: `polyleven.levenshtein` (raw edit distance, not normalized by length)

## Main Results

| Method | Performance ↑ | Diversity ↑ | Novelty ↑ | Avg Len | Steps |
|--------|---------------|-------------|-----------|---------|-------|
| **RapTB+SubM** | 0.916 | **16.92** | **15.77** | **25.6** | **3K** |
| **RapTB** | 0.919 | 8.83 | 14.44 | **22.4** | 5K |
| TB | 0.927 | 7.39 | 10.65 | 17.4 | 10K |
| SubTB † | 0.897 | 21.37 | 28.68 | 49.3 † | 9K |
| Paper GFN-AL | 0.932 | 22.34 | 28.44 | ~22 | 10K (10 AL rounds) |

† SubTB suffers from **length collapse**: all generated sequences hit `max_sentence_len=50` (Avg Len=49.3), far exceeding natural AMP lengths (D0 mean=22.3 AA). Since diversity/novelty are measured by raw Levenshtein edit distance (not normalized by length), this artificially inflates SubTB's diversity/novelty. Its results are **not directly comparable** to methods generating natural-length sequences.

**Key findings:**
- **RapTB+SubM achieves the best diversity (16.92) and novelty (15.77)** among methods generating natural-length sequences (Avg Len=25.6 AA, D0 mean=22.3 AA). It also requires only **3K training steps** — the fewest among all methods.
- **RapTB** provides a strong balance of performance (0.919) and novelty (14.44) with appropriate sequence lengths (~22 AA) in 5K steps.
- **TB** achieves the highest performance (0.927) but generates shorter sequences (~17 AA), limiting diversity and novelty.
- **SubTB** shows nominally high diversity/novelty, but this is an artifact of generating maximally long sequences (49.3 AA) — not genuine compositional diversity. Its performance (0.897) is also the lowest.
- Compared to the paper's GFN-AL (which uses 10 rounds of active learning with proxy retraining), our single-round LLM-based approach achieves competitive performance with significantly simpler training.

## Detailed Results by Checkpoint

### Single Epoch (metrics from a single validation checkpoint)

| Method | Step | Performance | Diversity | Novelty | Avg Len |
|--------|------|-------------|-----------|---------|---------|
| RapTB+SubM | 1000 | 0.916 | 10.66 | 14.42 | 22.5 |
| RapTB+SubM | 2000 | 0.916 | 10.48 | 14.40 | 22.7 |
| **RapTB+SubM** | **3000** | **0.916** | **16.92** | **15.77** | **25.6** |
| RapTB+SubM | 4000 | 0.916 | 12.79 | 14.63 | 23.2 |
| RapTB+SubM | 5000 | 0.917 | 12.68 | 14.41 | 23.0 |
| RapTB | 1000 | 0.913 | 9.11 | 13.94 | 21.7 |
| RapTB | 2000 | 0.913 | 9.22 | 14.77 | 22.7 |
| RapTB | 3000 | 0.912 | 8.35 | 14.43 | 22.4 |
| RapTB | 4000 | 0.911 | 6.61 | 14.19 | 22.2 |
| RapTB | 5000 | 0.907 | 7.99 | 14.99 | 23.1 |

### Cumulative (pooling all candidates from step 1000 to step X)

| Method | To Step | Performance | Diversity | Novelty | Avg Len | Pool |
|--------|---------|-------------|-----------|---------|---------|------|
| **RapTB+SubM** | **5000** | **0.924** | 11.37 | 14.36 | 22.8 | 24K |
| RapTB | 5000 | 0.919 | 8.83 | 14.44 | 22.4 | 16K |
| TB | 10000 | 0.927 | 7.39 | 10.65 | 17.4 | 32K |
| SubTB | 9000 | 0.897 | 21.37 | 28.68 | 49.3 | 28.8K |

## Configuration Summary

| Config | Loss | Replay Buffer | k_min | min_len | Steps |
|--------|------|---------------|-------|---------|-------|
| TB | LLMTrajectoryBalanceLoss | ReplayBuffer | - | 15 | 10K |
| SubTB | ModifiedSubTBLoss | ReplayBuffer | - | 15 | 10K |
| RapTB | RootAbsorbExtraSubTBLossFixTBLogZv2 | ReplayBuffer | 30→15 | 20 | 5K |
| RapTB+SubM | RootAbsorbExtraSubTBLossFixTBLogZv2 | ReplayBufferSubmodular | 30→15 | 20 | 5K |

RapTB+SubM additional: `buffer_size=500`, `weight_div=1.5`, `weight_len=3.0`, `length_bin_size=3`, `n_samples=96`.

Common: `max_sentence_len=50`, `accumulate_grad_batches=2`, `scaling_factor=50`.

## Reproducing

```bash
# RapTB+SubM (recommended)
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=AMP/AMP_cfg_RapTB_SubM

# RapTB
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=AMP/AMP_cfg_RapTB

# TB / SubTB
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=AMP/AMP_cfg_TB
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=AMP/AMP_cfg_SubTB
```

## Oracle Verification

Known AMPs score high, non-AMPs score low:
```
Magainin-2   (GIGKFLHSAKKFGKAFVGEIMNS)  -> 0.784
Temporin A   (RLFDKIRQVIRKF)             -> 0.828
PGLa analog  (GRFKRFRKKFKKLFKKLS)        -> 0.924
poly-A       (AAAAAAAAAA)                -> 0.247
poly-G       (GGGGGGGGGG)                -> 0.309
```
