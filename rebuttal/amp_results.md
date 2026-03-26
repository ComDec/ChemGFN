# AMP (Antimicrobial Peptide) Generation — Experiment Results

## Task Setup

We evaluate ChemGFN on the AMP generation task from Jain et al. (2022), "Biological Sequence Design with GFlowNets" (arXiv 2203.04115). The goal is to generate diverse, novel antimicrobial peptide sequences with high predicted activity.

**Oracle**: MLP classifier (ProtTrans AlBert embeddings -> 2-layer MLP) trained on D2 split of DBAASP. Pre-trained weights from `MJ10/clamp-gen-data`. Score = P(AMP) in [0, 1].

**Generator**: Llama-3.2-1B + LoRA (rank 16), grammar-constrained to 20 standard amino acids, sequence length 15-50.

**Training**: 10,000 steps, batch size 64, accumulate_grad_batches 2, bf16, single GPU.

## Metrics (aligned with paper)

All metrics computed on **D_Best = Top-100 from cumulative generated candidates (excluding D0)**, following the paper exactly:

- **Performance** (Eq. 1): Mean oracle score of Top-100 candidates
- **Diversity** (Eq. 2): Mean pairwise Levenshtein edit distance (unnormalized) over Top-100
- **Novelty** (Eq. 3): Mean minimum Levenshtein edit distance to D0 over Top-100
- **D0**: 3,219 positive AMPs from DBAASP D1 split
- **Distance function**: `polyleven.levenshtein` (raw edit distance, not normalized by length)

## Results

| Method | Performance | Diversity | Novelty | TopK Avg Len | All Avg Len | Status |
|--------|-------------|-----------|---------|--------------|-------------|--------|
| TB | 0.9267 | 7.39 | 10.65 | 17.4 | 17.2 | Complete (10K steps) |
| SubTB | 0.8965 | 21.37 | 28.68 | 49.3 | 49.9 | Complete (10K steps) |
| RapTB | 0.9266 | 7.96 | 11.52 | 18.4 | 18.7 | Complete (10K steps) |
| RapTB+SubM | 0.9252 | 9.67 | 11.65 | 18.7 | 22.7 | **In progress (~5K/10K steps)** |
| Paper GFN-AL | 0.932 | 22.34 | 28.44 | ~22.0 | ~22.0 | 10 AL rounds x 1000 candidates |

## Analysis

**Performance**: All methods achieve high AMP prediction scores (0.90-0.93), close to the paper's 0.932. TB and RapTB lead at 0.927.

**Diversity & Novelty**: There is a clear trade-off driven by sequence length:
- **TB/RapTB** generate shorter sequences (~17-18 AA, near `min_sentence_len=15`), yielding high performance but low diversity/novelty (max edit distance is bounded by sequence length).
- **SubTB** generates sequences at `max_sentence_len=50`, achieving diversity (21.37) and novelty (28.68) comparable to the paper (22.34 / 28.44), but with slightly lower performance (0.897).
- **RapTB+SubM** (still training) shows intermediate behavior with diversity gradually increasing.

**Key difference from paper**: The paper uses 10 rounds of active learning with proxy retraining each round. Our approach is single-round GFlowNet training without active learning. Despite this, SubTB already matches the paper's diversity/novelty metrics.

## Configuration Summary

| Config | Loss | Replay Buffer | k_min |
|--------|------|---------------|-------|
| TB | LLMTrajectoryBalanceLoss | ReplayBuffer | - |
| SubTB | ModifiedSubTBLoss | ReplayBuffer | - |
| RapTB | RootAbsorbExtraSubTBLossFixTBLogZv2 | ReplayBuffer | 25->10 |
| RapTB+SubM | RootAbsorbExtraSubTBLossFixTBLogZv2 | ReplayBufferSubmodular | 25->10 |

Common: `min_sentence_len=15`, `max_sentence_len=50`, `n_samples=64`, `accumulate_grad_batches=2`, `max_steps=10000`, `scaling_factor=50`.

## Reproducing

```bash
# Single experiment
CUDA_VISIBLE_DEVICES=0 python chemgfn/train.py experiment=AMP/AMP_cfg_TB

# All experiments (GPU 0/1/2)
bash scripts/run_amp_tmux.sh
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
