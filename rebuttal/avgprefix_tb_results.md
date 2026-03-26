# AvgPrefixTB Baseline Results

> Reviewer QHmk requested baseline: "just averaging the usual TB loss over all prefixes of the sequence"

## Method

**AvgPrefixTBLoss** (`chemgfn/models/losses.py`): Strict averaged-prefix TB.

$$\mathcal{L}_{\text{AvgPrefixTB}}(\xi) = \frac{1}{|K(\xi)|} \sum_{k \in K(\xi)} \left(\Delta_k^{\text{TB}}(\xi)\right)^2$$

where $K(\xi) = \{0, 1, \ldots, \tau\}$ and $\Delta_k^{\text{TB}} = \log Z_\theta + \sum_{t=0}^{k-1} \log p_F[t] + \log p_{\text{term}}[k] - \log r[k]$.

Config: `mode="avgprefix"`, `k_min=0`, `detach_pterm_in_aux=false`, learnable $\log Z$.

## Table 3: VarExpr24 (N=6400 samples)

| Method | Acc ↑ | Unique | NormCov ↑ | KL(π→p\*) ↓ | KL(p\*→π) ↓ | JS ↓ | log p\_term(τ) ↑ |
|---|---|---|---|---|---|---|---|
| **AvgPrefixTB (RP)** | 0.998 | 142 | 0.016 | 0.808 | 8.908 | 0.213 | -0.560 |
| **AvgPrefixTB (PRT)** | 1.000 | 108 | 0.012 | 0.944 | 10.989 | 0.249 | -0.606 |
| ~~AvgPrefixTB (SubM v1)~~ | 1.000 | 7 | 0.000 | 1.442 | 14.506 | 0.372 | -0.000 |
| **AvgPrefixTB (SubM)** | 0.993 | 902 | 0.050 | 0.190 | 0.819 | 0.051 | -0.181 |
| **AvgPrefixTB (Oracle)** | 0.922 | 3369 | **0.183** | **0.052** | **0.052** | **0.013** | -1.105 |

## Table 2: SMILES (L=10, N=3200 samples × 3 repeats)

| Method | Acc ↑ | Score (QED) ↑ | Diversity (Ent) ↑ | FPDiv ↑ | Len\_μ |
|---|---|---|---|---|---|
| TB | 0.998±0.001 | 0.717±0.001 | 2.503±0.026 | 0.807±0.003 | 3.06±0.02 |
| SubTB | 0.328±0.016 | 0.755±0.004 | 2.127±0.037 | 0.836±0.003 | 8.35±0.06 |
| TB+SubM | 0.996±0.000 | 0.842±0.001 | 2.775±0.002 | 0.889±0.002 | 6.56±0.09 |
| SubTB+SubM | 0.298±0.006 | 0.736±0.005 | 2.165±0.006 | 0.851±0.002 | 8.73±0.06 |
| RapTB | 0.996±0.001 | 0.740±0.004 | 2.448±0.017 | 0.860±0.001 | 6.14±0.03 |
| **RapTB+SubM** | 0.988±0.003 | **0.844±0.001** | **2.726±0.017** | **0.898±0.001** | 7.44±0.05 |
| **AvgPrefixTB** | **1.000±0.000** | 0.661±0.002 | 0.665±0.035 | 0.649±0.003 | 2.89±0.03 |

### SMILES AvgPrefixTB Per-Length QED Scores

| Len | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| QED | 0.644 | 0.645 | 0.737 | 0.661 | 0.722 | 0.621 | 0.637 | 0.560 | 0.518 | 0.438 |
| Count | 1186 | 535 | 514 | 258 | 263 | 166 | 147 | 76 | 40 | 15 |

### SMILES val metrics (training-time reference)

| Method | val/Acc | val/Diversity | val/Loss |
|---|---|---|---|
| SMILES AvgPrefixTB (RP) | 1.000 | 0.570 | 11.1 |
| SMILES AvgPrefixTB (SubM) | 0.999 | 2.372 | 23.5 |

## Key Observations

### VarExpr24
1. **RP/PRT: High Acc but severe mode collapse** — Acc ≥0.998 but NormCov ≤0.016, only 108–142 unique valid out of 6400 samples. The model converges to a few high-reward modes.
2. **Oracle dramatically improves diversity** — NormCov=0.183 (11× RP), 3369 unique valid, KL nearly zero. This shows AvgPrefixTB *can* learn diverse policies when given oracle data guidance.
3. **SubM collapsed on Expr24** — Only 7 unique valid. Root cause: `diversity_valid_only=true` starved the SubM buffer during early training when AvgPrefixTB is still learning to produce valid expressions. Buffer only reached 31/200 items. Re-run with `diversity_valid_ratio=0.5` also collapsed (76/200 items, but all length-3).

### SMILES (L=10)
1. **AvgPrefixTB achieves near-perfect accuracy (1.000)** but worst QED (0.661) and diversity (0.665) among all methods — even TB (Diversity=2.503, QED=0.717) is far superior.
2. **Severe length collapse** — mean length 2.89 (vs TB's 3.06 and RapTB+SubM's 7.44). 54% of samples are length 1–2, only 1.8% reach length 9–10.
3. **FPDiv (0.649)** is the lowest, confirming AvgPrefixTB generates structurally similar molecules.
4. **SubM partially rescues diversity** (val/diversity=2.37) — buffer filled to 200/200 since SMILES validity is easier.
5. **Pattern consistent with VarExpr24**: AvgPrefixTB converges to a narrow set of short, high-accuracy modes.

### SubM Tuning History (Expr24)

| SubM Run | valid_ratio | buffer_size | weight_len | alpha_power | val/diversity | Unique |
|---|---|---|---|---|---|---|
| v1 (original) | 1.0 (valid only) | 200 | 1.0 | 1.0 | 0.002 | 7 |
| v2 (ratio fix) | 0.5 | 200 | 1.0 | 1.0 | -0.000 | — |
| **v3 (final)** | **0.3** | **500** | **5.0** | **2.0** | **1.595** | **902** |

v1/v2 collapsed to length-3 expressions. v3 fix: lower valid_ratio (0.3), larger buffer (500), stronger length incentive (weight_len=5, alpha_power=2). This allows the buffer to collect longer invalid candidates early, which the length function then promotes.

## Interpretation for Rebuttal

These results directly support the paper's claims:
- Simple prefix-averaged TB **does not match RapTB** on diversity/coverage metrics
- AvgPrefixTB achieves high accuracy but collapses to few modes (RP: 142 unique / 6400 sampled)
- Oracle access partially rescues diversity (NormCov: 0.016 → 0.183)
- RapTB's rooted residual + absorbed target + pterm detach are **not cosmetic improvements** — they solve the mode collapse that AvgPrefixTB suffers from

## Experiment Configs

- RP: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB.yaml`
- PRT: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_PRT.yaml`
- SubM: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_subM_div_on_valid.yaml`
- Oracle: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_oracle.yaml`
- SMILES RP: `configs/experiment/SMILES_basic/SMILES_cfg_AvgPrefixTB.yaml`
- SMILES SubM: `configs/experiment/SMILES_SubM/SMILES_cfg_AvgPrefixTB_subM_replay_add_len_func.yaml`

## Eval Paths

- SMILES AvgPrefixTB eval: `logs/eval/smiles_CFG_AvgPrefixTB/eval/runs/2026-03-27_00-42-44/`
- SMILES AvgPrefixTB train ckpt: `logs/train/smiles_CFG_AvgPrefixTB/train/runs/2026-03-26_08-32-35/checkpoints/last.ckpt`
- WandB: https://wandb.ai/comdec/ChemGFN_eval/runs/vtquhva4

## Pending

- [x] ~~AvgPrefixTB SubM v3~~ — Done: NormCov=0.050, 902 unique (up from 7)
- [x] ~~SMILES eval~~ — Done: Acc=1.000, QED=0.661, Diversity=0.665, FPDiv=0.649
