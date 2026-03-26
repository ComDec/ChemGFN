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

## SMILES (val metrics, training complete)

| Method | val/Acc | val/Diversity | val/Loss |
|---|---|---|---|
| SMILES AvgPrefixTB (RP) | 1.000 | 0.570 | 11.1 |
| SMILES AvgPrefixTB (SubM) | 0.999 | 2.372 | 23.5 |

## Key Observations

### VarExpr24
1. **RP/PRT: High Acc but severe mode collapse** — Acc ≥0.998 but NormCov ≤0.016, only 108–142 unique valid out of 6400 samples. The model converges to a few high-reward modes.
2. **Oracle dramatically improves diversity** — NormCov=0.183 (11× RP), 3369 unique valid, KL nearly zero. This shows AvgPrefixTB *can* learn diverse policies when given oracle data guidance.
3. **SubM collapsed on Expr24** — Only 7 unique valid. Root cause: `diversity_valid_only=true` starved the SubM buffer during early training when AvgPrefixTB is still learning to produce valid expressions. Buffer only reached 31/200 items. Re-run with `diversity_valid_ratio=0.5` also collapsed (76/200 items, but all length-3).

### SMILES
- **SubM works on SMILES** (diversity=2.37) — buffer filled to 200/200 with valid_frac=1.0. SMILES is easier to learn initially, so the buffer fills quickly.
- SubM collapse is specific to **Expr24 + AvgPrefixTB** interaction, not a general SubM failure.

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

## Pending

- [x] ~~AvgPrefixTB SubM v3~~ — Done: NormCov=0.050, 902 unique (up from 7)
- [ ] SMILES eval (test metrics)
