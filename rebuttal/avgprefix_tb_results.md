# AvgPrefixTB Baseline Results — VarExpr24

> Reviewer QHmk requested baseline: "just averaging the usual TB loss over all prefixes of the sequence"

## Method

**AvgPrefixTBLoss** (`chemgfn/models/losses.py`): Strict averaged-prefix TB.

$$\mathcal{L}_{\text{AvgPrefixTB}}(\xi) = \frac{1}{|K(\xi)|} \sum_{k \in K(\xi)} \left(\Delta_k^{\text{TB}}(\xi)\right)^2$$

where $K(\xi) = \{0, 1, \ldots, \tau\}$ and $\Delta_k^{\text{TB}} = \log Z_\theta + \sum_{t=0}^{k-1} \log p_F[t] + \log p_{\text{term}}[k] - \log r[k]$.

Config: `mode="avgprefix"`, `k_min=0`, `detach_pterm_in_aux=false`, learnable $\log Z$.

## Table 3: VarExpr24 (N=6400 samples)

| Method | Acc ↑ | Unique Valid | NormCov ↑ | KL(π→p\*) ↓ | KL(p\*→π) ↓ | JS ↓ | log p\_term(τ) ↑ |
|---|---|---|---|---|---|---|---|
| **AvgPrefixTB (RP)** | 0.998 | 142 | 0.016 | 0.808 | 8.908 | 0.213 | -0.560 |
| **AvgPrefixTB (PRT)** | 1.000 | 108 | 0.012 | 0.944 | 10.989 | 0.249 | -0.606 |
| **AvgPrefixTB (SubM)** | 1.000 | 7 | 0.000 | 1.442 | 14.506 | 0.372 | -0.000 |

## Key Observations

1. **High Acc (0.998–1.000)** — nearly all generated expressions are valid
2. **Very low NormCov (0.000–0.016)** — severe mode collapse, covering <2% of oracle solutions
3. **Large KL(p\*→π)** — oracle modes not covered by the learned policy
4. **SubM collapsed** — only 7 unique valid expressions; root cause: `diversity_valid_only=true` + slow early convergence of AvgPrefixTB starved the submodular buffer (buffer_total=31/200). Fix: set `diversity_valid_ratio=0.5`.

## Interpretation for Rebuttal

These results directly support the paper's claims:
- Simple prefix-averaged TB **does not match RapTB** on diversity/coverage metrics
- The naive averaging drives high accuracy but fails to maintain mode diversity
- This validates that RapTB's rooted residual + absorbed target + pterm detach are not cosmetic improvements
- SubM failure is a config interaction issue (being re-run with `diversity_valid_ratio=0.5`)

## Experiment Configs

- RP: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB.yaml`
- PRT: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_PRT.yaml`
- SubM: `configs/experiment/VarExpr24/VarExpr24_AvgPrefixTB_subM_div_on_valid.yaml`

## Eval Commands

```bash
PYTHON=/home/xiwang/miniforge3/envs/chemgfn/bin/python
# 200 test batches × 32 samples = 6400
$PYTHON chemgfn/eval.py experiment=VarExpr24/VarExpr24_AvgPrefixTB \
  trainer.devices=1 "+trainer.limit_test_batches=200" \
  ckpt_path=<path_to_last.ckpt>

# Then compute Table 3 metrics:
$PYTHON scripts/sweep/eval_expr24_table3.py \
  --csv-path <eval_run_dir>/test_samples/samples_test_0.csv \
  --buffer-path data/24_points/buffer_24_non_zero.pt
```

## Pending

- [ ] AvgPrefixTB SubM re-run with `diversity_valid_ratio=0.5`
- [ ] AvgPrefixTB Oracle
- [ ] SMILES results (running on GPU7)
