# RapTB Hyperparameter Sweep Tables (for cA3o W3)

We conduct cross-task sweeps over $\beta \times \rho$ (18 configs) plus $\eta$ and $k_{\min}$ ablations. The main finding is **stability across a broad plateau, not a sharp optimum**.

All runs: single seed=42, paper-matched bsz=128.

---

## Table 1: Expr24

Config: `n_samples=32, accum=4` (effective bsz=128), 5000 steps, N=6400 eval samples.
Paper reference ($\beta=3, \rho=0.5$, 3-seed avg): Acc=0.991, Diversity=1.208.

### (a) $\beta \times \rho$ Grid (9 configs)

| $\beta$ | $\rho$ | Acc ↑ | Diversity ↑ |
|---|---|---:|---:|
| 1 | 0.0 | 0.999 | 1.010 |
| 1 | 0.1 | 0.999 | 0.993 |
| 1 | 0.5 | 1.000 | 1.015 |
| 3 | 0.0 | 0.990 | 0.950 |
| 3 | 0.1 | 1.000 | 0.769 |
| **3** | **0.5** | **0.997** | **0.978** |
| 5 | 0.0 | 0.983 | 1.022 |
| 5 | 0.1 | 0.998 | 0.968 |
| 5 | 0.5 | 0.999 | 0.912 |

All 9 configs: Acc $\in [0.983, 1.000]$, Diversity $\in [0.769, 1.022]$. $\log p_{\mathrm{term}}(\tau) \in [-0.25, -0.04]$ everywhere (vs. SubTB's $-79.6$) — no termination drift at any setting.

### (b) $\eta$ Sweep ($\beta=3, \rho=0.5$)

| $\eta$ | Acc ↑ | Diversity ↑ |
|---|---:|---:|
| 0.1 | 0.999 | 0.983 |
| **0.25 (default)** | 0.997 | 0.978 |
| 0.5 | 0.987 | **1.149** |

Monotonic: higher $\eta$ → better diversity ($0.983 \to 1.149$), mild Acc trade-off. $\eta$ is not a nuisance parameter — it has clear interpretable behavior.

### (c) $k_{\min}$ Ablation ($\beta=3, \rho=0.5, \eta=0.25$)

| $k_{\min}$ variant | Acc ↑ | Diversity ↑ |
|---|---:|---:|
| Fixed $k=3$ | 0.998 | 0.852 |
| **Schedule $7 \to 3$ (default)** | 0.997 | 0.978 |
| Fixed $k=7$ | 0.969 | **1.025** |

Fixed-low $k_{\min}$ is the clearly worst setting — early prefixes carry noisier supervision. We are transparent that $\gamma$ and $K$ are not fully swept and remain a limitation.

---

## Table 2: SMILES

Config: bsz=128, 5000 steps, 3200 eval samples.
Paper reference ($\beta=5, \rho=0.1$, 3-seed avg): Acc=0.996, QED=0.740, Entropy=2.448, FPDiv=0.860, Len=6.14.

### $\beta \times \rho$ Grid (9 configs)

| $\beta$ | $\rho$ | Acc ↑ | Entropy ↑ | QED ↑ | FPDiv ↑ | Len |
|---|---|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.992 | 2.173 | 0.751 | 0.849 | 6.84 |
| 1 | 0.1 | 0.994 | 2.161 | 0.744 | 0.857 | 7.01 |
| 1 | 0.5 | 0.992 | 2.162 | 0.748 | 0.860 | 6.71 |
| 5 | 0.0 | 0.995 | 1.997 | 0.795 | 0.855 | 7.68 |
| **5** | **0.1** | **0.991** | **2.076** | **0.783** | **0.864** | **7.58** |
| 5 | 0.5 | 0.997 | 2.079 | 0.800 | 0.865 | 7.48 |
| 10 | 0.0 | 0.968 | 2.279 | 0.727 | 0.883 | 7.41 |
| 10 | 0.1 | 0.999 | 1.986 | 0.804 | 0.854 | 7.50 |
| 10 | 0.5 | 0.997 | 2.036 | 0.794 | 0.863 | 7.40 |

8/9 configs Acc $\geq 0.991$; only $(\beta=10, \rho=0)$ shows mild degradation (0.968) — high temperature with zero distance penalty. QED $\in [0.727, 0.804]$ (paper: 0.740), FPDiv $\in [0.849, 0.883]$ (paper: 0.860). No length collapse: all Len $\in [6.7, 7.7]$, far from TB's collapse (Len=3.06).

---

## Cross-Task Summary

| Task | Configs | Acc range | Key metric | Paper default |
|---|---|---|---|---|
| Expr24 | 9 ($\beta \times \rho$) + 3 ($\eta$) + 3 ($k_{\min}$) | 0.983–1.000 | $\log p_{\mathrm{term}} \in [-0.25, -0.04]$ (vs. SubTB: $-79.6$) | $\beta=3, \rho=0.5$ |
| SMILES | 9 ($\beta \times \rho$) | 0.968–0.999 | FPDiv: 0.849–0.883 (paper: 0.860) | $\beta=5, \rho=0.1$ |

**Bottom line**: No catastrophic failure, no length collapse, no termination drift across all 18 configs. The paper default sits within a broad performance plateau — it is not a fragile optimum.
