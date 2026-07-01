# Building Mean-Variance Portfolios That Perform Out-of-Sample

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

_Author: Kiril Ivanov · Supervisor: Dr. Sanders Barendse_

---

## HMVA Pipeline

### Stage 1 — Top-Down Correlation-Minimising Tree

Builds a complete binary tree over assets using top-down recursive bipartition. At each node the split that minimises the estimated correlation between the two equal-weighted child portfolios is selected:

```
f(L, R) = Corr_EW(L, R) = Cov_EW(L, R) / √(Var_EW(L) · Var_EW(R))
```

**Motivation:** for any interior allocation α ∈ (0,1) between branches L and R, parent variance is monotonically increasing in the cross-branch correlation. Minimising cross-branch correlation is therefore aligned with constructing a hierarchy whose upper-level branches are maximally diversifying. Volatility and return information are then introduced only in the recursive allocation step, creating a clean separation of roles.

Clusters of up to n*bf = 10 assets use exhaustive search; larger clusters use an O(n²) contiguous-cut heuristic via 2D prefix sums (see \_Split Criterion Walkthrough* below).

### Stage 2 — Sharpe-Ratio Capital Allocation

At each internal node, equal-weighted cluster Sharpe proxies are formed to reduce signal-estimation noise through cluster averaging:

```
Ŝ(C) = max( μ̄_BL(C) / σ_EW(C), 0 )
```

where μ̄_BL(C) is the equal-weight mean BL return and σ_EW(C) is the equal-weight volatility of cluster C. The truncation at zero preserves the long-only nature of the strategy.

The allocation fraction going to the left child L is Ŝ(L) / (Ŝ(L) + Ŝ(R)). If both proxies are zero or negative, allocation falls back to inverse-volatility bisection (risk-only mode). This risk-only variant is denoted **HMVA-mv**.

### Stage 3     — Latent Weight Smoother

Interprets rebalancing as a linear filtering problem for the latent portfolio weights. The smoothed allocation is:

```
w̃ = K · w_new + (1 − K) · w_old
```

where K ∈ [0,1] is a data-adaptive Kalman gain computed from two uncertainty proxies:

- **Y** = normalised return-signal velocity = ‖μ̂*new − μ̂_old‖ / ‖μ̂_old‖ — high when the return environment has changed materially (proxy for \_process* uncertainty).
- **X** = normalised spectral entropy of Σ̂ eigenvalues ∈ [0,1] — high when the covariance structure is diffuse, low when one eigenvalue dominates (proxy for _measurement_ uncertainty).

```
K = Y / (X + Y)
```

When the return signal shifts sharply (large Y) relative to a concentrated covariance structure (small X), the gain rises and the portfolio updates aggressively. When the signal is quiet and the covariance structure is diffuse, the gain falls and weights barely move.

---

## Project Layout

```
analysis/
├── utils/
│   ├── backtest.py          # HMVA pipeline, all strategy constructors, statistical tests
│   ├── data.py              # CRSP loading, point-in-time universe (UniverseFn)
│   └── plotting.py          # visualisation helpers
├── run_backtest.py          # main strategy comparison
├── run_cost_robustness.py   # transaction-cost robustness sweep
├── run_crisis_analysis.py   # NBER regime / sub-period analysis
└── run_split_comparison.py  # heuristic vs brute-force split accuracy simulation
thesis/
├── thesis.tex               # LaTeX thesis source
└── references.bib           # bibliography
data/                        # CRSP files
results/
├── backtest/                # metrics, statistical_tests, equity/DD plots
├── cost_robustness/         # robustness tables, Sharpe heatmaps
├── crisis_analysis/         # period_metrics, regime_summary, rolling plots
└── split_comparison/        # summary.csv, raw.csv, accuracy plots
requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
cd analysis
```

### To only replicate the papers results

Run all cells in makeall.ipynb

### Main strategy comparison

```bash
python run_backtest.py
```

Outputs: `results/backtest/metrics.csv`, `statistical_tests.csv`, equity/drawdown/Sharpe plots.

### Transaction-cost robustness sweep

```bash
python run_cost_robustness.py
```

Outputs: `results/cost_robustness/robustness_long.csv`, `robustness_summary.csv`, Sharpe heatmaps.

### Regime/crisis-period analysis

```bash
python run_crisis_analysis.py
```

Outputs: `results/crisis_analysis/period_metrics.csv`, `regime_summary.csv`, rolling Sharpe plots.

### Split criterion accuracy simulation

```bash
python run_split_comparison.py
```

Outputs: `results/split_comparison/summary.csv`, `raw.csv`, approximation-ratio plots.

---
