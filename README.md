# Hierarchical Minimum Variance Allocation (HMVA)

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

**Central question:** Can a principled, multi-stage modification of Hierarchical Risk Parity (HRP) — integrating nonlinear shrinkage covariance estimation, a vol-balanced top-down tree, Black-Litterman return forecasts, and Kalman-filter weight smoothing — deliver superior out-of-sample risk-adjusted performance on a point-in-time S&P 500 universe?

---

## What this thesis does

Standard HRP allocates risk through a hierarchical clustering tree without inverting the covariance matrix. This paper proposes **HMVA** (Hierarchical Minimum Variance Allocator), a six-stage extension of HRP that addresses each of its known failure modes:

| Stage | Modification | Failure mode addressed |
|-------|----------------------------------------------|----------------------------------------------------------------|
| 1 | EWMA pseudo-returns (halflife = 21 days) | Equal weighting ignores volatility clustering |
| 2 | Nonlinear shrinkage covariance (NLS) | Sample Σ has biased eigenvalues at any N/T ratio |
| 3 | Black-Litterman expected returns (τ = 0.05) | Pure-risk allocation ignores return expectations |
| 4 | Vol-balanced top-down tree (λ = 0.25) | Static correlation tree conflates correlation and risk balance |
| 5 | Sharpe-ratio bisection | Variance bisection ignores expected risk-adjusted return |
| 6 | Kalman-filter weight smoother | Excessive turnover and noise-chasing in weight updates |

An ablation study on the top-100 S&P 500 stocks (2002–2024, T = 504) documents the marginal Sharpe contribution of each stage.

---

## Key results

### Ablation study — cumulative Sharpe improvement over HRP baseline

| Strategy | Sharpe | Max DD | Ann. Ret. | Ann. Vol. | Turnover | Marginal ΔSharpe |
|-----------------|--------|--------|-----------|-----------|----------|-----------------|
| **HMVA** | **0.809** | **−38.1%** | **14.0%** | 17.3% | 1.09 | +0.105 (Stage 6) |
| +BL+Sharpe | 0.704 | −35.4% | 13.6% | 19.3% | 1.22 | +0.068 (Stages 3+5) |
| +VBTree | 0.636 | −31.8% | 9.7% | 15.2% | 1.07 | +0.042 (Stage 4) |
| +NLS+EWMA | 0.594 | −42.5% | 9.3% | 15.6% | 0.42 | +0.019 (Stages 1+2) |
| HRP (baseline) | 0.574 | −41.8% | 9.4% | 16.3% | 0.32 | — |
| EW (1/N) | 0.477 | −55.2% | 9.0% | 18.8% | 0.06 | — |

HMVA achieves a cumulative Sharpe improvement of +0.235 over standard HRP and +0.332 over equal-weight. The Kalman-filter smoother (Stage 6) is the single largest contributor (+0.105).

### Cost robustness (HMVA vs EW)

Across 20 robustness cells (4 lookbacks × 5 cost levels, 0–20 bps):

| Metric | Value |
|----------------------|-------|
| Mean ΔSharpe vs EW | +0.301 |
| Pct. cells beating EW | 100% |
| Mean HMVA Sharpe | 0.773 |

HMVA's mean Sharpe advantage persists even at 20 bps per unit |Δw| (still +0.272 above EW at lookback = 252 days).

### Crisis-period performance (annualised Sharpe)

| Period | HMVA | HRP | EW |
|----------------------|-------|------|------|
| Dot-com trough (2002)| −0.95 | −0.88 | −0.92 |
| GFC (2007–2009) | **−0.31** | −0.88 | −0.89 |
| COVID crash (2020 Q1)| **−0.34** | −0.42 | −0.53 |
| Rate-hike cycle (2022)| −0.64 | −0.36 | −0.59 |

HMVA strongly outperforms in the GFC and COVID drawdowns; performance is mixed in the dot-com and rate-hike episodes.

---

## Project layout

```
analysis/
├── utils/
│   ├── strategy.py          # estimators, HMVA, allocators, backtest engine, tests
│   ├── data.py              # CRSP CSV loading + point-in-time universe (UniverseFn)
│   └── plotting.py          # visualisation helpers
├── run_backtest.py          # main strategy comparison (HMVA, HMVA-mv, HRP, MVO, EW, SPY-K)
├── run_ablation.py          # stage-by-stage ablation study
├── run_cost_robustness.py   # cost × lookback robustness sweep (20 cells)
├── run_crisis_analysis.py   # crisis-period and calm-period regime analysis
└── run_param_robustness.py  # one-at-a-time hyperparameter sensitivity
thesis/
├── main.tex                 # LaTeX thesis source
└── references.bib           # bibliography
data/                        # CRSP files (not committed — see Data section)
results/
├── backtest/                # main backtest outputs
├── ablation/                # stage-by-stage ablation outputs
├── cost_robustness/         # robustness sweep outputs
├── crisis_analysis/         # crisis-period analysis outputs
└── param_robustness/        # parameter sensitivity outputs
requirements.txt
```

---

## HMVA pipeline

### Stage 1 — EWMA pseudo-returns

Weights each observation exponentially with half-life h = 21 trading days (λ = 0.5^(1/21) ≈ 0.967). The pseudo-return matrix `R̃` satisfies `R̃ᵀR̃/T = Σ̂_EWMA`, allowing NLS in Stage 2 to operate on exponentially weighted data.

### Stage 2 — Nonlinear shrinkage (NLS) covariance

Applies Ledoit-Wolf (2020) analytical nonlinear shrinkage to `R̃`. Each sample eigenvalue receives an optimal asymptotic correction; eigenvectors are unchanged. NLS is positive-definite at any N/T ratio.

### Stage 3 — Black-Litterman expected returns

Constructs a return prior from the CAPM equilibrium `π = δ Σ̂ w_ew` (δ = 2.5, w_ew = 1/N), blends with a skip-1-month momentum signal as absolute views, and solves the BL posterior with uncertainty τ = 0.05. This regularises raw sample means toward the equilibrium, reducing noise in the return signal.

### Stage 4 — Vol-balanced top-down tree (λ = 0.25)

Replaces agglomerative clustering with a top-down binary tree that minimises at each split:

```
f(A,B) = λ × VB(A,B) + (1−λ) × ρ(A,B)
```

where VB(A,B) = |vol(A) − vol(B)| / (vol(A) + vol(B)) is the equal-weight volatility imbalance and ρ(A,B) is the equal-weight inter-cluster correlation. With λ = 0.25, the criterion balances vol-equity (25%) against diversification (75%). Clusters ≤ 10 assets use exhaustive search; larger clusters use an O(n²) heuristic via 2-D prefix sums.

### Stage 5 — Sharpe-ratio bisection

At each internal node, allocates weight between left and right subtrees proportional to their cluster Sharpe proxies:

```
Ŝ(C) = max(μ̄_BL,C / σ_EW(C), 0)
```

Falls back to inverse-variance split when both proxies are non-positive.

### Stage 6 — Kalman-filter weight smoother

Computes a time-varying gain K_t based on two signals:
- R_t = spectral entropy of Σ̂ eigenvalues (normalised to [0,1]; high = stable structure)
- Q_t = BL signal velocity / previous signal magnitude (high = large return signal)

```
K_t = Q_t / (Q_t + R_t)
w_t = (1 − K_t) × w_{t−1} + K_t × w_raw
```

When the covariance structure is stable and the return signal is small, K_t is small and the portfolio barely moves (low turnover). When the signal is large and the structure is changing, K_t is large and the portfolio updates aggressively.

---

## Strategies in `make_crsp_strategies()`

| Label | Tree | Covariance | Bisection |
|---------|-----------|-----------|-----------|
| `HMVA` | Vol-balanced (λ=0.25) | NLS + EWMA | Sharpe |
| `HMVA-mv` | Vol-balanced (λ=0.25) | NLS + EWMA | Vol |
| `HRP` | Agglomerative | NLS + EWMA | Vol |
| `MVO` | — | NLS + EWMA | MV utility |
| `EW` | — | — | Equal |
| `SPY-K` | — | — | Market cap |

The ablation baseline `HRP` uses sample covariance and agglomerative clustering (standard Lopez de Prado 2016).

---

## Parameter sensitivity

Two hyperparameters are swept in `run_param_robustness.py`:

| Parameter | Default | Grid | Key finding |
|-------------|---------|------|-------------|
| ewma_halflife | 21 | [5, 10, 14, 21, 30, 42, 63] | Sharpe stable at 0.72–0.76 for h ≥ 10; h = 5 gives EW-like performance |
| lam_cov | 0.25 | [0.0, 0.05, ..., 1.0] | Sharpe highest at λ = 1.0 (pure vol-balance); default 0.25 is conservative |

Note: Black-Litterman δ and τ cancel analytically in the posterior formula (P = I, Ω = τ diag(Σ)) so they are not swept.

---

## How to run

```bash
pip install -r requirements.txt
cd analysis
```

### Stage-by-stage ablation (primary result)

```bash
python run_ablation.py
```

Outputs: `results/ablation/metrics.csv`, `sharpe_delta.csv`, equity/drawdown/Sharpe-bar/waterfall plots.

### Main strategy comparison

```bash
python run_backtest.py
```

Outputs: `results/backtest/metrics.csv`, `statistical_tests.csv`, all diagnostic plots.

### Cost × lookback robustness sweep

```bash
python run_cost_robustness.py
```

Outputs: `results/cost_robustness/robustness_long.csv`, `robustness_summary.csv`, Sharpe heatmaps.

### Crisis-period analysis

```bash
python run_crisis_analysis.py
```

Outputs: `results/crisis_analysis/period_metrics.csv`, `regime_summary.csv`, rolling Sharpe and equity plots.

### Hyperparameter sensitivity

```bash
python run_param_robustness.py
```

Outputs: `results/param_robustness/sensitivity.csv`, per-parameter sensitivity curves.

---

## Data inputs

### `data/stock_daily_returns.csv` — CRSP daily file (CIZ format)

| Column | Notes |
|-----------|---------------------------------------------------------------------------|
| `PERMNO` | CRSP unique stock identifier |
| `DlyCalDt` | Calendar date (YYYY-MM-DD) |
| `DlyClose` | Split-adjusted closing price (used if DlyRet missing) |
| `DlyRet` | Daily total return (preferred over price-derived return) |
| `DlyCap` | Market capitalisation (used for top-K filtering) |

### `data/constiuents.csv` — S&P 500 historical membership

Range format: one row per (PERMNO, membership-spell) with columns `permno`, `start`, `ending`. `UniverseFn(date)` intersects the membership table with top-K market cap to give the exact investable universe at each rebalance date.

### Point-in-time construction

At each 21-day rebalance:

1. Find PERMNOs with active S&P 500 membership on that date.
2. Restrict to top 100 by market capitalisation.
3. Require complete non-NaN return history for the full lookback window.
4. Estimate Σ̂ on [t − T, t − 1]. Delisted stocks earn return 0 from the delisting date.

---

## Statistical inference

All comparisons use:

| Test | Description |
|-------------------------------|---------------------------------------------------|
| **LW (2008) block-bootstrap** | Circular blocks, b = 21; compares Sharpe ratios |
| **Holm-Bonferroni** | Step-down FWER correction across strategies |
| **Benjamini-Hochberg** | FDR correction (more powerful for multiple H1) |

---

## Limitations

1. **Transaction costs**: Main ablation uses zero costs. HMVA's monthly turnover ≈ 1.09; at 10 bps per unit |Δw|, annual cost ≈ 130 bps, reducing Sharpe by ~0.16. Cost-robustness sweep confirms HMVA still beats EW at all cost levels up to 20 bps.
2. **Price returns**: `DlyClose` excludes dividends. Total-return comparison would raise all strategies by approximately the same dividend yield (~1.5–2%/year).
3. **Universe**: Top-100 S&P 500 universe is dominated by large-cap stocks; results may differ for broader or international universes.
4. **Attribution**: HMVA combines six stages; the ablation quantifies marginal contributions, but joint attribution (interaction effects) is not separately measured.

---

## Bibliography (key references)

- Ledoit, O., & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix.* Journal of Portfolio Management.
- Ledoit, O., & Wolf, M. (2008). *Robust Performance Hypothesis Testing with the Sharpe Ratio.* Journal of Empirical Finance.
- Ledoit, O., & Wolf, M. (2020). *Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices.* Annals of Statistics.
- Fan, J., Liao, Y., & Mincheva, M. (2013). *Large Covariance Estimation by Thresholding Principal Orthogonal Complements.* JRSS-B.
- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out of Sample.* Journal of Portfolio Management.
- Molyboga, M. (2020). *A Modified Hierarchical Risk Parity Framework for Portfolio Management.* Journal of Financial Data Science.
- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive Diversification.* Review of Financial Studies.
- Diebold, F.X., & Mariano, R.S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics.
