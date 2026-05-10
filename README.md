# Hierarchical Minimum Variance Allocation (HMVA)

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

**Central question:** Can a principled, multi-stage modification of Hierarchical Risk Parity (HRP) — integrating nonlinear shrinkage, regime-adaptive tree construction, and Black-Litterman return forecasts — deliver superior out-of-sample risk-adjusted performance on a point-in-time S&P 500 universe?

---

## What this thesis does

Standard HRP allocates risk through a hierarchical clustering tree without inverting the covariance matrix. This paper proposes **HMVA** (Hierarchical Minimum Variance Allocator), a six-stage extension of HRP that addresses each of its known failure modes:

| Stage | Modification                                      | Failure mode addressed                                           |
| ----- | ------------------------------------------------- | ---------------------------------------------------------------- |
| 1     | EWMA pseudo-returns (halflife=21 days)            | All observations equally weighted; ignores volatility clustering |
| 2     | Nonlinear shrinkage covariance (NLS)              | Sample Σ biased at any N/T ratio                                 |
| 3     | Black-Litterman expected returns (δ=2.5, τ=0.05)  | Pure-risk allocation ignores return expectations                 |
| 4     | Regime-adaptive vol-balanced tree                 | Static tree conflates correlation and risk                       |
| 5     | Sharpe-ratio bisection                            | Variance bisection ignores expected risk-adjusted return         |
| 6     | L2 blend to 1/N (10%) + L1 turnover filter (5 pp) | Concentration and unnecessary rebalancing                        |

A factor-model simulation study validates the NLS covariance choice across two data-generating processes. The main empirical experiment compares HMVA against HRP (sample covariance), GMV (sample covariance), equal-weight (1/N), and a market-cap top-K benchmark on the top 100 S&P 500 stocks by market cap under a fully point-in-time constituent membership from 2002 to 2024.

---

## Key results

| Strategy        | Sharpe (T=252) | Sharpe (T=504) | Ann. Ret. (T=252) | Max DD (T=252) |
| --------------- | :------------: | :------------: | :---------------: | :------------: |
| **HMVA**        |   **0.807**    |   **0.804**    |     **13.7%**     |   **−34.2%**   |
| HRP (sample)    |     0.605      |     0.574      |       9.6%        |     −42.2%     |
| GMV (sample)    |     0.544      |     0.462      |       7.9%        |     −38.6%     |
| EW (1/N)        |     0.477      |     0.477      |       9.0%        |     −55.2%     |
| SPY-K (mkt-cap) |     0.503      |     0.503      |       9.5%        |     −52.3%     |

HMVA's Sharpe ratio advantage over EW (+0.33 at T=252) carries a raw block-bootstrap p-value of 0.062, which falls to 0.186 after Holm correction. HMVA beats EW in 100% of the 8 robustness cells (2 lookbacks × 4 cost levels), with a mean ΔSharpe of 0.287.

---

## Project layout

```
analysis/
├── utils/
│   ├── strategy.py       # estimators, HMVA, allocators, backtest engine, tests
│   ├── data.py           # CRSP CSV loading + point-in-time universe (UniverseFn)
│   └── plotting.py       # 17 visualisation functions
├── run_backtest.py       # *** main experiment: 2-lookback CRSP PIT backtest ***
├── run_robustness.py     # robustness sweep (lookback × cost × linkage)
└── run_simulation.py     # factor-model simulation study (2 DGPs × 2 N/T ratios)
thesis/
├── main.tex
└── references.bib
data/                     # CRSP files (not committed — see Data section)
results/
├── crsp_lb252/           # T=252 backtest outputs
├── crsp_lb504/           # T=504 backtest outputs
├── crsp_summary.csv      # cross-lookback merged table
├── simulation/           # simulation outputs
└── crsp_robustness/      # robustness sweep outputs
requirements.txt
```

---

## Strategies

| Label   | Allocator                           | Covariance                    |
| ------- | ----------------------------------- | ----------------------------- |
| `HMVA`  | Vol-balanced HRP + Sharpe bisection | NLS (with EWMA pre-weighting) |
| `HRP`   | Standard HRP (average linkage)      | **Sample**                    |
| `GMV`   | Long-only minimum variance          | **Sample**                    |
| `EW`    | Equal weight (1/N)                  | —                             |
| `SPY-K` | Market-cap top-K                    | —                             |

`EW` is the primary statistical baseline. `HRP` and `GMV` show what the same approach achieves with plain sample covariance.

---

## HMVA design: stage 4 in detail — regime-adaptive lambda

The key novelty beyond a standard NLS-covariance HRP is the **regime-adaptive tree construction criterion**. At each rebalance, the vol-balanced split objective uses a dynamic λ:

```
λ_eff = clip(λ_base + λ_scale × CV_vol + λ_corr × avg_corr, 0, 1)
      = clip(0.0    + 0.5     × CV_vol + 0.2    × avg_corr, 0, 1)
```

where `CV_vol = std(vols) / mean(vols)` is the cross-sectional coefficient of variation of asset volatilities and `avg_corr` is the mean off-diagonal correlation. This makes the tree:

- **More correlation-focused** (λ_eff ≈ 0.2) in calm markets — groups uncorrelated assets for maximum diversification
- **More vol-balance focused** (λ_eff ≈ 0.5) in crises — balances risk across clusters when vol dispersion is high

The resulting split objective is `λ_eff × vol_balance + (1−λ_eff) × ρ(A,B)`.

---

## Simulation study

Two synthetic DGPs provide a theoretical anchor for the NLS choice:

| DGP              | True Σ structure                            | Predicted winner | Rationale                                              |
| ---------------- | ------------------------------------------- | ---------------- | ------------------------------------------------------ |
| `factor_sparse`  | K=3 low-rank + banded sparse residual       | POET             | Exploits the exact structure POET assumes              |
| `dispersed_eigs` | Dense power-law eigenvalue spectrum (α=0.7) | NLS              | No factor structure; NLS eigenvalue correction optimal |

Parameters: N=200, T=300, n_reps=50, N/T ∈ {2.52, 5.04}.

**Results (min-var portfolio variance, relative to Sample baseline):**

| DGP              | N/T  | LW improvement | NLS improvement | POET improvement |
| ---------------- | ---- | :------------: | :-------------: | :--------------: |
| `dispersed_eigs` | 2.52 |     −25.6%     |   **−30.2%**    |      −21.9%      |
| `dispersed_eigs` | 5.04 |     −7.5%      |   **−11.7%**    |      −4.1%       |
| `factor_sparse`  | 2.52 |     −27.5%     |     −33.8%      |    **−37.2%**    |
| `factor_sparse`  | 5.04 |     −8.5%      |     −14.1%      |    **−18.4%**    |

All improvements are vs Sample. All are statistically significant (Wilcoxon p < 10⁻¹⁵).
NLS significantly beats LW in all (regime, DGP) cells (p < 10⁻¹⁵).
POET beats LW significantly in `factor_sparse`; POET is significantly worse than LW in `dispersed_eigs`.

**Takeaway:** NLS is the universally safe choice — never worst, always improves over sample. POET is optimal only when the true Σ has a genuine factor-plus-sparse structure. HMVA uses NLS.

---

## Robustness sweep

Sweeps 2 lookbacks × 4 cost levels = 8 cells (single linkage):

| Axis             | Values                          |
| ---------------- | ------------------------------- |
| Lookback T       | 252, 504 (days)                 |
| Transaction cost | 0, 2, 5, 10 bps per unit \|Δw\| |
| Linkage          | single                          |

Results from `results/crsp_robustness/robustness_summary.csv`:

| Strategy | Mean ΔSharpe vs EW | Pct cells beating EW | Mean Sharpe | Mean turnover |
| -------- | :----------------: | :------------------: | :---------: | :-----------: |
| HMVA     |     **+0.287**     |       **100%**       |    0.734    |     1.008     |
| SPY-K    |       +0.026       |         100%         |    0.473    |     0.066     |
| EW       |       0.000        |          —           |    0.447    |     0.057     |

---

## How to run

```bash
pip install -r requirements.txt
```

### Main experiment

```bash
cd analysis
python run_backtest.py    # uses defaults: lookbacks=[252,504], cost=0, top-k=100
```

Outputs: `results/crsp_lb{252,504}/metrics.csv`, `statistical_tests.csv`, PNG plots, and `results/crsp_summary.csv`.

### Robustness sweep

```bash
python run_robustness.py  # 8-cell sweep: 2 lookbacks × 4 cost levels
```

### Simulation

```bash
python run_simulation.py  # no data files needed
```

---

## Data inputs

### `data/stock_daily_returns.csv` — CRSP daily file

| Column     | Notes                                                                        |
| ---------- | ---------------------------------------------------------------------------- |
| `PERMNO`   | CRSP unique stock identifier                                                 |
| `DlyCalDt` | Calendar date (YYYY-MM-DD)                                                   |
| `DlyClose` | Split-adjusted closing price                                                 |
| `DlyRet`   | Daily total return (used in preference to price-derived return when present) |

### `data/constiuents.csv` — S&P 500 historical membership

Range format: one row per (PERMNO, membership-spell). `UniverseFn(date)` intersects with top-K market cap to give the exact investable universe at each rebalance date.

### Point-in-time construction

At each 21-day rebalance:

1. Find PERMNOs in S&P 500 on that date (from `constiuents.csv`).
2. Restrict to top 100 by market cap.
3. Keep only stocks with complete data for `max(lookbacks) = 504` days (ensures fair comparison between lookback configurations).
4. Estimate covariance on the lookback window. Delisted stocks earn 0 return from the delisting date.

---

## Expected output structure

```
results/crsp_lb252/
├── metrics.csv              # 5 strategies × 14 metrics
├── statistical_tests.csv    # DM + LW Sharpe tests, Holm & BH corrected, vs EW
├── daily_excess_returns.csv # daily portfolio excess returns
└── *.png                    # equity curves, drawdowns, Sharpe bars, etc.
results/crsp_summary.csv     # both lookbacks side-by-side
results/simulation/
├── sim_summary.csv          # mean min-var variance by (DGP, N/T, estimator)
├── sim_paired_tests_vs_sample.csv
├── sim_paired_tests_vs_lw.csv
└── *.png                    # improvement bar chart with 95% CIs
results/crsp_robustness/
├── robustness_long.csv      # one row per (cell, strategy)
├── robustness_summary.csv   # aggregated: mean ΔSharpe, pct-positive, turnover
└── heatmap_*.png
```

---

## Statistical inference

All pairwise comparisons vs EW use:

| Test                          | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| **Diebold-Mariano (HAC)**     | Paired loss test; Newey-West HAC variance         |
| **LW (2008) block-bootstrap** | Circular blocks, block=21; compares Sharpe ratios |
| **Holm-Bonferroni**           | Step-down FWER correction                         |
| **Benjamini-Hochberg**        | FDR correction (more power)                       |

---

## Limitations

1. **Zero transaction costs** in main run (`COST_BPS=0.00`). HMVA's monthly turnover is ~1.0 (vs 0.06 for EW); at 5 bps per unit turnover, annual cost ≈ 60 bps, reducing HMVA net Sharpe by ~0.35. Robustness sweep shows Sharpe still above EW at all cost levels.
2. **N/T regime**: with N=100 and T=252, N/T≈0.40. Both lookbacks are in the moderate (non-singular) regime; the high-dimensional advantage of NLS is demonstrated in simulation only.
3. **Price returns**: `DlyClose` is split-adjusted but excludes dividends. CRSP `RET` (total return including dividends) would be more accurate.
4. **Single linkage** in robustness sweep. The main run uses average linkage for HRP.
5. **Attribution**: HMVA outperforms HRP (sample) by combining NLS covariance, EWMA, BL returns, adaptive tree, and Sharpe bisection. This paper does not ablate each component separately.

---

## Bibliography (key references)

- Ledoit, O., & Wolf, M. (2004). _A well-conditioned estimator for large-dimensional covariance matrices._ JMVA.
- Ledoit, O., & Wolf, M. (2020). _Analytical nonlinear shrinkage of large-dimensional covariance matrices._ Annals of Statistics.
- Fan, J., Liao, Y., & Mincheva, M. (2013). _Large covariance estimation by thresholding principal orthogonal complements._ JRSS-B.
- López de Prado, M. (2016). _Building diversified portfolios that outperform out of sample._ Journal of Portfolio Management.
- Molyboga, M. (2020). _A modified hierarchical risk parity framework._ Journal of Financial Data Science.
- Black, F., & Litterman, R. (1992). _Global portfolio optimization._ Financial Analysts Journal.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). _Optimal versus naive diversification._ Review of Financial Studies.
