# BSc Thesis — Regime-Aware Portfolio Optimisation

**Topic:** Optimal portfolio strategies over a large equity universe with market-regime conditioning

**Core idea:** Can a machine-learning return-forecasting model paired with mean-variance optimisation outperform simple benchmarks, and does augmenting the model with market-regime information produce a further, consistent improvement?

---

## Data

| Source | Description |
|---|---|
| Yahoo Finance | Adjusted daily OHLCV for 44 large-cap S&P 500 constituents, 2014–2026 |
| S&P 500 index (^GSPC) | Used as the benchmark and as the input series for regime detection |

Rebalancing is **monthly**; a 36-month rolling window is used for model training.

---

## Methodology

### 1. Market-Regime Detection (Sparse Jump Model)

A **Sparse Jump Model** (Nystrup et al., 2021) is fitted on EWM features of S&P 500 log-returns — drift, log-volatility, and downside semi-deviation — across three half-lives (5, 21, 63 days). The model identifies two hidden states:

- **State 0 — Bear:** high volatility, negative drift, large downside
- **State 1 — Bull:** low volatility, positive drift

Training uses data through end-2015; online (look-ahead-free) predictions are generated from 2016 onward. The dominant features selected by the sparse penalty are `downside_63` (weight 0.898), `logvol_63` (0.711), and `downside_21`.

### 2. Feature Engineering

For each asset and each month-end the following characteristics are computed:

- **Momentum:** 3-, 6-, and 12-month price momentum
- **Volatility:** 21-day realised volatility (annualised); idiosyncratic volatility (CAPM residual)
- **Liquidity:** Amihud illiquidity ratio; volume trend (12-month change)
- **Tail risk:** Maximum single-day return over the prior month
- **Fundamentals:** E/P ratio, debt-to-equity, ROE, revenue growth (from yfinance), log market cap
- **Seasonality:** Month-of-year dummies
- **Rolling aggregates:** 3-, 6-, 12-month rolling mean and standard deviation of key series
- **Interaction terms:** size × momentum, volatility × momentum
- **Regime features** *(regime-augmented variant only)*: current regime label (0/1), days continuously in the current regime, and regime × each major characteristic interaction

### 3. Return Forecasting (XGBoost, Walk-Forward)

An **XGBoost regressor** is trained each month on the rolling 36-month window to predict next-month stock returns. Input features are augmented with the top 3 PCA factors (fit on the training window only) before model training to capture latent cross-sectional structure. Forecasts are then **shrunk toward the cross-sectional mean** (shrink factor 0.25) to reduce overfitting.

### 4. Portfolio Construction (Mean-Variance + Ledoit-Wolf)

Expected returns from XGBoost are combined with a **Ledoit-Wolf shrunk covariance** (12-month lookback) in a long-only mean-variance optimisation:

$$\max_w \; \mu^\top w - \tfrac{\lambda}{2}\, w^\top \Sigma w - \gamma \,\|w - w_{t-1}\|_1$$

Parameters: risk aversion λ = 2.5, turnover penalty γ = 0.01. The problem is solved with SLSQP.

### 5. Benchmarks

| Strategy | Description |
|---|---|
| **XGBoost + Markowitz** | Full ML + MV pipeline (baseline, no regime features) |
| **1/n Equal Weight** | Naive equal allocation, monthly rebalanced |
| **SPY** | S&P 500 index (passive, market-cap weighted) |

---

## Results (Backtest period: Jul 2018 – Nov 2024, 77 months)

| Strategy | Ann. Return | Ann. Vol | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| **XGBoost + Markowitz** | **57.3%** | 42.6% | **1.250** | -43.6% |
| 1/n Equal Weight | 26.8% | 18.4% | 1.238 | -23.0% |
| SPY | 14.0% | 17.6% | 0.570 | -24.8% |

Key observations:

- The ML model achieves a materially higher Sharpe ratio (1.25) than the passive SPY (0.57), demonstrating the value of cross-sectional return prediction in portfolio construction.
- Equal-weight achieves a comparable Sharpe (1.24) at roughly half the volatility, suggesting the ML model's excess return comes with concentrated risk.
- The regime-augmented variant (with regime labels and interaction features fed as XGBoost inputs) is currently under evaluation to assess whether systematic regime conditioning further improves risk-adjusted performance.

---

## Project Structure

```
analysis/
  research.ipynb   — main backtest and regime-detection notebook
utils/
  plotting.py      — shared visualisation helpers
  data_mining.py   — data utilities
Bsc_Thesis.pdf     — thesis document (in progress)
```
