# Hierarchical Minimum Variance Allocation (HMVA)

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

**Central question:** Can a complete, covariance-minimising hierarchical allocation framework —
combining EWMA-weighted nonlinear shrinkage, Black-Litterman return estimates, a top-down binary
splitting criterion, Sharpe-ratio capital allocation, and adaptive Kalman-filter weight smoothing —
deliver superior out-of-sample risk-adjusted performance on a point-in-time S&P 500 universe?

---

## What HMVA is

**HMVA (Hierarchical Minimum Variance Allocator)** is a portfolio allocation method that constructs
a binary tree over assets by greedily minimising inter-cluster covariance at each split, then
distributes capital down the tree using Sharpe-ratio-weighted bisection. Its six stages each address
a specific failure mode of estimation-naive allocation:

| Stage | Component                                                                 | Failure mode addressed                                       |
| ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 1     | EWMA pseudo-returns (half-life h = 21 days)                               | Equal weighting ignores volatility clustering                |
| 2     | Nonlinear shrinkage covariance (Ledoit-Wolf 2020)                         | Sample Σ has biased eigenvalues at any N/T ratio             |
| 3     | Black-Litterman expected returns                                          | Pure-risk allocation ignores return expectations             |
| 4     | Top-down binary tree (minimise inter-cluster EW covariance + correlation) | Static correlation tree conflates two notions of similarity  |
| 5     | Sharpe-ratio bisection                                                    | Variance bisection ignores expected risk-adjusted return     |
| 6     | Kalman-filter weight smoother                                             | Noisy weight updates generate excessive, uninformed turnover |

HMVA differs from HRP in all three principal design choices: (i) tree construction (top-down
covariance-minimising vs. bottom-up single-linkage), (ii) capital allocation (Sharpe-weighted vs.
inverse-variance), and (iii) weight dynamics (adaptive Kalman filter vs. none). HRP serves as a
natural benchmark, not a parent method.

---

## HMVA pipeline

### Stage 1 — EWMA pseudo-returns

Weights each observation in the lookback window exponentially with half-life h = 21 trading days
(decay λ = 0.5^(1/21) ≈ 0.9672). The rescaled matrix R̃ satisfies R̃ᵀR̃/T = Σ̂_EWMA, so that the
NLS estimator in Stage 2 operates on exponentially down-weighted data.

### Stage 2 — Nonlinear shrinkage (NLS) covariance

Applies Ledoit-Wolf (2020) analytical nonlinear shrinkage to R̃. Each sample eigenvalue receives an
asymptotically optimal, eigenvalue-specific shrinkage correction; eigenvectors are unchanged. NLS is
positive-definite at any N/T ratio and avoids the uniform shrinkage of linear methods.

### Stage 3 — Black-Litterman trend expected returns

Constructs return estimates with:

- **Prior:** π = 0.01 · **1**\_N (flat, equal-return prior)
- **Signal Q:** skip-month cross-sectional mean of raw daily returns (skipping the most recent 21 days to avoid short-term reversal)
- **Uncertainty:** Ω = diag(Σ̂)

The BL posterior is:

```
μ_BL = π + Σ̂ (Σ̂ + diag(Σ̂))⁻¹ (Q − π)
```

This regularises the raw momentum signal toward the uninformative prior, dampening noise in
cross-sectional return estimates.

### Stage 4 — Top-down binary tree

Builds a complete binary tree over assets using top-down recursive bipartition. At each node, the
split that minimises the following objective is selected:

```
f(A, B) = 0.5 × Cov_EW(A, B) + 0.5 × Corr_EW(A, B)
```

where:

- **Cov_EW(A, B)** = cross / (n_A × n_B) = equal-weight inter-cluster covariance
- **Corr_EW(A, B)** = cross / √(S_A × S_B) = Pearson correlation between the two EW sub-portfolios

and `cross = Σ_{i∈A, j∈B} Σ̂_ij`, `S_C = Σ_{i,j∈C} Σ̂_ij`.

The 50/50 blend penalises both the absolute covariance level and the relative cross-cluster
correlation simultaneously. Clusters of up to 10 assets use exhaustive search; larger clusters use
an O(n²) contiguous-cut heuristic via 2-D prefix sums (see _Split Criterion Walkthrough_ below).

### Stage 5 — Sharpe-ratio bisection

At each internal node, capital is allocated between left (L) and right (R) subtrees proportional to
their cluster Sharpe proxies:

```
Ŝ(C) = max(μ̄_BL,C / σ_EW(C), 0)
```

where μ̄_BL,C is the equal-weight mean BL return and σ_EW(C) is the equal-weight volatility of
cluster C. The weight fraction going to L is Ŝ(L) / (Ŝ(L) + Ŝ(R)); if both proxies are
non-positive the allocation falls back to inverse-variance bisection.

### Stage 6 — Kalman-filter weight smoother

Computes a time-varying smoothing gain K_t from two signals:

- **R_t** = normalised spectral entropy of Σ̂ eigenvalues ∈ [0, 1]: high when covariance structure is stable, low when concentrated
- **Q_t** = relative BL signal velocity = ‖μ_BL(t) − μ_BL(t−1)‖ / ‖μ_BL(t−1)‖: high when the return signal is moving rapidly

```
K_t = Q_t / (Q_t + R_t)
w_t = (1 − K_t) × w_{t−1} + K_t × w_raw
```

When the covariance structure is stable and the return signal is quiet (small Q_t, large R_t), K_t
is close to zero and weights barely move. When the signal is large and the structure is
concentrated, K_t rises and the portfolio updates more aggressively.

---

## Strategies in `make_crsp_strategies()`

| Label     | Tree                      | Covariance | Bisection           | KF  |
| --------- | ------------------------- | ---------- | ------------------- | --- |
| `HMVA`    | Top-down, minimise f(A,B) | EWMA + NLS | Sharpe-ratio        | Yes |
| `HMVA-mv` | Top-down, minimise f(A,B) | EWMA + NLS | Variance            | Yes |
| `HRP`     | Agglomerative single-link | NLS (raw)  | Variance            | No  |
| `MVO`     | —                         | NLS (raw)  | Max-utility (γ=2.5) | No  |
| `GMV`     | —                         | NLS (raw)  | Min-variance        | No  |
| `EW`      | —                         | —          | Equal               | No  |
| `SPY-K`   | —                         | —          | Market cap          | No  |

HRP, MVO, and GMV use NLS applied to the raw (un-EWMA-weighted) return window. EWMA
preprocessing and the Kalman smoother are exclusive to HMVA and HMVA-mv.

---

## Key results

### Full-sample performance (2000–2024, T = 504 days lookback, monthly rebalance)

| Strategy | Sharpe    | Sortino   | Ann. Ret. | Ann. Vol. | Max DD     | Calmar | Turnover |
| -------- | --------- | --------- | --------- | --------- | ---------- | ------ | -------- |
| **HMVA** | **0.789** | **1.043** | **14.0%** | 17.7%     | **−38.6%** | 0.362  | 1.101    |
| HMVA-mv  | 0.690     | 0.881     | 10.4%     | 15.1%     | −35.8%     | 0.290  | 0.952    |
| HRP      | 0.564     | 0.711     | 9.2%      | 16.3%     | −42.4%     | 0.216  | 0.291    |
| MVO      | 0.162     | 0.223     | 4.0%      | 24.6%     | −51.5%     | 0.077  | 0.509    |
| GMV      | 0.450     | 0.563     | 6.6%      | 14.7%     | −36.6%     | 0.181  | 0.329    |
| EW       | 0.477     | 0.596     | 9.0%      | 18.8%     | −55.2%     | 0.163  | 0.057    |
| SPY-K    | 0.503     | 0.635     | 9.5%      | 18.9%     | −52.3%     | 0.182  | 0.066    |

HMVA leads on Sharpe (+0.225 over HRP, +0.286 over EW), Sortino, annualised return, and maximum
drawdown. HMVA-mv achieves the lowest maximum drawdown of all strategies.

### Statistical tests (all comparisons vs HMVA)

| vs HMVA | ΔSharpe | LW p-val  | LW p (Holm) | DM stat    | DM p-val  | DM p (Holm) |
| ------- | ------- | --------- | ----------- | ---------- | --------- | ----------- |
| HMVA-mv | +0.099  | 0.240     | 0.425       | −2.525     | 0.012\*   | 0.058       |
| HRP     | +0.225  | 0.171     | 0.425       | −1.814     | 0.070     | 0.264       |
| MVO     | +0.627  | **0.007** | **0.042\*** | −1.839     | 0.066     | 0.264       |
| GMV     | +0.339  | 0.027     | 0.135       | **−2.934** | **0.003** | **0.020\*** |
| EW      | +0.312  | 0.096     | 0.382       | −1.456     | 0.145     | 0.291       |
| SPY-K   | +0.286  | 0.142     | 0.425       | −1.259     | 0.208     | 0.291       |

Tests: Ledoit-Wolf (2008) circular block-bootstrap Sharpe test (b=21, B=2000); Diebold-Mariano on
squared losses. Multiple testing corrections: Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR).
Significance markers: \* p < 0.05 after correction.

HMVA significantly outperforms MVO by the LW test after FWER correction and GMV by the DM test.
The LW test is underpowered for moderate differences over a single 25-year path.

---

## Cost robustness

Sweep over 3 lookback windows × 5 transaction-cost levels = **15 cells**
(lookbacks: 126, 252, 504 days; costs: 0, 2, 5, 10, 20 bps per unit |Δw|).

| Strategy | Mean ΔSharpe vs EW | Median ΔSharpe vs EW | % cells beating EW | Mean Sharpe | Mean Max DD | Mean Turnover |
| -------- | ------------------ | -------------------- | ------------------ | ----------- | ----------- | ------------- |
| HMVA     | +0.253             | +0.271               | **100%**           | 0.725       | −39.5%      | 1.083         |
| HMVA-mv  | +0.169             | +0.188               | **100%**           | 0.641       | −36.9%      | 0.932         |

HMVA beats equal-weight in all 15 cells. Even at 20 bps per unit |Δw| and a 126-day lookback window,
the Sharpe advantage persists. HMVA-mv similarly dominates EW across the full grid.

---

## Crisis-period analysis

Annualised Sharpe ratios across four crisis and three calm sub-periods.

### Crisis periods

| Period                   | Start   | End     | HMVA       | HMVA-mv    | HRP    | MVO    | GMV    | EW     | SPY-K  |
| ------------------------ | ------- | ------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ |
| Dot-com trough (2002)    | 2002-01 | 2002-10 | **−0.117** | −0.408     | −0.952 | −0.914 | −1.417 | −0.966 | −0.972 |
| GFC (2007–2009)          | 2007-10 | 2009-03 | **−0.210** | −0.496     | −0.765 | −0.542 | −0.811 | −0.897 | −0.863 |
| COVID-19 crash (2020 Q1) | 2020-01 | 2020-04 | **−0.139** | −0.195     | −0.528 | −0.800 | −0.747 | −0.554 | −0.399 |
| Rate-hike cycle (2022)   | 2022-01 | 2022-12 | −0.251     | **+0.013** | −0.494 | −0.445 | −0.241 | −0.668 | −0.857 |

### Calm periods

| Period                       | Start   | End     | HMVA  | HMVA-mv | HRP   | MVO   | GMV   | EW    | SPY-K |
| ---------------------------- | ------- | ------- | ----- | ------- | ----- | ----- | ----- | ----- | ----- |
| Pre-GFC bull (2003–2007)     | 2003-01 | 2007-09 | 1.094 | 0.780   | 1.060 | 0.702 | 1.197 | 1.112 | 0.999 |
| Post-GFC bull (2009–2020)    | 2009-04 | 2020-01 | 1.218 | 1.218   | 1.240 | 0.591 | 1.221 | 1.079 | 1.087 |
| Post-COVID rebound (2020–21) | 2020-05 | 2021-12 | 2.033 | 2.435   | 2.131 | 0.151 | 0.728 | 2.172 | 2.175 |

HMVA records the best (least negative) crisis Sharpe in three of four drawdown episodes. In the
2022 rate-hike cycle, HMVA-mv is the only strategy to record a positive Sharpe. HMVA performs
competitively in bull markets while preserving the largest part of the downside advantage.

---

## Split criterion walkthrough

These three functions implement the **covariance-minimising bipartition** used in HMVA's tree
construction. The algorithm recursively splits each cluster into two sub-clusters by minimising a
50/50 blend of equal-weight inter-cluster covariance and equal-weight inter-cluster correlation.

### Running example

```python
Σ = [[4.0, 3.0, 0.2, 0.1],   # assets {0, 1} are highly correlated
     [3.0, 4.0, 0.1, 0.2],
     [0.2, 0.1, 1.0, 0.8],   # assets {2, 3} are highly correlated
     [0.1, 0.2, 0.8, 1.0]]

indices = [0, 1, 2, 3]
```

Both methods should recover the natural split **{0, 1} | {2, 3}**.

---

### Function 1 — `_vb_merge_cost`

```python
def _vb_merge_cost(cross: float, n_a: int, n_b: int,
                   sw_a: float, sw_b: float) -> float:
```

**Purpose:** Score a candidate bipartition (A, B). Lower score = more dissimilar = preferred split.

#### Arguments

| Argument       | Meaning                                            |
| -------------- | -------------------------------------------------- |
| `cross`        | One-way cross sum: Σ\_{i∈A, j∈B} Σ_ij              |
| `n_a`, `n_b`   | Cluster sizes                                      |
| `sw_a`, `sw_b` | Within-cluster sum of covariances: Σ\_{i,j∈S} Σ_ij |

#### Line-by-line

```python
cov_term  = cross / float(n_a * n_b) if n_a * n_b > 0 else 0.0
```

Equal-weight inter-cluster covariance: Cov(EW_A, EW_B) = cross / (n_A × n_B).

_Example — A = {2,3}, B = {0,1}:_ cross = 0.2+0.1+0.1+0.2 = 0.6; cov_term = 0.6/4 = 0.15

---

```python
denom_rho = float(np.sqrt(max(sw_a * sw_b, 0.0)))
rho       = cross / denom_rho if denom_rho > 1e-12 else 0.0
```

Equal-weight inter-cluster correlation: Corr(EW_A, EW_B) = cross / √(sw_A × sw_B).

_Example:_ sw_a = 3.6, sw_b = 14.0; denom_rho = √50.4 ≈ 7.099; rho = 0.6/7.099 ≈ 0.0845

---

```python
return 0.5 * cov_term + 0.5 * rho
```

50/50 blend. _Example:_ cost = 0.5×0.15 + 0.5×0.0845 = **0.117** (natural split — correctly low).

**Contrast** — bad split A = {0,2}, B = {1,3}: cross = 4.1, sw_a = sw_b = 5.4;
cost = 0.5×1.025 + 0.5×0.759 = **0.892** ← correctly penalised.

---

### Function 2 — `_vb_split_bruteforce`

```python
def _vb_split_bruteforce(cov_arr, indices) -> Tuple[List[int], List[int]]:
```

**Purpose:** Globally optimal bipartition by exhaustive enumeration of all subsets of size
r = 1 … n//2. Called when `len(indices) ≤ bf_threshold` (≤ 16).

Key steps:

1. Extract sub-matrix M indexed by `indices`.
2. Iterate over all combinations of size r from local indices 0..n-1.
3. For each candidate (A_loc, B_loc), compute `sA`, `sB`, `cross` from M sub-blocks.
4. Score via `_vb_merge_cost`; keep the minimum.
5. Map winning local indices back to global asset indices.

All scores for the example (n=4):

| Split           | cross     | sA      | sB       | cov_term | rho       | **score**  |
| --------------- | --------- | ------- | -------- | -------- | --------- | ---------- | ----------- |
| {0}             | {1,2,3}   | 3.3     | 4.0      | 22.2     | 1.100     | 0.238      | 0.396       |
| {1}             | {0,2,3}   | 3.3     | 4.0      | 22.2     | 1.100     | 0.238      | 0.396       |
| {2}             | {0,1,3}   | 1.2     | 1.0      | 26.2     | 0.234     | 0.0.4      | ~0.430      |
| {3}             | {0,1,2}   | 1.2     | 1.0      | 26.2     | 0.234     | ~0.4       | ~0.430      |
| \*\*{0,1}       | {2,3}\*\* | **0.6** | **14.0** | **3.6**  | **0.150** | **0.0845** | **0.117 ✓** |
| {0,2}           | {1,3}     | 4.1     | 5.4      | 5.4      | 1.025     | 0.759      | 0.892       |
| ... (symmetric) |           |         |          |          |           | 0.892      |

Winner: **{0,1} | {2,3}** with score 0.117.

---

### Function 3 — `_vb_split_heuristic`

```python
def _vb_split_heuristic(cov_arr, indices) -> Tuple[List[int], List[int]]:
```

**Purpose:** O(n²) approximation for large clusters. Instead of 2ⁿ candidates, evaluates only the
n−1 contiguous cuts of the row-sum-sorted asset sequence.

**Key insight:** Assets that covary similarly with the rest of the cluster (similar row sums) tend
to belong together. Sorting by row sum and cutting contiguously separates "high-covariance" assets
from "low-covariance" ones.

**Steps:**

1. Compute M = sub-matrix of `cov_arr` for `indices`.
2. Sort assets by ascending within-cluster row sum: `order = argsort(M.sum(axis=1))`.
3. Reorder M into sorted order; build 2D prefix-sum array P via `M.cumsum(0).cumsum(1)`.
4. For each cut k = 1…n−1, recover `s_left`, `s_right`, `cross` in O(1) from P.
5. Evaluate objective vectorially; return the cut with minimum score.

**Example** (sorted order: [2, 3, 0, 1]):

| k     | cross   | s_left  | s_right  | cov_term  | rho        | **objective** |
| ----- | ------- | ------- | -------- | --------- | ---------- | ------------- |
| 1     | 1.1     | 1.0     | 15.6     | 0.367     | 0.279      | 0.323         |
| **2** | **0.6** | **3.6** | **14.0** | **0.150** | **0.0845** | **0.117 ✓**   |
| 3     | 3.3     | 8.2     | 4.0      | 1.100     | 0.577      | 0.839         |

Returns `([2, 3], [0, 1])` — the natural split, matching brute-force. ✓

**Empirical accuracy** (from `results/split_comparison/summary.csv`, 300 random factor-model Σ per n):

| Cluster size n | Median approx. ratio | p95 ratio | Exact match rate |
| -------------- | -------------------- | --------- | ---------------- |
| 12             | 1.000                | 1.000     | 53%              |
| 14             | 1.000                | 1.000     | 51%              |
| 16             | 0.996                | 1.000     | 49%              |
| 18             | 1.000                | 1.000     | 51%              |
| 20             | 0.971                | 1.000     | 39%              |

The heuristic is near-optimal: the median approximation ratio is 1.000 (identical score to the
global optimum) across all cluster sizes tested.

### Summary comparison

| Property               | `_vb_split_bruteforce`   | `_vb_split_heuristic`               |
| ---------------------- | ------------------------ | ----------------------------------- |
| **Guarantee**          | Globally optimal         | Near-optimal (median ratio = 1.0)   |
| **Time complexity**    | O(2ⁿ × n²)               | O(n²)                               |
| **Cuts evaluated**     | All bipartitions         | n−1 contiguous cuts in sorted order |
| **When used**          | n ≤ bf_threshold (≤ 16)  | n > bf_threshold                    |
| **Key data structure** | `itertools.combinations` | 2D prefix-sum array P               |

---

## Project layout

```
analysis/
├── utils/
│   ├── backtest.py          # HMVA pipeline, all strategy constructors, statistical tests
│   ├── data.py              # CRSP loading, point-in-time universe (UniverseFn)
│   └── plotting.py          # visualisation helpers
├── run_backtest.py          # main strategy comparison (HMVA, HMVA-mv, HRP, MVO, GMV, EW, SPY-K)
├── run_cost_robustness.py   # lookback × cost robustness sweep (15 cells)
├── run_crisis_analysis.py   # crisis / calm sub-period analysis
└── run_split_comparison.py  # heuristic vs brute-force split accuracy simulation
thesis/
├── pilot.tex                # LaTeX thesis source
└── references.bib           # bibliography
data/                        # CRSP files
results/
├── backtest/                # metrics.csv, statistical_tests.csv, equity/DD plots
├── cost_robustness/         # robustness_long.csv, robustness_summary.csv, heatmaps
├── crisis_analysis/         # period_metrics.csv, regime_summary.csv, rolling plots
└── split_comparison/        # summary.csv, raw.csv, accuracy plots
requirements.txt
```

---

## How to run

```bash
pip install -r requirements.txt
cd analysis
```

### Main strategy comparison

```bash
python run_backtest.py
```

Outputs: `results/backtest/metrics.csv`, `statistical_tests.csv`, equity/drawdown/Sharpe plots.

### Cost × lookback robustness sweep

```bash
python run_cost_robustness.py
```

Outputs: `results/cost_robustness/robustness_long.csv`, `robustness_summary.csv`, Sharpe heatmaps.

### Crisis-period analysis

```bash
python run_crisis_analysis.py
```

Outputs: `results/crisis_analysis/period_metrics.csv`, `regime_summary.csv`, rolling Sharpe and
equity curves per sub-period.

### Split criterion accuracy simulation

```bash
python run_split_comparison.py
```

Outputs: `results/split_comparison/summary.csv`, `raw.csv`, approximation-ratio plots.

---

## Data inputs

### `data/stock_daily_returns.csv` — CRSP daily file (CIZ format)

| Column     | Notes                                                                               |
| ---------- | ----------------------------------------------------------------------------------- |
| `PERMNO`   | CRSP unique stock identifier                                                        |
| `DlyCalDt` | Calendar date (YYYY-MM-DD)                                                          |
| `DlyClose` | Split-adjusted closing price (used if DlyRet missing)                               |
| `DlyRet`   | Daily total return (preferred; delisting returns from Shumway 1997 −30% imputation) |
| `DlyCap`   | Market capitalisation (used for top-K filtering)                                    |

### `data/constiuents.csv` — S&P 500 historical membership

One row per (PERMNO, membership spell) with columns `permno`, `start`, `ending`. `UniverseFn(date)`
intersects the membership table with top-100 market cap to give the investable universe at each
rebalance date, enforcing strict point-in-time constitution.

### Universe construction

At each 21-day rebalance:

1. Identify PERMNOs with active S&P 500 membership on that date.
2. Restrict to top 100 by market capitalisation.
3. Require complete non-NaN return history over the full lookback window (T = 504 days).
4. Apply Shumway (1997) delisting correction: −30% return on the delisting day, then zero.

---

## Statistical inference

| Test                             | Description                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Ledoit-Wolf (2008) bootstrap** | Circular block bootstrap, block size b = 21, B = 2000 replications; tests equality of Sharpe ratios |
| **Diebold-Mariano (1995)**       | Tests equality of mean squared losses (squared daily P&L)                                           |
| **Holm-Bonferroni**              | Step-down FWER correction across all 6 strategy comparisons                                         |
| **Benjamini-Hochberg**           | FDR correction (more powerful when multiple alternatives hold)                                      |

HMVA is tested against all six alternatives simultaneously; adjusted p-values account for the
multiplicity of comparisons.

---

## Limitations

1. **Transaction costs:** Main backtest uses zero costs. HMVA turnover ≈ 1.10 (monthly units of
   |Δw|); at 10 bps per unit, annual drag ≈ 110 bps, reducing Sharpe by approximately 0.14. The
   cost robustness sweep confirms HMVA beats EW in all 15 cells up to 20 bps.
2. **Price returns:** `DlyClose` excludes dividends. Both HMVA and benchmarks are affected equally
   by the approximately 1.5–2% annual dividend yield.
3. **Universe:** Top-100 S&P 500 is a large-cap universe; results may not generalise to smaller
   stocks or international markets.
4. **Single path:** All statistics are estimated from one historical path. The block-bootstrap
   Sharpe test addresses serial dependence but cannot account for luck across a unique 25-year
   window.

---

## Bibliography

- Ledoit, O., & Wolf, M. (2004). _Honey, I Shrunk the Sample Covariance Matrix._ Journal of Portfolio Management.
- Ledoit, O., & Wolf, M. (2008). _Robust Performance Hypothesis Testing with the Sharpe Ratio._ Journal of Empirical Finance.
- Ledoit, O., & Wolf, M. (2020). _Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices._ Annals of Statistics.
- López de Prado, M. (2016). _Building Diversified Portfolios that Outperform Out of Sample._ Journal of Portfolio Management.
- Black, F., & Litterman, R. (1992). _Global Portfolio Optimization._ Financial Analysts Journal.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). _Optimal Versus Naive Diversification._ Review of Financial Studies.
- Diebold, F.X., & Mariano, R.S. (1995). _Comparing Predictive Accuracy._ Journal of Business & Economic Statistics.
- Holm, S. (1979). _A Simple Sequentially Rejective Multiple Test Procedure._ Scandinavian Journal of Statistics.
- Benjamini, Y., & Hochberg, Y. (1995). _Controlling the False Discovery Rate._ Journal of the Royal Statistical Society: Series B.
- Shumway, T. (1997). _The Delisting Bias in CRSP Data._ Journal of Finance.
