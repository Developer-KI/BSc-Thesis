# Building Mean-Variance Portfolios That Perform Out-of-Sample

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

_Author: Kiril Ivanov · Supervisor: Dr. Sanders Barendse_

**Research question:** Can a hierarchical mean-variance allocation framework deliver superior out-of-sample risk-adjusted performance on a point-in-time S&P 500 universe?

---

## Abstract

Mean-variance optimisation often performs poorly out-of-sample because inverting the covariance matrix amplifies estimation error in expected returns and covariances into unstable, concentrated portfolios. This thesis introduces Hierarchical Mean Variance Allocation (HMVA), a long-only portfolio-construction method designed to target risk-adjusted performance without directly inverting an estimated covariance matrix.

HMVA organises the asset universe into a top-down binary tree whose splits minimise the correlation between equal-weighted sub-portfolios. It allocates capital recursively using cluster-level Sharpe proxies formed from return estimates, reverting to inverse-volatility bisection when no positive return estimate is available. An adaptive Kalman-filter smoothing step then blends each rebalance's target weights with the previous holdings, reducing turnover when the estimated risk structure is diffuse and updating more aggressively when the return signal changes.

HMVA is evaluated in a fully point-in-time backtest of the top-100 S&P 500 constituents from 2000 to 2024 against mean-variance, minimum-variance, hierarchical risk-parity, modified hierarchical risk-parity, and passive benchmarks, with all active strategies using identical regularised inputs. HMVA achieves the highest full-sample point-estimate Sharpe ratio of **0.746**, exceeding equal weighting by 0.317 and mean-variance optimisation from the same inputs by 0.377 Sharpe points, while reducing maximum drawdown by about 18 percentage points relative to equal weighting. Statistical inference provides the strongest evidence against classical mean-variance optimisation: the HMVA vs MVO-EK difference is significant at the 5% level after Holm correction. The equal-weight comparison reaches the conventional rejection threshold; the SPY-100 comparison is borderline; comparisons against MHRP-EK and the risk-only benchmarks remain positive but statistically inconclusive.

---

## What HMVA Is

**HMVA (Hierarchical Mean Variance Allocation)** constructs an investment decision in four stages:

1. Apply exponential weighting to the return window and estimate the mean vector and covariance matrix from the weighted data.
2. Construct a top-down binary asset hierarchy by recursively separating clusters whose equal-weighted returns have low cross-correlation.
3. Allocate capital through the hierarchy using cluster-level Sharpe approximations.
4. Apply a latent weight filter that smooths the target allocation between rebalances.

Its key departure from classical mean-variance optimisation is that it never inverts the covariance matrix and never loads a return vector directly onto individual asset weights. Covariance estimates are used to form a diversification hierarchy; return estimates enter only at the cluster level; and the Kalman filter limits unnecessary reallocation when the estimated environment appears noisy.

HMVA differs from HRP in all three principal design choices: (i) tree construction (top-down correlation-minimising vs. bottom-up single-linkage), (ii) capital allocation (Sharpe-weighted vs. inverse-variance), and (iii) weight dynamics (adaptive Kalman filter vs. none). HRP serves as a natural benchmark, not a parent method.

---

## HMVA Pipeline

### Stage 1 — EWMA Pseudo-Returns

Following the RiskMetrics (1996) approach, the return window is rescaled into a pseudo-return matrix with decay parameter θ = 2^(−1/T*θ) and half-life T*θ = T_r (set equal to the rebalance interval):

```
R̃_t = √(s_t · T) · R_t,   s_t = (1−θ)·θ^(T−1−t) / Σ(1−θ)·θ^i
```

The sample moments of R̃ recover the exponentially weighted moments of R. Setting the half-life equal to the rebalance interval means recent periods receive the largest weight, while older observations decay gradually rather than being discarded abruptly. This allows shrinkage estimators to be applied to R̃ as if it were an ordinary return matrix while inheriting exponential weighting automatically.

### Stage 2 — Nonlinear Shrinkage (NLS) Covariance

Applies the Ledoit-Wolf (2020) analytical nonlinear shrinkage estimator to R̃. Following random matrix theory (Marchenko-Pastur), sample eigenvalues are systematically biased — large ones inflated, small ones deflated — and portfolio optimisers are especially sensitive to the smallest eigenvalues. NLS corrects each eigenvalue individually:

```
λ_i* = λ_i / |1 − N/T − (N/T)·λ_i·m̃_F(λ_i)|²
```

where m̃_F(λ) is the boundary value of the Stieltjes transform of the limiting spectral distribution, estimated nonparametrically. The shrunk covariance Σ̂^NLS = U·diag(λ_1*,…,λ_N*)·Uᵀ shares the sample eigenvectors but replaces eigenvalues with their asymptotically optimal corrections under Frobenius loss.

### Stage 3 — Black-Litterman Return Estimates

Constructs return estimates as a Bayesian blend of a conservative cross-sectional prior with skip-month momentum:

- **View Q:** cross-sectional mean of raw daily returns over [t−T_h, t−T_r], skipping the most recent T_r days to avoid short-term reversal (Jegadeesh 1993).
- **Prior π:** flat equal-return prior set to the cross-sectional average over the estimation window (rounded upward to the nearest 0.01%).
- **Uncertainty Ω:** diag(Σ̂) — asset-specific, proportional to each asset's own variance.

With P = I_N and τ = 1, the BL posterior simplifies to:

```
μ̂^BL = π + Σ̂ · (Σ̂ + diag(Σ̂))⁻¹ · (Q − π)
```

This regularises the raw momentum signal toward the uninformative prior, dampening noise in cross-sectional return estimates.

### Stage 4 — Top-Down Correlation-Minimising Tree

Builds a complete binary tree over assets using top-down recursive bipartition. At each node the split that minimises the estimated correlation between the two equal-weighted child portfolios is selected:

```
f(L, R) = Corr_EW(L, R) = Cov_EW(L, R) / √(Var_EW(L) · Var_EW(R))
```

**Motivation:** for any interior allocation α ∈ (0,1) between branches L and R, parent variance is monotonically increasing in the cross-branch correlation. Minimising cross-branch correlation is therefore aligned with constructing a hierarchy whose upper-level branches are maximally diversifying. Volatility and return information are then introduced only in the recursive allocation step, creating a clean separation of roles.

Clusters of up to n*bf = 10 assets use exhaustive search; larger clusters use an O(n²) contiguous-cut heuristic via 2D prefix sums (see \_Split Criterion Walkthrough* below).

### Stage 5 — Sharpe-Ratio Capital Allocation

At each internal node, equal-weighted cluster Sharpe proxies are formed to reduce signal-estimation noise through cluster averaging:

```
Ŝ(C) = max( μ̄_BL(C) / σ_EW(C), 0 )
```

where μ̄_BL(C) is the equal-weight mean BL return and σ_EW(C) is the equal-weight volatility of cluster C. The truncation at zero preserves the long-only nature of the strategy.

The allocation fraction going to the left child L is Ŝ(L) / (Ŝ(L) + Ŝ(R)). If both proxies are zero or negative, allocation falls back to inverse-volatility bisection (risk-only mode). This risk-only variant is denoted **HMVA-mv**.

### Stage 6 — Kalman-Filter Weight Smoother

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

## Key Results

### Full-sample performance (July 2000 – December 2024, T_h = 126 days, monthly rebalance)

| Strategy | Ann. Ret. | Ann. Vol. | Sharpe    | Sortino   | Max DD     | Ñ     |
| -------- | --------- | --------- | --------- | --------- | ---------- | ----- |
| **HMVA** | **13.2%** | 17.7%     | **0.746** | **0.976** | **−37.0%** | 4.8   |
| MVO-EK   | 7.4%      | 24.8%     | 0.299     | 0.375     | −60.4%     | 3.2   |
| MHRP-EK  | 8.9%      | 16.2%     | 0.547     | 0.685     | −46.2%     | 47.2  |
| HMVA-mv  | 10.0%     | 15.3%     | 0.652     | 0.837     | −37.5%     | 9.5   |
| GMV-EK   | 8.2%      | 14.6%     | 0.560     | 0.712     | −38.9%     | 26.3  |
| HRP-E    | 8.6%      | 15.6%     | 0.554     | 0.697     | −43.2%     | 62.5  |
| EW       | 7.4%      | 19.0%     | 0.392     | 0.497     | −55.2%     | 100.0 |
| SPY-100  | 7.6%      | 19.1%     | 0.398     | 0.509     | −53.3%     | 52.1  |

Suffix **E** denotes EWMA preprocessing; suffix **K** denotes the adaptive Kalman filter. Every active strategy draws on an identical information set (T_h = 126 days, T_r = 21 days, top-100 point-in-time S&P 500 universe) so that any difference in realized performance is attributable to the construction mechanism.

Ñ is the mean inverse Herfindahl-Hirschman Index (mean effective number of holdings). HMVA leads on Sharpe (+0.317 over EW, +0.377 over MVO-EK), Sortino, annualised return, and maximum drawdown. The estimation regime is N/T = 100/126 ≈ 0.8 — a difficult setting ideal for testing strategy robustness.

---

## Statistical Tests

Tests use the Ledoit-Wolf (2008) circular block bootstrap (block size b = T_r = 21, B = 2,000 replications), testing equality of Sharpe ratios pairwise. Multiple-testing correction uses Holm-Bonferroni FWER within pre-specified hypothesis families.

### HMVA vs. return-based active strategies

| Comparator | SR diff | LW 95% C.I.     | LW p  | Holm p      |
| ---------- | ------- | --------------- | ----- | ----------- |
| MVO-EK     | +0.377  | [0.064, 0.673]  | 0.016 | **0.031\*** |
| MHRP-EK    | +0.185  | [−0.080, 0.439] | 0.157 | 0.157       |

HMVA significantly outperforms MVO-EK after FWER correction. The advantage over MHRP-EK is positive but statistically inconclusive.

### HMVA vs. passive benchmarks

| Comparator | SR diff | LW 95% C.I.     | LW p  | Holm p      |
| ---------- | ------- | --------------- | ----- | ----------- |
| EW         | +0.317  | [0.009, 0.601]  | 0.037 | **0.050\*** |
| SPY-100    | +0.311  | [−0.010, 0.602] | 0.050 | 0.050       |

HMVA reaches the 5% rejection threshold against equal weighting. The SPY-100 result is borderline (rounded C.I. slightly includes zero).

### HMVA-mv vs. risk-only strategies

| Comparator | SR diff | LW 95% C.I.     | LW p  | Holm p |
| ---------- | ------- | --------------- | ----- | ------ |
| GMV-EK     | +0.087  | [−0.069, 0.248] | 0.273 | 0.379  |
| HRP-E      | +0.090  | [−0.106, 0.292] | 0.379 | 0.379  |

HMVA-mv ranks above both risk-only comparators on a point-estimate basis, but the differences are not statistically significant.

\* p < 0.05 after Holm-Bonferroni correction within the hypothesis family.

---

## Regime Analysis

NBER recession windows in the evaluation sample: Dot-com (2001-03 – 2001-11), GFC (2007-12 – 2009-06), COVID-19 (2020-02 – 2020-04).

### Mean annualised Sharpe by regime

| Strategy | Crisis | Calm  |     | Strategy | Crisis | Calm  |
| -------- | ------ | ----- | --- | -------- | ------ | ----- |
| HMVA     | +0.024 | 0.962 |     | HMVA-mv  | −0.146 | 0.942 |
| MVO-EK   | −0.582 | 0.505 |     | GMV-EK   | −0.153 | 0.864 |
| MHRP-EK  | −0.188 | 0.858 |     | HRP-E    | −0.212 | 0.854 |
| SPY-100  | −0.172 | 0.643 |     | EW       | −0.326 | 0.676 |

HMVA is the only strategy with a positive crisis-period Sharpe ratio, and it also leads in calm periods. The results suggest an overall increase in risk-adjusted returns rather than a regime-specific specialisation.

---

## Transaction-Cost Robustness

After-cost Sharpe ratios under proportional one-way transaction costs (basis points per unit |Δw|), compared against equal weighting.

| Strategy | κ=0   | κ=2   | κ=5   | κ=10  | κ=20  | κ=30  | Turnover |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- | -------- |
| HMVA     | 0.746 | 0.729 | 0.703 | 0.660 | 0.574 | 0.489 | 1.133    |
| HMVA-mv  | 0.652 | 0.636 | 0.612 | 0.572 | 0.492 | 0.412 | 0.941    |
| EW       | 0.392 | 0.391 | 0.389 | 0.385 | 0.379 | 0.372 | 0.057    |
| SPY-100  | 0.398 | 0.397 | 0.397 | 0.396 | 0.393 | 0.391 | 0.068    |

HMVA maintains the highest after-cost Sharpe ratio across all examined cost levels. Even at 30 bps per unit turnover, HMVA exceeds both passive benchmarks. The advantage naturally compresses as costs increase, but the ranking remains favorable throughout the tested range.

---

## Split Criterion Walkthrough

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
r = 1 … n//2. Called when `len(indices) ≤ bf_threshold` (≤ 10).

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

**Empirical accuracy** (from `results/split_comparison/summary.csv`, 100 random factor-model Σ per n):

| Cluster size n | Median approx. ratio | 95th pct. ratio | Exact match rate |
| -------------- | -------------------- | --------------- | ---------------- |
| 12             | 1.000                | 1.000           | 71%              |
| 14             | 1.000                | 1.000           | 69%              |
| 16             | 1.000                | 1.000           | 69%              |
| 18             | 1.000                | 1.000           | 69%              |
| 20             | 1.000                | 1.000           | 56%              |

The heuristic is near-optimal: the median approximation ratio is 1.000 (identical score to the
global optimum) across all cluster sizes tested.

### Summary comparison

| Property               | `_vb_split_bruteforce`   | `_vb_split_heuristic`               |
| ---------------------- | ------------------------ | ----------------------------------- |
| **Guarantee**          | Globally optimal         | Near-optimal (median ratio = 1.0)   |
| **Time complexity**    | O(2ⁿ × n²)               | O(n²)                               |
| **Cuts evaluated**     | All bipartitions         | n−1 contiguous cuts in sorted order |
| **When used**          | n ≤ n_bf (≤ 10)          | n > n_bf                            |
| **Key data structure** | `itertools.combinations` | 2D prefix-sum array P               |

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

## Data Inputs

### `data/stock_daily_returns.csv` — CRSP daily file (not included)

| Column     | Notes                                                 |
| ---------- | ----------------------------------------------------- |
| `PERMNO`   | CRSP unique stock identifier                          |
| `DlyCalDt` | Calendar date (YYYY-MM-DD)                            |
| `DlyClose` | Split-adjusted closing price (used if DlyRet missing) |
| `DlyRet`   | Daily total return (preferred)                        |
| `DlyCap`   | Market capitalisation (used for top-100 filtering)    |

### `data/constituents.csv` — S&P 500 historical membership

One row per (PERMNO, membership spell) with columns `permno`, `start`, `ending`. `UniverseFn(date)` intersects the membership table with the top-100 market cap filter to give the investable universe at each rebalance date, enforcing strict point-in-time constitution (no survivorship bias).

### Universe construction

At each 21-day rebalance:

1. Identify PERMNOs with active S&P 500 membership on that date.
2. Restrict to top 100 by market capitalisation.
3. Require complete non-NaN return history over the full lookback window (T_h = 126 days).
4. Align weight vectors across consecutive rebalances on the union of assets, with missing entries set to zero, so that newly opened and fully liquidated positions both contribute to turnover.

The backtest period spans July 3, 2000 to December 31, 2024. The first usable rebalance occurs after the 126-day warm-up period from the data start of January 1, 2000. The rolling sample size is chosen to produce a difficult estimation regime of N/T = 100/126 ≈ 0.8.

---

## Statistical Inference

| Test                   | Description                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Ledoit-Wolf (2008)** | Circular block bootstrap, block size b = T_r = 21, B = 2,000 replications; studentized with Newey-West HAC std. err. |
| **Holm-Bonferroni**    | Step-down FWER correction applied within pre-specified hypothesis families (not across all comparisons jointly)      |

Four hypothesis families are tested separately: (1) HMVA vs. return-based active strategies (MVO-EK, MHRP-EK); (2) HMVA vs. passive benchmarks (EW, SPY-100); (3) HMVA-mv vs. risk-only strategies (GMV-EK, HRP-E); (4) HMVA-mv vs. passive benchmarks. Rejection within one family is not interpreted as evidence for a global claim of superiority across all benchmarks.

---

## Limitations

1. **Single market/asset class:** The evaluation covers large-cap US equities only; results may not generalise to international markets, small-caps, or multi-asset universes.
2. **Transaction costs:** The main backtest uses zero costs. HMVA turnover ≈ 1.13 monthly; at 10 bps per unit, annual drag reduces the Sharpe advantage but the ranking remains favorable through 30 bps.
3. **Price returns:** `DlyClose` excludes dividends. Both HMVA and benchmarks are affected equally by the approximately 1.5–2% annual dividend yield.
4. **Single path:** All statistics are estimated from one historical path. The block-bootstrap Sharpe test addresses serial dependence but cannot account for luck across a unique 25-year window.
5. **Design choices:** The brute-force threshold, EWMA half-life, cluster Sharpe proxies, and Kalman gain proxies have not been fully sensitivity-tested. Some in-sample design bias may remain despite the point-in-time backtest.
6. **Heuristic validation:** The contiguous-cut split heuristic is validated on the split objective, not on realized portfolio performance.

---

## Bibliography

- Markowitz, H. (1952). _Portfolio Selection._ Journal of Finance.
- Ledoit, O., & Wolf, M. (2004). _Honey, I Shrunk the Sample Covariance Matrix._ Journal of Portfolio Management.
- Ledoit, O., & Wolf, M. (2008). _Robust Performance Hypothesis Testing with the Sharpe Ratio._ Journal of Empirical Finance.
- Ledoit, O., & Wolf, M. (2020). _Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices._ Annals of Statistics.
- Black, F., & Litterman, R. (1992). _Global Portfolio Optimization._ Financial Analysts Journal.
- López de Prado, M. (2016). _Building Diversified Portfolios that Outperform Out of Sample._ Journal of Portfolio Management.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). _Optimal Versus Naive Diversification._ Review of Financial Studies.
- Jegadeesh, N., & Titman, S. (1993). _Returns to Buying Winners and Selling Losers._ Journal of Finance.
- Michaud, R.O. (1989). _The Markowitz Optimization Enigma: Is Optimized Optimal?_ Financial Analysts Journal.
- Kalman, R.E. (1960). _A New Approach to Linear Filtering and Prediction Problems._ Journal of Basic Engineering.
- Holm, S. (1979). _A Simple Sequentially Rejective Multiple Test Procedure._ Scandinavian Journal of Statistics.
- Newey, W., & West, K. (1994). _Automatic Lag Selection in Covariance Matrix Estimation._ Review of Economic Studies.
- Marchenko, V.A., & Pastur, L.A. (1967). _Distribution of eigenvalues for some sets of random matrices._ Mathematics of the USSR-Sbornik.
- RiskMetrics Group (1996). _RiskMetrics Technical Document._ J.P. Morgan/Reuters.
