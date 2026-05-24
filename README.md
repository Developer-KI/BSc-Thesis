# Hierarchical Minimum Variance Allocation (HMVA)

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

**Central question:** Can a principled, multi-stage modification of Hierarchical Risk Parity (HRP) — integrating nonlinear shrinkage covariance estimation, a vol-balanced top-down tree, Black-Litterman return forecasts, and Kalman-filter weight smoothing — deliver superior out-of-sample risk-adjusted performance on a point-in-time S&P 500 universe?

---

## What this thesis does

Standard HRP allocates risk through a hierarchical clustering tree without inverting the covariance matrix. This paper proposes **HMVA** (Hierarchical Minimum Variance Allocator), a six-stage extension of HRP that addresses each of its known failure modes:

| Stage | Modification                                | Failure mode addressed                                         |
| ----- | ------------------------------------------- | -------------------------------------------------------------- |
| 1     | EWMA pseudo-returns (halflife = 21 days)    | Equal weighting ignores volatility clustering                  |
| 2     | Nonlinear shrinkage covariance (NLS)        | Sample Σ has biased eigenvalues at any N/T ratio               |
| 3     | Black-Litterman expected returns (τ = 0.05) | Pure-risk allocation ignores return expectations               |
| 4     | Vol-balanced top-down tree (λ = 0.25)       | Static correlation tree conflates correlation and risk balance |
| 5     | Sharpe-ratio bisection                      | Variance bisection ignores expected risk-adjusted return       |
| 6     | Kalman-filter weight smoother               | Excessive turnover and noise-chasing in weight updates         |

An ablation study on the top-100 S&P 500 stocks (2002–2024, T = 504) documents the marginal Sharpe contribution of each stage.

---

## Key results

### Ablation study — cumulative Sharpe improvement over HRP baseline

| Strategy       | Sharpe    | Max DD     | Ann. Ret. | Ann. Vol. | Turnover | Marginal ΔSharpe    |
| -------------- | --------- | ---------- | --------- | --------- | -------- | ------------------- |
| **HMVA**       | **0.809** | **−38.1%** | **14.0%** | 17.3%     | 1.09     | +0.105 (Stage 6)    |
| +BL+Sharpe     | 0.704     | −35.4%     | 13.6%     | 19.3%     | 1.22     | +0.068 (Stages 3+5) |
| +VBTree        | 0.636     | −31.8%     | 9.7%      | 15.2%     | 1.07     | +0.042 (Stage 4)    |
| +NLS+EWMA      | 0.594     | −42.5%     | 9.3%      | 15.6%     | 0.42     | +0.019 (Stages 1+2) |
| HRP (baseline) | 0.574     | −41.8%     | 9.4%      | 16.3%     | 0.32     | —                   |
| EW (1/N)       | 0.477     | −55.2%     | 9.0%      | 18.8%     | 0.06     | —                   |

HMVA achieves a cumulative Sharpe improvement of +0.235 over standard HRP and +0.332 over equal-weight. The Kalman-filter smoother (Stage 6) is the single largest contributor (+0.105).

### Cost robustness (HMVA vs EW)

Across 20 robustness cells (4 lookbacks × 5 cost levels, 0–20 bps):

| Metric                | Value  |
| --------------------- | ------ |
| Mean ΔSharpe vs EW    | +0.301 |
| Pct. cells beating EW | 100%   |
| Mean HMVA Sharpe      | 0.773  |

HMVA's mean Sharpe advantage persists even at 20 bps per unit |Δw| (still +0.272 above EW at lookback = 252 days).

### Crisis-period performance (annualised Sharpe)

| Period                 | HMVA      | HRP   | EW    |
| ---------------------- | --------- | ----- | ----- |
| Dot-com trough (2002)  | −0.95     | −0.88 | −0.92 |
| GFC (2007–2009)        | **−0.31** | −0.88 | −0.89 |
| COVID crash (2020 Q1)  | **−0.34** | −0.42 | −0.53 |
| Rate-hike cycle (2022) | −0.64     | −0.36 | −0.59 |

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

| Label     | Tree                  | Covariance | Bisection  |
| --------- | --------------------- | ---------- | ---------- |
| `HMVA`    | Vol-balanced (λ=0.25) | NLS + EWMA | Sharpe     |
| `HMVA-mv` | Vol-balanced (λ=0.25) | NLS + EWMA | Vol        |
| `HRP`     | Agglomerative         | NLS + EWMA | Vol        |
| `MVO`     | —                     | NLS + EWMA | MV utility |
| `EW`      | —                     | —          | Equal      |
| `SPY-K`   | —                     | —          | Market cap |

The ablation baseline `HRP` uses sample covariance and agglomerative clustering (standard Lopez de Prado 2016).

---

## Parameter sensitivity

Two hyperparameters are swept in `run_param_robustness.py`:

| Parameter     | Default | Grid                        | Key finding                                                                |
| ------------- | ------- | --------------------------- | -------------------------------------------------------------------------- |
| ewma_halflife | 21      | [5, 10, 14, 21, 30, 42, 63] | Sharpe stable at 0.72–0.76 for h ≥ 10; h = 5 gives EW-like performance     |
| lam_cov       | 0.25    | [0.0, 0.05, ..., 1.0]       | Sharpe highest at λ = 1.0 (pure vol-balance); default 0.25 is conservative |

Note: Black-Litterman δ and τ cancel analytically in the posterior formula (P = I, Ω = τ diag(Σ)) so they are not swept.

---

# VB Split Functions — Line-by-Line Walkthrough

These three functions implement the **splitting criterion** for the **Variance-Bisection (VB)** hierarchical clustering used in portfolio construction. The algorithm recursively bisects a cluster of assets into two sub-clusters by minimising a cost that blends inter-cluster covariance and correlation.

---

## Running Example

We'll use the same concrete 4×4 covariance matrix throughout all three functions.

```
Σ = [[4.0, 3.0, 0.2, 0.1],   # assets {0, 1} are highly correlated
     [3.0, 4.0, 0.1, 0.2],
     [0.2, 0.1, 1.0, 0.8],   # assets {2, 3} are highly correlated
     [0.1, 0.2, 0.8, 1.0]]

indices = [0, 1, 2, 3]        # all four assets are in the cluster
```

We expect both methods to discover the natural split: **{2, 3} | {0, 1}**.

---

## Function 1 — `_vb_merge_cost`

```python
def _vb_merge_cost(cross: float, n_a: int, n_b: int,
                   sw_a: float, sw_b: float) -> float:
```

**Purpose:** Given two candidate sub-clusters A and B, score how "expensive" it would be to merge them. A lower score means they are more similar and belong together; a higher score means they are more different and should stay apart. The VB algorithm _minimises_ this score, so it finds the most dissimilar bipartition.

### Arguments

| Argument       | Meaning                                                         |
| -------------- | --------------------------------------------------------------- |
| `cross`        | One-way sum of covariances between A and B: `Σ_{i∈A, j∈B} Σ_ij` |
| `n_a`, `n_b`   | Number of assets in each cluster                                |
| `sw_a`, `sw_b` | Within-cluster sum-of-covariances: `Σ_{i,j∈S} Σ_ij`             |

### Line-by-Line

```python
cov_term  = cross / float(n_a * n_b) if n_a * n_b > 0 else 0.0
```

**What it does:** Computes the equal-weight covariance between the two clusters.

- An equal-weight portfolio of cluster A has variance `sw_A / n_A²`
- The EW covariance between A and B is `Cov(EW_A, EW_B) = cross / (n_A × n_B)`
- The guard `if n_a * n_b > 0` prevents division by zero for empty clusters.

**Example** — A = {2, 3}, B = {0, 1}:

```
cross  = Σ[2,0] + Σ[2,1] + Σ[3,0] + Σ[3,1]
       = 0.2 + 0.1 + 0.1 + 0.2 = 0.6   (one-way)

cov_term = 0.6 / (2 × 2) = 0.15
```

---

```python
denom_rho = float(np.sqrt(max(sw_a * sw_b, 0.0)))
```

**What it does:** Computes the denominator for the correlation term.

- The EW portfolio variance of cluster S is `sw_S / n_S²`, so the EW volatility is `sqrt(sw_S) / n_S`.
- The EW cross-correlation between A and B is `cross / sqrt(sw_A × sw_B)`.
- `max(..., 0.0)` clips any floating-point negative near zero before `sqrt`.

**Example:**

```
sw_a = Σ[2,2] + Σ[2,3] + Σ[3,2] + Σ[3,3]
     = 1.0 + 0.8 + 0.8 + 1.0 = 3.6

sw_b = Σ[0,0] + Σ[0,1] + Σ[1,0] + Σ[1,1]
     = 4.0 + 3.0 + 3.0 + 4.0 = 14.0

denom_rho = sqrt(3.6 × 14.0) = sqrt(50.4) ≈ 7.099
```

---

```python
rho       = cross / denom_rho if denom_rho > 1e-12 else 0.0
```

**What it does:** Computes the EW cross-correlation.

- The threshold `1e-12` avoids division by zero when a cluster has zero total variance (e.g. a single constant asset).
- When `denom_rho` is negligibly small the cross-correlation is treated as 0.

**Example:**

```
rho = 0.6 / 7.099 ≈ 0.0845
```

---

```python
return 0.5 * cov_term + 0.5 * rho
```

**What it does:** Returns the 50/50 blend of the two terms.

**Example:**

```
cost = 0.5 × 0.15 + 0.5 × 0.0845 = 0.075 + 0.042 = 0.117
```

This is a low score — assets {2,3} and {0,1} have very little in common, which is exactly right.

**Contrast** — A = {0, 2}, B = {1, 3} (a bad split mixing correlated assets):

```
cross    = Σ[0,1] + Σ[0,3] + Σ[2,1] + Σ[2,3]
         = 3.0 + 0.2 + 0.1 + 0.8 = 4.1

sw_a     = Σ[0,0]+Σ[0,2]+Σ[2,0]+Σ[2,2] = 4.0+0.2+0.2+1.0 = 5.4
sw_b     = Σ[1,1]+Σ[1,3]+Σ[3,1]+Σ[3,3] = 4.0+0.2+0.2+1.0 = 5.4

cov_term = 4.1 / 4 = 1.025
rho      = 4.1 / sqrt(5.4 × 5.4) = 4.1 / 5.4 ≈ 0.759
cost     = 0.5 × 1.025 + 0.5 × 0.759 = 0.892   ← much higher, correctly penalised
```

---

## Function 2 — `_vb_split_bruteforce`

```python
def _vb_split_bruteforce(cov_arr: np.ndarray,
                          indices: List[int]) -> Tuple[List[int], List[int]]:
```

**Purpose:** Find the **globally optimal** bipartition of `indices` by exhaustive enumeration. Only called for small clusters (`len(indices) <= bf_threshold`, typically ≤ 16) because cost is O(2ⁿ).

### Line-by-Line

```python
n = len(indices)
if n <= 1:
    return list(indices), []
```

**Base case:** A single asset (or empty) cannot be split — return it as A and B as empty.

---

```python
M = cov_arr[np.ix_(indices, indices)]
```

**What it does:** Extracts the sub-matrix of `cov_arr` whose rows and columns correspond to the current cluster's assets.

`np.ix_([2,3,0,1], [2,3,0,1])` creates an open mesh for fancy indexing.

**Example** — `indices = [0, 1, 2, 3]`:

```
M = cov_arr[[0,1,2,3], :][:, [0,1,2,3]] = Σ  (the full 4×4 matrix)

M = [[4.0, 3.0, 0.2, 0.1],
     [3.0, 4.0, 0.1, 0.2],
     [0.2, 0.1, 1.0, 0.8],
     [0.1, 0.2, 0.8, 1.0]]
```

---

```python
local = list(range(n))
```

**What it does:** Creates local 0-based indices into M (not into `cov_arr`). We enumerate over these and map back to global indices later.

**Example:** `local = [0, 1, 2, 3]`

---

```python
best_score = np.inf
best_A, best_B = local[:1], local[1:]
```

**What it does:** Initialises the best seen score to ∞ and sets a default split (first asset alone vs the rest) as a fallback.

---

```python
for r in range(1, n // 2 + 1):
    for A_tup in combinations(local, r):
```

**What it does:** Iterates over all subsets of size `r = 1, 2, …, n//2`.

- Stopping at `n//2` exploits A/B symmetry: the split {A, B} is identical to {B, A}, so we never need to enumerate the larger half.
- For n=4, r ∈ {1, 2} → we check subsets of size 1 and size 2.
- Total: C(4,1) + C(4,2) = 4 + 6 = 10 candidates, but only C(4,2)/2 = 3 unique pairs at r=2 (the rest are duplicates of the r=1 case mirrored). Actually r goes up to 2 inclusive, so 4+6=10 iterations, but {A,B} and {B,A} only appear once each here because we fix |A|≤|B|.

**Example iterations at r=2:**

| A_tup | A_loc | B_loc |
| ----- | ----- | ----- |
| (0,1) | [0,1] | [2,3] |
| (0,2) | [0,2] | [1,3] |
| (0,3) | [0,3] | [1,2] |
| (1,2) | [1,2] | [0,3] |
| (1,3) | [1,3] | [0,2] |
| (2,3) | [2,3] | [0,1] |

---

```python
        A_loc = list(A_tup)
        B_loc = [i for i in local if i not in set(A_tup)]
```

**What it does:** Forms complementary local index lists. `set(A_tup)` gives O(1) membership check.

**Example** — A_tup = (0, 1): `A_loc = [0, 1]`, `B_loc = [2, 3]`

---

```python
        nA, nB  = len(A_loc), len(B_loc)
        sA      = float(M[np.ix_(A_loc, A_loc)].sum())
        sB      = float(M[np.ix_(B_loc, B_loc)].sum())
        cross   = float(M[np.ix_(A_loc, B_loc)].sum())
```

**What it does:** Computes the three raw statistics needed by `_vb_merge_cost`:

| Variable | Formula                                        | Example (A={0,1}, B={2,3}) |
| -------- | ---------------------------------------------- | -------------------------- |
| `sA`     | sum of all entries in A×A sub-matrix           | 4+3+3+4 = 14.0             |
| `sB`     | sum of all entries in B×B sub-matrix           | 1+0.8+0.8+1 = 3.6          |
| `cross`  | sum of all entries in A×B sub-matrix (one-way) | 0.2+0.1+0.1+0.2 = 0.6      |

Note: `M[np.ix_(A_loc, B_loc)]` picks rows from A and columns from B, giving the off-diagonal rectangular block.

---

```python
        score   = _vb_merge_cost(cross, nA, nB, sA, sB)
        if score < best_score:
            best_score = score
            best_A, best_B = A_loc, B_loc
```

**What it does:** Scores this split and keeps track of the global minimum.

**All scores for our example:**

| Split | cross   | sA  | sB   | cov  | rho   | score  |
| ----- | ------- | --- | ---- | ---- | ----- | ------ | ----------- |
| {0}   | {1,2,3} | 3.3 | 4.0  | 22.2 | 1.1   | 0.238  | 0.396       |
| {1}   | {0,2,3} | 3.3 | 4.0  | 22.2 | 1.1   | 0.238  | 0.396       |
| {2}   | {0,1,3} | 1.2 | 1.0  | 26.2 | 0.4   | 0.234  | 0.430       |
| {3}   | {0,1,2} | 1.2 | 1.0  | 26.2 | 0.4   | 0.234  | 0.430       |
| {0,1} | {2,3}   | 0.6 | 14.0 | 3.6  | 0.15  | 0.0845 | **0.117** ✓ |
| {0,2} | {1,3}   | 4.1 | 5.4  | 5.4  | 1.025 | 0.759  | 0.892       |
| {0,3} | {1,2}   | 4.1 | 5.4  | 5.4  | 1.025 | 0.759  | 0.892       |
| {1,2} | {0,3}   | 4.1 | 5.4  | 5.4  | 1.025 | 0.759  | 0.892       |
| {1,3} | {0,2}   | 4.1 | 5.4  | 5.4  | 1.025 | 0.759  | 0.892       |
| {2,3} | {0,1}   | 0.6 | 3.6  | 14.0 | 0.15  | 0.0845 | **0.117** ✓ |

**Winner:** split {0,1} | {2,3} (or equivalently {2,3} | {0,1}) with score **0.117**.

---

```python
return [indices[i] for i in best_A], [indices[i] for i in best_B]
```

**What it does:** Translates local indices back to global asset indices.

**Example:** `best_A = [0, 1]` (local), `best_B = [2, 3]` (local).
Since `indices = [0, 1, 2, 3]`, `indices[i] == i` here, so:

```
return [0, 1], [2, 3]
```

---

## Function 3 — `_vb_split_heuristic`

```python
def _vb_split_heuristic(cov_arr: np.ndarray,
                         indices: List[int]) -> Tuple[List[int], List[int]]:
```

**Purpose:** An **O(n²)** approximation to `_vb_split_bruteforce` for large clusters. Instead of checking all 2ⁿ subsets, it:

1. Sorts assets by their within-cluster row-sum.
2. Considers only the n−1 **contiguous cuts** of the sorted sequence.
3. Evaluates each cut in O(1) using a precomputed 2D prefix-sum array.

The intuition: assets that covary similarly with the rest of the cluster (similar row-sums) tend to belong together. A contiguous cut of the sorted sequence separates "high covariance" assets from "low covariance" ones.

### Line-by-Line

```python
n = len(indices)
if n <= 1:
    return list(indices), []
```

Same base case as the brute-force version.

---

```python
M_raw      = cov_arr[np.ix_(indices, indices)]
```

**What it does:** Extracts the sub-matrix for the current cluster.

**Example** — same full 4×4 Σ as before.

---

```python
order      = np.argsort(M_raw.sum(axis=1))        # ascending within-cluster row-sum
```

**What it does:** Computes each asset's **total within-cluster covariance** (how much it covaries with every other asset in the cluster) and sorts from lowest to highest.

- `M_raw.sum(axis=1)` gives each row's sum — this is proportional to the asset's EW covariance with the cluster.
- `np.argsort` returns the _permutation_ that sorts these sums in ascending order.
- Low row-sum = low "systemic" covariance with the cluster → likely belongs to the quieter sub-cluster.

**Example:**

```
Row sums of M_raw:
  asset 0 (row 0): 4.0 + 3.0 + 0.2 + 0.1 = 7.3
  asset 1 (row 1): 3.0 + 4.0 + 0.1 + 0.2 = 7.3
  asset 2 (row 2): 0.2 + 0.1 + 1.0 + 0.8 = 2.1
  asset 3 (row 3): 0.1 + 0.2 + 0.8 + 1.0 = 2.1

order = np.argsort([7.3, 7.3, 2.1, 2.1])
      = [2, 3, 0, 1]   (ascending: assets 2 and 3 come first)
```

---

```python
sorted_idx = [indices[int(i)] for i in order]
```

**What it does:** Translates the local sorted permutation to global asset indices.

**Example:** `order = [2, 3, 0, 1]`, `indices = [0,1,2,3]`

```
sorted_idx = [indices[2], indices[3], indices[0], indices[1]]
           = [2, 3, 0, 1]
```

---

```python
M          = M_raw[np.ix_(order, order)]
```

**What it does:** Reorders both rows and columns of M_raw so that the sorted asset order is along both axes. This is required so that prefix-sum lookups correspond to contiguous cuts.

**Example:**

```
Reordered M (rows/cols in order [2, 3, 0, 1]):

         asset2  asset3  asset0  asset1
asset2 [  1.0     0.8     0.2     0.1  ]
asset3 [  0.8     1.0     0.1     0.2  ]
asset0 [  0.2     0.1     4.0     3.0  ]
asset1 [  0.1     0.2     3.0     4.0  ]
```

The top-left 2×2 block is the {2,3} sub-cluster; the bottom-right 2×2 block is {0,1}. The off-diagonal blocks hold the cross-covariances.

---

```python
P       = M.cumsum(axis=0).cumsum(axis=1)
```

**What it does:** Builds the **2D prefix sum** of M.

`P[i, j]` = sum of all elements in `M[0:i+1, 0:j+1]`.

This is computed in two passes:

1. `cumsum(axis=0)` — running sum down each column.
2. `cumsum(axis=1)` — running sum across each row of the result.

**Example step-by-step:**

_After `cumsum(axis=0)`:_

```
C0 = [[1.0,  0.8,  0.2,  0.1],
      [1.8,  1.8,  0.3,  0.3],
      [2.0,  1.9,  4.3,  3.3],
      [2.1,  2.1,  7.3,  7.3]]
```

_After `cumsum(axis=1)` on C0:_

```
P  = [[1.0,  1.8,  2.0,  2.1],
      [1.8,  3.6,  3.9,  4.2],
      [2.0,  3.9,  8.2, 11.5],
      [2.1,  4.2, 11.5, 18.8]]
```

Verification: `P[1,1]` = sum of M[0:2, 0:2] = 1.0+0.8+0.8+1.0 = 3.6 ✓
`P[3,3]` = sum of entire M = total variance = 18.8 ✓

---

```python
total   = float(P[-1, -1])
```

Sum of all covariances in the cluster. **Example:** `total = 18.8`

---

```python
ks      = np.arange(1, n)
```

**What it does:** The candidate cut points: `k=1` means "first 1 asset | last n-1 assets", …, `k=n-1` means "first n-1 | last 1".

**Example:** `ks = [1, 2, 3]`

---

```python
s_left  = P[ks - 1, ks - 1]
```

**What it does:** Within-cluster sum for the **left** sub-group (assets 0..k−1 in sorted order).

By the prefix sum definition: `P[k-1, k-1]` = sum of `M[0:k, 0:k]` = sum of all covariances within the left group.

**Example:**

```
k=1: s_left = P[0,0] = 1.0          (just asset 2: Σ[2,2])
k=2: s_left = P[1,1] = 3.6          (assets {2,3}: 1+0.8+0.8+1)
k=3: s_left = P[2,2] = 8.2          (assets {2,3,0})
```

---

```python
s_right = total - P[-1, ks - 1] - P[ks - 1, -1] + s_left
```

**What it does:** Within-cluster sum for the **right** sub-group using the inclusion-exclusion principle.

The sum of M restricted to the right block (rows k..n-1, cols k..n-1) is:

```
s_right = total
        - (sum of first k cols of all rows)   → P[n-1, k-1]
        - (sum of first k rows of all cols)   → P[k-1, n-1]
        + (sum of top-left k×k block counted twice)  → P[k-1, k-1]
```

**Example:**

```
k=1: s_right = 18.8 - P[3,0] - P[0,3] + P[0,0]
             = 18.8 - 2.1    - 2.1    + 1.0 = 15.6
              (assets {1,0,3} → wait, sorted order {3,0,1})

k=2: s_right = 18.8 - P[3,1] - P[1,3] + P[1,1]
             = 18.8 - 4.2    - 4.2    + 3.6 = 14.0   (assets {0,1}: 4+3+3+4=14 ✓)

k=3: s_right = 18.8 - P[3,2] - P[2,3] + P[2,2]
             = 18.8 - 11.5   - 11.5   + 8.2 = 4.0    (just asset 1: Σ[1,1]=4 ✓)
```

---

```python
cross   = P[ks - 1, -1] - P[ks - 1, ks - 1]      # one-way cross sum
```

**What it does:** Sum of the off-diagonal rectangular block — the covariance between the left and right sub-groups.

`P[k-1, n-1]` = sum of rows 0..k-1, all columns.
`P[k-1, k-1]` = sum of rows 0..k-1, columns 0..k-1 (within-left block).
Their difference = sum of rows 0..k-1, columns k..n-1 = one-way cross sum.

**Example:**

```
k=1: cross = P[0,3] - P[0,0] = 2.1 - 1.0 = 1.1
k=2: cross = P[1,3] - P[1,1] = 4.2 - 3.6 = 0.6   ← very small!
k=3: cross = P[2,3] - P[2,2] = 11.5 - 8.2 = 3.3
```

---

```python
cov_term  = cross / (ks * (n - ks))
denom_rho = np.sqrt(np.maximum(s_left * s_right, 0.0))
rho       = np.where(denom_rho > 1e-12, cross / denom_rho, 0.0)
objective = 0.5 * cov_term + 0.5 * rho
```

**What it does:** Vectorised evaluation of `_vb_merge_cost` for all n−1 cuts simultaneously.

- `ks * (n - ks)` = `n_left × n_right` for each cut (broadcast).
- `np.maximum(..., 0.0)` clips floating-point negatives before `sqrt`.
- `np.where` applies the same 1e-12 guard as the scalar version.

**Example (all three cuts):**

| k     | cross   | s_left  | s_right  | cov_term            | denom_rho       | rho        | **objective** |
| ----- | ------- | ------- | -------- | ------------------- | --------------- | ---------- | ------------- |
| 1     | 1.1     | 1.0     | 15.6     | 1.1/(1×3)=0.367     | √15.6≈3.950     | 0.279      | 0.323         |
| **2** | **0.6** | **3.6** | **14.0** | **0.6/(2×2)=0.150** | **√50.4≈7.099** | **0.0845** | **0.117** ✓   |
| 3     | 3.3     | 8.2     | 4.0      | 3.3/(3×1)=1.100     | √32.8≈5.727     | 0.577      | 0.839         |

---

```python
k = int(np.argmin(objective)) + 1
return sorted_idx[:k], sorted_idx[k:]
```

**What it does:**

- `np.argmin(objective)` returns the 0-based index of the minimum, so we add 1 to recover the cut position `k`.
- `sorted_idx[:k]` and `sorted_idx[k:]` partition the globally-indexed sorted list at the winning cut.

**Example:**

```
np.argmin([0.323, 0.117, 0.839]) = 1  →  k = 2

sorted_idx[:2] = [2, 3]   →  cluster A
sorted_idx[2:] = [0, 1]   →  cluster B
```

Return value: `([2, 3], [0, 1])` — exactly the natural split. ✓

---

## Summary Comparison

| Property               | `_vb_split_bruteforce`                | `_vb_split_heuristic`                                                          |
| ---------------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| **Guarantee**          | Globally optimal                      | Approximately optimal                                                          |
| **Time complexity**    | O(2ⁿ × n²)                            | O(n²)                                                                          |
| **Cuts evaluated**     | All bipartitions                      | n−1 contiguous cuts in sorted order                                            |
| **When used**          | `len(cluster) ≤ bf_threshold`         | `len(cluster) > bf_threshold`                                                  |
| **Key data structure** | `itertools.combinations`              | 2D prefix-sum array P                                                          |
| **Key insight**        | No shortcut exists for global optimum | Sorting by row-sum aligns cohesive assets; prefix sums give O(1) block queries |

Both functions call `_vb_merge_cost` as their scoring primitive, ensuring the same objective is minimised regardless of which search strategy is used.

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

| Column     | Notes                                                    |
| ---------- | -------------------------------------------------------- |
| `PERMNO`   | CRSP unique stock identifier                             |
| `DlyCalDt` | Calendar date (YYYY-MM-DD)                               |
| `DlyClose` | Split-adjusted closing price (used if DlyRet missing)    |
| `DlyRet`   | Daily total return (preferred over price-derived return) |
| `DlyCap`   | Market capitalisation (used for top-K filtering)         |

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

| Test                          | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| **LW (2008) block-bootstrap** | Circular blocks, b = 21; compares Sharpe ratios |
| **Holm-Bonferroni**           | Step-down FWER correction across strategies     |
| **Benjamini-Hochberg**        | FDR correction (more powerful for multiple H1)  |

---

## Limitations

1. **Transaction costs**: Main ablation uses zero costs. HMVA's monthly turnover ≈ 1.09; at 10 bps per unit |Δw|, annual cost ≈ 130 bps, reducing Sharpe by ~0.16. Cost-robustness sweep confirms HMVA still beats EW at all cost levels up to 20 bps.
2. **Price returns**: `DlyClose` excludes dividends. Total-return comparison would raise all strategies by approximately the same dividend yield (~1.5–2%/year).
3. **Universe**: Top-100 S&P 500 universe is dominated by large-cap stocks; results may differ for broader or international universes.
4. **Attribution**: HMVA combines six stages; the ablation quantifies marginal contributions, but joint attribution (interaction effects) is not separately measured.

---

## Bibliography (key references)

- Ledoit, O., & Wolf, M. (2004). _Honey, I Shrunk the Sample Covariance Matrix._ Journal of Portfolio Management.
- Ledoit, O., & Wolf, M. (2008). _Robust Performance Hypothesis Testing with the Sharpe Ratio._ Journal of Empirical Finance.
- Ledoit, O., & Wolf, M. (2020). _Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices._ Annals of Statistics.
- Fan, J., Liao, Y., & Mincheva, M. (2013). _Large Covariance Estimation by Thresholding Principal Orthogonal Complements._ JRSS-B.
- López de Prado, M. (2016). _Building Diversified Portfolios that Outperform Out of Sample._ Journal of Portfolio Management.
- Molyboga, M. (2020). _A Modified Hierarchical Risk Parity Framework for Portfolio Management._ Journal of Financial Data Science.
- Black, F., & Litterman, R. (1992). _Global Portfolio Optimization._ Financial Analysts Journal.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). _Optimal Versus Naive Diversification._ Review of Financial Studies.
- Diebold, F.X., & Mariano, R.S. (1995). _Comparing Predictive Accuracy._ Journal of Business & Economic Statistics.
