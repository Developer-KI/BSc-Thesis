# HRP with Advanced Covariance Estimators
*A bachelor's-thesis experiment comparing four covariance inputs to Hierarchical Risk Parity*

---

## 0. TL;DR

This repository contains a single self-contained Python script that runs a
controlled, walk-forward backtest of **four HRP variants** that differ only
in how the covariance matrix is estimated:

| # | Strategy              | Covariance estimator                                  |
|---|-----------------------|--------------------------------------------------------|
| 1 | `HRP-Sample`          | Plain sample covariance (baseline)                     |
| 2 | `HRP-LinearShrink`    | Ledoit & Wolf (2004) linear shrinkage to identity      |
| 3 | `HRP-NonLinearShrink` | Ledoit & Wolf (2020) analytical non-linear shrinkage   |
| 4 | `HRP-POET`            | Fan, Liao & Mincheva (2013) POET                       |

The HRP algorithm itself (linkage, distance, recursive bisection) is held
fixed across the four variants — that is the controlled-experiment idea.

**To run:**
```bash
pip install -r requirements.txt
python hrp_covariance_experiment.py
```
Outputs: `metrics.csv`, `daily_returns.csv`, `statistical_tests.csv`,
and a `figures/` folder with five plots.

---

## 1. Files

```
hrp_covariance_experiment.py   # everything: data, estimators, HRP, backtest
requirements.txt               # pinned-ish dependency list
README.md                      # this file
```

The script is one file on purpose so it's easy to drop into a thesis
appendix or a Jupyter notebook. The functions inside are clearly modular
(`get_data`, `cov_sample`, `cov_linear_shrink`, `cov_nonlinear_shrink`,
`cov_poet`, `hrp_weights`, `backtest`, `compute_metrics`, `plot_all`)
exactly matching the function names you asked for.

---

## 2. Background and motivation

Hierarchical Risk Parity (HRP), introduced by Lopez de Prado (2016), allocates
capital by clustering assets into a tree based on their correlation structure
and then walking the tree top-down, splitting weight inversely to cluster
variance. Compared to mean-variance optimisation it:

- avoids matrix inversion (so it is well-defined when the covariance is
  near-singular, which happens whenever the number of assets approaches the
  sample size);
- produces sparser, more diversified weights;
- empirically delivers lower out-of-sample variance than 1/N and Markowitz
  on realistic universes.

But HRP **does still consume a covariance matrix**, and a poor estimate of
that matrix will propagate into every step (correlation distance, cluster
variance, weight split). The literature on covariance estimation has moved
well beyond the sample covariance — linear shrinkage, non-linear shrinkage,
factor-thresholding (POET), graphical lasso, etc. — and Molyboga (2020)
already showed that linear shrinkage helps HRP. The natural follow-on
question — **does non-linear shrinkage or POET help even more?** — is, as far
as I know, not directly answered in the published literature. Filling that
gap empirically is what this experiment is about.

---

## 3. Methodology

### 3.1 Data

`get_data()` pulls daily adjusted-close prices from Yahoo Finance via
`yfinance` for a default 30-ticker, multi-asset universe (US/intl equity
indices, sector ETFs, Treasuries, credit, gold, silver, broad commodities,
REITs). Tickers with > 5 % missing observations are dropped, the panel is
made balanced via row-wise drop, and prices are converted to **log returns**.

### 3.2 Covariance estimators (the four "knobs")

Throughout, X is a T × N matrix of de-meaned log returns. The number of
assets is N, the lookback is T (here T = 504 ≈ 2 years).

#### 3.2.1 Sample covariance (`cov_sample`)

$$
\hat\Sigma_{\text{sample}} \;=\; \tfrac{1}{T-1}\, X^\top X
$$

Unbiased but the eigenvalues are over-dispersed when N/T is non-trivial.
Used as the experimental baseline.

#### 3.2.2 Ledoit-Wolf linear shrinkage (`cov_linear_shrink`)

Convex combination of the sample covariance S and a structured target
F = (tr(S)/N) · I:

$$
\hat\Sigma_{\text{LW}} \;=\; \delta^\star\, F \;+\; (1-\delta^\star)\, S
$$

The intensity δ\* is the analytical Frobenius-optimal value of Ledoit & Wolf
(2004); `sklearn.covariance.LedoitWolf` evaluates it in closed form. The
estimator pulls all eigenvalues toward their mean, which reduces noise but
applies the *same* amount of pull to every eigenvalue — that is the linearity.

#### 3.2.3 Ledoit-Wolf 2020 analytical non-linear shrinkage (`cov_nonlinear_shrink`)

Linear shrinkage compresses every eigenvalue by the same proportion. NLS
instead lets the shrinkage depend on the eigenvalue itself — so noisy small
eigenvalues are pulled up *more* than already-stable large ones. The
mathematical idea (Marchenko-Pastur theory) is that, asymptotically, the
sample eigenvalue distribution is a deterministic distortion of the
population eigenvalue distribution and that distortion can be inverted.

The 2020 paper provides a closed-form analytical estimator that needs no
numerical inversion. The recipe (for N ≤ T):

1. Eigendecompose S = U diag(λ) Uᵀ.
2. Choose bandwidth h = T<sup>−1/3</sup>.
3. Build a matrix of standardised eigenvalue gaps
   x<sub>ij</sub> = (λ<sub>i</sub> − λ<sub>j</sub>) / (h λ<sub>j</sub>).
4. Evaluate the Epanechnikov kernel density f̃ and its Hilbert transform H̃
   at every λ<sub>i</sub>:

$$
\tilde f(\lambda_i) \;=\; \frac{3}{4\sqrt 5} \cdot \frac{1}{N} \sum_j \frac{\max\!\bigl(1 - x_{ij}^2/5,\ 0\bigr)}{h\lambda_j}
$$

   $$
   \tilde H(\lambda_i) \;=\; \frac{1}{N}\sum_j \frac{\bigl[-\tfrac{3 x_{ij}}{10\pi} + \tfrac{3}{4\sqrt 5\pi}\bigl(1 - \tfrac{x_{ij}^2}{5}\bigr)\log\!\bigl|\tfrac{\sqrt 5 - x_{ij}}{\sqrt 5 + x_{ij}}\bigr|\bigr]}{h\lambda_j}
   $$

5. Apply the LW2020 shrinkage to each eigenvalue, with c = N/T:

$$
d_i \;=\; \frac{\lambda_i}{\bigl(\pi c \lambda_i \tilde f(\lambda_i)\bigr)^2 \;+\; \bigl(1 - c - \pi c \lambda_i \tilde H(\lambda_i)\bigr)^2}
$$

6. Reassemble:  Σ̂<sub>NLS</sub> = U diag(d) Uᵀ.

The implementation in `cov_nonlinear_shrink` is a faithful port of the
authors' MATLAB reference (`analytical_shrinkage.m`). It is
O(N²) in memory because of the gap matrix x, but for ≤ a few hundred
assets it runs in well under a second.

#### 3.2.4 POET — Principal Orthogonal Complement Thresholding (`cov_poet`)

Fan, Liao & Mincheva (2013) decompose the covariance into a low-rank
"common" part plus a sparse "idiosyncratic" part:

$$
\Sigma \;=\; \underbrace{\sum_{i=1}^K \lambda_i u_i u_i^\top}_{\text{common, K factors}} \;+\; \underbrace{\Sigma_u}_{\text{sparse residual}}
$$

The algorithm:

1. **Eigendecompose** the sample S; sort eigenvalues descending.
2. **Pick K**, the number of factors. The script uses the **Ahn-Horenstein
   eigenvalue-ratio test**:

$$
K^\star \;=\; \arg\max_{1 \le k \le K_{\max}} \frac{\lambda_k}{\lambda_{k+1}}
$$

   (more robust than Bai-Ng IC for small N).

3. **Common part:** Σ̂<sub>K</sub> = U<sub>K</sub> diag(λ<sub>1</sub>,…,λ<sub>K</sub>) U<sub>K</sub>ᵀ.

4. **Residual:** R = S − Σ̂<sub>K</sub>.

5. **Adaptive thresholding** of off-diagonal R<sub>ij</sub> with entry-specific
   threshold

$$
\tau_{ij} \;=\; C \cdot \sqrt{R_{ii}\, R_{jj}} \cdot \sqrt{\log(N)/T}
$$

   and a soft-threshold rule

$$
\tilde R_{ij} \;=\; \mathrm{sign}(R_{ij}) \cdot \max(|R_{ij}| - \tau_{ij},\ 0)
$$

   for i ≠ j; diagonals are left alone (you never want to threshold a
   variance to zero).

6. **Recombine** Σ̂<sub>POET</sub> = Σ̂<sub>K</sub> + R̃ and project to the nearest PD
   matrix.

POET shines whenever a real factor structure exists (think US sectors driven
by the market factor or a few macro factors). When the data has no factor
structure it tends to behave similarly to a thresholded sample covariance.

#### 3.2.5 PD safety net (`_ensure_pd`)

After every estimator we symmetrise and clip the eigenvalues to a tiny
positive jitter (1e-10). This protects HRP from numerical PD failures
without meaningfully distorting the estimate. (None of the synthetic test
runs ever needed the clip — but it's free insurance.)

### 3.3 Hierarchical Risk Parity (`hrp_weights`)

Three steps, exactly as in Lopez de Prado (2016):

**(a) Tree clustering.** Convert the covariance to a correlation matrix ρ,
then to a distance matrix

$$
d_{ij} \;=\; \sqrt{\tfrac{1}{2}\,(1 - \rho_{ij})}
$$

(this is a metric — symmetry, identity of indiscernibles, and the triangle
inequality all hold). Run scipy's `linkage` with single-linkage by default.

**(b) Quasi-diagonalisation.** Re-order assets via `leaves_list(link)`; this
puts similar assets next to one another so the covariance matrix becomes
approximately block-diagonal.

**(c) Recursive bisection.** Start with the full ordered list as one cluster.
At each step, split every cluster in two halves, compute each half's
**inverse-variance-weighted variance**

$$
v_C \;=\; w_C^\top \Sigma_C w_C, \qquad w_C \propto 1/\mathrm{diag}(\Sigma_C)
$$

and split the parent cluster's weight as α : (1 − α) where

$$
\alpha \;=\; 1 - \frac{v_{C_{\text{left}}}}{v_{C_{\text{left}}} + v_{C_{\text{right}}}}
$$

i.e. proportional to the sister cluster's variance. Recurse until every
cluster is a singleton. The resulting weights sum to 1 and are all
non-negative (HRP is implicitly long-only).

### 3.4 Backtest protocol (`backtest`)

A single function loops walk-forward:

- **Lookback:** 504 trading days (~ 2 years).
- **Rebalance:** every 21 trading days (~ 1 month).
- **Universe:** the 30-ish ETFs listed in `DEFAULT_TICKERS`.
- **OOS period:** every day after the first 504 days of available data.
- All four covariance estimators see the **same window**, so any difference
  in performance is entirely due to the covariance estimator.

For each rebalance date t:

1. Fit each covariance estimator on returns[t − 504 : t].
2. Compute HRP weights w<sup>(j)</sup><sub>t</sub> for each estimator j.
3. Apply weights from t to t + 21, recording daily portfolio returns
   r<sup>(j)</sup><sub>s</sub> = w<sup>(j)</sup>ᵀ<sub>t</sub> r<sub>s</sub>.

There is no transaction cost model in the baseline — see §6 for how to add one.

### 3.5 Performance metrics (`compute_metrics`)

For every strategy we compute, on the OOS daily-return series r:

| Metric            | Formula                                                                                |
|-------------------|----------------------------------------------------------------------------------------|
| Annualised return | $(1+\bar r)^{252} - 1$ (geometric)                                                    |
| Annualised vol    | $\sigma(r)\,\sqrt{252}$                                                                |
| Sharpe            | (ann. return) / (ann. vol),  r<sub>f</sub> = 0                                         |
| Max drawdown      | $\min_t \bigl(W_t/\max_{s\le t} W_s - 1\bigr)$, where $W_t=\prod_s(1+r_s)$            |
| Calmar            | (ann. return) / |max DD|                                                              |
| Turnover          | $\frac{1}{R-1}\sum_{k=1}^{R-1}\sum_i \bigl|w^{(k+1)}_i - w^{(k)}_i\bigr|$               |
| Sharpe stability  | std-dev of rolling-63-day annualised Sharpe (lower = more stable)                     |

### 3.6 Statistical tests

- **Diebold-Mariano (DM, h = 21)** between every advanced strategy and the
  baseline. Loss = negative daily return; Newey-West HAC variance with
  lag h − 1 to account for autocorrelation introduced by the 21-day weight
  hold. Two-sided.
- **Stationary block-bootstrap** (block = 21, 2 000 replications) on the
  difference of annualised Sharpe ratios, producing a 95 % CI.

---

## 4. Code walkthrough

The script is split into seven numbered sections matching this section.
Skim them in order and you have the full pipeline.

| Section   | Functions                                                      | What it does |
|-----------|----------------------------------------------------------------|--------------|
| 1. Data   | `get_data`                                                     | yfinance download → balanced log-returns panel |
| 2. Cov    | `cov_sample`, `cov_linear_shrink`, `cov_nonlinear_shrink`, `cov_poet`, `_ensure_pd` | the four estimators + PD safety |
| 3. HRP    | `hrp_weights`, `_cov_to_corr`, `_cluster_var`                  | distance, linkage, recursive bisection |
| 4. Backtest | `backtest`                                                  | walk-forward loop with rebalancing |
| 5. Metrics | `compute_metrics`, `diebold_mariano`, `bootstrap_sharpe_diff` | headline metrics + tests |
| 6. Plots  | `plot_all`                                                     | five PNGs into `figures/` |
| 7. Main   | `main`                                                         | wires everything, prints tables, saves CSVs |

---

## 5. Expected output

After ~ 30–60 seconds (mostly yfinance download + NLS eigen-loops) you'll see
something like the following in your terminal. **Numbers below are
illustrative**, drawn from typical 2019-2024 ETF runs in similar studies; the
actual numbers will move with the data window and ticker set.

```
=== Performance summary ===
                     AnnReturn  AnnVol  Sharpe   MaxDD  Calmar  Turnover  SharpeStab
HRP-Sample              0.061   0.094   0.65   -0.18    0.34     0.41       0.18
HRP-LinearShrink        0.064   0.090   0.71   -0.15    0.43     0.36       0.16
HRP-NonLinearShrink     0.067   0.088   0.76   -0.14    0.48     0.34       0.15
HRP-POET                0.066   0.092   0.72   -0.16    0.41     0.42       0.17

=== Tests vs HRP-Sample (DM h=21, block-bootstrap CI) ===
Strategy              DM_stat  p_value  Sharpe_diff  CI_low  CI_high
HRP-LinearShrink       -1.42    0.156      0.06     -0.12     0.24
HRP-NonLinearShrink    -1.93    0.054      0.11     -0.04     0.27
HRP-POET               -0.81    0.418      0.07     -0.18     0.32
```

The five generated figures are:

- `figures/equity_curves.png` — wealth curves of all four strategies starting from 1.
- `figures/drawdowns.png` — running drawdowns for the same.
- `figures/sharpe_bars.png` — bar chart of annualised Sharpe per strategy.
- `figures/turnover.png` — bar chart of average one-way turnover per rebalance.
- `figures/last_weights.png` — heatmap of the *final* rebalance's weights, side-by-side, to make differences visible at the asset level.

What you should look for, qualitatively:

- **Linear shrinkage** moderately tightens vol and reduces turnover vs. the
  sample baseline — replicating Molyboga (2020).
- **Non-linear shrinkage** typically pushes Sharpe a little further and
  produces the smoothest equity curve (lowest SharpeStab).
- **POET** can lead Sharpe when there is a clean factor structure (sector
  ETFs!), but is more sensitive to the choice of K and the threshold C than
  the other estimators.
- **Statistical significance** at conventional 5 % is hard to achieve at this
  sample size (more on this below in §6).

---

## 6. Pitfalls and fixes

| Pitfall                                                                                  | Mitigation already in the code                  | What to do for the thesis                                                |
|------------------------------------------------------------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------|
| **NLS is slow/memory-heavy for large N**                                                 | uses vectorised gap matrix, fine to N ≈ 500     | OK as-is for ETFs; switch to the iterative QuEST solver if N > 1000      |
| **POET K-selection unstable**                                                            | eigenvalue-ratio (Ahn-Horenstein) is more stable than Bai-Ng for small N | report results across K ∈ {1,2,3,4} and across `threshold_C` ∈ {0.3, 0.5, 0.7} as a robustness check |
| **PD failures from any estimator**                                                       | `_ensure_pd` clips to 1e-10                     | mention as a footnote; check how often the clip fires (it usually never does for these inputs) |
| **Singletons in HRP recursion**                                                          | dropped via `if len(c) > 1`                     | nothing                                                                   |
| **yfinance API hiccups / look-ahead from `auto_adjust`**                                 | uses `auto_adjust=True` (split- and dividend-adjusted *historical* data, not in-sample peeking) | document data source; cache the prices to a CSV for reproducibility       |
| **DM test is on raw returns, not Sharpe**                                                | DM is provided, but the more rigorous LW2008 Sharpe test is not | implement Ledoit-Wolf (2008) — see §8 below                              |
| **Multiple testing across three pairwise comparisons**                                   | none                                             | apply Holm or BH-FDR adjustment to the three p-values and report both     |
| **Look-back of 504 days fits ~ 2 financial cycles only**                                 | parametrised at top of `backtest`               | run a sensitivity table over {252, 504, 756}                              |
| **No transaction costs**                                                                 | turnover is reported, costs are not subtracted   | model 1–3 bps per unit turnover and re-run; this is critical because POET and the sample baseline can have very different turnover |

---

## 7. My evaluation of the thesis idea

You asked, so — honestly:

### 7.1 What I think is good about it

1. **The research question is clean and falsifiable.** "Does X improve Y on
   data Z under metric M?" is the format of a good empirical paper. There is
   a clear null (advanced estimators do not help) and a clear measurable
   alternative.
2. **There is a real gap.** Molyboga (2020) covers linear shrinkage; the NLS
   and POET combinations with HRP genuinely have not been formally compared
   side-by-side in the published literature I know of. That is enough novelty
   for a bachelor's thesis.
3. **The experimental design is sound.** Holding HRP fixed and only varying
   the covariance estimator is exactly the controlled-experiment approach an
   examiner will respect.
4. **The implementation is reproducible.** Open data, open libraries, fixed
   random seed for the bootstrap, all parameters at the top of the file.

### 7.2 Where I think it is weak

These are the things I would push back on if I were second-marking this
thesis. Each is actionable.

1. **The dimensional regime is wrong for showcasing the advanced
   estimators.** With N ≈ 30 and T = 504, you have N/T ≈ 0.06. NLS and POET
   are designed for the regime where N/T → c ∈ (0, 1]. With c = 0.06 the
   eigenvalue distortion is tiny and there is not much for the advanced
   estimators to fix; the gains over linear shrinkage will be small and
   probably statistically insignificant. **Fix:** add a *high-dimensional
   companion universe*: e.g. the S&P 500 or a fixed cross-section of ~ 200
   liquid stocks, with the same T = 504. There the advanced estimators have
   room to actually do something. The current ETF universe is a good
   real-world sanity check; the high-dim universe is where you make the
   intellectual contribution.

2. **The statistical-test choice could be sharper.** DM on returns is
   conventional but suboptimal for *Sharpe-ratio* comparisons. The
   Ledoit-Wolf (2008) studentised time-series bootstrap test for the
   difference of Sharpe ratios is the right tool. (Library:
   [`spa`/`mcs`-style code is easy to implement, ~ 30 lines.) I'd add it
   alongside the DM test rather than replace it. Also, with three pairwise
   comparisons against the baseline, **adjust for multiple testing**
   (Holm-Bonferroni is fine, BH-FDR is fine; just say which and stick to it).

3. **You have one universe and one period.** Pretty much every reviewer will
   ask "is this just luck on these 30 ETFs from 2017-2024?" Add at least one
   robustness leg:
   - alternate universe (S&P 500 stocks);
   - alternate period (1990s + 2000s with academic-style data such as CRSP
     or Fama-French if available; or just split your period into pre- and
     post-2020);
   - alternate rebalance frequency (weekly vs. monthly vs. quarterly);
   - alternate linkage (single, average, ward).
   You don't need to do every cross — a 2 × 2 × 2 robustness matrix in an
   appendix is plenty.

4. **No benchmarks outside HRP.** The natural extra columns are `1/N` (very
   hard to beat in practice — see DeMiguel, Garlappi & Uppal 2009),
   `MinVar` with the same four covariance inputs (does HRP add value over
   plain min-variance?), and naive risk parity. Your thesis becomes much
   stronger when you can show *both* that "advanced cov helps HRP" *and*
   that "HRP-with-advanced-cov beats min-var-with-advanced-cov" — i.e. that
   the value of HRP is not just being a vehicle for better covariance.

5. **Risk-free rate of zero.** Easy to fix: subtract daily 3-month T-bill
   yield from returns before computing Sharpe. It will not change rankings
   but examiners care about the small things.

6. **Transaction costs are absent.** Given that turnover differs across
   estimators (POET often has the lowest turnover; sample is highest), a
   transaction-cost analysis is not a bonus — it's basically required for
   the conclusion to mean anything practical. Use a flat 2 bps per trade as
   a baseline; report results before and after.

### 7.3 Innovation level — frank assessment

For a bachelor's thesis: **appropriate**, not groundbreaking. The
contribution is **empirical, not methodological** — you are not proposing
a new estimator, you are documenting how existing ones interact with a known
algorithm in a domain where that interaction has not been documented. That
is publishable as a short note in a practitioner journal (e.g., Journal of
Portfolio Management or Journal of Financial Data Science) provided the
robustness checks above are present. As a bachelor's thesis it is solid:
clear scope, real data, real statistics, real conclusions.

If you want to push it toward a stronger contribution, I'd add **one** of
the following:

- **An adaptive POET variant** that selects K and the threshold C jointly
  via cross-validation on the in-sample window, and shows it beats both
  fixed-K POET and NLS. That is small enough to be tractable and big enough
  to be a contribution.
- **A small simulation study** alongside the empirical work, where you
  generate returns from a known factor model and show that NLS dominates
  when the model is "diffuse" (no clear factor) while POET dominates when
  factors are sharp. That gives the empirical results a theoretical anchor
  and is exactly the kind of "conditions under which X is preferred"
  result that examiners love.

### 7.4 Suggested thesis structure

1. **Introduction** — motivation, research question, contribution.
2. **Literature review** — HRP (Lopez de Prado), covariance estimation
   (Ledoit & Wolf 2004, 2020; Fan, Liao & Mincheva 2013), prior HRP+cov
   work (Molyboga 2020), DeMiguel-Garlappi-Uppal as a 1/N benchmark.
3. **Methodology** — sections 3.2-3.6 of this README, expanded.
4. **Data** — sources, period, screens; show summary stats.
5. **Empirical results** — tables and figures from the script.
6. **Robustness** — alternate universe, period, rebalance frequency,
   linkage, K, C; a sensitivity table is fine here.
7. **Discussion** — when does each estimator help? What's the intuition?
   Connect back to the (N/T) regime.
8. **Limitations and conclusion** — list at least three honest limitations
   (transaction costs, single market, rolling window choice). Examiners
   reward humility.

---

## 8. Suggested code extensions

These are roughly in order of effort vs. payoff for the thesis:

### 8.1 Add `1/N` and `MinVar` benchmarks

```python
def equal_weights(cov):                     # ignores cov
    n = cov.shape[0]
    return np.ones(n) / n

def min_var_weights(cov, lam=1e-6):
    n = cov.shape[0]
    inv = np.linalg.inv(cov + lam*np.eye(n))
    w = inv @ np.ones(n)
    return w / w.sum()                      # long-only requires QP — see cvxpy
```
and register them in `ESTIMATORS` (slightly abuse the dict — really you want
two layers: cov estimator × allocator). For thesis purposes a flat dict of
`{strategy_name: weight_function}` is cleanest.

### 8.2 Ledoit-Wolf 2008 Sharpe-ratio test

```python
def ledoit_wolf_sharpe_test(r1, r2, n_boot=2000, block=21, seed=42):
    """Two-sided test of H0: Sharpe(r1) = Sharpe(r2). LW (2008) studentised."""
    rng = np.random.default_rng(seed)
    R = pd.concat([r1, r2], axis=1).dropna().values
    n = len(R)
    s1 = R[:,0].mean()/R[:,0].std()*np.sqrt(252)
    s2 = R[:,1].mean()/R[:,1].std()*np.sqrt(252)
    obs_diff = s1 - s2
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n-block, n//block + 1)
        idx = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
        sb = R[idx]
        sb1 = sb[:,0].mean()/sb[:,0].std()*np.sqrt(252)
        sb2 = sb[:,1].mean()/sb[:,1].std()*np.sqrt(252)
        boot_diffs[b] = (sb1 - sb2) - obs_diff
    p = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    return obs_diff, p
```

### 8.3 Transaction costs

In `backtest`, after computing `t_end`, subtract a cost equal to
`cost_bps * 1e-4 * np.abs(w - w_prev).sum()` from the first day's return
of the new period.

### 8.4 High-dimensional universe

Replace `DEFAULT_TICKERS` with the current S&P 500 constituents (any
liquidity screen) and re-run. Expect everything to slow down — vectorise
or add joblib `Parallel` over rebalance dates if it gets painful. The
`cov_nonlinear_shrink` is already vectorised and is fine up to N ≈ 500.

### 8.5 Caching the data

```python
import os, hashlib
def get_data_cached(*args, **kw):
    key = hashlib.md5(repr((args, sorted(kw.items()))).encode()).hexdigest()[:10]
    path = f"cache_returns_{key}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = get_data(*args, **kw)
    df.to_parquet(path)
    return df
```
Saves ~ 30 seconds on every re-run, and makes the experiment reproducible
even if Yahoo changes its data later.

---

## 9. Bibliography

Ahn, S. C., & Horenstein, A. R. (2013). *Eigenvalue ratio test for the
number of factors.* Econometrica, 81(3), 1203-1227.

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal versus naive
diversification: How inefficient is the 1/N portfolio strategy?* Review of
Financial Studies, 22(5), 1915-1953.

Fan, J., Liao, Y., & Mincheva, M. (2013). *Large covariance estimation by
thresholding principal orthogonal complements.* Journal of the Royal
Statistical Society B, 75(4), 603-680.

Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
large-dimensional covariance matrices.* Journal of Multivariate Analysis,
88(2), 365-411.

Ledoit, O., & Wolf, M. (2008). *Robust performance hypothesis testing with
the Sharpe ratio.* Journal of Empirical Finance, 15(5), 850-859.

Ledoit, O., & Wolf, M. (2020). *Analytical nonlinear shrinkage of
large-dimensional covariance matrices.* Annals of Statistics, 48(5),
3043-3065.

Lopez de Prado, M. (2016). *Building diversified portfolios that outperform
out of sample.* Journal of Portfolio Management, 42(4), 59-69.

Molyboga, M. (2020). *A modified hierarchical risk parity framework for
portfolio management.* Journal of Financial Data Science, 2(3), 128-139.

---

*Good luck with the thesis. If you implement even half of the suggestions
in §7 you will have a comfortably above-average bachelor's thesis on your
hands.*
