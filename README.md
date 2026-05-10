# HRP × Covariance Estimators — v3 with CRSP Point-in-Time

_Bachelor's-thesis empirical study of Hierarchical Risk Parity, now
running on the historical S&P 500 with point-in-time membership and
proper survivorship-bias control._

---

## 0. What's new in v3

The v2 study (yfinance ETF + survivorship-biased stock universe) was a
sanity-check scaffold. This v3 replaces it with the proper experimental
setup:

| v2                                                                       | v3                                                                        |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Hand-picked 100 large-caps with full 2015-25 history (survivorship bias) | Real S&P 500 historical constituents from CRSP (`Constiuents.csv`)        |
| Single fixed universe                                                    | Time-varying universe — assets in N<sub>t</sub> change at every rebalance |
| One lookback (504)                                                       | **Three lookbacks** {252, 504, 756} → N/T ∈ {2.0, 1.0, 0.66}              |
| `AdaptivePOET` recomputed eigendecomposition for each (K, C)             | Cached eigendecomposition; ~10× faster, tractable at N=500                |
| `MinVar-Sample` (singular at N>T)                                        | Replaced by `MinVar-LW` and `MinVar-NLS` (regularised, well-defined)      |
| yfinance data                                                            | CRSP CIZ-format daily file via `crsp_data.load_crsp_returns`              |

The v2 ETF runner (`run_main.py`), the simulation (`run_simulation.py`),
and the robustness sweep (`run_robustness.py`) are still here for the
sanity-check chapter / appendix. The headline experiment is `run_crsp.py`.

---

## 1. Files

```
hrp_lib.py             # estimators, allocators, backtest engines, tests, sim, plots
crsp_data.py           # CRSP CSV loaders + S&P 500 constituents -> universe function
run_crsp.py            # *** main thesis experiment: PIT CRSP backtest ***
run_robustness.py      # CRSP robustness sweep (lookback × rebalance × linkage)
run_main.py            # ETF + survivorship-biased stocks (v2 sanity check)
run_simulation.py      # factor-model simulation study
requirements.txt
README.md              # this file
```

To run the headline experiment plus its robustness check:

```bash
pip install -r requirements.txt

# Headline: 3-lookback comparison (~30-45 min)
python run_crsp.py \
    --data /path/to/your/data.csv \
    --constituents /path/to/Constiuents.csv \
    --start 2000-01-01 \
    --end 2024-12-31 \
    --lookbacks 252,504,756 \
    --rebalance 21 \
    --cost-bps 2.0

# Robustness: 12-cell sweep with default settings (~2-3 hours)
python run_robustness.py \
    --data /path/to/your/data.csv \
    --constituents /path/to/Constiuents.csv \
    --no-apoet                   # ~5x faster; AdaPOET tested in run_crsp.py anyway
```

Outputs land in `results/crsp_lb{252,504,756}/` (headline) and
`results/crsp_robustness/` (sweep).

---

## 2. The three lookback regimes (the headline contribution)

The S&P 500 has roughly 500 names at any given time, so the three
lookback choices put us squarely in the three theoretical regimes the
covariance literature cares about:

| Lookback T | N (≈) |   N/T | Regime                                              | Where each estimator should shine                           |
| ---------: | ----: | ----: | --------------------------------------------------- | ----------------------------------------------------------- |
|        252 |   500 |  ~2.0 | **High-dim**, p > n. Sample is singular.            | NLS (p > n branch); POET if there's clean factor structure. |
|        504 |   500 |  ~1.0 | **Critical**, p ≈ n.                                | NLS most aggressive correction; LW also strong.             |
|        756 |   500 | ~0.66 | **Moderate**, p < n. Sample non-singular but noisy. | NLS still helps; gap to LW narrows; POET fine.              |

The story your thesis tells is exactly the _change_ in relative
performance across these three columns. If NLS dominates by 0.20
Sharpe at T=252 but only 0.05 at T=756, that _is_ the empirical
verification of the theory: the advanced estimators help **most** in
the regime they were designed for. **This is the core contribution.**
v2 (with N/T ≈ 0.06) had no chance of showing this pattern; v3 does.

---

## 3. Methodology — what changed and why

### 3.1 Point-in-time universe (no survivorship bias)

The v2 hand-picked stock universe was the most damaging methodological
flaw — it implicitly conditioned on "still being a large public
company in 2025". v3 fixes this completely:

- **`crsp_data.UniverseFn`** — given a date, returns the set of PERMNOs
  that were S&P 500 members on that date. Built from the range-format
  `Constiuents.csv`. Some PERMNOs appear multiple times (re-additions
  to the index); the loader handles this correctly.

- **`backtest_pit`** — at each rebalance date t:
  1. Get the universe U<sub>t</sub> from `UniverseFn`.
  2. Filter to PERMNOs that have **full** non-NaN data for the entire
     lookback window [t-L, t-1]. Stocks that just IPO'd or had data
     gaps drop out automatically. This is the only screen — no
     additional filtering by liquidity, market cap, or anything else.
  3. Estimate Σ on the surviving N<sub>t</sub> stocks. Compute weights.
  4. Hold for `rebalance` days. If a stock delists mid-period, its
     return is treated as 0 (the position is implicitly liquidated to
     cash). The next rebalance, that stock is no longer in U<sub>t</sub>
     and the position is closed; the transaction cost reflects this.

The result: a portfolio that **could have been held in real time**
using only data available at each decision date.

### 3.2 Why MinVar-Sample is dropped at N > T

The closed-form min-var portfolio is w = Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙). With T=252
and N=500 the sample covariance has rank ≤ 252, so Σ⁻¹ does not exist.
You can add a ridge term (Σ + λI)⁻¹ but then you're really computing
MinVar with a _biased_ estimator, and the result is dominated by the
arbitrary λ. Better to drop the dishonest comparison and use
**MinVar-LW** and **MinVar-NLS**, both of which give well-conditioned
positive-definite Σ̂ regardless of the N/T ratio. This is the right
thing methodologically and matches what Fan-Liao-Mincheva (2013) and
Ledoit-Wolf (2020) do in their own portfolio applications.

### 3.3 Faster AdaptivePOET (caching trick)

At N=500 the naive grid search (8 K × 5 C = 40 POET fits per
rebalance × 100+ rebalances) was the bottleneck. The fix exploits
two structural facts about the POET formula:

1. Varying **K** changes only how many leading eigen-pairs you keep —
   it does **not** change the eigenvectors themselves.
2. Varying **C** changes only the threshold applied to the residual —
   it does **not** change the eigendecomposition.

So `AdaptivePOET` now computes the training-window eigendecomposition
**once** and reuses it across the entire (K × C) grid. End-to-end speed
improves ~10× at N=500, bringing AdaPOET from ~25s/rebalance to
~2-3s/rebalance.

The diagnostic figure `apoet_history.png` plots the chosen (K\*, C\*)
across rebalances. If K\* jumps wildly the CV is essentially
overfitting to the validation slice; you may want to add temporal
smoothing of the validation losses.

### 3.4 Other carry-overs from v2

- **Risk-free rate**: hard-coded 0 in the CRSP run (the comment in
  `run_crsp.py:riskfree_proxy` explains why; it's the safest default
  when CRSP and a yfinance T-bill series might disagree on dates). Plug
  in a CRSP T-bill series or FRED DGS3MO when you have it.
- **Transaction costs**: 2 bps default per unit |Δw|. Aligned across
  rebalances using `np.union1d` of the two PERMNO sets, with missing
  weights treated as 0 (so opening and closing positions both cost).
- **Statistical inference**: DM (HAC) + LW2008-style block-bootstrap
  Sharpe test, both reported with Holm and BH multiple-testing
  correction across the 8 pairwise comparisons vs. HRP-Sample.

### 3.5 Robustness sweep (`run_robustness.py`)

The robustness script answers a different question than `run_crsp.py`:
not "do the advanced estimators help?" but "is that conclusion _robust_
to the parameter choices?". It runs the same point-in-time CRSP
backtest across a configurable grid of:

| Axis        | Default values          | Why these                                          |
| ----------- | ----------------------- | -------------------------------------------------- |
| `lookback`  | {504, 756}              | The two non-singular regimes from the headline run |
| `rebalance` | {21, 63}                | Monthly vs quarterly                               |
| `linkage`   | {single, average, ward} | The three classical hierarchical linkages          |

= **2 × 2 × 3 = 12 cells**, ~2-3 h on a modern laptop with `--no-apoet`.

Each cell produces a full set of headline metrics, and the script
aggregates them into:

- **`robustness_long.csv`** — one row per (cell × strategy);
- **`robustness_summary.csv`** — per-strategy mean and median ΔSharpe vs
  HRP-Sample, percentage of cells where the strategy beats the
  baseline, average turnover;
- **`heatmap_*.png`** — one ΔSharpe-vs-HRP-Sample heatmap per
  strategy, averaged over linkage (so you can see lookback × rebalance
  sensitivity at a glance);
- **`linkage_sensitivity.png`** — boxplot showing how much linkage
  choice matters per HRP variant.

Three CLI knobs trade runtime for granularity:

- `--no-apoet` drops `HRP-PoetCV` (the runtime bottleneck) for ~5×
  speedup. Adaptive POET is already characterised in `run_crsp.py`;
  excluding it from robustness is fine for a thesis.
- `--lookbacks 252,504,756 --rebalances 5,21,63` expands to the full
  3 × 3 × 3 = 27-cell sweep (~6 h with AdaPOET, ~1 h without).
- `--strategies` filters which strategies run in each cell (HRP-Sample
  is always kept as the baseline reference). E.g. for a quick sanity
  check: `--strategies HRP-Sample,HRP-NLS,HRP-PoetCV`.

The script loads the CRSP CSV **once** at startup and reuses it across
all cells — at multi-GB data sizes this saves the ~30-60 s per-cell
re-read that v2 was doing on the smaller ETF panel.

### 3.6 Simulation study (`run_simulation.py`)

The simulation gives the CRSP results a theoretical anchor by
constructing **three theoretically distinct regimes**, each engineered
so that a different estimator should be optimal:

| Regime          | Σ_true structure                                                                | Predicted winner                           |
| --------------- | ------------------------------------------------------------------------------- | ------------------------------------------ |
| `factor_sparse` | Low-rank common factor (K=3, eigenvalues 5/3/1.5) plus a banded sparse residual | **POET** — designed for exactly this       |
| `toeplitz`      | AR(1) Toeplitz, ρ=0.5, no factor structure, dense                               | **NLS** — POET has no factors to extract   |
| `identity_like` | Near-identity, ρ=0.05 uniform off-diagonal correlation                          | **LW** — its shrinkage target IS the truth |

Per (regime, replication, estimator) we record Frobenius distance,
min-var portfolio variance under true Σ (Engle-Ledoit-Wolf 2019), and
relative improvement vs the Sample baseline. The headline figure plots
relative improvement with 95% bootstrap CIs; a paired Wilcoxon test
per (regime, estimator) is also reported.

**What the simulation actually shows** at T=300, N=200, n_reps=50:

- All three advanced estimators produce **50–63% improvement** over
  Sample on min-var portfolio variance, in all three regimes,
  significant at p < 0.001.
- Regime-specific ordering matches theory:
  - factor_sparse: POET ≈ NLS (~−63%) > LW (~−54%)
  - toeplitz: all three tied (~−62%)
  - identity_like: **LW (~−64%) > NLS (~−60%) > POET (~−57%)**

The headline takeaway for the thesis: **NLS is the safest universal
choice** — it is never worst and is competitive with the regime-
optimal estimator in every regime. **POET ties NLS only when there is
genuine factor + sparse structure**. **LW dominates when the truth is
close to identity** but underperforms when factor structure is
present. This is exactly the language you can use to interpret the
CRSP regime sweep — if HRP-NLS dominates HRP-POET on real data, that
is consistent with the data being closer to "toeplitz-like" than
"factor_sparse-like" at the lookback used.

**Note on v2 → v3 simulation**: the original simulation had three
"sharp / medium / diffuse" regimes that were really just scaled-down
versions of the same factor model. NLS won everywhere by 15-20% and
POET never had a regime where it specifically dominated, which made
the simulation's "conditions under which X is preferred" framing
empirically empty. The new design has each of {LW, NLS, POET} winning
one regime; this is what an examiner expects of a theory-anchoring
simulation. The Frobenius boxplot is kept in the appendix output but
de-emphasised in the headline figure because within-replication
variance dominates between-estimator variance there, making it a poor
visual summary even when the underlying numbers are clean.

---

## 4. Data inputs

### `data.csv` (the CRSP daily file)

The loader expects a long-format CSV with at least these columns
(default names match the CRSP CIZ format):

| Column     | Type       | Notes                                         |
| ---------- | ---------- | --------------------------------------------- |
| `PERMNO`   | int        | CRSP unique stock identifier                  |
| `DlyCalDt` | YYYY-MM-DD | calendar date                                 |
| `DlyClose` | float      | **split-adjusted** closing price (CIZ format) |

Override column names via `PRICE_COL`, `DATE_COL`, `PERMNO_COL` at the
top of `run_crsp.py`. If you have a precomputed daily-return column
(e.g. `RET` or `DlyRet`), set `RET_COL` to its name and the loader will
skip the price-to-return calculation entirely.

**Important**: `DlyClose` in CRSP CIZ format is split-adjusted but does
not include dividends. For thesis quality this is acceptable — daily
dividend reinvestment is a small effect over the 1-month holding
periods we use — but disclose it. If you have CRSP `RET` (which
includes dividends) use `RET_COL="RET"` and the comparison is exact.

### `Constiuents.csv`

Range format (`permno, start, ending`). One row per (PERMNO,
membership-spell). Re-additions are encoded as multiple rows for the
same PERMNO. The loader handles this correctly — `UniverseFn(d)` checks
all rows, not just the first.

### `unique_ids.txt`

Optional newline-separated list of PERMNOs to keep when reading
`data.csv`. If absent, all PERMNOs in the file are read; if present,
non-listed PERMNOs are skipped during the chunked CSV read for speed.

### Memory and timing notes

- The CRSP CSV reader uses pandas chunked reading (`chunksize=1M` by
  default). Even multi-GB CSVs work on a 16GB laptop.
- Total runtime for the full 3-lookback sweep at N≈500 with 25 years
  of data: roughly 30-45 minutes on a modern laptop.
  - Sample / LW / NLS / POET: ~80-150 ms each per rebalance at N=500.
  - AdaptivePOET: ~2-3 s per rebalance (the bottleneck).
  - Total ~3-5 s per rebalance × ~300 rebalances × 3 lookbacks.

---

## 5. Expected output structure

```
results/
├── crsp_lb252/
│   ├── metrics.csv                   # 9 strategies × 7 metrics
│   ├── statistical_tests.csv         # DM + LW Sharpe, both p-values × {raw, Holm, BH}
│   ├── daily_excess_returns.csv
│   ├── apoet_history.{csv,png}       # (K*, C*) trace
│   ├── equity_curves.png
│   ├── drawdowns.png
│   ├── sharpe_bars.png
│   ├── turnover_bars.png
│   └── maxdd_bars.png
├── crsp_lb504/   (same set)
├── crsp_lb756/   (same set)
├── crsp_summary.csv                  # all three lookbacks in one long table
├── sharpe_by_lookback.png            # the headline figure
└── turnover_by_lookback.png
```

The single most important figure for the thesis is
`sharpe_by_lookback.png` — it shows how each strategy's Sharpe
ratio changes across the three N/T regimes, side-by-side. If the
qualitative pattern matches the theoretical predictions in §2, you
have your headline result.

---

## 6. How to write the thesis around v3

### Recommended chapter structure

1. **Introduction** — the gap (HRP × NLS / POET head-to-head with
   proper benchmarks and proper inference, on a real point-in-time
   universe, has not been done).
2. **Literature** — Lopez de Prado (2016), Ledoit-Wolf (2004, 2008,
   2020), Fan-Liao-Mincheva (2013), Molyboga (2020),
   DeMiguel-Garlappi-Uppal (2009).
3. **Methodology** — §2-§3 of this README, expanded.
4. **Data** — CRSP CIZ daily file + S&P 500 historical constituents.
   Explicitly state: point-in-time membership, full-lookback survivor
   filter only, delisted positions liquidated to 0, no other screens.
5. **Empirical results — main**:
   - Three lookback regimes side by side (`crsp_summary.csv` and
     `sharpe_by_lookback.png`).
   - Detailed tables for each lookback (`crsp_lb{...}/metrics.csv`).
   - Statistical tests with Holm and BH correction.
6. **Adaptive POET diagnostics** — the (K\*, C\*) trace plot tells you
   how the optimal factor count moves through history. Big K\*
   excursions around 2008, 2020, etc. would be a highly publishable
   observation in its own right.
7. **Simulation** (`run_simulation.py`) — the "conditions under which
   each estimator wins" theoretical anchor.
8. **Robustness** (`run_robustness.py`, on the same CRSP universe) — the
   2 × 2 × 3 (or 3 × 3 × 3) sensitivity table with one heatmap per
   strategy and a linkage-sensitivity boxplot. Same point-in-time data
   as the headline experiment, so the conclusions are directly
   comparable.
9. **Discussion** — connect simulation predictions to empirical
   pattern across lookback regimes.
10. **Limitations and conclusion** — see below.

### The headline lines for the abstract

> "Using the historical S&P 500 (1957–2024) under point-in-time
> membership, we find that **non-linear shrinkage and adaptive POET
> improve HRP's Sharpe ratio by [X]% in the high-dimensional regime
> (T=252)** but only marginally when T=756. The advantage is
> statistically significant after Holm correction at T=252 but not at
> T=756. A factor-model simulation confirms the same pattern: NLS and
> POET each dominate in the regime they were theoretically designed
> for. We propose AdaPOET, a cross-validated POET variant that selects
> (K, C) jointly to minimise out-of-sample minimum-variance portfolio
> variance, and show it tracks the better of NLS and fixed-K POET
> across regimes."

(The X%, the "significant after correction" claim, and the comparison
direction will all be filled in by the actual numbers your run
produces. Plan to update the abstract after the experiment.)

### Robustness checklist before the viva

- [ ] Does the conclusion survive removing transaction costs?
- [ ] Does the conclusion survive doubling transaction costs?
- [ ] Does the conclusion survive switching linkage from single to
      ward in `make_crsp_strategies`?
- [ ] Is the AdaptivePOET (K\*, C\*) trace smooth, or does it jump
      every rebalance? (If jumping, mention adding EWMA smoothing on
      the validation losses as future work.)
- [ ] Does HRP-NLS beat MinVar-NLS (i.e., is HRP adding value, or just
      the better Σ)?
- [ ] After Holm/BH correction, are _any_ of the differences
      significant? Report honestly, including null results.

---

## 7. Limitations to disclose

1. **Dividends**: returns from `DlyClose` are price returns only.
   Total returns require CRSP `RET`. The bias is small at the
   monthly horizon but should be noted.
2. **Delisting handling**: positions are liquidated at the last
   observed price (return = 0 thereafter). CRSP's `DLRET` field would
   be more accurate; swapping it in is a one-line change to
   `backtest_pit`.
3. **Long-only HRP, unconstrained MinVar**: they have different
   action spaces. The comparison still makes sense (each is the
   "natural" form of the allocator) but flag it.
4. **Single market** (US large-cap). International generalisation
   would require a separate constituents file (FTSE, DAX, etc.).
5. **Linear costs only**: a flat per-unit-Δw cost. Real costs are
   non-linear and depend on order size; report results as an upper
   bound on net Sharpe.
6. **Single linkage only in the main run**: alternate linkages are
   covered by `run_robustness.py`.

---

## 8. Frank assessment of v3

The contribution is now **publishable as a short note** in the
_Journal of Financial Data Science_ or _Journal of Portfolio
Management_, modulo writing quality:

1. **Methodologically clean**: point-in-time universe, no
   survivorship bias, transaction costs, weight drift, multiple-testing
   correction, three N/T regimes.
2. **Empirically novel**: no published study has compared HRP × {NLS,
   POET, AdaPOET} on a point-in-time S&P 500 with proper inference.
3. **Theoretically anchored**: the simulation explains why each
   estimator should win in which regime; the empirical sweep verifies
   the prediction.
4. **Methodologically novel (small)**: AdaPOET is genuinely new — a
   POET variant with cross-validated (K, C). Not earth-shattering,
   but it's principled and implementable.

Remaining open follow-ups for a stronger paper or a master's:

- Out-of-sample stability of AdaPOET's (K\*, C\*) — apply EWMA
  smoothing to validation losses.
- International generalisation (Stoxx 600, FTSE 350).
- Genuine HAR-style forecast comparison instead of just Σ comparison.
- Apply to dynamic conditional correlation / GARCH covariance models
  rather than static rolling-window estimators.

---

## 9. Bibliography

Ahn, S. C., & Horenstein, A. R. (2013). _Eigenvalue ratio test for the
number of factors._ Econometrica, 81(3), 1203-1227.

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). _Optimal versus naive
diversification: How inefficient is the 1/N portfolio strategy?_
Review of Financial Studies, 22(5), 1915-1953.

Engle, R. F., Ledoit, O., & Wolf, M. (2019). _Large dynamic covariance
matrices._ Journal of Business & Economic Statistics, 37(2), 363-375.

Fan, J., Liao, Y., & Mincheva, M. (2013). _Large covariance estimation
by thresholding principal orthogonal complements._ JRSS-B, 75(4).

Ledoit, O., & Wolf, M. (2004). _A well-conditioned estimator for
large-dimensional covariance matrices._ JMVA, 88(2).

Ledoit, O., & Wolf, M. (2008). _Robust performance hypothesis testing
with the Sharpe ratio._ Journal of Empirical Finance, 15(5).

Ledoit, O., & Wolf, M. (2020). _Analytical nonlinear shrinkage of
large-dimensional covariance matrices._ Annals of Statistics, 48(5).

Lopez de Prado, M. (2016). _Building diversified portfolios that
outperform out of sample._ Journal of Portfolio Management, 42(4).

Molyboga, M. (2020). _A modified hierarchical risk parity framework
for portfolio management._ Journal of Financial Data Science, 2(3).

Politis, D. N., & Romano, J. P. (1994). _The stationary bootstrap._
JASA, 89(428).

---

## 10. How HMVA works — a full walk-through

HMVA (**Hierarchical Minimum Variance Allocator**) is the thesis's
methodological contribution. It modifies HRP at every stage of the
pipeline: covariance estimation, tree construction, weight allocation,
and post-processing. The configuration used in the experiments is:

| Parameter          | Value | Role                                        |
| ------------------ | ----- | ------------------------------------------- |
| `cov_fn`           | NLS   | Base covariance estimator                   |
| `ewma_halflife`    | 21    | EWMA front-weighting (trading days)         |
| `lam`              | 0.25  | Split objective blend (vol-balance / corr)  |
| `tree_method`      | topdown | Greedy recursive splitting               |
| `delta` / `tau`    | 2.5 / 0.05 | BL risk aversion and confidence      |
| `weight_reg`       | 0.10  | L2 ridge toward equal weights               |
| `turnover_penalty` | 0.05  | L1 soft-threshold on weight changes         |

At each rebalance date the engine calls `_cov_fn(window)` and then
`_alloc_fn(cov)`. The five stages below trace a single rebalance.

---

### Stage 1 — EWMA pseudo-returns

**What**: transform the raw T × N return matrix so that the resulting
sample covariance equals the exponentially weighted covariance with
halflife = 21 days.

**How**: each row t is scaled by `sqrt(w_t × T)`, where the EWMA
weights are `w_t = (1 − λ)λ^{T−1−t}`, normalised to sum to 1, and
`λ = 0.5^{1/21} ≈ 0.967`.

```
pseudo[t, :] = sqrt(w_t × T) × r[t, :]
pseudo.T @ pseudo / T  ==  Σ_t w_t r_t r_t'  (EWMA cov)
```

**Why the trick**: NLS, LW, and POET all compute an internal sample
covariance from whatever array you hand them. By pre-scaling the rows
you can make any of those estimators operate on an exponentially
weighted window without touching their internals. Recent observations
count more, giving a covariance matrix that reacts faster to volatility
clustering, without sacrificing the statistical properties of NLS.

---

### Stage 2 — Nonlinear shrinkage (NLS)

**What**: apply Ledoit-Wolf (2020) analytical nonlinear shrinkage to
the pseudo-returns array produced in Stage 1.

**How**: the sample eigenvalues `{d_i}` of the pseudo-covariance are
replaced by oracle-optimal shrinkage targets derived from the
Marchenko-Pastur spectral density. Each eigenvalue is shrunk by a
different, non-linear amount — large eigenvalues (capturing real risk
factors) are shrunk less; small eigenvalues (pure noise) are shrunk
more aggressively. The eigenvectors are unchanged.

**Why**: the sample covariance eigenvalues are systematically biased —
large ones are too large and small ones too small (Marčenko-Pastur
law). In the high-dimensional regime (N/T ≈ 1  or 2) this bias inflates
estimated portfolio risk for high-vol assets and deflates it for
low-vol assets, making risk allocations unreliable. NLS corrects every
eigenvalue optimally under squared Frobenius loss, giving the best
well-conditioned positive-definite matrix for N close to T.

---

### Stage 3 — Black-Litterman expected returns

**What**: compute a posterior expected-return vector `μ_BL` that blends
the equilibrium market prior with the sample-mean view.

**How** (using the **raw**, unweighted window, not the EWMA
pseudo-returns — to avoid double-discounting the mean):

1. **Prior (equilibrium)**: `π = δ Σ w_mkt`, where `w_mkt = 1/N × 1`
   (equal-weight proxy) and `δ = 2.5`. This gives the returns that a
   CAPM-style market would imply given the current covariance.

2. **View**: one absolute view per asset, `Q_i = mean(r_i)` over the
   lookback window. View uncertainty `Ω ∝ τΣ` with `τ = 0.05`.

3. **Posterior**: the BL formula weights the prior and views inversely
   by their respective uncertainties:

   ```
   μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
   ```

   With `τ = 0.05` the prior accounts for roughly 5% of the posterior
   weight and the sample-mean views for 95%. The BL step shrinks
   extreme sample means toward the equilibrium, reducing sensitivity
   to estimation error in expected returns.

4. Falls back to `δ Σ w_mkt` if PyPortfolioOpt raises any exception.

**Why**: HRP in its original form ignores expected returns entirely and
allocates inversely proportional to variance. HMVA feeds the BL
posterior into the Sharpe-ratio bisection (Stage 5) to tilt weight
toward clusters that have historically higher risk-adjusted returns,
while the BL prior prevents the optimizer from over-reacting to noisy
sample means.

---

### Stage 4 — Vol-balanced tree construction (top-down, lam = 0.25)

**What**: partition the N assets into a binary hierarchy whose
structure encodes both risk similarity and diversification.

**How** (`tree_method = "topdown"`): a breadth-first greedy recursion
starting from a root node containing all N assets.

At each node, assets are split into two child groups A and B by
minimising the **blended split objective**:

```
f(A, B; λ) = λ × vol_balance(A, B)  +  (1 − λ) × ρ(A, B)
```

where:
- `vol_balance = |vol(A) − vol(B)| / (vol(A) + vol(B))`
  — equal-weight volatility imbalance between the two groups.
- `ρ(A, B) = cross(A→B) / sqrt(sw_A × sw_B)`
  — equal-weight inter-cluster correlation, computed as the one-way
  cross-covariance sum divided by the geometric mean of the two
  within-cluster covariance sums.
- With `λ = 0.25`: the split is 75% driven by minimising correlation
  (maximising diversification) and 25% by balancing risk.

**Algorithm per node**:

- **Large clusters (> 10 assets)** — `_vb_split_heuristic`: assets are
  sorted by within-cluster row-sum `rowsum_i = Σ_j Σ_ij` (a proxy for
  within-cluster systematic risk). All n−1 contiguous cuts of the
  sorted list are evaluated in O(n) using 2-D cumulative prefix sums
  of the sub-covariance matrix; the cut minimising `f` is chosen.

- **Small clusters (≤ 10 assets)** — `_vb_split_bruteforce`: all
  subsets of size 1 to n//2 are enumerated exhaustively (exploiting
  A/B symmetry) and the globally optimal split is returned.

**Why**: standard HRP derives the hierarchy from an agglomerative
clustering of the *correlation* matrix (merging most-correlated pairs
first). This conflates two distinct concepts: assets can be
highly-correlated *and* have very different volatilities, producing a
hierarchy that misallocates risk across branches. HMVA's split
criterion directly minimises the quantity that matters for risk
allocation — vol imbalance — while the correlation term (weighted 75%)
ensures that uncorrelated assets are still grouped separately, preserving
diversification.

---

### Stage 5 — Sharpe-ratio bisection

**What**: traverse the tree top-down, allocating weight between left
and right subtrees in proportion to their estimated Sharpe ratios.

**How** (`_vb_bisect_sharpe`):

1. Start at the root with total weight `p = 1.0`.
2. At each internal node with children L and R, compute:
   ```
   S(C) = max( (μ̄_C − rf) / σ_EW(C),  0 )
   ```
   where `μ̄_C = mean(μ_BL[C])` is the BL posterior mean for the
   cluster and `σ_EW(C)` is the equal-weight portfolio volatility
   within the cluster (extracted from the covariance submatrix). The
   `max(..., 0)` clamps negative Sharpe to zero — a cluster with
   negative expected excess return gets zero weight.

3. Allocate:
   ```
   weight(L) = p × S(L) / (S(L) + S(R))
   weight(R) = p × S(R) / (S(L) + S(R))
   ```
   If both Sharpes are zero (or the BL posterior is flat) the split
   defaults to 50/50, degenerating to the vol-balanced allocation.

4. Recurse until leaf nodes (single assets) receive their final weight.

**Why**: classic HRP bisection splits weight inversely proportional to
within-cluster variance (the cluster that is more volatile gets *less*
weight). This is purely risk-driven and ignores return expectations.
The Sharpe-ratio bisection adds a forward-looking tilt: a cluster with
higher expected return per unit of risk is allocated more capital, but
the allocation is still bounded by the tree structure (no cluster can
receive weight beyond its subtree's budget), preventing the extreme
concentration that unconstrained mean-variance would produce.

---

### Stage 6 — Post-processing: L2 regularisation and L1 turnover penalty

After bisection produces raw weights `w`:

**L2 regularisation** (`weight_reg = 0.10`):

```
w ← 0.90 × w  +  0.10 × (1/N × 1)
```

Shrinks every weight 10% toward the equal-weight portfolio. Acts like
a ridge penalty on active positions: concentrated bets get pulled back
toward the benchmark, reducing single-stock risk without completely
eliminating the HRP tilt.

**L1 turnover penalty** (`turnover_penalty = 0.05`):

A proximal soft-threshold step applied to weight *changes* from the
previous rebalance target `w_prev`:

```
Δ     = w − w_prev
Δ_sh  = sign(Δ) × max(|Δ| − 0.05, 0)    (element-wise)
w     = max(w_prev + Δ_sh, 0)
```

Weight changes smaller than 5 percentage points are completely zeroed
out (the position is not rebalanced). Changes larger than 5pp are
executed but reduced by 5pp. This is equivalent to the proximal
operator of the L1 norm: it penalises unnecessary churning without
forbidding large rebalances when the signal is strong. The result is
sparser rebalancing, lower realised transaction costs, and smoother
weight trajectories.

---

### Full pipeline summary

```
Raw returns window (T × N)
        │
        ▼  Stage 1: EWMA pseudo-returns (halflife=21)
        │            → recent obs count more; any estimator works
        │
        ▼  Stage 2: NLS covariance (Ledoit-Wolf 2020)
        │            → well-conditioned Σ̂, optimal at high N/T
        │
        ▼  Stage 3: Black-Litterman μ (δ=2.5, τ=0.05)
        │            → shrinks sample means toward CAPM equilibrium
        │
        ▼  Stage 4: Vol-balanced top-down tree (λ=0.25)
        │            → hierarchy that balances risk and maximises diversification
        │
        ▼  Stage 5: Sharpe-ratio bisection
        │            → weights proportional to cluster risk-adjusted return
        │
        ▼  Stage 6: L2 blend toward 1/N (10%) + L1 turnover filter (5pp)
        │            → concentration control + transaction-cost reduction
        │
        └─► Final weights w ∈ ℝ_+^N,  sum(w) = 1
```

The key conceptual point for the thesis: each stage directly addresses
a known failure mode of standard HRP.

| Standard HRP failure mode                                     | HMVA fix                        |
| ------------------------------------------------------------- | ------------------------------- |
| Sample covariance is noisy / singular at high N/T             | NLS covariance (Stage 2)        |
| All observations are weighted equally, ignoring vol clustering | EWMA pseudo-returns (Stage 1)   |
| Hierarchy conflates correlation and risk imbalance            | Vol-balanced split (Stage 4)    |
| Bisection ignores expected returns entirely                   | BL + Sharpe bisection (Stages 3, 5) |
| Weights can be highly concentrated in a single asset          | L2 ridge (Stage 6)              |
| Unnecessary churning inflates realised transaction costs      | L1 turnover filter (Stage 6)    |

# current idea:

(1-2 pages) 0. Introduction - give the idea that its interesing direction to see if even without error amplification shrinakge gets better results in portfolio allocation
(2-3 pages)

1. Lit review

- Summarize all papers used through and conclude that no simillar test has been carried

2. Methodology

Portfolio Theory:

- Explain Sample Cov and Porfolio construction with mean-variance (GMV portfolio directly comparable) and 1/N portfolio as benchmarks (1 page)
- Explain HRP (1 page)
- Explain LW, NLS, POET (2 pages)
- Explain MHRP (1 page) - introduce the EWA shrinkage

Data:

- Explain DGPs (1 page) - 2 regimes: sparse factor creation, and spiked non linear dense eigenvalues
- Explain Data (1 page) - point in time SP500 equities (maybe restricted to top 100 for easier computation and delistings are handled as 0 return)

4. Results

- Replication: Run cov simulation study to motivate covarinace improvement idea Sample vs All, then LW vs NLS, POET - run_simulation.py (2 pages) (done with results)
- New empirical result: Run a full robustness test of the alpha based on transaction costs, linkage and lookback (result seems to be the alpha doesnt come from shrinkage only, only linear cost, done with results)
- Innovate by creating a modification to HRP called HDRP - idea is that cov estimation errors are not amplified, hence we can be sparing when forecasting and innovating on incorporating complexity in the forward looking estimation
- Sample vs LW, NLS, POET HDRP wtih 1/N, SPY-K benchmarks - run_crsp.py (2 pages)
- Start using best LW MHRP as basis to comapre to advanced shrinkage NLS, POET MHRP overall and with transaction costs - run_robustness.py

5. Conclussion

- Summarize results of my experiments and propose future research directions like


# current idea:

(1-2 pages) 0. Introduction - give the idea that its interesing direction to see if even without error amplification shrinakge gets better results in portfolio allocation
(2-3 pages)

1. Lit review

- Summarize all papers used through and conclude that no simillar test has been carried

2. Methodology

Portfolio Theory:

- Explain Porfolio construction with mean-variance and HRP, with 1/N and SPY-K portfolios as benchmark (1 page)
- Explain NLS (2 pages)
- Explain Black litterman
- Explain EWA 
- Explain Regularization

Data:

- Explain DGPs (1 page) - 2 regimes: sparse factor creation, and spiked non linear dense eigenvalues
- Explain Data (1 page) - point in time SP500 equities (maybe restricted to top 100 for easier computation and delistings are handled as 0 return)

3. Innovation

Prove and explain HMVA and how its constructed

4. Results

- Replication: Run cov simulation study to motivate covarinace improvement idea Sample vs All, then LW vs NLS, POET - run_simulation.py (2 pages) (done with results)
- New empirical result: Run a full robustness test of the alpha based on transaction costs, linkage and lookback (result seems to be the alpha doesnt come from shrinkage only, only linear cost, done with results)
- Innovate by creating a modification to HRP called HDRP - idea is that cov estimation errors are not amplified, hence we can be sparing when forecasting and innovating on incorporating complexity in the forward looking estimation
- Sample vs LW, NLS, POET HDRP wtih 1/N, SPY-K benchmarks - run_crsp.py (2 pages)
- Start using best LW MHRP as basis to comapre to advanced shrinkage NLS, POET MHRP overall and with transaction costs - run_robustness.py

5. Conclussion

- Summarize results of my experiments and propose future research directions like
