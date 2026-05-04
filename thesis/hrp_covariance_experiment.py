"""
=============================================================================
HRP with Advanced Covariance Estimators - Bachelor's Thesis Experiment
=============================================================================

Compares four Hierarchical Risk Parity (HRP) variants that differ ONLY in
how the covariance matrix is estimated:

    1. HRP-Sample           - raw sample covariance (baseline)
    2. HRP-LinearShrink     - Ledoit-Wolf (2004) linear shrinkage
    3. HRP-NonLinearShrink  - Ledoit-Wolf (2020) analytical NLS
    4. HRP-POET             - Fan, Liao & Mincheva (2013) POET

The HRP algorithm itself is held fixed (single-linkage, distance metric,
recursive bisection); all that changes is the covariance matrix that feeds
into HRP. This is the controlled-experiment principle.

References
----------
- Lopez de Prado, M. (2016) "Building diversified portfolios that outperform
  out of sample", Journal of Portfolio Management 42(4).
- Ledoit, O. and Wolf, M. (2004) "A well-conditioned estimator for
  large-dimensional covariance matrices", J. Multivariate Analysis 88.
- Ledoit, O. and Wolf, M. (2020) "Analytical nonlinear shrinkage of
  large-dimensional covariance matrices", Annals of Statistics 48(5).
- Fan, J., Liao, Y. and Mincheva, M. (2013) "Large covariance estimation
  by thresholding principal orthogonal complements", JRSS-B 75(4).
- Molyboga, M. (2020) "A modified hierarchical risk parity framework for
  portfolio management", Journal of Financial Data Science.

Author: Bachelor's Thesis Code, 2026.
=============================================================================
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.covariance import LedoitWolf


# =============================================================================
# 1. DATA
# =============================================================================

DEFAULT_TICKERS: List[str] = [
    # US equity (broad + style + small)
    "SPY", "QQQ", "IWM", "MDY",
    # International equity
    "EFA", "EEM", "EWJ", "EWG", "EWU",
    # Fixed income
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP",
    # Commodities
    "GLD", "SLV", "DBC", "USO",
    # US sectors (SPDR Select Sector ETFs)
    "XLE", "XLF", "XLV", "XLK", "XLI", "XLP", "XLU", "XLY", "XLB",
    # Real estate / themes
    "VNQ", "SMH",
]


def get_data(tickers: List[str] = DEFAULT_TICKERS,
             start: str = "2015-01-01",
             end: str = "2025-01-01",
             min_obs_ratio: float = 0.95) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance and convert to log-returns.

    Parameters
    ----------
    tickers : list of Yahoo Finance tickers
    start, end : ISO date strings
    min_obs_ratio : drop tickers with fewer than this fraction of observations

    Returns
    -------
    pd.DataFrame of daily log-returns, columns = tickers, index = dates
    """
    print(f"[data] Downloading {len(tickers)} tickers from {start} to {end} ...")
    data = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=True)

    # yfinance returns a multi-index (field, ticker) when many tickers are
    # requested.  We only need adjusted close.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Drop tickers with too many missing observations, then drop dates with
    # any remaining NaN to obtain a balanced panel.
    keep_cols = prices.columns[prices.notna().mean() >= min_obs_ratio]
    prices = prices[keep_cols].dropna()

    if prices.shape[1] < 5:
        raise RuntimeError("Not enough tickers with sufficient history "
                           "after filtering.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"[data] Final panel: {log_returns.shape[0]} days "
          f"x {log_returns.shape[1]} assets")
    return log_returns


# =============================================================================
# 2. COVARIANCE ESTIMATORS
# =============================================================================

# ----- 2a. Sample covariance -------------------------------------------------

def cov_sample(X: np.ndarray) -> np.ndarray:
    """Plain-vanilla sample covariance.

    Σ̂ = 1/(T-1) * (X - x̄)' (X - x̄)
    Inputs X is a (T, N) array of returns.
    """
    return np.cov(X, rowvar=False)


# ----- 2b. Ledoit-Wolf LINEAR shrinkage --------------------------------------

def cov_linear_shrink(X: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf (2004) linear shrinkage to a multiple of the identity.

        Σ̂_LW = δ * F + (1 - δ) * S

    where:
        S = sample covariance,
        F = (trace(S)/N) * I  (target: scaled identity),
        δ = analytically-optimal intensity that minimises Frobenius
            distance to the (unknown) true covariance.

    sklearn implements the closed-form intensity exactly.
    """
    return LedoitWolf().fit(X).covariance_


# ----- 2c. Ledoit-Wolf 2020 NON-LINEAR analytical shrinkage ------------------

def cov_nonlinear_shrink(X: np.ndarray) -> np.ndarray:
    """
    Analytical non-linear shrinkage (Ledoit & Wolf, 2020).

    Sample eigenvalues are over-dispersed: large ones are too large, small
    ones too small.  NLS *individually* shrinks every eigenvalue using a
    formula derived from the Marchenko-Pastur theory, with a kernel-density
    estimate of the limiting eigenvalue distribution.

    Steps (for the standard p <= n case):
        1. Eigendecompose S = U diag(λ) U'.
        2. Bandwidth h = T^(-1/3); evaluate the Epanechnikov density f̃ and
           its Hilbert transform H̃ at every λ_i.
        3. Shrunk eigenvalues:
              d_i = λ_i / [ (π c λ_i f̃(λ_i))² + (1 - c - π c λ_i H̃(λ_i))² ]
           with c = N/T.
        4. Σ̂_NLS = U diag(d) U'.

    This function is a faithful Python port of the authors' MATLAB
    `analytical_shrinkage.m` (https://www.econ.uzh.ch/en/people/faculty/
    wolf/publications.html).  It is O(N²) in memory (an N x N matrix of
    eigenvalue differences) but is fast for N <= a few hundred.
    """
    X = X - X.mean(axis=0)
    T, N = X.shape
    # MLE-style normalisation matches the LW reference implementation.
    S = (X.T @ X) / T
    S = (S + S.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(S)            # ascending
    n_pos = min(N, T)
    lam = eigvals[N - n_pos:]                       # keep positive ones
    u_pos = eigvecs[:, N - n_pos:]

    # --- kernel density f̃ at each eigenvalue ----------------------------
    h = T ** (-1.0 / 3.0)
    L = np.tile(lam.reshape(-1, 1), (1, n_pos))     # (n_pos, n_pos)
    H = h * L.T                                     # local bandwidth
    x = (L - L.T) / H                               # standardised gaps

    # Epanechnikov kernel K(u) = (3/(4√5)) (1 - u²/5)_+ for |u| <= √5
    inside = np.maximum(1.0 - x ** 2 / 5.0, 0.0)
    ftilde = (3.0 / (4.0 * np.sqrt(5))) * np.mean(inside / H, axis=1)

    # --- Hilbert transform H̃ of the Epanechnikov kernel ----------------
    eps = 1e-15
    log_arg = np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x + eps))
    H_kernel = (-3.0 / (10.0 * np.pi)) * x \
        + (3.0 / (4.0 * np.sqrt(5) * np.pi)) * (1.0 - x ** 2 / 5.0) \
        * np.log(log_arg + eps)
    # closed-form limit at the kernel boundary
    bnd = np.abs(np.abs(x) - np.sqrt(5)) < 1e-10
    H_kernel[bnd] = (-3.0 / (10.0 * np.pi)) * x[bnd]
    Hftilde = np.mean(H_kernel / H, axis=1)

    # --- LW2020 shrinkage formula --------------------------------------
    if N <= T:
        c = N / T
        denom = (np.pi * c * lam * ftilde) ** 2 \
            + (1.0 - c - np.pi * c * lam * Hftilde) ** 2
        d = lam / denom
        sigma = u_pos @ np.diag(d) @ u_pos.T
    else:
        # high-dim p > n branch (kept for safety; not exercised here)
        Hftilde0 = (1.0 / np.pi) * (
            3.0 / (10.0 * h ** 2)
            + (3.0 / (4.0 * np.sqrt(5) * h)) * (1.0 - 1.0 / (5.0 * h ** 2))
            * np.log((1.0 + np.sqrt(5) * h) / (1.0 - np.sqrt(5) * h))
        ) * np.mean(1.0 / lam)
        d0 = 1.0 / (np.pi * (N - T) / T * Hftilde0)
        d1 = lam / (np.pi ** 2 * lam ** 2 * (ftilde ** 2 + Hftilde ** 2))
        d_full = np.concatenate([np.full(N - T, d0), d1])
        sigma = eigvecs @ np.diag(d_full) @ eigvecs.T

    return _ensure_pd(sigma)


# ----- 2d. POET (Fan, Liao, Mincheva 2013) -----------------------------------

def cov_poet(X: np.ndarray,
             K: Optional[int] = None,
             K_max: int = 8,
             threshold_C: float = 0.5) -> np.ndarray:
    """
    POET = Principal Orthogonal complEment Thresholding.

    Models  Σ = Λ_K Λ_K' + Σ_u
    where the first part is a low-rank "factor" component and Σ_u is a
    sparse idiosyncratic covariance.

    Algorithm
    ---------
    1. S = sample covariance, eigendecompose S = U diag(λ) U'.
    2. Choose K (#factors) by the eigenvalue-ratio test
       (Ahn & Horenstein 2013) - more robust than Bai & Ng IC for small N.
    3. Common part:        Σ̂_K = U[:, :K] diag(λ[:K]) U[:, :K]'.
    4. Residual:           R = S - Σ̂_K.
    5. Adaptive (correlation-)thresholding:
           τ_ij = C * sqrt(R_ii R_jj) * sqrt(log(N) / T)
           R̃_ij = sign(R_ij) * max(|R_ij| - τ_ij, 0)         (off-diag)
           R̃_ii = R_ii                                       (variances kept)
    6. Σ̂_POET = Σ̂_K + R̃ ; project to nearest PD.

    Parameters
    ----------
    K : int, optional
        Manually specify number of factors (None = auto via ER test).
    K_max : int
        Upper bound on K used by the eigenvalue-ratio search.
    threshold_C : float
        Multiplier in front of √(log N / T).  Smaller -> less thresholding
        (closer to S).  Larger -> sparser residual.  Cross-validate in
        practice; 0.5 is a reasonable default for ETF-scale N.
    """
    T, N = X.shape
    S = np.cov(X, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(S)
    # sort descending for clarity
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # --- Number of factors via Ahn-Horenstein eigenvalue ratio ---------
    if K is None:
        K_max = max(1, min(K_max, N - 1))
        ratios = eigvals[:K_max] / np.maximum(eigvals[1:K_max + 1], 1e-12)
        K = int(np.argmax(ratios) + 1)
        K = max(1, min(K, K_max))

    # --- common (low-rank) part ----------------------------------------
    U_K = eigvecs[:, :K]
    Lam_K = np.diag(eigvals[:K])
    common = U_K @ Lam_K @ U_K.T

    # --- idiosyncratic residual + adaptive thresholding ----------------
    R = S - common
    diag_R = np.maximum(np.diag(R), 1e-12)          # numerical guard
    theta = np.outer(np.sqrt(diag_R), np.sqrt(diag_R))
    tau = threshold_C * theta * np.sqrt(np.log(N) / T)
    R_thresh = np.sign(R) * np.maximum(np.abs(R) - tau, 0.0)
    np.fill_diagonal(R_thresh, np.diag(R))          # do NOT threshold variances

    Sigma = common + R_thresh
    return _ensure_pd(Sigma)


# ----- helpers ---------------------------------------------------------------

def _ensure_pd(M: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Symmetrise and clip eigenvalues to enforce positive definiteness."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, jitter)
    return V @ np.diag(w) @ V.T


# =============================================================================
# 3. HIERARCHICAL RISK PARITY
# =============================================================================

def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Covariance -> correlation (clipped to [-1, 1])."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


def _cluster_var(cov: np.ndarray, items: List[int]) -> float:
    """Variance of the inverse-variance-weighted portfolio of one cluster."""
    sub = cov[np.ix_(items, items)]
    inv_var = 1.0 / np.diag(sub)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray,
                linkage_method: str = "single") -> np.ndarray:
    """
    Lopez de Prado (2016) Hierarchical Risk Parity weights.

    Three steps:
      (a) tree clustering on the correlation distance
              d_ij = sqrt(0.5 * (1 - ρ_ij))
      (b) quasi-diagonalisation: order assets by tree leaf order so that
          similar assets are adjacent in the covariance matrix
      (c) recursive bisection: split the ordered list in half, allocate
          weight inversely proportional to the cluster's IVP variance,
          recurse.

    Parameters
    ----------
    cov : (N, N) covariance matrix
    linkage_method : 'single' | 'average' | 'complete' | 'ward'

    Returns
    -------
    np.ndarray of length N - portfolio weights summing to 1.
    """
    N = cov.shape[0]
    corr = _cov_to_corr(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    link = linkage(squareform(dist, checks=False), method=linkage_method)
    sort_ix = list(leaves_list(link))

    w = np.ones(N)
    clusters: List[List[int]] = [sort_ix]
    while clusters:
        # split each cluster in two halves (drop singletons)
        clusters = [c[i:j]
                    for c in clusters
                    for i, j in [(0, len(c) // 2), (len(c) // 2, len(c))]
                    if len(c) > 1]
        # walk pairs: cluster 2k vs cluster 2k+1
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)            # inverse-variance split
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================

ESTIMATORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "HRP-Sample":          cov_sample,
    "HRP-LinearShrink":    cov_linear_shrink,
    "HRP-NonLinearShrink": cov_nonlinear_shrink,
    "HRP-POET":            cov_poet,
}


def backtest(returns: pd.DataFrame,
             lookback: int = 504,
             rebalance: int = 21,
             estimators: Dict[str, Callable] = ESTIMATORS,
             ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Walk-forward backtest with monthly rebalancing.

    For every rebalance index t starting at `lookback`:
      * estimate covariance on returns[t-lookback : t] (in-sample window)
      * compute HRP weights from each estimator
      * apply those weights from day t to t+rebalance-1 (out-of-sample)

    Returns
    -------
    daily_returns : DataFrame of OOS daily portfolio returns per strategy
    weights_log   : dict mapping strategy -> DataFrame of weights per
                    rebalance date (rows = rebalance dates, cols = assets)
    """
    T, N = returns.shape
    rebal_idx = list(range(lookback, T, rebalance))
    print(f"[backtest] {len(rebal_idx)} rebalances over {T - lookback} OOS days")

    daily_pnl = {name: np.full(T, np.nan) for name in estimators}
    weights_records: Dict[str, List[pd.Series]] = {n: [] for n in estimators}
    arr = returns.values

    for k, t in enumerate(rebal_idx):
        window = arr[t - lookback:t]
        for name, fn in estimators.items():
            try:
                cov = fn(window)
                w = hrp_weights(cov)
            except Exception as e:
                # very rare: fall back to equal weights and log
                print(f"[warn] {name} @ t={t}: {e}; using 1/N fallback")
                w = np.ones(N) / N

            t_end = min(t + rebalance, T)
            daily_pnl[name][t:t_end] = arr[t:t_end] @ w
            weights_records[name].append(
                pd.Series(w, index=returns.columns, name=returns.index[t]))

        if (k + 1) % 12 == 0 or k == len(rebal_idx) - 1:
            print(f"[backtest]   completed rebalance {k + 1}/{len(rebal_idx)}")

    daily_returns = pd.DataFrame(daily_pnl, index=returns.index).dropna(how="all")
    daily_returns = daily_returns.dropna()
    weights_log = {k: pd.DataFrame(v) for k, v in weights_records.items()}
    return daily_returns, weights_log


# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================

def compute_metrics(daily_returns: pd.DataFrame,
                    weights_log: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per-strategy table of headline metrics.

    All annualised at 252 trading days.  Risk-free rate is 0 for simplicity;
    swap in a T-bill series for production work (see README, "Improvements").
    """
    rows: Dict[str, Dict[str, float]] = {}
    for name, r in daily_returns.items():
        ann_ret = (1.0 + r).prod() ** (252.0 / len(r)) - 1.0
        ann_vol = r.std() * np.sqrt(252.0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1.0 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1.0).min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

        # one-way turnover per rebalance, averaged
        W = weights_log[name]
        turnover = W.diff().abs().sum(axis=1).iloc[1:].mean()

        # stability: std of rolling 63-day Sharpe (lower = more stable)
        rolling_sharpe = (r.rolling(63).mean() / r.rolling(63).std()).std()

        rows[name] = dict(
            AnnReturn=ann_ret,
            AnnVol=ann_vol,
            Sharpe=sharpe,
            MaxDD=max_dd,
            Calmar=calmar,
            Turnover=turnover,
            SharpeStab=rolling_sharpe,
        )
    return pd.DataFrame(rows).T


def diebold_mariano(d1: pd.Series, d2: pd.Series,
                    h: int = 1) -> Tuple[float, float]:
    """
    Diebold-Mariano test, paired, with Newey-West variance.

    The "loss" function used here is the negative daily portfolio return,
    i.e. larger losses = worse performance.  H0: equal loss.  Two-sided.

    Note: a more robust alternative for portfolio comparison is the
    Ledoit-Wolf (2008) Sharpe ratio test - see README discussion.
    """
    loss1, loss2 = -d1, -d2
    d = (loss1 - loss2).dropna().values
    n = len(d)
    mean_d = d.mean()
    g0 = np.var(d, ddof=1)
    var_d = g0
    for k in range(1, max(1, h)):
        gk = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        var_d += 2.0 * (1.0 - k / h) * gk
    var_d = max(var_d, 1e-12)
    dm_stat = mean_d / np.sqrt(var_d / n)
    pvalue = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return dm_stat, pvalue


def bootstrap_sharpe_diff(r1: pd.Series, r2: pd.Series,
                          n_boot: int = 2000,
                          block: int = 21,
                          seed: int = 42
                          ) -> Tuple[float, Tuple[float, float]]:
    """
    Stationary block-bootstrap confidence interval for Sharpe(r1) - Sharpe(r2).
    """
    rng = np.random.default_rng(seed)
    a = pd.concat([r1, r2], axis=1).dropna().values
    n = len(a)
    if n < 2 * block:
        return np.nan, (np.nan, np.nan)
    n_blocks = n // block + 1
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = a[idx]
        s1 = sample[:, 0].mean() / sample[:, 0].std() * np.sqrt(252)
        s2 = sample[:, 1].mean() / sample[:, 1].std() * np.sqrt(252)
        diffs[b] = s1 - s2
    return float(diffs.mean()), tuple(np.quantile(diffs, [0.025, 0.975]))


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_all(daily_returns: pd.DataFrame,
             weights_log: Dict[str, pd.DataFrame],
             metrics: pd.DataFrame,
             outdir: str = "figures") -> None:
    """Save the full set of figures referenced in the README."""
    os.makedirs(outdir, exist_ok=True)
    sns.set_style("whitegrid")

    # --- Equity curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + daily_returns).cumprod().plot(ax=ax, lw=1.5)
    ax.set_title("Out-of-sample equity curves")
    ax.set_ylabel("Wealth (start = 1)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{outdir}/equity_curves.png", dpi=120)
    plt.close()

    # --- Drawdowns -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    cum = (1 + daily_returns).cumprod()
    dd = cum / cum.cummax() - 1
    dd.plot(ax=ax)
    ax.set_title("Out-of-sample drawdowns")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(f"{outdir}/drawdowns.png", dpi=120)
    plt.close()

    # --- Sharpe ratios -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics["Sharpe"].plot.bar(ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Annualised Sharpe ratio by strategy")
    ax.set_ylabel("Sharpe")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sharpe_bars.png", dpi=120)
    plt.close()

    # --- Turnover ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics["Turnover"].plot.bar(ax=ax, color="darkorange", edgecolor="black")
    ax.set_title("Average one-way turnover per rebalance")
    ax.set_ylabel("Sum of |Δw|")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/turnover.png", dpi=120)
    plt.close()

    # --- Last-rebalance weights heat-map -------------------------------
    n_strat = len(weights_log)
    fig, axes = plt.subplots(1, n_strat,
                             figsize=(3.2 * n_strat, 7),
                             sharey=True)
    if n_strat == 1:
        axes = [axes]
    for ax, (name, W) in zip(axes, weights_log.items()):
        last = W.iloc[-1].sort_values(ascending=False).to_frame("w")
        sns.heatmap(last, annot=True, fmt=".2%", cmap="Blues",
                    cbar=False, ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
    plt.suptitle("Final-rebalance weights by strategy", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/last_weights.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[plots] Figures written to '{outdir}/'")


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    np.random.seed(42)

    # -- 1. data ---------------------------------------------------------
    returns = get_data()

    # -- 2. backtest -----------------------------------------------------
    daily, wts = backtest(returns, lookback=504, rebalance=21)

    # -- 3. headline metrics --------------------------------------------
    metrics = compute_metrics(daily, wts)
    print("\n=== Performance summary ===")
    print(metrics.round(4).to_string())
    metrics.to_csv("metrics.csv")
    daily.to_csv("daily_returns.csv")

    # -- 4. statistical tests vs sample baseline ------------------------
    print("\n=== Tests vs HRP-Sample (DM h=21, block-bootstrap CI) ===")
    base = daily["HRP-Sample"]
    rows = []
    for name in daily.columns:
        if name == "HRP-Sample":
            continue
        dm, p = diebold_mariano(daily[name], base, h=21)
        d_sharpe, ci = bootstrap_sharpe_diff(daily[name], base)
        rows.append(dict(
            Strategy=name, DM_stat=dm, p_value=p,
            Sharpe_diff=d_sharpe,
            CI_low=ci[0], CI_high=ci[1],
        ))
    test_df = pd.DataFrame(rows)
    print(test_df.round(4).to_string(index=False))
    test_df.to_csv("statistical_tests.csv", index=False)

    # -- 5. figures ------------------------------------------------------
    plot_all(daily, wts, metrics)

    print("\nDone. Outputs: metrics.csv, daily_returns.csv, "
          "statistical_tests.csv, figures/*.png")


if __name__ == "__main__":
    main()
