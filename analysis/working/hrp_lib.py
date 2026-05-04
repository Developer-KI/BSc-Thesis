"""
=============================================================================
hrp_lib.py - Building blocks for the HRP x Covariance Estimator experiment
=============================================================================

Contents
--------
1.  Universe definitions and data loaders
2.  Covariance estimators (sample, LW linear, LW non-linear, POET, adaptive POET)
3.  Portfolio allocators (HRP, MinVar long-only, naive risk parity, equal-weight)
4.  Backtest engine with weight drift and proportional transaction costs
5.  Performance metrics (return, vol, Sharpe, drawdown, Calmar, turnover, stability)
6.  Statistical tests (DM, LW2008-style block-bootstrap Sharpe test, Holm and BH)
7.  Factor-model simulation engine
8.  Plot helpers

This module is import-only.  See run_main.py / run_robustness.py /
run_simulation.py for the experiments.

Author: Bachelor's thesis code, 2026.
=============================================================================
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


# =============================================================================
# 1. UNIVERSES AND DATA
# =============================================================================

ETF_UNIVERSE: List[str] = [
    # Broad US equity / style
    "SPY", "QQQ", "IWM", "MDY",
    # International equity
    "EFA", "EEM", "EWJ", "EWG", "EWU",
    # Fixed income
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP",
    # Commodities
    "GLD", "SLV", "DBC", "USO",
    # US sectors (Select Sector SPDRs)
    "XLE", "XLF", "XLV", "XLK", "XLI", "XLP", "XLU", "XLY", "XLB",
    # Real estate / themes
    "VNQ", "SMH",
]


# Roughly 100 large-cap US stocks chosen for stable history 2015-present.
# This list contains survivorship bias by construction (these are firms
# that existed and were liquid for the full period).  Document this in
# the thesis and replace with a properly point-in-time CRSP universe later.
STOCKS_UNIVERSE: List[str] = [
    # Tech / Communication
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM",
    "ADBE", "CSCO", "ACN", "AMD", "INTC", "IBM", "TXN", "QCOM", "INTU",
    "NOW", "AMAT", "ADI", "MU", "LRCX", "KLAC", "PANW",
    "NFLX", "T", "VZ", "TMUS", "CMCSA", "DIS",
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
    "USB", "PNC", "SCHW", "BK", "CME", "ICE", "SPGI",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG", "REGN", "VRTX", "BSX",
    # Consumer staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "TGT",
    "KMB", "GIS", "SYY",
    # Consumer discretionary
    "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "TSLA", "F", "GM",
    # Industrials
    "GE", "CAT", "RTX", "HON", "UPS", "BA", "DE", "LMT", "UNP", "ETN",
    "FDX", "MMM", "CSX", "NSC", "EMR",
    # Energy & Utilities
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "OXY", "MPC", "VLO",
    "NEE", "SO", "DUK", "AEP", "EXC",
    # Materials & Real estate
    "LIN", "SHW", "FCX", "APD", "ECL", "AMT", "PLD", "EQIX", "CCI",
]


def get_returns(tickers: List[str],
                start: str = "2015-01-01",
                end: str = "2025-01-01",
                min_obs_ratio: float = 0.95) -> pd.DataFrame:
    """
    Download adjusted-close prices via yfinance, filter, return log-returns.

    Drops tickers with fewer than `min_obs_ratio` of observations, then drops
    rows containing any remaining NaN, producing a balanced panel.
    """
    print(f"[data] Downloading {len(tickers)} tickers from {start} to {end} ...")
    data = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=True)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    keep = prices.columns[prices.notna().mean() >= min_obs_ratio]
    prices = prices[keep].dropna()
    if prices.shape[1] < 5:
        raise RuntimeError("Too few tickers with sufficient history.")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"[data] Final panel: {log_returns.shape[0]} days x "
          f"{log_returns.shape[1]} assets   "
          f"(N/T at 504 lookback = {log_returns.shape[1]/504:.2f})")
    return log_returns


def get_riskfree(returns_index: pd.DatetimeIndex) -> pd.Series:
    """
    Daily risk-free rate from the 13-week T-bill yield (^IRX).

    ^IRX is quoted in percent (e.g. 5.0 means 5 %).  We convert to a daily
    decimal rate using the simple 252-day approximation, forward-filling
    weekends/holidays.  Returns 0.0 if download fails.
    """
    try:
        rf = yf.download("^IRX", start=returns_index[0] - pd.Timedelta(days=10),
                         end=returns_index[-1] + pd.Timedelta(days=2),
                         progress=False, auto_adjust=True)
        rf = rf["Close"] if isinstance(rf.columns, pd.MultiIndex) else rf
        rf = rf.squeeze()
        rf_daily = (rf / 100.0) / 252.0
        rf_daily = rf_daily.reindex(returns_index, method="ffill").fillna(0.0)
        return rf_daily
    except Exception as e:
        print(f"[data] WARN: ^IRX download failed ({e}); using rf=0.")
        return pd.Series(0.0, index=returns_index)


# =============================================================================
# 2. COVARIANCE ESTIMATORS
# =============================================================================

def cov_sample(X: np.ndarray) -> np.ndarray:
    """Plain-vanilla sample covariance Σ̂ = (T-1)^{-1} (X-μ)' (X-μ)."""
    return np.cov(X, rowvar=False)


def cov_linear_shrink(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) linear shrinkage to scaled identity."""
    return LedoitWolf().fit(X).covariance_


def cov_nonlinear_shrink(X: np.ndarray) -> np.ndarray:
    """
    Analytical non-linear shrinkage of Ledoit & Wolf (2020).

    Each sample eigenvalue λ_i is mapped to a non-linear function of the
    full empirical eigenvalue distribution (kernel-density based), which
    corrects the over-dispersion that plagues the sample covariance.

    Faithful Python port of the authors' MATLAB analytical_shrinkage.m.
    O(N^2) memory; fine up to N ~ a few hundred.
    """
    X = X - X.mean(axis=0)
    T, N = X.shape
    S = (X.T @ X) / T
    S = (S + S.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(S)            # ascending
    n_pos = min(N, T)
    lam = eigvals[N - n_pos:]
    u_pos = eigvecs[:, N - n_pos:]

    h = T ** (-1.0 / 3.0)
    L = np.tile(lam.reshape(-1, 1), (1, n_pos))
    H = h * L.T
    x = (L - L.T) / H

    inside = np.maximum(1.0 - x ** 2 / 5.0, 0.0)
    ftilde = (3.0 / (4.0 * np.sqrt(5))) * np.mean(inside / H, axis=1)

    eps = 1e-15
    log_arg = np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x + eps))
    H_kernel = (-3.0 / (10.0 * np.pi)) * x \
        + (3.0 / (4.0 * np.sqrt(5) * np.pi)) * (1.0 - x ** 2 / 5.0) \
        * np.log(log_arg + eps)
    bnd = np.abs(np.abs(x) - np.sqrt(5)) < 1e-10
    H_kernel[bnd] = (-3.0 / (10.0 * np.pi)) * x[bnd]
    Hftilde = np.mean(H_kernel / H, axis=1)

    if N <= T:
        c = N / T
        d = lam / ((np.pi * c * lam * ftilde) ** 2 +
                   (1.0 - c - np.pi * c * lam * Hftilde) ** 2)
        sigma = u_pos @ np.diag(d) @ u_pos.T
    else:
        Hf0 = (1.0 / np.pi) * (
            3.0 / (10.0 * h ** 2) +
            (3.0 / (4.0 * np.sqrt(5) * h)) * (1.0 - 1.0 / (5.0 * h ** 2))
            * np.log((1.0 + np.sqrt(5) * h) / (1.0 - np.sqrt(5) * h))
        ) * np.mean(1.0 / lam)
        d0 = 1.0 / (np.pi * (N - T) / T * Hf0)
        d1 = lam / (np.pi ** 2 * lam ** 2 * (ftilde ** 2 + Hftilde ** 2))
        d_full = np.concatenate([np.full(N - T, d0), d1])
        sigma = eigvecs @ np.diag(d_full) @ eigvecs.T

    return _ensure_pd(sigma)


def cov_poet(X: np.ndarray,
             K: Optional[int] = None,
             K_max: int = 8,
             threshold_C: float = 0.5) -> np.ndarray:
    """
    POET = Principal Orthogonal complEment Thresholding (Fan-Liao-Mincheva 2013).

    1. Eigendecompose S, sort eigenvalues descending.
    2. Pick K via the eigenvalue-ratio test (Ahn-Horenstein 2013) unless given.
    3. Common part = first K eigen-pairs.
    4. Adaptive correlation thresholding of the residual:
            τ_ij = C * sqrt(R_ii R_jj) * sqrt(log N / T)
       Soft threshold off-diagonal entries; keep variances unchanged.
    5. Project sum to nearest PD.
    """
    T, N = X.shape
    S = np.cov(X, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if K is None:
        Kmax = max(1, min(K_max, N - 1))
        ratios = eigvals[:Kmax] / np.maximum(eigvals[1:Kmax + 1], 1e-12)
        K = int(np.argmax(ratios) + 1)
        K = max(1, min(K, Kmax))

    U_K = eigvecs[:, :K]
    Lam_K = np.diag(eigvals[:K])
    common = U_K @ Lam_K @ U_K.T
    R = S - common

    diag_R = np.maximum(np.diag(R), 1e-12)
    theta = np.outer(np.sqrt(diag_R), np.sqrt(diag_R))
    tau = threshold_C * theta * np.sqrt(np.log(N) / T)
    R_thresh = np.sign(R) * np.maximum(np.abs(R) - tau, 0.0)
    np.fill_diagonal(R_thresh, np.diag(R))

    return _ensure_pd(common + R_thresh)


# ---- adaptive POET --------------------------------------------------------

class AdaptivePOET:
    """
    Cross-validated POET that picks (K, C) jointly to minimise the realised
    variance of the implied minimum-variance portfolio on a held-out slice
    of the in-sample window.

    Why minimum-variance loss?  Frobenius-to-validation-sample-cov is too
    noisy when the validation slice is short.  The implied MVP variance is
    a more discriminating loss because it weights the parts of Σ that
    actually matter for inversion (Engle, Ledoit & Wolf 2019).

    The estimator is callable like the others (X -> cov) and exposes a
    `history` list of (K*, C*) selections for diagnostic plotting.
    """

    def __init__(self,
                 K_grid: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
                 C_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5),
                 train_frac: float = 0.7):
        self.K_grid = K_grid
        self.C_grid = C_grid
        self.train_frac = train_frac
        self.history: List[Dict] = []

    def __call__(self, X: np.ndarray) -> np.ndarray:
        T, N = X.shape
        T1 = int(T * self.train_frac)
        X_tr, X_va = X[:T1], X[T1:]

        best = (np.inf, self.K_grid[0], self.C_grid[0])
        for K in self.K_grid:
            for C in self.C_grid:
                try:
                    Sigma = cov_poet(X_tr, K=K, threshold_C=C)
                    inv = np.linalg.inv(Sigma + 1e-8 * np.eye(N))
                    ones = np.ones(N)
                    w = inv @ ones / (ones @ inv @ ones)
                    val_var = float(np.var(X_va @ w))
                    if np.isfinite(val_var) and val_var < best[0]:
                        best = (val_var, K, C)
                except Exception:
                    continue

        _, K_star, C_star = best
        self.history.append({"K": K_star, "C": C_star})
        return cov_poet(X, K=K_star, threshold_C=C_star)


def _ensure_pd(M: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Symmetrise then clip eigenvalues to enforce PD."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, jitter)
    return V @ np.diag(w) @ V.T


# =============================================================================
# 3. PORTFOLIO ALLOCATORS
# =============================================================================

def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    return np.clip(cov / np.outer(std, std), -1.0, 1.0)


def _cluster_var(cov: np.ndarray, items: List[int]) -> float:
    sub = cov[np.ix_(items, items)]
    inv_var = 1.0 / np.diag(sub)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray,
                linkage_method: str = "single") -> np.ndarray:
    """Lopez de Prado (2016) Hierarchical Risk Parity weights."""
    N = cov.shape[0]
    corr = _cov_to_corr(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method=linkage_method)
    sort_ix = list(leaves_list(link))

    w = np.ones(N)
    clusters: List[List[int]] = [sort_ix]
    while clusters:
        clusters = [c[i:j]
                    for c in clusters
                    for i, j in [(0, len(c) // 2), (len(c) // 2, len(c))]
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def equal_weights(cov: np.ndarray) -> np.ndarray:
    """1/N benchmark - DeMiguel, Garlappi & Uppal (2009)."""
    n = cov.shape[0]
    return np.ones(n) / n


def min_var_weights(cov: np.ndarray) -> np.ndarray:
    """
    Long-only minimum-variance portfolio via SLSQP.

        min w' Σ w     s.t.   sum(w)=1,   0 <= w_i <= 1
    """
    n = cov.shape[0]
    cov_pd = _ensure_pd(cov)
    obj = lambda w: w @ cov_pd @ w
    grad = lambda w: 2.0 * cov_pd @ w
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(n)}]
    bnds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    res = minimize(obj, x0, jac=grad, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-10})
    if not res.success:
        return x0
    w = np.maximum(res.x, 0.0)
    return w / w.sum() if w.sum() > 0 else x0


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Naive risk parity:  w_i ∝ 1 / σ_i ."""
    sigma = np.sqrt(np.diag(cov))
    sigma = np.maximum(sigma, 1e-12)
    w = 1.0 / sigma
    return w / w.sum()


# =============================================================================
# 4. BACKTEST ENGINE WITH WEIGHT DRIFT AND TRANSACTION COSTS
# =============================================================================

# A "strategy" is a (cov_estimator, allocator) pair.
StrategyMap = Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                              Callable[[np.ndarray], np.ndarray]]]


def make_default_strategies(linkage_method: str = "single") -> StrategyMap:
    """
    The 9 default strategies for the main experiment.

    Five HRP variants spanning the covariance estimators we want to compare,
    plus four out-of-family benchmarks.  AdaptivePOET is instantiated fresh
    so its `history` list does not leak across calls.
    """
    apoet = AdaptivePOET()

    def hrp_with(cov_fn):
        return cov_fn, lambda c: hrp_weights(c, linkage_method=linkage_method)

    return {
        "HRP-Sample":      hrp_with(cov_sample),
        "HRP-LW":          hrp_with(cov_linear_shrink),
        "HRP-NLS":         hrp_with(cov_nonlinear_shrink),
        "HRP-POET":        hrp_with(cov_poet),
        "HRP-PoetCV":      hrp_with(apoet),       # adaptive POET
        "EW":              (cov_sample, equal_weights),
        "MinVar-Sample":   (cov_sample, min_var_weights),
        "MinVar-NLS":      (cov_nonlinear_shrink, min_var_weights),
        "RP-Sample":       (cov_sample, risk_parity_weights),
    }, apoet


def backtest(returns: pd.DataFrame,
             strategies: StrategyMap,
             lookback: int = 504,
             rebalance: int = 21,
             cost_bps: float = 0.0,
             rf_daily: Optional[pd.Series] = None,
             ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Walk-forward backtest with drifting weights and transaction costs.

    Mechanics
    ---------
    At each rebalance date t:
      * estimate covariance from returns[t-lookback : t]
      * compute target weights w_t for each strategy
      * pay transaction cost  c_t = cost_bps * 1e-4 * sum(|w_t - w_drifted|)
        on day t  (deducted from that day's portfolio return)
    Between rebalances the position drifts:
      w[s+1] = w[s] * (1 + r_asset[s+1]) / (1 + r_port[s+1]).
    """
    arr = returns.values
    T, N = arr.shape
    asset_names = returns.columns
    cost_rate = cost_bps * 1e-4

    rebal_idx = list(range(lookback, T, rebalance))
    print(f"[bt] lookback={lookback}, rebal={rebalance}, "
          f"cost={cost_bps}bps, rebalances={len(rebal_idx)}")

    daily_pnl = {n: np.full(T, np.nan) for n in strategies}
    weights_log = {n: [] for n in strategies}
    drifted_w: Dict[str, Optional[np.ndarray]] = {n: None for n in strategies}

    for k, t in enumerate(rebal_idx):
        window = arr[t - lookback:t]
        for name, (cov_fn, alloc_fn) in strategies.items():
            try:
                cov = cov_fn(window)
                w_target = alloc_fn(cov)
            except Exception as e:
                print(f"[bt] WARN {name} t={t}: {e}; falling back to 1/N")
                w_target = np.ones(N) / N

            # Transaction cost on rebalance day
            tc = (cost_rate * np.abs(w_target - drifted_w[name]).sum()
                  if drifted_w[name] is not None else 0.0)

            weights_log[name].append(
                pd.Series(w_target, index=asset_names, name=returns.index[t]))

            # Apply weights with drift
            t_end = min(t + rebalance, T)
            w_curr = w_target.copy()
            for s in range(t, t_end):
                r_port = float(arr[s] @ w_curr)
                daily_pnl[name][s] = r_port - (tc if s == t else 0.0)
                # weight drift:  w_{s+1} = w_s * (1 + r_asset_s) / (1 + r_port_s)
                denom = 1.0 + r_port
                if denom > 1e-8:
                    w_curr = w_curr * (1.0 + arr[s]) / denom
            drifted_w[name] = w_curr

        if (k + 1) % 12 == 0 or k == len(rebal_idx) - 1:
            print(f"[bt]   completed rebalance {k + 1}/{len(rebal_idx)}")

    daily = pd.DataFrame(daily_pnl, index=returns.index).dropna()
    weights_log = {k: pd.DataFrame(v) for k, v in weights_log.items()}

    # subtract risk-free if provided  -> excess returns
    if rf_daily is not None:
        rf = rf_daily.reindex(daily.index).fillna(0.0)
        daily = daily.subtract(rf, axis=0)

    return daily, weights_log


# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================

def compute_metrics(daily_returns: pd.DataFrame,
                    weights_log: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Headline metrics table.  Returns assumed to be excess if rf was supplied."""
    rows: Dict[str, Dict[str, float]] = {}
    for name, r in daily_returns.items():
        ann_ret = (1.0 + r).prod() ** (252.0 / len(r)) - 1.0
        ann_vol = r.std() * np.sqrt(252.0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1.0 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1.0).min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
        W = weights_log[name]
        turnover = W.diff().abs().sum(axis=1).iloc[1:].mean()
        sharpe_stab = (r.rolling(63).mean() / r.rolling(63).std()).std()
        rows[name] = dict(AnnReturn=ann_ret, AnnVol=ann_vol,
                          Sharpe=sharpe, MaxDD=max_dd, Calmar=calmar,
                          Turnover=turnover, SharpeStab=sharpe_stab)
    return pd.DataFrame(rows).T


# =============================================================================
# 6. STATISTICAL TESTS
# =============================================================================

def diebold_mariano(d1: pd.Series, d2: pd.Series,
                    h: int = 1) -> Tuple[float, float]:
    """Paired DM test on negative-return loss with Newey-West variance."""
    loss1, loss2 = -d1, -d2
    d = (loss1 - loss2).dropna().values
    n = len(d)
    g0 = np.var(d, ddof=1)
    var_d = g0
    for k in range(1, max(1, h)):
        gk = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        var_d += 2.0 * (1.0 - k / h) * gk
    var_d = max(var_d, 1e-12)
    dm_stat = d.mean() / np.sqrt(var_d / n)
    pval = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return dm_stat, pval


def lw_sharpe_test(r1: pd.Series, r2: pd.Series,
                   n_boot: int = 2000, block: int = 21,
                   seed: int = 42) -> Tuple[float, float]:
    """
    Block-bootstrap test in the spirit of Ledoit & Wolf (2008) for
    H0: SR(r1) = SR(r2).

    We use a circular block bootstrap and base the p-value on the centred
    bootstrap distribution of the Sharpe-ratio difference.  This is the
    practical version recommended by LW2008 when the analytical HAC SE is
    awkward; it preserves serial dependence in the joint return process.
    """
    rng = np.random.default_rng(seed)
    R = pd.concat([r1, r2], axis=1).dropna().values
    n = len(R)

    def sr_diff(x):
        s1 = x[:, 0].mean() / x[:, 0].std(ddof=1) * np.sqrt(252) if x[:, 0].std(ddof=1) > 0 else 0.0
        s2 = x[:, 1].mean() / x[:, 1].std(ddof=1) * np.sqrt(252) if x[:, 1].std(ddof=1) > 0 else 0.0
        return s1 - s2

    obs = sr_diff(R)
    n_blocks = n // block + 1
    boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boot[b] = sr_diff(R[idx])
    centred = boot - boot.mean()
    p = float(np.mean(np.abs(centred) >= np.abs(obs)))
    return obs, p


def adjust_pvalues(pvals: np.ndarray, method: str = "holm") -> np.ndarray:
    """
    Holm-Bonferroni (step-down) or Benjamini-Hochberg FDR adjustment.

    Use Holm if you want strong family-wise error rate control;
    use BH if you want FDR control (less conservative, more power).
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if method == "holm":
        order = np.argsort(pvals)
        adjusted_sorted = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            adjusted_sorted[rank] = max(running, min(pvals[i] * (m - rank), 1.0))
            running = adjusted_sorted[rank]
        out = np.empty(m)
        out[order] = adjusted_sorted
        return out
    elif method == "bh":
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        scaled = sorted_p * m / (np.arange(m) + 1)
        # enforce monotonicity from the right
        for k in range(m - 2, -1, -1):
            scaled[k] = min(scaled[k], scaled[k + 1])
        out = np.empty(m)
        out[order] = np.minimum(scaled, 1.0)
        return out
    else:
        raise ValueError(f"unknown method {method!r}")


# =============================================================================
# 7. SIMULATION
# =============================================================================

def simulate_factor_returns(T: int, N: int, K_true: int,
                            structure: str = "sharp",
                            seed: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic returns from a strict factor model.

        r_t = B f_t + ε_t,    Σ = B Λ_F B' + σ_ε² I

    Parameters
    ----------
    structure : 'sharp' | 'medium' | 'diffuse'
        Controls how separated factor variance is from idiosyncratic.

        sharp   : few, high-variance factors with a clear eigenvalue gap
                  (POET should dominate)
        diffuse : many small factors blending into noise
                  (NLS should dominate)
        medium  : intermediate

    Returns
    -------
    X : (T, N) simulated returns
    Sigma_true : (N, N) true covariance
    """
    rng = np.random.default_rng(seed)
    if structure == "sharp":
        # eigenvalue gap is huge -- ideal for POET
        factor_var = np.array([5.0, 3.0, 1.5, 0.7, 0.3])[:K_true]
        loading_scale = 1.0
        idio_var = 0.5
    elif structure == "medium":
        factor_var = np.linspace(2.5, 0.5, K_true)
        loading_scale = 1.0
        idio_var = 1.0
    elif structure == "diffuse":
        # gradual eigen-tail; no clear factor cut-off
        factor_var = np.linspace(1.0, 0.4, K_true)
        loading_scale = 0.6
        idio_var = 1.5
    else:
        raise ValueError(f"unknown structure {structure!r}")

    B = rng.standard_normal((N, K_true)) * loading_scale
    F = rng.standard_normal((T, K_true)) * np.sqrt(factor_var)
    eps = rng.standard_normal((T, N)) * np.sqrt(idio_var)
    X = F @ B.T + eps
    Sigma_true = (B * factor_var) @ B.T + idio_var * np.eye(N)
    return X, Sigma_true


def evaluate_sigma(Sigma_hat: np.ndarray,
                   Sigma_true: np.ndarray) -> Dict[str, float]:
    """Frobenius loss + minimum-variance portfolio variance under true Σ."""
    frob = float(np.linalg.norm(Sigma_hat - Sigma_true, ord="fro"))
    N = Sigma_hat.shape[0]
    try:
        inv = np.linalg.inv(Sigma_hat + 1e-10 * np.eye(N))
        ones = np.ones(N)
        w = inv @ ones / (ones @ inv @ ones)
        mv = float(w @ Sigma_true @ w)
    except Exception:
        mv = np.nan
    return {"frobenius": frob, "minvar_true_var": mv}


def run_simulation_study(T: int = 504, N: int = 100, K_true: int = 3,
                         structures: Tuple[str, ...] = ("sharp", "medium", "diffuse"),
                         n_reps: int = 30,
                         seed0: int = 0) -> pd.DataFrame:
    """
    Sweep cov estimators over scenarios and replications.  Returns a long
    DataFrame with one row per (structure, rep, estimator) combination.
    """
    estimators = {
        "Sample": cov_sample,
        "LW":     cov_linear_shrink,
        "NLS":    cov_nonlinear_shrink,
        "POET":   cov_poet,
    }
    rows: List[Dict] = []
    for s in structures:
        for r in range(n_reps):
            X, Sigma_true = simulate_factor_returns(
                T, N, K_true, structure=s, seed=seed0 + r * 17 + hash(s) % 1000)
            for name, fn in estimators.items():
                try:
                    Sigma_hat = fn(X)
                    ev = evaluate_sigma(Sigma_hat, Sigma_true)
                    rows.append(dict(structure=s, rep=r, estimator=name, **ev))
                except Exception as e:
                    print(f"[sim] {s}/{r}/{name} failed: {e}")
        print(f"[sim] structure={s} done ({n_reps} reps)")
    return pd.DataFrame(rows)


# =============================================================================
# 8. PLOTTING
# =============================================================================

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_equity_and_drawdown(daily: pd.DataFrame, outdir: str,
                             title_suffix: str = "") -> None:
    _ensure_dir(outdir)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + daily).cumprod().plot(ax=ax, lw=1.4)
    ax.set_title(f"Out-of-sample equity curves {title_suffix}".strip())
    ax.set_ylabel("Wealth (start = 1)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/equity_curves.png", dpi=120)
    plt.close()

    cum = (1 + daily).cumprod()
    dd = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 4))
    dd.plot(ax=ax)
    ax.set_title(f"Drawdowns {title_suffix}".strip())
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/drawdowns.png", dpi=120)
    plt.close()


def plot_metric_bars(metrics: pd.DataFrame, outdir: str,
                     title_suffix: str = "") -> None:
    _ensure_dir(outdir)

    for col, colour in [("Sharpe", "steelblue"),
                        ("Turnover", "darkorange"),
                        ("MaxDD", "indianred")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics[col].plot.bar(ax=ax, color=colour, edgecolor="black")
        ax.set_title(f"{col} by strategy {title_suffix}".strip())
        ax.set_ylabel(col)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{col.lower()}_bars.png", dpi=120)
        plt.close()


def plot_weights_heatmap(weights_log: Dict[str, pd.DataFrame], outdir: str,
                         max_strats: int = 6) -> None:
    _ensure_dir(outdir)
    items = list(weights_log.items())[:max_strats]
    fig, axes = plt.subplots(1, len(items),
                             figsize=(3.0 * len(items), 8), sharey=False)
    if len(items) == 1:
        axes = [axes]
    for ax, (name, W) in zip(axes, items):
        last = W.iloc[-1].sort_values(ascending=False).head(20).to_frame("w")
        sns.heatmap(last, annot=True, fmt=".2%", cmap="Blues",
                    cbar=False, ax=ax)
        ax.set_title(name, fontsize=10)
    plt.suptitle("Top-20 weights at final rebalance", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/last_weights.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_simulation_results(sim_df: pd.DataFrame, outdir: str) -> None:
    _ensure_dir(outdir)
    sns.set_style("whitegrid")
    for metric, ylabel in [("frobenius", "Frobenius loss"),
                           ("minvar_true_var", "Min-Var portfolio variance")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=sim_df, x="structure", y=metric, hue="estimator", ax=ax,
                    order=["sharp", "medium", "diffuse"])
        ax.set_title(f"Estimator comparison: {ylabel}")
        ax.set_xlabel("Factor structure")
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{outdir}/sim_{metric}.png", dpi=120)
        plt.close()
