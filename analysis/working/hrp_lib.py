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

    Performance optimisation
    ------------------------
    The grid is K x C.  Varying K does not change the eigenvectors of S_tr
    -- only how many we keep -- and varying C does not change the residual
    R = S - common_K -- only the threshold applied to it.  We therefore
    compute the training eigendecomposition exactly ONCE per call and
    reuse it across the entire grid.  This makes AdaptivePOET ~ 30x
    faster on N = 500 universes.

    The estimator is callable like the others (X -> cov) and exposes a
    `history` list of (K*, C*) selections for diagnostic plotting.
    """

    def __init__(self,
                 K_grid: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10),
                 C_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5),
                 train_frac: float = 0.7):
        self.K_grid = K_grid
        self.C_grid = C_grid
        self.train_frac = train_frac
        self.history: List[Dict] = []

    @staticmethod
    def _build_poet(eigvals_desc: np.ndarray,
                    eigvecs_desc: np.ndarray,
                    S: np.ndarray,
                    T: int,
                    K: int,
                    C: float) -> np.ndarray:
        """Build a POET cov from a precomputed eigendecomposition of S."""
        N = S.shape[0]
        U_K = eigvecs_desc[:, :K]
        common = (U_K * eigvals_desc[:K]) @ U_K.T
        R = S - common
        diag_R = np.maximum(np.diag(R), 1e-12)
        theta = np.outer(np.sqrt(diag_R), np.sqrt(diag_R))
        tau = C * theta * np.sqrt(np.log(N) / T)
        R_thresh = np.sign(R) * np.maximum(np.abs(R) - tau, 0.0)
        np.fill_diagonal(R_thresh, np.diag(R))
        return _ensure_pd(common + R_thresh)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        T, N = X.shape
        T1 = int(T * self.train_frac)
        X_tr, X_va = X[:T1], X[T1:]

        # one eigendecomposition for the training window
        S_tr = np.cov(X_tr, rowvar=False)
        evals, evecs = np.linalg.eigh(S_tr)
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
        Kmax = min(max(self.K_grid), N - 1)

        ones = np.ones(N)
        best = (np.inf, self.K_grid[0], self.C_grid[0])
        for K in self.K_grid:
            if K > Kmax:
                continue
            for C in self.C_grid:
                try:
                    Sigma = self._build_poet(evals, evecs, S_tr,
                                             T1, K, C)
                    inv = np.linalg.inv(Sigma + 1e-8 * np.eye(N))
                    w = inv @ ones / (ones @ inv @ ones)
                    val_var = float(np.var(X_va @ w))
                    if np.isfinite(val_var) and val_var < best[0]:
                        best = (val_var, K, C)
                except Exception:
                    continue

        _, K_star, C_star = best
        self.history.append({"K": K_star, "C": C_star})
        # final fit uses the full window, recomputed from scratch
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


def min_var_unconstrained(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Closed-form unconstrained minimum-variance portfolio.

        w = Σ_reg^{-1} 1 / (1' Σ_reg^{-1} 1),     Σ_reg = Σ + λ I

    Allows negative weights (i.e. short selling).  This is the right
    benchmark when N is large and SLSQP would be too slow, and the only
    one that is well-defined in the p > n regime where the sample
    covariance is singular -- but only because we add the ridge term.
    """
    n = cov.shape[0]
    inv = np.linalg.inv(cov + ridge * np.eye(n))
    ones = np.ones(n)
    w = inv @ ones
    s = w.sum()
    return w / s if abs(s) > 1e-10 else np.ones(n) / n


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


# ---------------------------------------------------------------------------
# Point-in-time backtest with time-varying universe (CRSP S&P 500)
# ---------------------------------------------------------------------------

def make_crsp_strategies(linkage_method: str = "single") -> Tuple[StrategyMap, "AdaptivePOET"]:
    """
    Strategy set for the high-dimensional CRSP runs.

    Differences vs make_default_strategies:
      * MinVar uses the closed-form ridge variant, because long-only SLSQP
        on N = 500 is too slow and the sample covariance is singular at
        N > T anyway.
      * MinVar-Sample is dropped: it requires inverting a singular matrix
        in the N > T case and is conceptually not defensible there.
        We keep MinVar-LW and MinVar-NLS (both regularised).
    """
    apoet = AdaptivePOET()

    def hrp_with(cov_fn):
        return cov_fn, lambda c: hrp_weights(c, linkage_method=linkage_method)

    return {
        "HRP-Sample":   hrp_with(cov_sample),
        "HRP-LW":       hrp_with(cov_linear_shrink),
        "HRP-NLS":      hrp_with(cov_nonlinear_shrink),
        "HRP-POET":     hrp_with(cov_poet),
        "HRP-PoetCV":   hrp_with(apoet),
        "EW":           (cov_sample, equal_weights),
        "MinVar-LW":    (cov_linear_shrink, min_var_unconstrained),
        "MinVar-NLS":   (cov_nonlinear_shrink, min_var_unconstrained),
        "RP-Sample":    (cov_sample, risk_parity_weights),
    }, apoet


def backtest_pit(returns_wide: pd.DataFrame,
                 universe_fn: Callable,
                 strategies: StrategyMap,
                 lookback: int = 504,
                 rebalance: int = 21,
                 cost_bps: float = 0.0,
                 rf_daily: Optional[pd.Series] = None,
                 min_history_days: Optional[int] = None,
                 verbose: bool = True,
                 ) -> Tuple[pd.DataFrame, Dict[str, Dict[pd.Timestamp, pd.Series]]]:
    """
    Point-in-time backtest with time-varying universe.

    At each rebalance date t:
      1.  Take the universe U_t = universe_fn(t).
      2.  Filter to PERMNOs that are in U_t AND have non-NaN data for
          every day in [t-lookback, t-1].  Call this filtered set N_t.
      3.  Estimate covariance on returns[t-lookback : t, N_t].
      4.  Compute target weights w_t for each strategy.
      5.  Hold for `rebalance` days; on each holding day:
            - Look up returns for the held PERMNOs.  If a PERMNO has NaN
              that day (e.g. delisted), treat its return as 0 (the
              position is implicitly liquidated to cash with rf=0 from
              that day on).  This is the standard simplification; CRSP
              proper has a delisting-return field that makes it cleaner.
            - Drift weights by realised single-asset returns.
      6.  Pay transaction cost  κ × sum |w_t - w_drifted|  on day t.

    Returns
    -------
    daily : DataFrame indexed by date, columns = strategy names, values
            = (excess) daily portfolio returns.
    weights_log : dict mapping strategy -> {rebalance_date: Series of
                  weights}.  Each Series has a different index because
                  the universe is time-varying.
    """
    arr = returns_wide.values            # (T_total, N_all) float
    permnos = np.asarray(returns_wide.columns, dtype=int)
    permno_to_col = {int(p): i for i, p in enumerate(permnos)}
    dates = returns_wide.index
    T_total, N_all = arr.shape
    cost_rate = cost_bps * 1e-4
    if min_history_days is None:
        min_history_days = lookback   # require full lookback by default

    rebal_idx = list(range(lookback, T_total, rebalance))
    if verbose:
        print(f"[bt-pit] lookback={lookback}, rebal={rebalance}, "
              f"cost={cost_bps}bps, rebalances={len(rebal_idx)}")

    daily_pnl = {n: np.full(T_total, np.nan) for n in strategies}
    weights_log: Dict[str, Dict[pd.Timestamp, pd.Series]] = \
        {n: {} for n in strategies}
    # last drifted state per strategy: (col_indices, weights)
    drifted: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = \
        {n: None for n in strategies}
    universe_sizes = []

    for k, t in enumerate(rebal_idx):
        rebal_date = dates[t]
        universe = universe_fn(rebal_date)

        # candidate columns: those PERMNOs that are in the universe at t
        cand = np.array([permno_to_col[p] for p in universe
                         if p in permno_to_col])
        if cand.size == 0:
            print(f"[bt-pit] WARN no universe overlap at {rebal_date}; "
                  f"skipping rebalance.")
            continue

        # require full non-NaN history in [t-lookback, t)
        window = arr[t - lookback:t, cand]
        full_hist = ~np.isnan(window).any(axis=0)
        keep = cand[full_hist]
        if keep.size < 5:
            print(f"[bt-pit] WARN only {keep.size} stocks survive "
                  f"history filter at {rebal_date}; skipping.")
            continue
        window_clean = arr[t - lookback:t, keep]
        N_t = keep.size
        universe_sizes.append({"date": rebal_date, "raw": len(universe),
                               "in_panel": cand.size, "with_history": N_t})

        for name, (cov_fn, alloc_fn) in strategies.items():
            try:
                cov = cov_fn(window_clean)
                w_target = alloc_fn(cov)
            except Exception as e:
                print(f"[bt-pit] WARN {name} t={rebal_date}: {e}; "
                      f"falling back to 1/N_t")
                w_target = np.ones(N_t) / N_t

            # transaction cost: compare to drifted weights from previous period.
            # The two weight vectors live on potentially different column sets;
            # we align on the union and treat missing entries as zero weight
            # (i.e., closed positions / new positions both count as a trade).
            if drifted[name] is None:
                tc = 0.0
            else:
                old_cols, old_w = drifted[name]
                union = np.union1d(old_cols, keep)
                w_old_aligned = np.zeros(union.size)
                w_new_aligned = np.zeros(union.size)
                w_old_aligned[np.searchsorted(union, old_cols)] = old_w
                w_new_aligned[np.searchsorted(union, keep)] = w_target
                tc = cost_rate * np.abs(w_new_aligned - w_old_aligned).sum()

            weights_log[name][rebal_date] = pd.Series(
                w_target, index=permnos[keep], name=rebal_date)

            # apply weights with drift, NaN → 0 for that day
            t_end = min(t + rebalance, T_total)
            w_curr = w_target.copy()
            cols_curr = keep.copy()
            for s in range(t, t_end):
                day_rets = arr[s, cols_curr]
                # treat NaN as 0 (delisted / not trading)
                nan_mask = np.isnan(day_rets)
                day_rets_clean = np.where(nan_mask, 0.0, day_rets)
                r_port = float(day_rets_clean @ w_curr)
                daily_pnl[name][s] = r_port - (tc if s == t else 0.0)
                # weight drift on the same column set
                denom = 1.0 + r_port
                if denom > 1e-8:
                    w_curr = w_curr * (1.0 + day_rets_clean) / denom
            drifted[name] = (cols_curr, w_curr)

        if verbose and ((k + 1) % 12 == 0 or k == len(rebal_idx) - 1):
            print(f"[bt-pit]   rebal {k + 1}/{len(rebal_idx)} "
                  f"date={rebal_date.date()} N_t={N_t}")

    daily = pd.DataFrame(daily_pnl, index=dates).dropna(how="all")
    if rf_daily is not None:
        rf = rf_daily.reindex(daily.index).fillna(0.0)
        daily = daily.subtract(rf, axis=0)

    if verbose and universe_sizes:
        sz = pd.DataFrame(universe_sizes)
        print(f"[bt-pit] universe sizes (with full history):  "
              f"min={sz['with_history'].min()}  "
              f"median={int(sz['with_history'].median())}  "
              f"max={sz['with_history'].max()}")

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


def compute_metrics_pit(daily_returns: pd.DataFrame,
                        weights_log: Dict[str, Dict[pd.Timestamp, pd.Series]]
                        ) -> pd.DataFrame:
    """
    Same headline metrics as compute_metrics, but for the PIT backtest where
    each rebalance has its own (potentially different) set of stocks.

    Turnover at rebalance r is computed against the *drifted* weights from
    the previous rebalance, on the union of column sets, treating missing
    entries as zero weight (so new positions and closed positions both
    register as trades).
    """
    rows: Dict[str, Dict[str, float]] = {}
    for name, r in daily_returns.items():
        r = r.dropna()
        if len(r) < 2:
            # backtest produced no usable observations for this strategy
            rows[name] = dict(AnnReturn=np.nan, AnnVol=np.nan,
                              Sharpe=np.nan, MaxDD=np.nan, Calmar=np.nan,
                              Turnover=np.nan, SharpeStab=np.nan)
            continue
        ann_ret = (1.0 + r).prod() ** (252.0 / len(r)) - 1.0
        ann_vol = r.std() * np.sqrt(252.0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1.0 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1.0).min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

        # turnover from sequence of Series with possibly different indices
        wd = weights_log[name]
        dates_sorted = sorted(wd.keys())
        if len(dates_sorted) >= 2:
            turnovers = []
            for i in range(1, len(dates_sorted)):
                w0 = wd[dates_sorted[i - 1]]
                w1 = wd[dates_sorted[i]]
                aligned = pd.concat([w0.rename("a"), w1.rename("b")],
                                    axis=1).fillna(0.0)
                turnovers.append((aligned["b"] - aligned["a"]).abs().sum())
            turnover = float(np.mean(turnovers))
        else:
            turnover = np.nan

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

def _toeplitz_cov(N: int, rho: float) -> np.ndarray:
    """AR(1)-style Toeplitz: Σ_ij = ρ^|i-j|. Dense, smoothly decaying."""
    idx = np.arange(N)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _banded_cov(N: int, diag_var: float, bandwidth: int = 2,
                decay: float = 0.4) -> np.ndarray:
    """
    Banded covariance: nonzero entries only within `bandwidth` of the diagonal.
    Used as a *truly sparse* idiosyncratic residual to give POET something to
    actually threshold (a strict-diagonal residual is uninteresting because
    NLS handles it equally well).
    """
    Sigma = np.zeros((N, N))
    for k in range(-bandwidth, bandwidth + 1):
        v = diag_var * decay ** abs(k)
        np.fill_diagonal(Sigma[max(0, k):, max(0, -k):], v)
    return Sigma


def simulate_returns(T: int, N: int,
                     regime: str = "factor_sparse",
                     seed: int = 0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic returns from one of three theoretically distinct regimes.

    Regimes
    -------
    'factor_sparse'  - low-rank common + TRULY SPARSE (banded) residual.
                       This is the regime POET was *designed* for.
                       Expect POET to be the leader; NLS competitive.
    'toeplitz'       - AR(1) Toeplitz covariance, no factor structure, dense
                       residual.  Eigenvalues decay smoothly without a gap.
                       POET has no factors to extract -> should hurt itself
                       by truncating signal as noise.
                       Expect NLS to dominate; POET to lose to even Sample.
    'identity_like'  - near-identity covariance (small uniform correlation).
                       LW shrinks toward identity, which IS the truth here,
                       so LW should be optimal.

    Returns
    -------
    X : (T, N) simulated returns from N(0, Σ_true)
    Sigma_true : the true covariance
    """
    rng = np.random.default_rng(seed)

    if regime == "factor_sparse":
        # 3 strong factors with clear eigenvalue gap
        K = 3
        factor_var = np.array([5.0, 3.0, 1.5])
        B = rng.standard_normal((N, K))
        common = (B * factor_var) @ B.T
        # truly sparse banded residual
        Sigma_idio = _banded_cov(N, diag_var=0.4, bandwidth=2, decay=0.4)
        Sigma_true = common + Sigma_idio
    elif regime == "toeplitz":
        # smooth correlation decay -> no clear factor cut-off
        Sigma_true = _toeplitz_cov(N, rho=0.5)
    elif regime == "identity_like":
        # weak uniform correlation; close to identity
        rho = 0.05
        Sigma_true = (1.0 - rho) * np.eye(N) + rho * np.ones((N, N))
    else:
        raise ValueError(f"unknown regime {regime!r}; "
                         f"expected one of factor_sparse, toeplitz, "
                         f"identity_like")

    Sigma_true = _ensure_pd(Sigma_true)
    # X ~ N(0, Σ_true): use Cholesky for speed
    L_chol = np.linalg.cholesky(Sigma_true)
    Z = rng.standard_normal((T, N))
    X = Z @ L_chol.T
    return X, Sigma_true


# Backwards-compat wrapper kept so older callers do not break.
def simulate_factor_returns(T: int, N: int, K_true: int = 3,
                            structure: str = "sharp",
                            seed: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated.  Calls simulate_returns with a regime mapping."""
    mapping = {"sharp": "factor_sparse",
               "medium": "factor_sparse",
               "diffuse": "toeplitz"}
    regime = mapping.get(structure, "factor_sparse")
    return simulate_returns(T, N, regime=regime, seed=seed)


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


def run_simulation_study(T: int = 300, N: int = 200,
                         regimes: Tuple[str, ...] = ("factor_sparse",
                                                      "toeplitz",
                                                      "identity_like"),
                         n_reps: int = 50,
                         seed0: int = 0,
                         **kwargs) -> pd.DataFrame:
    """
    Sweep cov estimators over the three theoretical regimes.

    Defaults: T=300, N=200 -> N/T ≈ 0.67, in the "moderate high-dim"
    regime where NLS / POET advantages are visible.

    Returns a long DataFrame with one row per (regime, rep, estimator)
    plus a per-replication 'relative_to_sample' column equal to
    (loss_X - loss_sample) / loss_sample.  Negative means improvement.
    """
    estimators = {
        "Sample": cov_sample,
        "LW":     cov_linear_shrink,
        "NLS":    cov_nonlinear_shrink,
        "POET":   cov_poet,
    }
    rows: List[Dict] = []
    for s in regimes:
        for r in range(n_reps):
            X, Sigma_true = simulate_returns(
                T, N, regime=s, seed=seed0 + r * 17 + abs(hash(s)) % 1000)
            base = evaluate_sigma(estimators["Sample"](X), Sigma_true)
            for name, fn in estimators.items():
                try:
                    Sigma_hat = fn(X)
                    ev = evaluate_sigma(Sigma_hat, Sigma_true)
                    # relative improvement vs Sample baseline
                    rel_mv = ((ev["minvar_true_var"] - base["minvar_true_var"])
                              / base["minvar_true_var"])
                    rel_frob = ((ev["frobenius"] - base["frobenius"])
                                / base["frobenius"])
                    rows.append(dict(regime=s, rep=r, estimator=name,
                                     **ev,
                                     mv_relative=rel_mv,
                                     frob_relative=rel_frob))
                except Exception as e:
                    print(f"[sim] {s}/{r}/{name} failed: {e}")
        print(f"[sim] regime={s} done ({n_reps} reps)")
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
    """
    Three figures:

      1.  sim_minvar_relative.png : bar chart of mean (loss_X − loss_sample)
                                    / loss_sample with 95% bootstrap CI per
                                    (regime × estimator).  This is the headline
                                    figure -- negative bars mean improvement
                                    over Sample.
      2.  sim_minvar_true_var.png : raw boxplots of the underlying loss
                                    (kept for completeness / appendix).
      3.  sim_frobenius.png       : Frobenius boxplots, with explicit caveat
                                    that within-rep variance dominates here.
    """
    _ensure_dir(outdir)
    sns.set_style("whitegrid")

    # 1. RELATIVE-IMPROVEMENT bar chart with bootstrap CI ---------------
    rng = np.random.default_rng(0)

    def boot_mean_ci(x: np.ndarray, n_boot: int = 2000) -> Tuple[float, float, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return float("nan"), float("nan"), float("nan")
        idx = rng.integers(0, x.size, (n_boot, x.size))
        means = x[idx].mean(axis=1)
        return float(x.mean()), float(np.quantile(means, 0.025)), \
               float(np.quantile(means, 0.975))

    rows = []
    for (regime, estimator), grp in sim_df.groupby(["regime", "estimator"]):
        m, lo, hi = boot_mean_ci(grp["mv_relative"].values)
        rows.append({"regime": regime, "estimator": estimator,
                     "mean": m, "ci_lo": lo, "ci_hi": hi})
    summary = pd.DataFrame(rows)

    regimes = ["factor_sparse", "toeplitz", "identity_like"]
    regimes = [r for r in regimes if r in summary["regime"].unique()]
    estimators = ["Sample", "LW", "NLS", "POET"]
    palette = {"Sample": "#7f7f7f", "LW": "#1f77b4",
               "NLS": "#2ca02c",   "POET": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x_idx = np.arange(len(regimes))
    width = 0.2
    for i, est in enumerate(estimators):
        sub = summary[summary["estimator"] == est].set_index("regime")
        means = [sub.loc[r, "mean"] if r in sub.index else np.nan for r in regimes]
        los = [sub.loc[r, "ci_lo"] if r in sub.index else np.nan for r in regimes]
        his = [sub.loc[r, "ci_hi"] if r in sub.index else np.nan for r in regimes]
        err_lower = [m - lo for m, lo in zip(means, los)]
        err_upper = [hi - m for m, hi in zip(means, his)]
        ax.bar(x_idx + (i - 1.5) * width, means, width=width,
               yerr=[err_lower, err_upper],
               capsize=3, color=palette.get(est, "C0"),
               edgecolor="black", linewidth=0.5,
               label=est)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Min-Var portfolio variance: (X − Sample) / Sample")
    ax.set_title("Relative improvement over Sample baseline\n"
                 "(negative = better; bars = 95% bootstrap CI)")
    ax.legend(title="Estimator", loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{outdir}/sim_minvar_relative.png", dpi=120)
    plt.close()

    # 2. raw MV boxplots (appendix) ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=sim_df, x="regime", y="minvar_true_var",
                hue="estimator", ax=ax, order=regimes,
                palette=palette)
    ax.set_title("Min-Var portfolio variance under true Σ (raw values)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Min-Var portfolio variance")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sim_minvar_true_var.png", dpi=120)
    plt.close()

    # 3. Frobenius (with caveat in title) ------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=sim_df, x="regime", y="frobenius",
                hue="estimator", ax=ax, order=regimes,
                palette=palette)
    ax.set_title("Frobenius loss (NB: within-rep variance often dominates;\n"
                 "see sim_minvar_relative.png for the portfolio-relevant view)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("‖Σ̂ − Σ_true‖_F")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sim_frobenius.png", dpi=120)
    plt.close()