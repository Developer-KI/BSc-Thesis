"""
=============================================================================
hrp_lib.py - Building blocks for the HRP x Covariance Estimator experiment
=============================================================================

Contents
--------
1.  Universe definitions and data loaders
2.  Covariance estimators (sample, LW linear, LW non-linear, POET, POET-CV)
3.  Portfolio allocators (HRP, MinVar long-only, naive risk parity, equal-weight)
4.  Backtest engine with weight drift and proportional transaction costs
5.  Performance metrics (return, vol, Sharpe, drawdown, Calmar, turnover, stability)
6.  Statistical tests (DM, LW2008-style block-bootstrap Sharpe test, Holm and BH)
7.  Factor-model simulation engine (including power-law decay & weak factor regimes)
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
from sklearn.model_selection import KFold

from sklearn.covariance import ledoit_wolf
import nonlinshrink as nls


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
    """
    Ledoit-Wolf linear shrinkage covariance estimator (2004) wrapper from sklearn
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    # ledoit_wolf automatically centers the data
    shrunk_cov, _ = ledoit_wolf(X, assume_centered=False)
    return shrunk_cov


def cov_nonlinear_shrink(X: np.ndarray) -> np.ndarray:
    """Wrapper for the nonlinshrink package"""

    # The package automatically demeans the data by default
    # 'k' is an optional parameter to specify effective degrees of freedom already subtracted
    return nls.shrink_cov(X, k=0)


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


def _poet_from_eig(eigvals_desc: np.ndarray,
                   eigvecs_desc: np.ndarray,
                   S: np.ndarray,
                   T: int,
                   K: int,
                   C: float) -> np.ndarray:
    """Build a POET estimate from a precomputed eigendecomposition of S."""
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


def cov_poet_cv(X: np.ndarray,
                K_max: int = 8,
                C_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2),
                train_frac: float = 0.7) -> np.ndarray:
    """
    POET with K selected analytically (eigenvalue-ratio test) and
    threshold constant C selected by time-series cross-validation.

    The training slice [0:T1] is used to fix K via the Ahn-Horenstein
    ratio test and to build candidate POET matrices across C_grid.  The
    validation slice [T1:T] picks C* by minimising the out-of-sample
    variance of the implied minimum-variance portfolio.  The final
    estimate uses the full window with K re-derived analytically.
    """
    T, N = X.shape
    T1 = int(T * train_frac)
    X_tr, X_va = X[:T1], X[T1:]

    S_tr = np.cov(X_tr, rowvar=False)
    evals, evecs = np.linalg.eigh(S_tr)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    Kmax = max(1, min(K_max, N - 1))
    ratios = evals[:Kmax] / np.maximum(evals[1:Kmax + 1], 1e-12)
    K = int(np.argmax(ratios) + 1)
    K = max(1, min(K, Kmax))

    ones = np.ones(N)
    best_C, best_loss = C_grid[0], np.inf
    for C in C_grid:
        try:
            Sigma = _poet_from_eig(evals, evecs, S_tr, T1, K, C)
            inv = np.linalg.inv(Sigma + 1e-8 * np.eye(N))
            w = inv @ ones / (ones @ inv @ ones)
            val_var = float(np.var(X_va @ w))
            if np.isfinite(val_var) and val_var < best_loss:
                best_loss, best_C = val_var, C
        except Exception:
            continue

    return cov_poet(X, K=None, K_max=K_max, threshold_C=best_C)

def _hard_threshold(matrix, tau):
    """Keep the tau largest absolute entries, set others to zero."""
    if tau <= 0 or tau >= matrix.size:
        return matrix.copy() if tau >= matrix.size else np.zeros_like(matrix)
    flat_abs = np.abs(matrix.ravel())
    # Find the tau-th largest absolute value (quickselect via partition)
    thresh = np.partition(flat_abs, -tau)[-tau]
    return np.where(np.abs(matrix) >= thresh, matrix, 0.0)


def _poetry_core(X, r, tau, T=10, eta=0.5, reg=1e-8):
    """
    Core POETRY estimator.

    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Training data.
    r : int
        Rank of low‑rank component.
    tau : int
        Number of largest entries to keep in sparse component.
    T : int
        Number of refinement iterations.
    eta : float
        Step size for scaled gradient descent (0.25 to 0.5 advised).
    reg : float
        Regularization for (B^T B) to avoid ill‑conditioning.

    Returns
    -------
    Sigma : ndarray of shape (d, d)
        Estimated covariance matrix.
    L : ndarray
        Low‑rank component.
    Psi : ndarray
        Sparse component.
    """
    n, d = X.shape
    S = (X.T @ X) / n

    # ---------- Initialisation: POET ----------
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    B = U[:, :r] @ np.diag(np.sqrt(s[:r]))
    O = S - B @ B.T
    Psi = _hard_threshold(O, tau)

    # ---------- Iterative refinement ----------
    for _ in range(T):
        # 1. Residual
        Res = B @ B.T + Psi - S

        # 2. Gradient w.r.t B: 2 * Res * B
        grad_B = 2.0 * Res @ B

        # 3. Scaled gradient descent with regularised inverse
        BtB = B.T @ B
        inv_BtB = np.linalg.inv(BtB + reg * np.eye(r))
        B = B - eta * grad_B @ inv_BtB

        # 4. Update sparse component
        new_residual = S - B @ B.T
        Psi = _hard_threshold(new_residual, tau)

    L = B @ B.T
    Sigma = L + Psi
    return Sigma, L, Psi

def estimate_factor_number(S, r_max=None):
    d = S.shape[0]
    if r_max is None:
        r_max = min(10, d // 2)
    eigvals = np.linalg.eigvalsh(S)[::-1]
    eigvals = np.maximum(eigvals, 1e-12)
    ratios = [eigvals[i-1] / eigvals[i] for i in range(1, min(r_max, len(eigvals)-1))]
    return np.argmax(ratios) + 1

def poetry_auto_covariance_estimation(X, tau_list=None, n_folds=4, r_max=None,
                                      T=200, eta=0.25, reg=1e-6, heuristic_tau_factor=0.02):
    n, d = X.shape
    S = (X.T @ X) / n

    # Analytical r
    r_est = estimate_factor_number(S, r_max)

    # Selection of tau
    if tau_list is not None:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        best_tau, best_score = None, np.inf
        for tau in tau_list:
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                Sigma_tr, _, _ = _poetry_core(X_train, r_est, tau, T, eta, reg)
                S_val = (X_val.T @ X_val) / len(val_idx)
                scores.append(np.linalg.norm(Sigma_tr - S_val, ord='fro'))
            mean_err = np.mean(scores)
            if mean_err < best_score:
                best_score, best_tau = mean_err, tau
        tau_est = best_tau
    else:
        # Heuristic from paper: tau ~ 0.02 * d^2
        tau_est = int(heuristic_tau_factor * d * d)

    Sigma, _, _ = _poetry_core(X, r_est, tau_est, T, eta, reg)
    return Sigma, r_est, tau_est


def cov_poetry(X: np.ndarray) -> np.ndarray:
    """POETRY covariance wrapper: auto-selects r and tau, returns Sigma only."""
    Sigma, _, _ = poetry_auto_covariance_estimation(X)
    return _ensure_pd(Sigma)


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
    Long-only minimum-variance portfolio.

    Despite the function name (kept for backwards-compatibility with earlier
    drafts of this code), this solves the LONG-ONLY minimum-variance QP:

        min  w' Σ w     s.t.   sum(w) = 1,   w_i >= 0

    via SLSQP.  We never want a closed-form Σ⁻¹𝟙 / 𝟙'Σ⁻¹𝟙 in this thesis
    because (a) it allows negative weights, and (b) at N > T the sample
    covariance is singular and "Σ + λI" introduces an arbitrary ridge.
    Long-only is the natural action space for HRP, so MinVar gets the
    same constraint for an apples-to-apples comparison.

    The `ridge` argument is retained as a small jitter on the diagonal to
    keep the optimiser numerically stable; it does not bias the solution
    materially because the regularised estimators (LW, NLS, POET) already
    produce well-conditioned Σ.
    """
    n = cov.shape[0]
    cov_pd = _ensure_pd(cov + ridge * np.eye(n))
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


# =============================================================================
# 3c. MHRP: Exponentially weighted covariance + equal-vol allocation + targeting
# =============================================================================

def cov_ewma(returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
    """EWMA covariance (RiskMetrics). Lower lambda_ = faster decay."""
    T, N = returns.shape
    if lambda_ <= 0 or lambda_ >= 1:
        raise ValueError("lambda_ must be in (0,1)")
    weights = np.array([(1 - lambda_) * lambda_**(T - 1 - i) for i in range(T)])
    weights = weights / weights.sum()
    demeaned = returns - returns.mean(axis=0)
    cov = demeaned.T @ (weights[:, None] * demeaned)
    return _ensure_pd(cov)


def _equal_vol_weights_from_cov(cov: np.ndarray, items: List[int]) -> np.ndarray:
    """Inverse-volatility weights for a subset of assets."""
    vols = np.sqrt(np.diag(cov)[items])
    inv_vol = 1.0 / np.maximum(vols, 1e-12)
    return inv_vol / inv_vol.sum()


def _cluster_vol_equal(cov: np.ndarray, items: List[int]) -> float:
    """Portfolio variance of a cluster under equal-volatility weights."""
    w = _equal_vol_weights_from_cov(cov, items)
    sub_cov = cov[np.ix_(items, items)]
    return float(w @ sub_cov @ w)


def _raw_returns(X: np.ndarray) -> np.ndarray:
    """
    Identity passthrough used as cov_fn for MHRP strategies.

    The backtest engine calls cov_fn(window) then alloc_fn(cov).  MHRP
    needs the raw return matrix, not a pre-computed covariance, so we pass
    the window straight through here and let mhrp_weights do its own
    EWMA covariance estimation internally.
    """
    return X


def mhrp_weights(returns: np.ndarray,
                 lambda_ewma: float = 0.94,
                 shrinkage_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 target_vol: float = 0.15,
                 linkage_method: str = "single") -> np.ndarray:
    """
    Modified HRP (Molyboga 2020): EWMA covariance, equal-volatility cluster
    allocation, and volatility targeting.

    Steps:
      1. EWMA covariance (optionally followed by a shrinkage pass).
      2. Hierarchical clustering on correlation distance.
      3. Recursive bisection with inverse-volatility weights per cluster.
      4. Scale final weights so ex-ante annualised vol = target_vol.

    The returned weights may sum to less than 1 (the remainder is cash
    earning rf=0 in the backtest engine).  This is the conventional MHRP
    convention; do not re-normalise before passing to the engine.
    """
    T, N = returns.shape

    cov_ew = cov_ewma(returns, lambda_=lambda_ewma)
    cov = shrinkage_func(cov_ew) if shrinkage_func is not None else cov_ew

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
            v0 = _cluster_vol_equal(cov, c0)
            v1 = _cluster_vol_equal(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha

    port_var = float(w @ cov @ w)
    ann_vol_forecast = np.sqrt(port_var * 252)
    if ann_vol_forecast > 1e-12:
        w = w * (target_vol / ann_vol_forecast)
    else:
        w = np.zeros_like(w)
    return w


# =============================================================================
# 4. BACKTEST ENGINE WITH WEIGHT DRIFT AND TRANSACTION COSTS
# =============================================================================

# A "strategy" is a (cov_estimator, allocator) pair.
StrategyMap = Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                              Callable[[np.ndarray], np.ndarray]]]


def make_default_strategies(linkage_method: str = "single") -> StrategyMap:
    """
    The default strategies for the main experiment.

    Five HRP variants spanning the covariance estimators we want to compare,
    three MHRP variants (EWMA + equal-vol + vol-targeting), plus EW benchmark.
    """
    def hrp_with(cov_fn):
        return cov_fn, lambda c: hrp_weights(c, linkage_method=linkage_method)

    def mhrp_with(shrinkage_fn=None):
        return (_raw_returns,
                lambda X: mhrp_weights(X, shrinkage_func=shrinkage_fn,
                                       linkage_method=linkage_method))

    return {
        "HRP-Sample":  hrp_with(cov_sample),
        "HRP-LW":      hrp_with(cov_linear_shrink),
        "HRP-NLS":     hrp_with(cov_nonlinear_shrink),
        "HRP-POET":    hrp_with(cov_poet_cv),
        "EW":          (cov_sample, equal_weights),
    }


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

def make_crsp_strategies(linkage_method: str = "single") -> StrategyMap:
    """
    Strategy set for the high-dimensional CRSP runs.

    Differences vs make_default_strategies:
      * HRP-POET uses POET-CV (cross-validated threshold) rather than
        analytic-only POET, because N > T makes CV selection more stable.
      * Three MHRP variants use _raw_returns as cov_fn so the backtest
        engine feeds the raw return window directly to mhrp_weights, which
        runs its own EWMA covariance internally.
    """
    def hrp_with(cov_fn):
        return cov_fn, lambda c: hrp_weights(c, linkage_method=linkage_method)

    def mhrp_with(shrinkage_fn=None):
        return (_raw_returns,
                lambda X: mhrp_weights(X, shrinkage_func=shrinkage_fn,
                                       linkage_method=linkage_method))

    return {
        "HRP-Sample":  hrp_with(cov_sample),
        "HRP-LW":      hrp_with(cov_linear_shrink),
        "HRP-NLS":     hrp_with(cov_nonlinear_shrink),
        "HRP-POET":    hrp_with(cov_poet_cv),
        "EW":          (cov_sample, equal_weights),
    }


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
            - Look up returns for the held PERMNOs.  When the daily
              return field already includes the CRSP delisting return
              (DlyRet / DLRET), the proper delisting return appears on
              the last trading day of the stock and the cell is NaN
              from the day after delisting onwards.  We treat any
              residual NaN as 0 (the position has been liquidated and
              the proceeds sit in cash earning rf=0 until the next
              rebalance).
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
                # NaN safety net: when DlyRet/DLRET is the source field,
                # the proper delisting return is already on the last
                # trading day; any further NaN means the position has
                # been liquidated and earns rf=0 until next rebalance.
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


def _dispersed_eig_cov(N: int, alpha: float = 0.7, seed: int = 0) -> np.ndarray:
    """
    Dense covariance with a power-law eigenvalue spectrum.

    Σ = U diag(λ) U' with λ_k = k^{-α} (then rescaled so trace = N).
    Eigenvectors U are a uniformly random orthogonal matrix.

    The point: eigenvalues are smoothly dispersed across several orders of
    magnitude with NO gap and NO sparsity.  Linear shrinkage (one constant
    proportion applied to every eigenvalue) is wasteful here -- the small
    eigenvalues need much more shrinkage than the large ones.  NLS handles
    each eigenvalue individually and should dominate LW.  POET has no
    factors to extract and no sparse residual to threshold.
    """
    rng = np.random.default_rng(seed)
    lam = (np.arange(1, N + 1)) ** (-alpha)
    lam = lam * (N / lam.sum())                # rescale so tr(Σ)=N
    A = rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(A)                     # uniform random orthogonal
    return (Q * lam) @ Q.T



def simulate_returns(T: int, N: int,
                     regime: str = "factor_sparse",
                     seed: int = 0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic returns from one of several theoretically distinct regimes.

    Regimes
    -------
    'factor_sparse'   : low-rank common + banded sparse residual.
                        POET expected to dominate, NLS competitive, LW worst.
    'dispersed_eigs'  : dense covariance with power-law eigenvalue spectrum.
                        NLS dominates LW; POET no advantage.
    'clustered_eigs'  : two eigenvalue clusters (few large, many small).
                        NLS should dominate LW; clean, stable DGP.
    'weak_factor'     : weak factors + sparse residual.  Robustness check.

    Returns
    -------
    X : (T, N) simulated returns from N(0, Σ_true)
    Sigma_true : the true covariance
    """
    rng = np.random.default_rng(seed)

    if regime == "factor_sparse":
        K = 3
        factor_var = np.array([5.0, 3.0, 1.5])
        B = rng.standard_normal((N, K))
        common = (B * factor_var) @ B.T
        Sigma_idio = _banded_cov(N, diag_var=0.4, bandwidth=2, decay=0.4)
        Sigma_true = common + Sigma_idio
    elif regime == "dispersed_eigs":
        Sigma_true = _dispersed_eig_cov(N, alpha=0.7, seed=seed+1)
    else:
        raise ValueError(f"unknown regime {regime!r}; "
                         f"expected one of factor_sparse, dispersed_eigs")

    Sigma_true = _ensure_pd(Sigma_true)
    L_chol = np.linalg.cholesky(Sigma_true)
    Z = rng.standard_normal((T, N))
    X = Z @ L_chol.T
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


def run_simulation_study(T: int = 300, N: int = 200,
                         regimes: Tuple[str, ...] = ("factor_sparse", "dispersed_eigs"),
                         n_reps: int = 50,
                         seed0: int = 0,
                         **kwargs) -> pd.DataFrame:
    """
    Sweep cov estimators over multiple theoretical regimes.

    Defaults: T=300, N=200 -> N/T ≈ 0.67, the moderate high-dim regime
    where NLS / POET advantages over LW are visible.

    Two relative-improvement columns are computed for downstream tests:
        mv_relative_to_sample : (loss_X - loss_Sample) / loss_Sample
        mv_relative_to_lw     : (loss_X - loss_LW)     / loss_LW
    Negative means improvement over the relevant baseline.
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
            base_sample = evaluate_sigma(estimators["Sample"](X), Sigma_true)
            base_lw = evaluate_sigma(estimators["LW"](X), Sigma_true)
            for name, fn in estimators.items():
                try:
                    Sigma_hat = fn(X)
                    ev = evaluate_sigma(Sigma_hat, Sigma_true)
                    rel_sample = ((ev["minvar_true_var"]
                                   - base_sample["minvar_true_var"])
                                  / base_sample["minvar_true_var"])
                    rel_lw = ((ev["minvar_true_var"]
                               - base_lw["minvar_true_var"])
                              / base_lw["minvar_true_var"])
                    rows.append(dict(regime=s, rep=r, estimator=name,
                                     **ev,
                                     mv_relative_to_sample=rel_sample,
                                     mv_relative_to_lw=rel_lw))
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
    Four figures:

      1. sim_minvar_relative_to_sample.png : bar chart of mean
            (loss_X − loss_Sample) / loss_Sample with 95% bootstrap CI.
            Sample is the universal baseline; this answers "do the
            advanced estimators all beat the trivial naive estimator?"
      2. sim_minvar_relative_to_lw.png : bar chart of mean
            (loss_X − loss_LW) / loss_LW with 95% bootstrap CI.
            LW is the relevant baseline once the universal Sample-vs-
            advanced gap is established; this answers "does going beyond
            linear shrinkage actually help?"
      3. sim_minvar_true_var.png : raw boxplots of the underlying loss
            (kept for completeness / appendix).
      4. sim_frobenius.png : Frobenius boxplots, with explicit caveat
            that within-rep variance dominates here.
    """
    _ensure_dir(outdir)
    sns.set_style("whitegrid")

    rng = np.random.default_rng(0)

    def boot_mean_ci(x: np.ndarray, n_boot: int = 2000
                     ) -> Tuple[float, float, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return float("nan"), float("nan"), float("nan")
        idx = rng.integers(0, x.size, (n_boot, x.size))
        means = x[idx].mean(axis=1)
        return (float(x.mean()),
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)))

    regimes = [r for r in sim_df["regime"].unique()]
    palette = {"Sample": "#7f7f7f", "LW": "#1f77b4",
               "NLS": "#2ca02c",   "POET": "#d62728"}

    def _bar_plot(rel_col: str, baseline_name: str,
                  excluded_estimators: Tuple[str, ...],
                  filename: str) -> None:
        rows = []
        for (regime, estimator), grp in sim_df.groupby(["regime", "estimator"]):
            if estimator in excluded_estimators:
                continue
            m, lo, hi = boot_mean_ci(grp[rel_col].values)
            rows.append({"regime": regime, "estimator": estimator,
                         "mean": m, "ci_lo": lo, "ci_hi": hi})
        summary = pd.DataFrame(rows)
        estimators = [e for e in ["Sample", "LW", "NLS", "POET"]
                      if e not in excluded_estimators]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_idx = np.arange(len(regimes))
        width = 0.8 / max(len(estimators), 1)
        for i, est in enumerate(estimators):
            sub = summary[summary["estimator"] == est].set_index("regime")
            means = [sub.loc[r, "mean"] if r in sub.index else np.nan for r in regimes]
            los   = [sub.loc[r, "ci_lo"] if r in sub.index else np.nan for r in regimes]
            his   = [sub.loc[r, "ci_hi"] if r in sub.index else np.nan for r in regimes]
            err_lo = [m - lo for m, lo in zip(means, los)]
            err_hi = [hi - m for m, hi in zip(means, his)]
            ax.bar(x_idx + (i - (len(estimators) - 1) / 2) * width, means,
                   width=width, yerr=[err_lo, err_hi],
                   capsize=3, color=palette.get(est, "C0"),
                   edgecolor="black", linewidth=0.5, label=est)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(regimes)
        ax.set_ylabel(f"Min-Var portfolio variance: (X − {baseline_name}) / {baseline_name}")
        ax.set_title(f"Relative improvement over {baseline_name} baseline\n"
                     f"(negative = better; bars = 95% bootstrap CI)")
        ax.legend(title="Estimator", loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{filename}", dpi=120)
        plt.close()

    # 1. relative to Sample (excludes Sample itself, which is trivially zero)
    _bar_plot("mv_relative_to_sample", "Sample",
              excluded_estimators=("Sample",),
              filename="sim_minvar_relative_to_sample.png")

    # 2. relative to LW (excludes Sample as not interesting and LW as trivial zero)
    _bar_plot("mv_relative_to_lw", "LW",
              excluded_estimators=("Sample", "LW"),
              filename="sim_minvar_relative_to_lw.png")

    # 3. raw MV boxplots (appendix) -----------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=sim_df, x="regime", y="minvar_true_var",
                hue="estimator", ax=ax, order=regimes, palette=palette)
    ax.set_title("Min-Var portfolio variance under true Σ (raw values)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Min-Var portfolio variance")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sim_minvar_true_var.png", dpi=120)
    plt.close()

    # 4. Frobenius (with caveat in title) -----------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=sim_df, x="regime", y="frobenius",
                hue="estimator", ax=ax, order=regimes, palette=palette)
    ax.set_title("Frobenius loss (NB: within-rep variance often dominates;\n"
                 "see relative-improvement figures for the portfolio-relevant view)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("‖Σ̂ − Σ_true‖_F")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sim_frobenius.png", dpi=120)
    plt.close()