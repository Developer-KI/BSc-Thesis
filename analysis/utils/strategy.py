"""
=============================================================================
hrp_lib.py - Building blocks for the HRP x Covariance Estimator experiment
=============================================================================

Contents
--------
1.  Covariance estimators (sample, LW linear, LW non-linear, POET, POET-CV, EWMA, ADCC-GARCH)
2.  Portfolio allocators (HRP, vol-balanced HRP+BL, equal-weight, SPYK market-cap)
3.  Backtest engine with weight drift and proportional transaction costs
4.  Performance metrics (return, vol, Sharpe, drawdown, Calmar, turnover, stability)
5.  Statistical tests (DM, LW2008-style block-bootstrap Sharpe test, Holm and BH)
6.  Factor-model simulation engine (power-law decay & factor-sparse regimes)
7.  Plot helpers

This module is import-only.  See run_main.py / run_robustness.py /
run_simulation.py for the experiments.

Author: Bachelor's thesis code, 2026.
=============================================================================
"""

from __future__ import annotations

import heapq
import os
import warnings
warnings.filterwarnings("ignore")

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils.plotting as _plt

from itertools import combinations
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf
import nonlinshrink as nls


# =============================================================================
# COVARIANCE ESTIMATORS
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

    return _cov_poet(X, K=None, K_max=K_max, threshold_C=best_C)


def cov_ewma(returns: np.ndarray,
             halflife: Optional[float] = None,
             lambda_: float = 0.97,
             shrink: Optional[str] = None) -> np.ndarray:
    """
    Vectorised EWMA covariance (RiskMetrics convention, zero-mean).

    halflife (trading days) overrides lambda_ when supplied:
        halflife=21  <->  lambda_ ≈ 0.967
        halflife=11  <->  lambda_ ≈ 0.939  (close to the classic 0.94)

    shrink applies a secondary shrinkage pass to the EWMA matrix:
        None   – no shrinkage (default)
        "lw"   – Ledoit-Wolf linear shrinkage toward scaled identity
        "nls"  – nonlinear eigenvalue shrinkage (Ledoit-Wolf 2020)
        "poet" – factor + adaptive soft-threshold (Fan-Liao-Mincheva 2013)

    All three shrinkage modes use the pseudo-returns trick: each row is
    scaled by sqrt(w_t * T) so that their (uncentred) sample covariance
    equals cov_ewma exactly, letting the existing estimators operate on
    the correct matrix without reimplementing their internals.
    """
    if halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must be positive")
        lambda_ = 0.5 ** (1.0 / halflife)
    if not (0 < lambda_ < 1):
        raise ValueError("lambda_ must be in (0, 1)")
    T, N = returns.shape
    exponents = np.arange(T - 1, -1, -1)
    w = (1.0 - lambda_) * np.power(lambda_, exponents)
    w /= w.sum()
    cov = returns.T @ (w[:, None] * returns)

    if shrink is None:
        return _ensure_pd(cov)

    # pseudo[t] = sqrt(w_t * T) * r_t  =>  pseudo.T @ pseudo / T = cov_ewma
    pseudo = np.sqrt(w * T)[:, None] * returns

    if shrink == "lw":
        _, alpha = ledoit_wolf(pseudo, assume_centered=True)
        mu = np.trace(cov) / N
        return _ensure_pd((1.0 - alpha) * cov + alpha * mu * np.eye(N))
    elif shrink == "nls":
        return _ensure_pd(nls.shrink_cov(pseudo, k=0))
    elif shrink == "poet":
        return cov_poet(pseudo)
    else:
        raise ValueError(f"unknown shrink={shrink!r}; expected None, 'lw', 'nls', or 'poet'")
    
# =============================================================================
# Estimation helpers
# =============================================================================

def _ensure_pd(M: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Symmetrise then clip eigenvalues to enforce PD."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, jitter)
    return V @ np.diag(w) @ V.T

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

def _cov_poet(X: np.ndarray,
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

def _ewma_pseudo(returns: np.ndarray, halflife: float) -> np.ndarray:
    """
    Rescale rows of *returns* so that pseudo.T @ pseudo / T == cov_ewma(returns, halflife).

    This lets any covariance estimator that internally computes a sample
    covariance (NLS, LW, POET, …) operate on exponentially front-weighted data
    without rewriting its internals.  The mapping is:

        pseudo[t] = sqrt(w_t * T) * r_t
        => pseudo.T @ pseudo / T = Σ_t w_t * r_t r_t' = EWMA cov

    where w_t = (1-λ) λ^{T-1-t}, normalised to sum to 1.
    """
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    T = returns.shape[0]
    lambda_ = 0.5 ** (1.0 / halflife)
    exponents = np.arange(T - 1, -1, -1, dtype=float)
    w = (1.0 - lambda_) * np.power(lambda_, exponents)
    w /= w.sum()
    return np.sqrt(w * T)[:, None] * returns

def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    return np.clip(cov / np.outer(std, std), -1.0, 1.0)

# =============================================================================
# Return estimation
# =============================================================================
def mu_black_litterman(
    window: np.ndarray,
    cov: np.ndarray,
    delta: float = 2.5,
    tau: float = 0.05,
    w_mkt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Black-Litterman posterior expected returns via PyPortfolioOpt.

    Prior:  π = δ Σ w_mkt  (equal-weight market proxy when w_mkt is None).
    Views:  one absolute view per asset, Q = sample mean, Ω = default (∝ τΣ).
    Falls back to the equilibrium π on any failure.
    """
    from pypfopt.black_litterman import BlackLittermanModel

    T, N = window.shape
    assets = list(range(N))
    cov_df = pd.DataFrame(cov, index=assets, columns=assets)

    if w_mkt is None:
        pi_arg: object = "equal"
        pi_fallback = np.ones(N) / N
    else:
        pi_arg = pd.Series(w_mkt, index=assets)
        pi_fallback = w_mkt

    absolute_views = dict(zip(assets, window.mean(axis=0)))

    try:
        bl = BlackLittermanModel(
            cov_matrix=cov_df,
            pi=pi_arg,
            absolute_views=absolute_views,
            risk_aversion=delta,
            tau=tau,
        )
        return bl.bl_returns().to_numpy()
    except Exception:
        return delta * cov @ pi_fallback

# =============================================================================
# PORTFOLIO ALLOCATORS
# =============================================================================

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


def equal_weights(cov: np.ndarray) -> np.ndarray:
    """1/N benchmark - DeMiguel, Garlappi & Uppal (2009)."""
    n = cov.shape[0]
    return np.ones(n) / n


def make_spyk_allocator(cap_wide: pd.DataFrame) -> Callable:
    """
    Market-cap weighted portfolio of the current top-K universe (SPYK benchmark).

    Returns a context-aware allocator (._context_aware = True) that receives
    the PERMNO array and rebalance date from backtest_pit and looks up the
    point-in-time market caps to form value-weighted portfolio weights.
    Falls back to equal-weight if cap data is unavailable.
    """
    def alloc_fn(data: np.ndarray, *,
                 permnos: np.ndarray, date: pd.Timestamp) -> np.ndarray:
        n = len(permnos)
        ew = np.ones(n) / n
        try:
            if date in cap_wide.index:
                row = cap_wide.loc[date]
            else:
                idx = cap_wide.index.searchsorted(date, side="right") - 1
                if idx < 0:
                    return ew
                row = cap_wide.iloc[idx]
        except Exception:
            return ew
        caps = row.reindex(permnos.tolist()).values.astype(float)
        caps = np.where(np.isnan(caps), 0.0, caps)
        total = caps.sum()
        if total <= 0:
            return ew
        return caps / total

    alloc_fn._context_aware = True
    return alloc_fn


# =============================================================================
# HMVA
# =============================================================================

def _vb_merge_cost(vol_a: float, vol_b: float,
                    sw_a: float, sw_b: float,
                    cross: float, lam_cov: float) -> float:
    """
    Scalar merge/split cost used by both top-down and bottom-up builders.

    lam_cov = 1  →  pure vol-balance:  |vol(A) − vol(B)| / (vol(A) + vol(B))
    lam_cov = 0  →  pure cross-cluster correlation:  ρ(A, B)
    0 < lam_cov < 1  →  convex blend of both terms.

    ρ(A, B) is the equal-weight inter-cluster correlation:
        ρ(A, B) = cross(A→B) / sqrt(sw_A × sw_B)
    where cross(A→B) = Σ_{i∈A, j∈B} Σ_ij  (one-way sum, by symmetry equal to
    the other direction) and sw_S = Σ_{i,j∈S} Σ_ij = (|S| vol(S))².

    This formula follows directly from
        Cov(EW_A, EW_B)   = cross / (|A| |B|)
        Vol(EW_A)         = sqrt(sw_A) / |A|
        Vol(EW_B)         = sqrt(sw_B) / |B|
    so ρ = [cross/(|A||B|)] / [sqrt(sw_A)/|A| × sqrt(sw_B)/|B|]
         = cross / sqrt(sw_A × sw_B).
    No matrix inversion or eigendecomposition is required.
    """
    vol_sum  = vol_a + vol_b
    vol_bal  = abs(vol_a - vol_b) / vol_sum if vol_sum > 1e-12 else 0.0
    if lam_cov >= 1.0:
        return vol_bal
    denom_rho = float(np.sqrt(max(sw_a * sw_b, 0.0)))
    rho       = cross / denom_rho if denom_rho > 1e-12 else 0.0
    return lam_cov * vol_bal + (1.0 - lam_cov) * rho


def _vb_split_bruteforce(cov_arr: np.ndarray,
                          indices: List[int],
                          lam_cov: float = 1.0) -> Tuple[List[int], List[int]]:
    """
    Exhaustive best bipartition for small clusters.

    Enumerates all subsets of size 1..n//2 (exploiting A/B symmetry) and
    returns the split that minimises _vb_merge_cost(lam_cov).
    Only called when len(indices) <= bf_threshold.
    """
    n = len(indices)
    if n <= 1:
        return list(indices), []
    M = cov_arr[np.ix_(indices, indices)]
    local = list(range(n))
    best_score = np.inf
    best_A, best_B = local[:1], local[1:]
    for r in range(1, n // 2 + 1):
        for A_tup in combinations(local, r):
            A_loc = list(A_tup)
            B_loc = [i for i in local if i not in set(A_tup)]
            nA, nB = len(A_loc), len(B_loc)
            sA    = float(M[np.ix_(A_loc, A_loc)].sum())
            sB    = float(M[np.ix_(B_loc, B_loc)].sum())
            cross = float(M[np.ix_(A_loc, B_loc)].sum())
            vA    = np.sqrt(max(sA / (nA * nA), 0.0))
            vB    = np.sqrt(max(sB / (nB * nB), 0.0))
            score = _vb_merge_cost(vA, vB, sA, sB, cross, lam_cov)
            if score < best_score:
                best_score = score
                best_A, best_B = A_loc, B_loc
    return [indices[i] for i in best_A], [indices[i] for i in best_B]


def _vb_split_heuristic(cov_arr: np.ndarray,
                         indices: List[int],
                         lam_cov: float = 1.0) -> Tuple[List[int], List[int]]:
    """
    O(n²) heuristic split via 2-D prefix sums.

    Sort order and objective both depend on lam_cov:

    lam_cov = 1 (pure vol-balance):
        Sort by individual asset vol.  Objective: |vol_L − vol_R|.
        Contiguous cuts in vol-sorted order capture the natural clustering
        of assets by risk level and are optimal for this criterion.

    lam_cov < 1 (correlation-aware blend):
        Sort by within-cluster row-sum  rowsum_i = Σ_j Σ_ij  (restricted to
        the current cluster).  This measures how much asset i co-moves with
        the rest of the cluster — its within-cluster systematic risk.
        Objective: lam_cov * vol_balance + (1-lam_cov) * ρ(A,B), where ρ is the
        equal-weight inter-cluster correlation computed from the same prefix
        sum P.  All n−1 cuts are evaluated in O(n) after the O(n²) matrix
        extraction:

            sum(M[:k, :k])   =  P[k-1, k-1]             ← within-left  sum
            sum(M[k:,  k:])  =  total − P[n-1,k-1]
                                       − P[k-1,n-1] + P[k-1,k-1]  ← within-right
            cross(left→right)=  P[k-1, n-1] − P[k-1, k-1]        ← one-way cross
            ρ(k)             =  cross / sqrt(s_left × s_right)

    Row-sum sort is also a good proxy for vol-sort (high-beta assets are
    typically high-vol), so it performs nearly as well for the vol-balance
    term while being strictly better for the cross-correlation term.
    """
    n = len(indices)
    if n <= 1:
        return list(indices), []

    # Extract submatrix once; derive sort key from it (no second cov lookup)
    M_raw = cov_arr[np.ix_(indices, indices)]
    if lam_cov >= 1.0:
        sort_key = np.sqrt(np.maximum(np.diag(M_raw), 0.0))  # individual vol
    else:
        sort_key = M_raw.sum(axis=1)                          # within-cluster rowsum

    order      = np.argsort(sort_key)
    sorted_idx = [indices[int(i)] for i in order]
    M          = M_raw[np.ix_(order, order)]                  # rearrange in-place

    P     = M.cumsum(axis=0).cumsum(axis=1)
    total = float(P[-1, -1])
    ks    = np.arange(1, n)
    s_left  = P[ks - 1, ks - 1]
    s_right = total - P[-1, ks - 1] - P[ks - 1, -1] + s_left
    n_l, n_r = ks.astype(float), (n - ks).astype(float)
    v_l  = np.sqrt(np.maximum(s_left  / (n_l * n_l), 0.0))
    v_r  = np.sqrt(np.maximum(s_right / (n_r * n_r), 0.0))

    if lam_cov >= 1.0:
        objective = np.abs(v_l - v_r)
    else:
        vol_bal   = np.abs(v_l - v_r) / np.maximum(v_l + v_r, 1e-12)
        cross     = P[ks - 1, -1] - P[ks - 1, ks - 1]          # one-way cross sum
        denom_rho = np.sqrt(np.maximum(s_left * s_right, 0.0))
        rho       = np.where(denom_rho > 1e-12, cross / denom_rho, 0.0)
        objective = lam_cov * vol_bal + (1.0 - lam_cov) * rho

    k = int(np.argmin(objective)) + 1
    return sorted_idx[:k], sorted_idx[k:]


def _build_vb_tree(cov_arr: np.ndarray, n: int,
                   bf_threshold: int = 10,
                   lam_cov: float = 1.0) -> dict:
    """BFS top-down construction of the vol-balanced binary tree."""
    root: dict = {"indices": list(range(n))}
    queue = [root]
    while queue:
        node = queue.pop(0)
        idx = node["indices"]
        if len(idx) <= 1:
            node["left"] = node["right"] = None
            continue
        if len(idx) <= bf_threshold:
            left_idx, right_idx = _vb_split_bruteforce(cov_arr, idx, lam_cov=lam_cov)
        else:
            left_idx, right_idx = _vb_split_heuristic(cov_arr, idx, lam_cov=lam_cov)
        node["left"]  = {"indices": left_idx}
        node["right"] = {"indices": right_idx}
        queue.append(node["left"])
        queue.append(node["right"])
    return root


def _build_vb_tree_bottomup(cov_arr: np.ndarray, n: int,
                              lam_cov: float = 1.0) -> dict:
    """
    Bottom-up agglomerative construction of the vol-balanced binary tree.

    Merge criterion controlled by lam_cov via _vb_merge_cost:
        lam_cov = 1  →  merge pairs with smallest |vol(A) − vol(B)| (vol-balance only)
        lam_cov = 0  →  merge pairs with smallest ρ(A, B), the equal-weight
                    inter-cluster correlation.  For singletons this equals the
                    standard pairwise correlation, so lam_cov=0 is equivalent to
                    average-linkage hierarchical clustering on the correlation
                    matrix — the same starting point as classic HRP, but now
                    used bottom-up rather than as an external dendrogram.
        0 < lam_cov < 1  →  blend: merge clusters that are both vol-similar AND
                    weakly correlated with each other.

    Implementation: lazy-deletion min-heap, O(N² log N).  The cross-sum
    cross(A,B) = Σ_{i∈A,j∈B} Σ_ij is already needed to update sw after each
    merge; for lam < 1 it is also used to compute ρ at no extra lookup cost.
    """
    idx_of = {i: [i]                                        for i in range(n)}
    sw     = {i: float(cov_arr[i, i])                      for i in range(n)}
    sz     = {i: 1                                          for i in range(n)}
    vol_c  = {i: float(np.sqrt(max(cov_arr[i, i], 0.0)))   for i in range(n)}
    nodes  = {i: {"indices": [i], "left": None, "right": None} for i in range(n)}
    active: set = set(range(n))
    next_id = n

    heap: List = []
    for i in range(n):
        for j in range(i + 1, n):
            cross_ij = float(cov_arr[i, j])
            cost = _vb_merge_cost(vol_c[i], vol_c[j], sw[i], sw[j], cross_ij, lam_cov)
            heapq.heappush(heap, (cost, i, j))

    while len(active) > 1:
        while heap:
            _, a, b = heapq.heappop(heap)
            if a in active and b in active:
                break
        else:
            break

        cross   = float(cov_arr[np.ix_(idx_of[a], idx_of[b])].sum())
        new_sw  = sw[a] + sw[b] + 2.0 * cross
        new_sz  = sz[a] + sz[b]
        new_vol = float(np.sqrt(max(new_sw / (new_sz * new_sz), 0.0)))
        new_idx = idx_of[a] + idx_of[b]

        nid = next_id
        next_id += 1
        idx_of[nid] = new_idx
        sw[nid]     = new_sw
        sz[nid]     = new_sz
        vol_c[nid]  = new_vol
        nodes[nid]  = {"indices": new_idx, "left": nodes[a], "right": nodes[b]}

        active.discard(a)
        active.discard(b)
        active.add(nid)

        for k in active:
            if k != nid:
                cross_k = float(cov_arr[np.ix_(new_idx, idx_of[k])].sum())
                cost_k  = _vb_merge_cost(new_vol, vol_c[k], new_sw, sw[k],
                                          cross_k, lam_cov)
                heapq.heappush(heap, (cost_k, nid, k))

    return nodes[next_id - 1]


def _vb_bisect_sharpe(node: dict, pw: float,
                       cov_arr: np.ndarray, mu_arr: np.ndarray,
                       rf: float) -> Dict[int, float]:
    """Allocate weight pw between subtrees proportional to non-negative cluster Sharpe."""
    if node is None or not node["indices"]:
        return {}
    idx = node["indices"]
    if len(idx) == 1:
        return {idx[0]: pw}
    left, right = node.get("left"), node.get("right")
    if left is None or right is None:
        return {i: pw / len(idx) for i in idx}

    def _sharpe(items: List[int]) -> float:
        n = len(items)
        sub = cov_arr[np.ix_(items, items)]
        sigma = float(np.sqrt(max(float(sub.sum()) / (n * n), 0.0)))
        mu = float(mu_arr[items].mean())
        return max((mu - rf) / sigma, 0.0) if sigma > 1e-12 else 0.0

    sl, sr = _sharpe(left["indices"]), _sharpe(right["indices"])
    tot = sl + sr
    alpha = sl / tot if tot > 1e-12 else 0.5
    return {
        **_vb_bisect_sharpe(left,  pw * alpha,       cov_arr, mu_arr, rf),
        **_vb_bisect_sharpe(right, pw * (1 - alpha), cov_arr, mu_arr, rf),
    }


def vol_hrp_bl_weights(cov: np.ndarray,
                        mu: np.ndarray,
                        rf: float = 0.0,
                        bf_threshold: int = 10,
                        cov_shrinkage: Optional[str] = None,
                        tree_method: str = "topdown",
                        lam_cov: float = 1.0,
                        turnover_penalty: float = 0.0,
                        weight_reg: float = 0.0,
                        w_prev: Optional[np.ndarray] = None,
                        regime_lam: bool = False,
                        lam_base: float = 0.0,
                        lam_scale: float = 0.5,
                        lam_corr: float = 0.0) -> np.ndarray:
    """
    Volatility-balanced HRP with Sharpe-ratio bisection and optional secondary
    covariance shrinkage and/or risk-contribution equalisation.

    Parameters
    ----------
    cov : (N, N) annualised covariance matrix.
    mu  : (N,) expected annual returns (e.g. Black-Litterman posterior).
    rf  : annual risk-free rate, same scale as mu.
    bf_threshold : cluster size at/below which exhaustive split search is used
        (only relevant when tree_method="topdown").
    cov_shrinkage : optional secondary shrinkage applied to cov before tree
        construction.  None | "lw" | "nls" | "poet"
    tree_method : "topdown"  — greedy recursive splitting (default, fast).
                  "bottomup" — agglomerative merging by min vol-difference
                               (O(N² log N), globally more coherent hierarchy).
    lam_cov : blend between vol-balance and cross-cluster correlation objectives.
        1.0 (default) — pure vol-balance, sort by individual vol.
        0.0           — minimise inter-cluster correlation ρ(A,B), sort by
                        within-cluster row-sum.
        (0, 1)        — convex blend; lam_cov=0.5 weights both objectives equally.
        See _vb_merge_cost and _vb_split_heuristic for the formula.
        Ignored when regime_lam=True (lam_base + lam_scale * CV_vol is used).
    turnover_penalty : L1 penalty strength on weight changes vs w_prev.
        Applied as a proximal soft-threshold step: deviations from w_prev
        smaller than this value are zeroed, larger ones are shrunk by it.
        Has no effect when w_prev is None.  Typical range: 0.01–0.05.
    weight_reg : L2 regularisation strength that blends weights toward
        the equal-weight portfolio.  weight_reg=0 (default) is pure HRP;
        weight_reg=1 collapses to 1/N.  Typical range: 0.05–0.3.
    w_prev : previous target weights (N,) for the L1 turnover penalty.
        Pass None (default) to skip the turnover penalty.
    regime_lam : when True, overrides lam at each call with a data-driven value
        computed from the cross-sectional dispersion of asset volatilities:
            CV_vol = std(vols) / mean(vols),  vols = sqrt(diag(Σ))
            lam_eff = clip(lam_base + lam_scale * CV_vol, 0, 1)
        High vol dispersion (crisis) → higher lam_eff (more vol-balance focus).
        Low vol dispersion (calm)    → lower lam_eff (more correlation focus).
        False (default) uses the fixed lam_cov parameter.
    lam_base  : intercept for the regime-conditional lam formula (default 0.0).
    lam_scale : slope on CV_vol for the regime-conditional lam formula (default 0.5).
        With lam_base=0.0, lam_scale=0.5: CV_vol=0.3 → contribution ≈0.15 (calm),
        CV_vol=0.8 → contribution ≈0.40 (crisis).
    lam_corr  : slope on average pairwise correlation for the regime-conditional
        lam formula (default 0.0 = disabled).  When non-zero, adds a second
        regime signal orthogonal to CV_vol:
            avg_corr = mean of off-diagonal entries of corr(Σ)
            lam_eff  = clip(lam_base + lam_scale*CV_vol + lam_corr*avg_corr, 0, 1)
        High avg_corr (crisis, everything co-moves) → higher lam_eff (vol-balance
        dominates because cluster structure breaks down).  Typical range: 0.1–0.3.
    """
    N = cov.shape[0]
    cov_pd = _ensure_pd(cov)

    if cov_shrinkage is not None:
        try:
            L = np.linalg.cholesky(cov_pd)
            n_pseudo = max(5 * N, 500)
            Z = np.random.default_rng(42).standard_normal((n_pseudo, N))
            pseudo = Z @ L.T
            if cov_shrinkage == "lw":
                _, alpha = ledoit_wolf(pseudo, assume_centered=True)
                mu_var = float(np.trace(cov_pd)) / N
                cov_pd = _ensure_pd(
                    (1.0 - alpha) * cov_pd + alpha * mu_var * np.eye(N))
            elif cov_shrinkage == "nls":
                cov_pd = _ensure_pd(nls.shrink_cov(pseudo, k=0))
            elif cov_shrinkage == "poet":
                cov_pd = cov_poet(pseudo)
            else:
                raise ValueError(f"cov_shrinkage={cov_shrinkage!r} not recognised")
        except Exception:
            pass

    if regime_lam:
        vols = np.sqrt(np.maximum(np.diag(cov_pd), 0.0))
        mean_vol = float(vols.mean())
        cv_vol = float(vols.std()) / mean_vol if mean_vol > 1e-12 else 0.0
        avg_corr = 0.0
        if lam_corr != 0.0:
            inv_vols = np.where(vols > 1e-12, 1.0 / vols, 0.0)
            corr_mat = cov_pd * np.outer(inv_vols, inv_vols)
            N_ = cov_pd.shape[0]
            avg_corr = float((corr_mat.sum() - N_) / max(N_ * (N_ - 1), 1))
        lam_cov = float(np.clip(lam_base + lam_scale * cv_vol + lam_corr * avg_corr, 0.0, 1.0))

    mu_arr = np.asarray(mu, dtype=float)
    if tree_method == "bottomup":
        tree = _build_vb_tree_bottomup(cov_pd, N, lam_cov=lam_cov)
    else:
        tree = _build_vb_tree(cov_pd, N, bf_threshold, lam_cov=lam_cov)

    w_dict = _vb_bisect_sharpe(tree, 1.0, cov_pd, mu_arr, rf)
    w = np.zeros(N)
    for i, wt in w_dict.items():
        w[i] = wt
    s = w.sum()
    w = w / s if s > 1e-12 else np.ones(N) / N

    # L2 weight regularisation: convex blend toward equal weights
    if weight_reg > 0.0:
        w = (1.0 - weight_reg) * w + weight_reg / N
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(N) / N

    # L1 turnover penalty: proximal soft-threshold deviations from w_prev
    if turnover_penalty > 0.0 and w_prev is not None:
        delta = w - w_prev
        delta = np.sign(delta) * np.maximum(np.abs(delta) - turnover_penalty, 0.0)
        w = np.maximum(w_prev + delta, 0.0)
        s = w.sum()
        w = w / s if s > 1e-12 else np.ones(N) / N

    return w


def vol_hrp_bl_strategy(
    cov_fn: Callable,
    rf: float = 0.0,
    delta: float = 2.5,
    tau: float = 0.05,
    bf_threshold: int = 10,
    cov_shrinkage: Optional[str] = None,
    tree_method: str = "topdown",
    lam_cov: float = 1.0,
    ewma_halflife: Optional[float] = None,
    turnover_penalty: float = 0.0,
    weight_reg: float = 0.0,
    vol_target: Optional[float] = None,
    adaptive_tau: bool = False,
    regime_lam: bool = False,
    lam_base: float = 0.0,
    lam_scale: float = 0.5,
    lam_corr: float = 0.0,
) -> Tuple[Callable, Callable]:
    """
    Factory: (cov_fn, alloc_fn) pair for vol-balanced HRP with BL returns.

    Parameters
    ----------
    cov_fn        : base covariance estimator, e.g. cov_sample.
    rf            : annual risk-free rate passed to vol_hrp_bl_weights.
    delta, tau    : BL risk-aversion and confidence parameters.
    bf_threshold  : exhaustive-search threshold (topdown only).
    cov_shrinkage : optional secondary shrinkage inside alloc_fn.
    tree_method   : "topdown" | "bottomup".
    lam_cov       : vol-balance / correlation blend (see vol_hrp_bl_weights).
    ewma_halflife : when set (trading days), front-weight observations via the
        EWMA pseudo-returns trick before calling cov_fn.  This makes recent
        returns count more without replacing the chosen estimator.  The BL
        expected-return estimate always uses the raw (uniform) window so that
        means are not double-discounted.  halflife=21 ≈ 1 month is a typical
        starting point; use halflife ≈ lookback/4 for a lookback-proportional
        decay.
    turnover_penalty : L1 penalty passed to vol_hrp_bl_weights; the closure
        tracks previous target weights across rebalances automatically.
    weight_reg    : L2 regularisation (ridge toward equal weights) passed to
        vol_hrp_bl_weights.
    vol_target    : annualised volatility target (e.g. 0.10 for 10%).  When set,
        weights are scaled after the L1/L2 step so that ex-ante annualised
        portfolio volatility equals vol_target (assumes daily cov, multiplied
        by 252 to annualise).  The portfolio may then be levered (weights sum
        > 1) or de-levered (weights sum < 1).  prev_w always stores the
        pre-scaling directional weights so the turnover penalty and BL prior
        are unaffected by the leverage level.  None (default) = no scaling.
    adaptive_tau  : when True, overrides tau with 1/T at each rebalance, where
        T is the lookback window length.  Longer windows give more confidence
        to historical-mean views; shorter windows lean on the equilibrium prior.
        Theoretically motivated: BL derivation sets tau ~ 1/T.  False = fixed tau.
    regime_lam, lam_base, lam_scale, lam_corr : passed to vol_hrp_bl_weights; see
        its docstring for the CV-vol + avg-corr formula.
        regime_lam=False (default) = fixed lam_cov; lam_corr=0.0 (default) disables
        the average-correlation term even when regime_lam=True.
    """
    cache: Dict = {}
    prev_w: Dict[str, Optional[np.ndarray]] = {"w": None}

    def _cov_fn(window: np.ndarray) -> np.ndarray:
        data = _ewma_pseudo(window, ewma_halflife) if ewma_halflife is not None else window
        cov = cov_fn(data)
        effective_tau = 1.0 / window.shape[0] if adaptive_tau else tau
        cache["mu"] = mu_black_litterman(window, cov, delta=delta, tau=effective_tau)
        return cov

    def _alloc_fn(cov: np.ndarray) -> np.ndarray:
        mu = cache.get("mu", np.zeros(cov.shape[0]))
        w = vol_hrp_bl_weights(cov, mu, rf=rf,
                               bf_threshold=bf_threshold,
                               cov_shrinkage=cov_shrinkage,
                               tree_method=tree_method,
                               lam_cov=lam_cov,
                               turnover_penalty=turnover_penalty,
                               weight_reg=weight_reg,
                               w_prev=prev_w["w"],
                               regime_lam=regime_lam,
                               lam_base=lam_base,
                               lam_scale=lam_scale,
                               lam_corr=lam_corr)
        prev_w["w"] = w  # store pre-scaling direction for BL prior & turnover
        if vol_target is not None:
            port_vol = float(np.sqrt(max(float(w @ cov @ w) * 252, 1e-12)))
            if port_vol > 1e-12:
                w = w * (vol_target / port_vol)
        return w

    return _cov_fn, _alloc_fn


# =============================================================================
# 4. BACKTEST ENGINE WITH WEIGHT DRIFT AND TRANSACTION COSTS
# =============================================================================

# A "strategy" is a (cov_estimator, allocator) pair.
StrategyMap = Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                              Callable[[np.ndarray], np.ndarray]]]


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

def make_crsp_strategies(market_cap_wide: Optional[pd.DataFrame] = None,
                         ) -> StrategyMap:
    """
    Strategy set for the high-dimensional CRSP runs
    """
    def hrp_with(cov_fn):
        return cov_fn, lambda c: hrp_weights(c)
    
    def miv_var_with(cov_fn):
        return cov_fn, lambda c: min_var_weights(c)

    def vb_hrp_with(cov_fn, cov_shrinkage=None, adaptive_tau=True, tree_method="topdown",
                    lam_cov=0.2, lam_corr=0.2, ewma_halflife=21,
                    turnover_penalty=0.05, weight_reg=0.10, regime_lam=True, vol_target=None):
        return vol_hrp_bl_strategy(cov_fn, cov_shrinkage=cov_shrinkage,
                                    adaptive_tau=adaptive_tau,
                                    tree_method=tree_method,
                                    lam_cov=lam_cov,
                                    lam_corr=lam_corr,
                                    ewma_halflife=ewma_halflife,
                                    turnover_penalty=turnover_penalty,
                                    weight_reg=weight_reg,
                                    regime_lam=regime_lam,
                                    vol_target=vol_target,)

    strategies: StrategyMap = {
        "HMVA":      vb_hrp_with(cov_nonlinear_shrink),
        "HRP":       hrp_with(cov_sample),
        "GMV":       miv_var_with(cov_sample),
        "EW":        (cov_sample, equal_weights),
    }
    if market_cap_wide is not None:
        strategies["SPY-K"] = (cov_sample, make_spyk_allocator(market_cap_wide))
    return strategies


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

    rebal_idx = list(range(min_history_days, T_total, rebalance))
    if verbose:
        print(f"[bt-pit] lookback={lookback}, min_history={min_history_days}, "
              f"rebal={rebalance}, cost={cost_bps}bps, rebalances={len(rebal_idx)}")

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

        # require full non-NaN history in [t-lookback, t); zero-fill the rest
        window = arr[t - lookback:t, cand]
        full_hist = ~np.isnan(window).any(axis=0)
        keep = cand[full_hist]
        if keep.size < 5:
            print(f"[bt-pit] WARN only {keep.size} stocks survive "
                  f"history filter at {rebal_date}; skipping.")
            continue
        partial = cand[~full_hist]
        n_zeroed = partial.size
        all_keep = np.concatenate([keep, partial]) if n_zeroed > 0 else keep
        window_clean = np.nan_to_num(arr[t - lookback:t, all_keep], nan=0.0)
        N_t = keep.size
        universe_sizes.append({"date": rebal_date, "raw": len(universe),
                               "in_panel": cand.size, "with_history": N_t,
                               "zeroed": n_zeroed})

        for name, (cov_fn, alloc_fn) in strategies.items():
            context_aware = getattr(alloc_fn, "_context_aware", False)
            try:
                cov = cov_fn(window_clean)
                if context_aware:
                    w_target = alloc_fn(cov, permnos=permnos[all_keep],
                                        date=rebal_date)
                else:
                    w_target = alloc_fn(cov)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"[bt-pit] strategy='{name}' date={rebal_date.date()} "
                    f"rebal={k+1}/{len(rebal_idx)}: {e} — "
                    f"falling back to equal-weight 1/N (N={all_keep.size})",
                    RuntimeWarning, stacklevel=2,
                )
                w_target = np.ones(all_keep.size) / all_keep.size

            # transaction cost: compare to drifted weights from previous period.
            # The two weight vectors live on potentially different column sets;
            # we align on the union and treat missing entries as zero weight
            # (i.e., closed positions / new positions both count as a trade).
            if drifted[name] is None:
                tc = 0.0
            else:
                old_cols, old_w = drifted[name]
                union = np.union1d(old_cols, all_keep)
                w_old_aligned = np.zeros(union.size)
                w_new_aligned = np.zeros(union.size)
                w_old_aligned[np.searchsorted(union, old_cols)] = old_w
                w_new_aligned[np.searchsorted(union, all_keep)] = w_target
                tc = cost_rate * np.abs(w_new_aligned - w_old_aligned).sum()

            weights_log[name][rebal_date] = pd.Series(
                w_target, index=permnos[all_keep], name=rebal_date)

            # apply weights with drift, NaN → 0 for that day
            t_end = min(t + rebalance, T_total)
            w_curr = w_target.copy()
            cols_curr = all_keep.copy()
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
            suffix = f" + {n_zeroed}" if n_zeroed > 0 else ""
            print(f"[bt-pit]   rebal {k + 1}/{len(rebal_idx)} "
                  f"date={rebal_date.date()} N_t={N_t}{suffix}")

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

def _extra_metrics(r: pd.Series) -> Dict[str, float]:
    """Sortino, Omega, VaR95, CVaR95, Skew, Kurt, HitRate from a daily return series."""
    ann_ret = (1.0 + r).prod() ** (252.0 / len(r)) - 1.0
    downside = r[r < 0].std() * np.sqrt(252.0)
    sortino = ann_ret / downside if downside > 0 else np.nan
    gains  = r[r > 0].sum()
    losses = (-r[r < 0]).sum()
    omega  = float(gains / losses) if losses > 0 else np.nan
    var95  = float(np.percentile(r.dropna(), 5))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
    return dict(Sortino=sortino, Omega=omega, VaR95=var95, CVaR95=cvar95,
                Skew=float(r.skew()), Kurt=float(r.kurtosis()),
                HitRate=float((r > 0).mean()))


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
                          Turnover=turnover, SharpeStab=sharpe_stab,
                          **_extra_metrics(r))
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
    _nan_row = dict(AnnReturn=np.nan, AnnVol=np.nan, Sharpe=np.nan,
                    MaxDD=np.nan, Calmar=np.nan, Turnover=np.nan,
                    SharpeStab=np.nan, Sortino=np.nan, Omega=np.nan,
                    VaR95=np.nan, CVaR95=np.nan, Skew=np.nan,
                    Kurt=np.nan, HitRate=np.nan)
    rows: Dict[str, Dict[str, float]] = {}
    for name, r in daily_returns.items():
        r = r.dropna()
        if len(r) < 2:
            rows[name] = _nan_row.copy()
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
                          Turnover=turnover, SharpeStab=sharpe_stab,
                          **_extra_metrics(r))
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


def plot_backtest_results(daily: pd.DataFrame,
                          weights_log,
                          metrics: pd.DataFrame,
                          outdir: str,
                          title_suffix: str = "") -> None:
    """
    Comprehensive backtest plot suite using plotting.py.

    Outputs (PNG, 150 dpi):
        equity_curves.png    cumulative wealth index
        drawdowns.png        underwater curve with fill
        rolling_sharpe.png   63-day rolling annualised Sharpe
        rolling_vol.png      63-day rolling annualised volatility
        annual_returns.png   grouped calendar-year bar chart
        risk_return.png      risk-return scatter with iso-Sharpe lines
        return_dist.png      histogram + KDE per strategy (monthly)
        strategy_corr.png    pairwise return-correlation heatmap
        monthly_<name>.png   calendar heatmap per strategy
        sharpe_bars.png      Sharpe bar chart
        maxdd_bars.png       Max-drawdown bar chart
        turnover_bars.png    Turnover bar chart
        metrics_table.png    colour-coded full metrics table

    Falls back to the legacy hrp_lib plots when plotting.py is unavailable.
    """
    _ensure_dir(outdir)
    suf = f" {title_suffix}".rstrip() if title_suffix else ""

    if _plt is None:
        warnings.warn("plotting.py not importable; using legacy hrp_lib plots.")
        plot_equity_and_drawdown(daily, outdir, title_suffix)
        plot_metric_bars(metrics, outdir, title_suffix)
        return

    # Resample to monthly for functions that assume monthly inputs
    monthly = daily.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)

    _plt.plot_cumulative_returns(
        daily, title=f"Equity Curves{suf}",
        save_path=f"{outdir}/equity_curves.png")
    plt.close("all")

    _plt.plot_drawdown(
        daily, title=f"Drawdowns{suf}",
        save_path=f"{outdir}/drawdowns.png")
    plt.close("all")

    _plt.plot_rolling_sharpe(
        daily, window=63, title=f"Rolling Sharpe (63-day){suf}",
        save_path=f"{outdir}/rolling_sharpe.png")
    plt.close("all")

    _plt.plot_rolling_volatility(
        daily, window=63, title=f"Rolling Volatility (63-day){suf}",
        save_path=f"{outdir}/rolling_vol.png")
    plt.close("all")

    _plt.plot_annual_returns(
        daily, title=f"Annual Returns{suf}",
        save_path=f"{outdir}/annual_returns.png")
    plt.close("all")

    _plt.plot_risk_return_scatter(
        monthly, title=f"Risk-Return Profile{suf}",
        save_path=f"{outdir}/risk_return.png")
    plt.close("all")

    _plt.plot_return_distribution(
        monthly, title=f"Monthly Return Distributions{suf}",
        save_path=f"{outdir}/return_dist.png")
    plt.close("all")

    _plt.plot_correlation_heatmap(
        daily, title=f"Strategy Return Correlations{suf}",
        save_path=f"{outdir}/strategy_corr.png")
    plt.close("all")

    for name in daily.columns:
        safe = name.replace(" ", "_").replace("/", "-")
        _plt.plot_monthly_returns_heatmap(
            monthly[name], title=f"Monthly Returns – {name}{suf}",
            save_path=f"{outdir}/monthly_{safe}.png")
        plt.close("all")

    # Metric bar charts
    if "Sharpe" in metrics.columns:
        _plt.plot_sharpe_bar(
            metrics["Sharpe"], title=f"Sharpe Ratio by Strategy{suf}",
            save_path=f"{outdir}/sharpe_bars.png")
        plt.close("all")

    for col, fname, colour in [("MaxDD",    "maxdd_bars",    "indianred"),
                                ("Turnover", "turnover_bars", "darkorange")]:
        if col in metrics.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            metrics[col].plot.bar(ax=ax, color=colour, edgecolor="black")
            ax.set_title(f"{col} by strategy{suf}".strip())
            ax.set_ylabel(col)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{fname}.png", dpi=150)
            plt.close()

    # Comprehensive colour-coded metrics table via plotting.py
    _rename = {
        "AnnReturn":  "Ann. Return (%)",
        "AnnVol":     "Ann. Volatility (%)",
        "Sharpe":     "Sharpe Ratio",
        "Sortino":    "Sortino Ratio",
        "Omega":      "Omega Ratio",
        "MaxDD":      "Max Drawdown (%)",
        "Calmar":     "Calmar Ratio",
        "VaR95":      "VaR 95% (%)",
        "CVaR95":     "CVaR 95% (%)",
        "HitRate":    "Hit Rate (%)",
        "Skew":       "Skewness",
        "Kurt":       "Excess Kurtosis",
        "Turnover":   "Turnover",
        "SharpeStab": "Sharpe Stability",
    }
    tbl = metrics.rename(
        columns={k: v for k, v in _rename.items() if k in metrics.columns})
    _plt.plot_metrics_table(
        tbl, title=f"Performance Metrics{suf}",
        save_path=f"{outdir}/metrics_table.png")
    plt.close("all")


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