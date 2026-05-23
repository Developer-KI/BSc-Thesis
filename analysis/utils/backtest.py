from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import utils.plotting as _plt

from itertools import combinations
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.optimize import minimize
import nonlinshrink as nls


# =============================================================================
# COVARIANCE ESTIMATORS
# =============================================================================

def cov_sample(X: np.ndarray) -> np.ndarray:
    """Plain-vanilla sample covariance Σ̂ = (T-1)^{-1} (X-μ)' (X-μ)."""
    return np.cov(X, rowvar=False)

def cov_nonlinear_shrink(X: np.ndarray) -> np.ndarray:
    """Wrapper for the nonlinshrink package"""

    # The package automatically demeans the data by default
    # 'k' is an optional parameter to specify effective degrees of freedom already subtracted
    return nls.shrink_cov(X, k=0)

def cov_ewa_nls(window: np.ndarray) -> np.ndarray:
    return cov_nonlinear_shrink(_ewma_pseudo(window, halflife=21))
    
# =============================================================================
# Estimation helpers
# =============================================================================

def _ensure_pd(M: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Symmetrise then clip eigenvalues to enforce PD."""
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, jitter)
    return V @ np.diag(w) @ V.T

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

def mu_BL_momentum(
    window: np.ndarray,
    cov: np.ndarray,
    skip_days: int = 21,
) -> np.ndarray:
    """
    Black-Litterman type posterior expected returns

    Prior:  π = 1
    Views:  one absolute view per asset, 
    Q = skip-x-days momentum signal
    P = I_N (absolute views)
    Ω = diag(Σ)  (proportional uncertainty)

    Final form:
    μ = π + Σ (Σ + diag(Σ))⁻¹ (Q - π)

    Falls back to π on any numerical failure
    """
    T, N = window.shape

    pi_mu = np.ones(N)

    signal_window = window[:-skip_days] if skip_days > 0 and T > skip_days else window
    Q = signal_window.mean(axis=0)

    try:
        D = np.diag(np.maximum(np.diag(cov), 1e-12))
        return pi_mu + cov @ np.linalg.solve(cov + D, Q - pi_mu)
    except np.linalg.LinAlgError:
        return pi_mu

# =============================================================================
# Estimation smoother
# =============================================================================
  
def _kf_smooth_weights(
    w_new: np.ndarray,
    w_prev: np.ndarray,
    cov: np.ndarray,
    mu: Optional[np.ndarray] = None,
    mu_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Kalman-filter weight smoother.

    Gain K_t = Q_t / (Q_t + R_t) blends the previous weights (state) with
    the new raw weights (observation), where:
        R_t = spectral entropy of Sigma eigenvalues (normalised to [0,1])
              -- high when the covariance is diffuse / estimation is noisy
        Q_t = relative BL signal velocity ||mu_t - mu_{t-1}|| / ||mu_{t-1}||
              -- high when the return signal has shifted sharply
              -- defaults to 1.0 when no return signal is available
    A large R_t (noisy estimate) or small Q_t (slow-moving signal) lowers K_t,
    so the smoother leans on the prior weights instead of the new ones.
    """
    N = len(w_new)

    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    s_sum = eigvals.sum()
    if s_sum > 1e-12:
        p = eigvals / s_sum
        H = -float(np.sum(p * np.log(p + 1e-12)))
        R_t = H / np.log(max(N, 2))
    else:
        R_t = 1.0

    if mu is not None and mu_prev is not None:
        dmu = float(np.linalg.norm(mu - mu_prev))
        mu_scale = float(np.linalg.norm(mu_prev)) + 1e-12
        Q_t = dmu / mu_scale
    else:
        Q_t = 1.0

    K_t = float(np.clip(Q_t / (Q_t + R_t + 1e-12), 0.0, 1.0))
    w = (1.0 - K_t) * w_prev + K_t * w_new
    s = w.sum()
    return w / s if s > 1e-12 else np.ones(N) / N

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

def hrp_strategy(
    cov_fn: Callable,
    linkage_method: str = "single",
    kf_tp: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Factory: (cov_fn, alloc_fn) pair for HRP with optional KF weight smoothing.

    No return signal is available so Q_t defaults to 1.0 in the KF smoother;
    smoothing is driven entirely by covariance spectral entropy (R_t).
    """
    prev_w: Dict[str, Optional[np.ndarray]] = {"w": None}

    def _alloc_fn(cov: np.ndarray) -> np.ndarray:
        w = hrp_weights(cov, linkage_method=linkage_method)
        if kf_tp and prev_w["w"] is not None:
            w = _kf_smooth_weights(w, prev_w["w"], cov)
        prev_w["w"] = w.copy()
        return w

    return cov_fn, _alloc_fn

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



def min_var_strategy(
    cov_fn: Callable,
    kf_tp: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Factory: (cov_fn, alloc_fn) pair for the long-only minimum-variance portfolio.

        min  w'Σw   s.t.  sum(w)=1,  0 ≤ w_i ≤ 1

    No return signal is available so Q_t defaults to 1.0 in the KF smoother;
    smoothing is driven entirely by covariance spectral entropy (R_t).
    """
    prev_w: Dict[str, Optional[np.ndarray]] = {"w": None}

    def _alloc_fn(cov: np.ndarray) -> np.ndarray:
        w = min_var_weights(cov)
        if kf_tp and prev_w["w"] is not None:
            w = _kf_smooth_weights(w, prev_w["w"], cov)
        prev_w["w"] = w.copy()
        return w

    return cov_fn, _alloc_fn


def max_utility_weights(cov: np.ndarray,
                        mu: np.ndarray,
                        gamma: float = 2.5) -> np.ndarray:
    """
    Long-only mean-variance utility maximisation via SLSQP.

        max  w'μ - (γ/2) w'Σw   s.t.  sum(w)=1,  0 ≤ w_i ≤ 1

    Falls back to equal-weight on solver failure.
    """
    n = cov.shape[0]
    cov_pd = _ensure_pd(cov)
    mu_arr = np.asarray(mu, dtype=float)
    x0 = np.ones(n) / n

    obj  = lambda w: -(w @ mu_arr) + 0.5 * gamma * float(w @ cov_pd @ w)
    grad = lambda w: -mu_arr + gamma * (cov_pd @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(n)}]
    bnds = [(0.0, 1.0)] * n
    res = minimize(obj, x0, jac=grad, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-10})
    if not res.success:
        return x0
    w = np.maximum(res.x, 0.0)
    return w / w.sum() if w.sum() > 0 else x0


def max_utility_strategy(
    cov_fn: Callable,
    gamma: float = 2.5,
    kf_tp: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Factory: (cov_fn, alloc_fn) pair for the long-only mean-variance utility
    portfolio.  Expected returns from Black-Litterman.

    Parameters
    ----------
    cov_fn : base covariance estimator.
    gamma  : risk-aversion coefficient.
    kf_tp  : if True, smooth weights each period with _kf_smooth_weights.
    """
    cache: Dict = {}
    prev_w: Dict[str, Optional[np.ndarray]] = {"w": None}
    kf_cache: Dict[str, Optional[np.ndarray]] = {"mu_prev": None}

    def _cov_fn(window: np.ndarray) -> np.ndarray:
        cov = cov_fn(window)
        cache["window"] = window
        return cov

    def _alloc_fn(cov: np.ndarray) -> np.ndarray:
        window = cache.get("window", np.zeros((1, cov.shape[0])))
        mu = mu_BL_momentum(window, cov)
        w = max_utility_weights(cov, mu, gamma=gamma)

        if kf_tp and prev_w["w"] is not None and kf_cache["mu_prev"] is not None:
            w = _kf_smooth_weights(w, prev_w["w"], cov, mu, kf_cache["mu_prev"])

        prev_w["w"] = w.copy()
        if kf_tp:
            kf_cache["mu_prev"] = mu.copy()

        return w

    return _cov_fn, _alloc_fn



def equal_weights(cov: np.ndarray) -> np.ndarray:
    """1/N benchmark"""
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
# HMVO
# =============================================================================

def _vb_merge_cost(cross: float, n_a: int, n_b: int,
                   sw_a: float, sw_b: float) -> float:
    """
    Equal-weight blend of inter-cluster covariance and correlation:

        0.5 * Cov(EW_A, EW_B) + 0.5 * rho(EW_A, EW_B)

    where
        Cov(EW_A, EW_B) = cross / (n_A * n_B)
        rho(EW_A, EW_B) = cross / sqrt(sw_A * sw_B)
        cross            = Σ_{i∈A, j∈B} Σ_ij  (one-way sum)
        sw_S             = Σ_{i,j∈S} Σ_ij = (n_S * Vol(EW_S))²
    """
    cov_term  = cross / float(n_a * n_b) if n_a * n_b > 0 else 0.0
    denom_rho = float(np.sqrt(max(sw_a * sw_b, 0.0)))
    rho       = cross / denom_rho if denom_rho > 1e-12 else 0.0
    return 0.5 * cov_term + 0.5 * rho


def _vb_split_bruteforce(cov_arr: np.ndarray,
                          indices: List[int]) -> Tuple[List[int], List[int]]:
    """
    Exhaustive best bipartition for small clusters.

    Enumerates all subsets of size 1..n//2 (exploiting A/B symmetry) and
    returns the split minimising _vb_merge_cost.
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
            nA, nB  = len(A_loc), len(B_loc)
            sA      = float(M[np.ix_(A_loc, A_loc)].sum())
            sB      = float(M[np.ix_(B_loc, B_loc)].sum())
            cross   = float(M[np.ix_(A_loc, B_loc)].sum())
            score   = _vb_merge_cost(cross, nA, nB, sA, sB)
            if score < best_score:
                best_score = score
                best_A, best_B = A_loc, B_loc
    return [indices[i] for i in best_A], [indices[i] for i in best_B]


def _vb_split_heuristic(cov_arr: np.ndarray,
                         indices: List[int]) -> Tuple[List[int], List[int]]:
    """
    O(n²) heuristic split via row-sum sort and 2-D prefix sums.

    Sort by within-cluster row-sum (rowsum_i = Σ_j Σ_ij): assets with high
    systematic within-cluster covariance are placed adjacent in the sorted
    sequence, so any contiguous cut produces internally cohesive groups.

    Objective at cut k  (all terms in O(1) from the 2-D prefix sum P):

        cross(k)   =  P[k-1, n-1] - P[k-1, k-1]        one-way inter-cluster sum
        s_left(k)  =  P[k-1, k-1]                       within-left sum
        s_right(k) =  total - P[n-1,k-1] - P[k-1,n-1] + P[k-1,k-1]

        Cov term   =  cross / (k * (n-k))
        rho term   =  cross / sqrt(s_left * s_right)
        objective  =  0.5 * Cov + 0.5 * rho

    All n-1 cuts are scored in O(n) after the O(n²) matrix extraction.
    """
    n = len(indices)
    if n <= 1:
        return list(indices), []

    M_raw      = cov_arr[np.ix_(indices, indices)]
    order      = np.argsort(M_raw.sum(axis=1))        # ascending within-cluster row-sum
    sorted_idx = [indices[int(i)] for i in order]
    M          = M_raw[np.ix_(order, order)]

    P       = M.cumsum(axis=0).cumsum(axis=1)
    total   = float(P[-1, -1])
    ks      = np.arange(1, n)
    s_left  = P[ks - 1, ks - 1]
    s_right = total - P[-1, ks - 1] - P[ks - 1, -1] + s_left
    cross   = P[ks - 1, -1] - P[ks - 1, ks - 1]      # one-way cross sum

    cov_term  = cross / (ks * (n - ks))
    denom_rho = np.sqrt(np.maximum(s_left * s_right, 0.0))
    rho       = np.where(denom_rho > 1e-12, cross / denom_rho, 0.0)
    objective = 0.5 * cov_term + 0.5 * rho

    k = int(np.argmin(objective)) + 1
    return sorted_idx[:k], sorted_idx[k:]


def _build_vb_tree(cov_arr: np.ndarray, n: int,
                   bf_threshold: int = 10) -> dict:
    root: dict = {"indices": list(range(n))}
    queue = [root]
    while queue:
        node = queue.pop(0)
        idx = node["indices"]
        if len(idx) <= 1:
            node["left"] = node["right"] = None
            continue
        if len(idx) <= bf_threshold:
            left_idx, right_idx = _vb_split_bruteforce(cov_arr, idx)
        else:
            left_idx, right_idx = _vb_split_heuristic(cov_arr, idx)
        node["left"]  = {"indices": left_idx}
        node["right"] = {"indices": right_idx}
        queue.append(node["left"])
        queue.append(node["right"])
    return root




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
    if tot > 1e-12:
        alpha = sl / tot
    else:
        # Both clusters have non-positive Sharpe — fall back to inverse-variance split
        L, R = left["indices"], right["indices"]
        v_L = float(cov_arr[np.ix_(L, L)].sum()) / (len(L) ** 2)
        v_R = float(cov_arr[np.ix_(R, R)].sum()) / (len(R) ** 2)
        v_tot = v_L + v_R
        alpha = v_R / v_tot if v_tot > 1e-12 else 0.5
    return {
        **_vb_bisect_sharpe(left,  pw * alpha,       cov_arr, mu_arr, rf),
        **_vb_bisect_sharpe(right, pw * (1 - alpha), cov_arr, mu_arr, rf),
    }




def _vb_bisect_vol(node: dict, pw: float,
                   cov_arr: np.ndarray) -> Dict[int, float]:
    """
    Allocate weight pw between subtrees by inverse cluster variance (standard HRP).

    Each subtree is treated as an equal-weighted sub-portfolio.  The left
    cluster receives weight proportional to 1/V_L so that higher-variance
    clusters get less capital:

        V_C  = sum(Σ[C,C]) / n_C²   (equal-weight cluster variance)
        alpha = V_R / (V_L + V_R)
    """
    if node is None or not node["indices"]:
        return {}
    idx = node["indices"]
    if len(idx) == 1:
        return {idx[0]: pw}
    left, right = node.get("left"), node.get("right")
    if left is None or right is None:
        return {i: pw / len(idx) for i in idx}

    L, R = left["indices"], right["indices"]
    v_L = float(cov_arr[np.ix_(L, L)].sum()) / (len(L) ** 2)
    v_R = float(cov_arr[np.ix_(R, R)].sum()) / (len(R) ** 2)
    tot = v_L + v_R
    alpha = v_R / tot if tot > 1e-12 else 0.5  # inverse-variance: left gets V_R share

    return {
        **_vb_bisect_vol(left,  pw * alpha,         cov_arr),
        **_vb_bisect_vol(right, pw * (1.0 - alpha), cov_arr),
    }


def vol_hrp_bl_weights(cov: np.ndarray,
                        mu: np.ndarray,
                        rf: float = 0.0,
                        bf_threshold: int = 10,
                        bisect_method: str = "sharpe",
                        ) -> np.ndarray:
    """
    HRP with equal-blend inter-cluster covariance/correlation tree.

    Parameters
    ----------
    cov : (N, N) annualised covariance matrix.
    mu  : (N,) expected annual returns
    rf  : annual risk-free rate, same scale as mu.
    bf_threshold : cluster size at/below which exhaustive split search is used.
    bisect_method : "sharpe" (default) | "vol"
        "sharpe" — proportional to each cluster's standalone Sharpe ratio.
        "vol"    — inverse cluster variance (standard HRP bisection)
    """
    N = cov.shape[0]
    cov_pd = _ensure_pd(cov)

    mu_arr = np.asarray(mu, dtype=float)
    tree = _build_vb_tree(cov_pd, N, bf_threshold)

    if bisect_method == "vol":
        w_dict = _vb_bisect_vol(tree, 1.0, cov_pd)
    else:
        w_dict = _vb_bisect_sharpe(tree, 1.0, cov_pd, mu_arr, rf)
    w = np.zeros(N)
    for i, wt in w_dict.items():
        w[i] = wt
    s = w.sum()
    w = w / s if s > 1e-12 else np.ones(N) / N

    return w


def vol_hrp_bl_strategy(
    cov_fn: Callable,
    rf: float = 0.0,
    bf_threshold: int = 10,
    bisect_method: str = "sharpe",
    ewma_halflife: Optional[float] = None,
    skip_days: Optional[int] = 21,
    kf_tp: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Factory: (cov_fn, alloc_fn) pair for HMVA with BL returns.

    Parameters
    ----------
    cov_fn        : base covariance estimator, e.g. cov_sample.
    rf            : annual risk-free rate passed to vol_hrp_bl_weights.
    bf_threshold  : exhaustive-search threshold for the tree splitter.
    bisect_method : "sharpe" (default) | "vol"
        "sharpe" — proportional to each cluster's standalone Sharpe ratio.
        "vol"    — inverse cluster variance (standard HRP bisection).
    ewma_halflife : EWMA pseudo-returns halflife in trading days; None = uniform window.
    kf_tp         : if True, smooth weights each period with _kf_smooth_weights.
    """
    cache: Dict = {}
    prev_w: Dict[str, Optional[np.ndarray]] = {"w": None}
    kf_cache: Dict[str, Optional[np.ndarray]] = {"mu_prev": None}

    def _cov_fn(window: np.ndarray) -> np.ndarray:
        data = _ewma_pseudo(window, ewma_halflife) if ewma_halflife is not None else window
        cov = cov_fn(data)
        cache["window"] = window
        return cov

    def _alloc_fn(cov: np.ndarray) -> np.ndarray:
        N = cov.shape[0]
        window = cache.get("window", np.zeros((1, N)))
        mu = mu_BL_momentum(window, cov, skip_days=skip_days)

        w = vol_hrp_bl_weights(
            cov, mu, rf=rf, bf_threshold=bf_threshold,
            bisect_method=bisect_method,
        )

        if kf_tp and prev_w["w"] is not None and kf_cache["mu_prev"] is not None:
            w = _kf_smooth_weights(w, prev_w["w"], cov, mu, kf_cache["mu_prev"])

        prev_w["w"] = w.copy()
        if kf_tp:
            kf_cache["mu_prev"] = mu.copy()

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
    def hrp_with(cov_fn, linkage_method="single"):
        return hrp_strategy(cov_fn, linkage_method=linkage_method, kf_tp=False)
    
    def mvo_with(cov_fn, risk_aversion=2.5):
        return max_utility_strategy(cov_fn, gamma=risk_aversion, kf_tp=False)
    
    def vb_hrp_with(cov_fn,
                    bisect_method,
                    # For portfolio construction always 1 month even for longer rebalances, due to the nature of tree allocations
                    ewma_halflife: float = 21, skip_days: int = 21,
                    kf_tp: bool = True):
        return vol_hrp_bl_strategy(
            cov_fn,
            bisect_method=bisect_method,
            ewma_halflife=ewma_halflife, skip_days=skip_days,
            kf_tp=kf_tp,
        )

    strategies: StrategyMap = {
        "HMVA":     vb_hrp_with(cov_nonlinear_shrink, bisect_method="sharpe"),
        "HMVA-mv":  vb_hrp_with(cov_nonlinear_shrink, bisect_method="vol"),
        "HRP":      hrp_with(cov_ewa_nls),
        "MVO":      mvo_with(cov_ewa_nls),
        "EW":       (cov_sample, equal_weights),
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
# PERFORMANCE METRICS
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
# STATISTICAL TESTS
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
# PLOTTING
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
        annual_returns.png   grouped calendar-year bar chart
        risk_return.png      risk-return scatter with iso-Sharpe lines
        return_dist.png      histogram + KDE per strategy (monthly)
        strategy_corr.png    pairwise return-correlation heatmap
        monthly_<name>.png   calendar heatmap per strategy
        sharpe_bars.png      Sharpe bar chart
        maxdd_bars.png       Max-drawdown bar chart
        turnover_bars.png    Turnover bar chart
        metrics_table.png    colour-coded full metrics table
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


def plot_holdings_concentration(
    weights_log: Dict[str, pd.DataFrame],
    outdir: str,
    top_ks: Tuple[int, ...] = (1, 3, 5, 10),
    title_suffix: str = "",
) -> None:
    """
    Three figures per strategy showing portfolio concentration over time.

    Outputs
    -------
    holdings_effective_n.png
        Effective number of assets (1 / HHI = 1 / Σwᵢ²) for all strategies
        on one axes — higher is more diversified.

    holdings_topk_<name>.png
        Cumulative weight of the top-k assets at each rebalance for a single
        strategy, one line per k value in `top_ks`.
    """
    _ensure_dir(outdir)
    suf = f" {title_suffix}".rstrip() if title_suffix else ""
    sns.set_style("whitegrid")

    # Normalise: backtest_pit returns Dict[str, Dict[date, Series]]; convert to DataFrame
    def _to_df(w) -> pd.DataFrame:
        if isinstance(w, pd.DataFrame):
            return w
        df = pd.DataFrame(list(w.values()), index=list(w.keys())).fillna(0.0)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    weights_log = {name: _to_df(w) for name, w in weights_log.items()}

    # ── 1. Effective N (all strategies on one plot) ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, W in weights_log.items():
        eff_n = 1.0 / (W ** 2).sum(axis=1)
        ax.plot(W.index, eff_n, label=name, lw=1.5)
    ax.set_title(f"Effective N (1 / HHI){suf}")
    ax.set_ylabel("Effective N")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/holdings_effective_n.png", dpi=150)
    plt.close()

    for name, W in weights_log.items():
        safe = name.replace(" ", "_").replace("/", "-")
        n_assets = W.shape[1]

        # ── 2. Top-k cumulative weight ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 4))
        for k in top_ks:
            if k > n_assets:
                continue
            top_k_w = W.apply(lambda row: row.nlargest(k).sum(), axis=1)
            ax.plot(W.index, top_k_w * 100, label=f"Top {k}", lw=1.4)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(f"Top-k Concentration — {name}{suf}")
        ax.set_ylabel("Cumulative weight (%)")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{outdir}/holdings_topk_{safe}.png", dpi=150)
        plt.close()

