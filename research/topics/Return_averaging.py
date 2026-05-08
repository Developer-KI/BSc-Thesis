"""
Predictor Combination for μ: Does Forecast Diversification Help?
=================================================================
BSc Thesis — Erasmus University Rotterdam, Econometrics and OR
Building on Hop (2017) "Efficient Portfolio Selection in a Large Market"
Supervisor: Sander Barendse

Central analogy
---------------
DeMiguel, Garlappi & Uppal (2009) show that 1/N equal-weight beats
optimised portfolio weights under estimation error in weight space.
This thesis tests the parallel question in FORECAST space:

    Does 1/K equal-weighted forecast combination beat the best single
    return predictor, just as 1/N beats optimised portfolios?

Three experimental layers
--------------------------
Layer 1 — Simulation
    Use the same FF3 factor-model DGP as Hop (2017, section 4.1).
    Add a known cross-sectional signal (true R²≈1%) to one predictor.
    Vary N ∈ {25,50,100} and T ∈ {60,120,240}. Test whether 1/K
    combination recovers the signal better than the best single predictor.

Layer 2 — Statistical evaluation (empirical, 44 S&P 500 stocks)
    Compute OOS cross-sectional R² and hit rate per predictor and per
    combination rule. Diebold-Mariano test for forecast accuracy
    differences.

Layer 3 — Economic evaluation
    Feed each forecast combination into the SAME MV optimizer and
    Ledoit-Wolf Σ (same as backtest.py). Report Sharpe, CER, turnover,
    and HHI concentration. Jobson-Korkie test for Sharpe differences.

Combination methods
-------------------
C0  Historical mean              — zero-signal R²≈0 baseline
P1–P7  Single predictors        — one per characteristic
C1  1/K equal-weighted           — the "1/N of forecasts" (main hypothesis)
C2  OOS-R²-weighted              — weight ∝ rolling 12m R²_k, clip at 0
C3  PCA of predicted returns     — 1st PC (μ-space analog of Hop's subspace)
C4  Adaptive best-single         — select predictor with highest recent R²

Predictors (cross-sectional OLS per month)
    P1 mom_3m      3-month price momentum
    P2 mom_6m      6-month price momentum
    P3 mom_12m     12-month price momentum  (skip-month convention: shift 1m)
    P4 vol_21d     21-day realised volatility (negative expected sign)
    P5 idio_vol    Idiosyncratic volatility
    P6 amihud      Amihud illiquidity ratio
    P7 ep_ratio    Earnings-to-price ratio

References
----------
DeMiguel et al. (2009) RFS — 1/N portfolio result
Welch & Goyal (2008) RFS — predictability puzzle
Campbell & Thompson (2008) RFS — sign restrictions
Hop (2017) Erasmus BSc — Σ-side baseline, same simulation DGP
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from research.engine.backtest import (
    RISK_AVERSION, TURNOVER_PENALTY, L2_REG, RISK_FREE_RATE,
    LOOKBACK_COV,
    fetch_and_engineer_features,
    get_shrunk_covariance,
    apply_transaction_costs,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","JNJ","V",
    "PG","UNH","HD","MA","DIS","BAC","XOM","PFE","KO","PEP",
    "INTC","CSCO","NFLX","ADBE","CRM","WMT","ABT","TMO","ACN","NKE",
    "AVGO","TXN","QCOM","MDT","COST","NEE","LIN","DHR","BMY","AMGN",
    "LOW","HON","UPS","RTX",
]
START_DATE = "2013-01-01"
END_DATE   = "2024-11-30"

# Predictor columns in the feature DataFrame (must exist after engineering)
PREDICTOR_COLS = {
    "P1_mom_3m":  "mom_3m",
    "P2_mom_6m":  "mom_6m",
    "P3_mom_12m": "mom_12m",
    "P4_vol_21d": "vol_21d",
    "P5_idio_vol": "idiosyncratic_vol",
    "P6_amihud":  "amihud_illiq",
}
# E/P computed separately below where available

TC_BPS        = 10
SLIPPAGE_BPS  = 5
R2_WINDOW     = 12        # months of rolling OOS R² for C2 and C4
MIN_R2_OBS    = 6         # minimum months before computing rolling R²
N_BOOT        = 2_000     # bootstrap iterations for Sharpe CI


# ═══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# LAYER 1 — SIMULATION (FF3 DGP, same as Hop 2017 section 4.1)
# ══════════════════════════════════════════════════════════════════

def simulate_ff3_returns(
    N: int,
    T: int,
    true_signal_r2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate N×T panel of stock returns from a three-factor model,
    with ONE known cross-sectional characteristic x carrying signal
    calibrated so E[R²_XS] ≈ true_signal_r2.

    Returns
    -------
    returns     (T, N) array of simulated excess returns
    chars       (T, N) array of the characteristic x_{i,t}

    DGP (exactly Hop 2017, equation 17 extended with characteristic):
        r_{i,t} = β_M * r_M,t + β_SMB * SMB_t + β_HML * HML_t
                  + γ * x_{i,t}    ← the signal to be recovered
                  + ε_{i,t}

    Factor parameters drawn from Ken French monthly data moments.
    β loadings uniform on Hop's bounds (section 4.1).
    """
    rng = np.random.default_rng(seed)

    # Factor moments (from July 1963 – Aug 2007 monthly FF3, Hop's calibration)
    factor_mean = np.array([0.006, 0.002, 0.004])   # MKT-RF, SMB, HML
    factor_cov  = np.array([
        [0.00195, 0.00023, -0.00018],
        [0.00023, 0.00110,  0.00012],
        [-0.00018, 0.00012,  0.00098],
    ])

    # Factor loadings (Hop's uniform bounds)
    beta_M   = rng.uniform(0.9, 1.2, N)
    beta_SMB = rng.uniform(-0.3, 1.4, N)
    beta_HML = rng.uniform(-0.5, 0.9, N)
    B = np.stack([beta_M, beta_SMB, beta_HML], axis=1)   # (N, 3)

    # Idiosyncratic variances
    sig2_eps = rng.uniform(0.1, 0.3, N)

    # Simulate T+1 months of factors and characteristics
    F = rng.multivariate_normal(factor_mean, factor_cov, T + 1)  # (T+1, 3)
    chars = rng.standard_normal((T + 1, N))   # characteristic x_{i,t} ~ N(0,1)

    # Calibrate γ so that R²_XS ≈ target
    # Var(γ*x) / (Var(γ*x) + Var(factor component) + Var(ε)) ≈ r²
    # γ² ≈ target_r2 * (Var(factor terms) + mean sig2_eps) / (1 - target_r2)
    var_factor  = float(np.mean([B[i] @ factor_cov @ B[i] for i in range(N)]))
    var_eps     = float(np.mean(sig2_eps))
    gamma = np.sqrt(true_signal_r2 * (var_factor + var_eps) / (1 - true_signal_r2 + 1e-8))

    eps = np.stack([rng.normal(0, np.sqrt(s), T + 1) for s in sig2_eps], axis=1)
    returns_full = F @ B.T + gamma * chars + eps  # (T+1, N)

    # Training: returns[0..T-1] use chars[0..T-1] to predict returns[1..T]
    chars_   = chars[:T]       # x_{i,t} used at month t to predict t+1
    returns_ = returns_full[1:T+1]   # realized at t+1
    return returns_, chars_, gamma


def run_simulation(
    N_list:   list[int] = (25, 50, 100),
    T_list:   list[int] = (60, 120, 240),
    n_sims:   int       = 500,
    true_r2:  float     = 0.01,
) -> pd.DataFrame:
    """
    For each (N, T) pair, run n_sims Monte Carlo iterations.
    In each iteration:
      - Simulate returns + characteristic (true signal γ known)
      - Walk-forward OOS: for each month t in [T_train, T], train on
        t observations, predict month t+1 using univariate OLS on x.
      - Also form 1/K combination (K=1 here — test with the known predictor)
        vs C0 historical mean.
    Reports: mean OOS R²_XS and mean Sharpe for each method.

    In a multi-predictor extension, add K-1 noise predictors with γ=0,
    then test whether 1/K equal combination still recovers the signal.
    """
    records = []
    train_window = 36   # mirror TRAIN_WINDOW_MONTHS from backtest.py

    for N in N_list:
        for T in T_list:
            for sim in range(n_sims):
                returns, chars, gamma = simulate_ff3_returns(
                    N=N, T=T, true_signal_r2=true_r2, seed=sim * 31 + N + T
                )

                # Also create K-1 noise predictors (γ=0) to form 1/K combination
                rng_noise = np.random.default_rng(sim * 17 + N)
                noise_chars = rng_noise.standard_normal((T, N, 5))  # 5 noise predictors

                # Walk-forward OLS
                xs_r2_signal, xs_r2_combo, xs_r2_hist = [], [], []
                port_signal, port_combo, port_hist = [], [], []

                for t in range(train_window, T - 1):
                    # ── Training data ────────────────────────────────────────
                    x_train_s  = chars[:t, :]    # (t, N)
                    r_train    = returns[:t, :]  # (t, N) — realized returns

                    # Stack: each (month, stock) is an observation
                    X_s = x_train_s.reshape(-1, 1)
                    y   = r_train.reshape(-1)

                    # Fit univariate OLS: r_{i,τ+1} = α + β * x_{i,τ}
                    X_aug = np.column_stack([np.ones(len(X_s)), X_s])
                    try:
                        coef_s = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                    except Exception:
                        continue

                    # Noise predictors: fit and get predicted returns
                    noise_preds = []
                    for k in range(noise_chars.shape[2]):
                        xn_train = noise_chars[:t, :, k].reshape(-1, 1)
                        Xn_aug   = np.column_stack([np.ones(len(xn_train)), xn_train])
                        try:
                            coef_n = np.linalg.lstsq(Xn_aug, y, rcond=None)[0]
                        except Exception:
                            coef_n = np.array([0.0, 0.0])
                        mu_n = coef_n[0] + coef_n[1] * noise_chars[t, :, k]
                        noise_preds.append(mu_n)

                    # ── Predict for month t ──────────────────────────────────
                    mu_signal = coef_s[0] + coef_s[1] * chars[t, :]  # (N,)
                    mu_hist   = r_train.mean(axis=0)                  # expanding mean
                    # 1/K combination: average signal predictor + noise predictors
                    all_preds  = [mu_signal] + noise_preds
                    mu_combo   = np.mean(all_preds, axis=0)

                    # ── Realized returns ─────────────────────────────────────
                    r_real = returns[t + 1, :]  # (N,)

                    # OOS cross-sectional R²
                    def xs_r2(mu_hat, r):
                        ss_res = np.sum((r - mu_hat) ** 2)
                        ss_tot = np.sum((r - r.mean()) ** 2)
                        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                    xs_r2_signal.append(xs_r2(mu_signal, r_real))
                    xs_r2_combo.append(xs_r2(mu_combo, r_real))
                    xs_r2_hist.append(xs_r2(mu_hist, r_real))

                    # Simple portfolio: long top-K, short bottom-K (long-only: top half)
                    for mu_hat, store in [
                        (mu_signal, port_signal),
                        (mu_combo,  port_combo),
                        (mu_hist,   port_hist),
                    ]:
                        w = np.zeros(N)
                        top = np.argsort(mu_hat)[-max(1, N // 4):]
                        w[top] = 1.0 / len(top)
                        store.append(float(w @ r_real))

                if not xs_r2_signal:
                    continue

                def sharpe(rets):
                    r = np.array(rets)
                    return r.mean() / (r.std() + 1e-8) * np.sqrt(12)

                records.append({
                    "N": N, "T": T, "sim": sim,
                    "r2_signal":  np.nanmean(xs_r2_signal),
                    "r2_combo":   np.nanmean(xs_r2_combo),
                    "r2_hist":    np.nanmean(xs_r2_hist),
                    "sharpe_signal": sharpe(port_signal),
                    "sharpe_combo":  sharpe(port_combo),
                    "sharpe_hist":   sharpe(port_hist),
                })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — CROSS-SECTIONAL OLS PER PREDICTOR
# (Statistical evaluation: OOS R² and hit rate)
# ═══════════════════════════════════════════════════════════════════════════

def _xs_ols_predict(
    x_cross: np.ndarray,
    y_cross: np.ndarray,
    x_pred:  np.ndarray,
) -> np.ndarray:
    """
    Fit cross-sectional OLS on (x_cross, y_cross) and predict for x_pred.
    All arrays are 1-D of length N (cross-section at one date).
    Returns predicted values (N,).
    """
    mask = np.isfinite(x_cross) & np.isfinite(y_cross)
    if mask.sum() < 5:
        return np.full(len(x_pred), np.nan)
    X  = np.column_stack([np.ones(mask.sum()), x_cross[mask]])
    y  = y_cross[mask]
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    X_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
    return X_pred @ coef


def compute_all_forecasts(
    features: pd.DataFrame,
    predictor_cols: dict[str, str],
    train_months: int = 36,
) -> dict[str, pd.Series]:
    """
    For each predictor in predictor_cols and for each month t:
      1. Train cross-sectional OLS on the past train_months of (x_k, r_{t+1}) pairs
         (cross-sectional regression per month, pooled over lookback)
      2. Predict r_{i,t+1} from x_{k,i,t}

    Also computes the historical mean forecast (C0 baseline).

    Returns dict: label → pd.Series with (date, ticker) MultiIndex.
    """
    ret_pivot  = features["ret_1m"].unstack("ticker").sort_index()
    dates      = ret_pivot.index
    forecasts  = {label: [] for label in predictor_cols}
    forecasts["C0_hist_mean"] = []

    for i in range(train_months, len(dates) - 1):
        forecast_date = dates[i]
        target_date   = dates[i + 1]

        if target_date not in ret_pivot.index:
            continue

        # Training window: [i-train_months, i-1], target is one step ahead
        train_dates  = dates[i - train_months : i]
        train_rets   = ret_pivot.loc[dates[i - train_months + 1] : target_date]

        # C0: cross-sectional mean of past train_months returns per ticker
        hist_mean = ret_pivot.loc[train_dates].mean()
        tickers   = ret_pivot.columns
        hist_fc   = hist_mean.reindex(tickers)
        forecasts["C0_hist_mean"].append(
            hist_fc.rename_axis("ticker").to_frame(name="fc")
            .assign(date=forecast_date).set_index("date", append=True)
            .swaplevel()["fc"]
        )

        # Per-predictor: pool cross-sections over the lookback window
        for label, col in predictor_cols.items():
            if col not in features.columns:
                continue

            # Build pooled training matrix: rows = (date, ticker) in lookback
            x_pool, y_pool = [], []
            for td in train_dates[:-1]:   # x at td predicts ret at td+1
                td_next = dates[dates.get_loc(td) + 1]
                if td_next not in ret_pivot.index:
                    continue
                xvals = features.loc[features.index.get_level_values("date") == td, col]
                yvals = ret_pivot.loc[td_next]
                common = xvals.index.get_level_values("ticker").intersection(yvals.index)
                if len(common) < 5:
                    continue
                xv = xvals.reindex(pd.MultiIndex.from_tuples(
                    [(td, t) for t in common], names=["date","ticker"])).values
                yv = yvals.loc[common].values
                finite = np.isfinite(xv) & np.isfinite(yv)
                x_pool.extend(xv[finite])
                y_pool.extend(yv[finite])

            if len(x_pool) < 10:
                continue

            x_arr, y_arr = np.array(x_pool), np.array(y_pool)
            coef, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(len(x_arr)), x_arr]),
                y_arr, rcond=None
            )

            # Predict for forecast_date cross-section
            xf = features.loc[features.index.get_level_values("date") == forecast_date, col]
            tickers_fc = xf.index.get_level_values("ticker")
            mu_hat = coef[0] + coef[1] * xf.values
            fc_series = pd.Series(
                mu_hat,
                index=pd.MultiIndex.from_tuples(
                    [(forecast_date, t) for t in tickers_fc],
                    names=["date","ticker"]
                ),
                name="fc",
            )
            forecasts[label].append(fc_series)

    # Concatenate
    out = {}
    for label, parts in forecasts.items():
        if parts:
            out[label] = pd.concat(parts)
    return out


def compute_oos_r2_and_hitrate(
    forecasts: dict[str, pd.Series],
    realized:  pd.Series,
) -> pd.DataFrame:
    """
    For each forecast series compute:
        OOS R²_XS  cross-sectional R² per month, averaged
        Hit rate   sign(forecast) == sign(realized)
        N obs      total aligned observations
    """
    records = []
    for label, fc in forecasts.items():
        df = pd.DataFrame({"fc": fc, "realized": realized}).dropna()
        if df.empty:
            continue

        r2_vals, hit_vals = [], []
        for date, grp in df.groupby(level="date"):
            y    = grp["realized"].values
            yhat = grp["fc"].values
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot > 0:
                r2_vals.append(1.0 - ss_res / ss_tot)
            hit_vals.extend((np.sign(yhat) == np.sign(y)).tolist())

        records.append({
            "label":    label,
            "r2_xs":    float(np.nanmean(r2_vals)),
            "hit_rate": float(np.mean(hit_vals)),
            "n_months": len(r2_vals),
            "n_obs":    len(df),
        })
    return pd.DataFrame(records).set_index("label")


# ═══════════════════════════════════════════════════════════════════════════
# COMBINATION RULES  (C1–C4)
# ═══════════════════════════════════════════════════════════════════════════

def _rolling_r2(fc: pd.Series, realized: pd.Series, window: int = R2_WINDOW) -> pd.Series:
    """Compute rolling OOS R²_XS for one forecast series."""
    df = pd.DataFrame({"fc": fc, "realized": realized}).dropna()
    r2_by_date = {}
    for date, grp in df.groupby(level="date"):
        y, yh = grp["realized"].values, grp["fc"].values
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_by_date[date] = (1.0 - np.sum((y - yh)**2) / ss_tot) if ss_tot > 0 else np.nan
    r2_series = pd.Series(r2_by_date).sort_index()
    return r2_series.rolling(window, min_periods=MIN_R2_OBS).mean()


def build_combination_forecasts(
    single_forecasts: dict[str, pd.Series],
    realized:         pd.Series,
    predictor_labels: list[str],
) -> dict[str, pd.Series]:
    """
    Build C1–C4 from the single-predictor forecasts.

    C1  Equal-weighted (1/K)
    C2  OOS-R²-weighted (rolling 12m R², clip at 0)
    C3  PCA of predicted returns (1st PC)
    C4  Adaptive best-single (predictor with highest recent R²)
    """
    preds = {k: single_forecasts[k] for k in predictor_labels if k in single_forecasts}
    if not preds:
        return {}

    # Align all predictors to a common (date, ticker) index
    all_dates = sorted(
        set.intersection(*[set(p.index.get_level_values("date")) for p in preds.values()])
    )
    combos = {}

    # Pre-compute rolling R² per predictor
    rolling_r2 = {
        k: _rolling_r2(v, realized) for k, v in preds.items()
    }

    c1_parts, c2_parts, c3_parts, c4_parts = [], [], [], []

    for date in all_dates:
        slices = {}
        for k, fc in preds.items():
            try:
                slices[k] = fc.xs(date, level="date")
            except KeyError:
                pass
        if not slices:
            continue

        tickers = sorted(set.intersection(*[set(s.index) for s in slices.values()]))
        if not tickers:
            continue

        mat = np.stack([slices[k].reindex(tickers).values for k in slices], axis=1)  # (N, K)
        idx = pd.MultiIndex.from_tuples([(date, t) for t in tickers], names=["date","ticker"])

        # C1: equal weight
        c1_parts.append(pd.Series(mat.mean(axis=1), index=idx))

        # C2: OOS-R²-weighted
        w = np.array([max(rolling_r2[k].get(date, 0.0) or 0.0, 0.0) for k in slices])
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(len(slices)) / len(slices)
        c2_parts.append(pd.Series(mat @ w, index=idx))

        # C3: PCA first component (if enough stocks)
        if mat.shape[0] >= 5 and mat.shape[1] >= 2:
            try:
                pca    = PCA(n_components=1).fit(mat)
                scores = pca.transform(mat).flatten()
                # align sign with average forecast
                if np.corrcoef(scores, mat.mean(axis=1))[0,1] < 0:
                    scores = -scores
                c3_parts.append(pd.Series(scores, index=idx))
            except Exception:
                pass

        # C4: adaptive best-single (highest rolling R² at this date)
        best_k  = max(slices.keys(),
                      key=lambda k: rolling_r2[k].get(date, -np.inf) or -np.inf)
        c4_parts.append(pd.Series(slices[best_k].reindex(tickers).values, index=idx))

    if c1_parts: combos["C1_equal_weight"] = pd.concat(c1_parts)
    if c2_parts: combos["C2_r2_weighted"]  = pd.concat(c2_parts)
    if c3_parts: combos["C3_pca"]          = pd.concat(c3_parts)
    if c4_parts: combos["C4_adaptive"]     = pd.concat(c4_parts)
    return combos


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — PORTFOLIO EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def _mv_optimise(
    mu: np.ndarray, cov: np.ndarray, prev_w: np.ndarray | None
) -> np.ndarray:
    n   = len(mu)
    w0  = np.ones(n) / n

    def neg_utility(w):
        util = mu @ w - 0.5 * RISK_AVERSION * (w @ cov @ w)
        if prev_w is not None:
            util -= TURNOVER_PENALTY * np.sum(np.abs(w - prev_w))
        util -= L2_REG * np.sum((w - w0) ** 2)
        return -util

    res = minimize(neg_utility, w0, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    return res.x if res.success else w0


def run_mv_backtest(
    monthly_returns: pd.DataFrame,
    forecasts:       pd.Series,
    cov_dict:        dict,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run MV backtest with given forecast series. Returns (net_returns, weights_df)."""
    dates        = sorted(forecasts.index.get_level_values("date").unique())
    port_rets    = []
    weights_list = []
    weight_dates = []
    prev_w       = None

    for i, date in enumerate(dates):
        try:
            mu_series = forecasts.loc[date]
        except KeyError:
            continue

        cov_date = date if date in cov_dict else max(
            (d for d in cov_dict if d <= date), default=None
        )
        if cov_date is None:
            continue

        entry       = cov_dict[cov_date]
        tickers_cov = entry["tickers"]
        cov_full    = entry["cov"]

        common = mu_series.index.intersection(tickers_cov)
        if len(common) < 2:
            continue

        mu      = mu_series.loc[common].values
        idx_map = {t: j for j, t in enumerate(tickers_cov)}
        idx     = [idx_map[t] for t in common]
        cov     = cov_full[np.ix_(idx, idx)]

        w_opt = _mv_optimise(mu, cov, prev_w)
        prev_w = w_opt

        next_date = dates[i + 1] if i + 1 < len(dates) else None
        if next_date is not None and next_date in monthly_returns.index:
            rets = monthly_returns.loc[next_date][common].values
            port_rets.append(float(w_opt @ rets))
            weights_list.append(dict(zip(common, w_opt)))
            weight_dates.append(date)

    idx_slice  = dates[1 : len(port_rets) + 1]
    gross      = pd.Series(port_rets, index=idx_slice)
    weights_df = pd.DataFrame(weights_list, index=weight_dates).fillna(0)
    net, _     = apply_transaction_costs(gross, weights_df,
                                         tc_bps=TC_BPS, slippage_bps=SLIPPAGE_BPS)
    return net, weights_df


def compute_portfolio_metrics(
    ret:        pd.Series,
    weights_df: pd.DataFrame,
    label:      str,
) -> dict:
    r          = ret.dropna()
    monthly_rf = RISK_FREE_RATE / 12
    ann_ret    = r.mean() * 12
    ann_vol    = r.std()  * np.sqrt(12)
    sharpe     = (r.mean() - monthly_rf) / r.std() * np.sqrt(12)
    cer        = ann_ret - 0.5 * RISK_AVERSION * ann_vol ** 2
    cum        = (1 + r).cumprod()
    max_dd     = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    hhi        = (weights_df ** 2).sum(axis=1).mean()
    turnover   = weights_df.diff().abs().sum(axis=1).mean()
    return {
        "label":         label,
        "ann_ret_pct":   round(ann_ret  * 100, 2),
        "ann_vol_pct":   round(ann_vol  * 100, 2),
        "sharpe":        round(sharpe, 3),
        "cer_pct":       round(cer      * 100, 2),
        "max_dd_pct":    round(max_dd   * 100, 2),
        "avg_hhi":       round(hhi, 4),
        "avg_turnover":  round(turnover, 4),
        "n_months":      len(r),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def diebold_mariano(
    fc_a: pd.Series, fc_b: pd.Series,
    realized: pd.Series, label: str
) -> dict:
    """
    Diebold-Mariano test: H0 = equal MSE for fc_a and fc_b.
    Positive DM stat means fc_a has higher MSE (fc_b is better).
    Uses cross-sectional squared error averaged per month.
    """
    df_a = pd.DataFrame({"fc": fc_a, "r": realized}).dropna()
    df_b = pd.DataFrame({"fc": fc_b, "r": realized}).dropna()

    e2_a, e2_b, dates_common = {}, {}, []
    for date in df_a.index.get_level_values("date").unique():
        if date not in df_b.index.get_level_values("date").unique():
            continue
        ga = df_a.xs(date, level="date")
        gb = df_b.xs(date, level="date")
        common = ga.index.intersection(gb.index)
        if len(common) < 3:
            continue
        e2_a[date] = np.mean((ga.loc[common, "r"] - ga.loc[common, "fc"]) ** 2)
        e2_b[date] = np.mean((gb.loc[common, "r"] - gb.loc[common, "fc"]) ** 2)
        dates_common.append(date)

    if len(dates_common) < 5:
        return {"comparison": label, "DM stat": "n/a", "DM p-value": "n/a"}

    d    = np.array([e2_a[d] - e2_b[d] for d in dates_common])
    n    = len(d)
    t_dm = d.mean() / (d.std() / np.sqrt(n))
    p    = 2 * stats.t.sf(abs(t_dm), df=n - 1)
    return {
        "comparison":       label,
        "DM stat":          round(t_dm, 3),
        "DM p-value":       round(p, 4),
        "mean loss diff A-B bps": round(d.mean() * 1e4, 3),
        "A worse than B?":  "yes" if d.mean() > 0 else "no",
    }


def jobson_korkie(r_a: pd.Series, r_b: pd.Series, label: str) -> dict:
    """Jobson-Korkie (1981) + Memmel (2003) correction for Sharpe ratio test."""
    common = r_a.index.intersection(r_b.index)
    a, b   = r_a.loc[common].values, r_b.loc[common].values
    rf     = RISK_FREE_RATE / 12
    n      = len(a)
    sa_m, sb_m = a.mean() - rf, b.mean() - rf
    sig_a,sig_b= a.std(), b.std()
    sr_a,  sr_b= sa_m / sig_a * np.sqrt(12), sb_m / sig_b * np.sqrt(12)
    rho        = np.corrcoef(a, b)[0, 1]
    theta      = (2 - 2*rho + 0.5*(sr_a**2 + sr_b**2)
                  - rho*sr_a*sr_b*(1 + rho**2)/2) / n
    se         = np.sqrt(max(theta, 1e-12))
    z          = (sr_a - sr_b) / se
    p          = 2 * stats.norm.sf(abs(z))
    return {
        "comparison":  label,
        "Sharpe A":    round(sr_a, 3),
        "Sharpe B":    round(sr_b, 3),
        "JK z-stat":   round(z, 3),
        "JK p-value":  round(p, 4),
    }


def bootstrap_sharpe_ci(ret: pd.Series, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    r   = ret.dropna().values
    rf  = RISK_FREE_RATE / 12
    boot = []
    for _ in range(n_boot):
        s = rng.choice(r, len(r), replace=True)
        boot.append((s.mean() - rf) / max(s.std(), 1e-8) * np.sqrt(12))
    return round(np.percentile(boot, 2.5), 3), round(np.percentile(boot, 97.5), 3)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_simulation_results(sim_df: pd.DataFrame, out: str = "sim_r2_sharpe.png"):
    """Replicate Hop's Figure 2 style but for R² and Sharpe across (N,T) grid."""
    Ns = sorted(sim_df["N"].unique())
    Ts = sorted(sim_df["T"].unique())
    fig, axes = plt.subplots(2, len(Ns), figsize=(5 * len(Ns), 8))

    for col, N in enumerate(Ns):
        sub = sim_df[sim_df["N"] == N].groupby("T").mean()

        # Row 0: OOS R²
        ax = axes[0, col]
        ax.plot(Ts, [sub.loc[T, "r2_signal"] * 100 for T in Ts],
                "o-", color="#185FA5", label="single predictor", lw=1.5)
        ax.plot(Ts, [sub.loc[T, "r2_combo"] * 100 for T in Ts],
                "s--", color="#1D9E75", label="1/K combination", lw=1.5)
        ax.plot(Ts, [sub.loc[T, "r2_hist"] * 100 for T in Ts],
                "x:", color="#888780", label="hist. mean", lw=1.2)
        ax.set_title(f"N={N}", fontsize=10)
        ax.set_xlabel("T (months)")
        if col == 0: ax.set_ylabel("OOS R²_XS (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Row 1: Sharpe
        ax = axes[1, col]
        ax.plot(Ts, [sub.loc[T, "sharpe_signal"] for T in Ts],
                "o-", color="#185FA5", label="single predictor", lw=1.5)
        ax.plot(Ts, [sub.loc[T, "sharpe_combo"] for T in Ts],
                "s--", color="#1D9E75", label="1/K combination", lw=1.5)
        ax.plot(Ts, [sub.loc[T, "sharpe_hist"] for T in Ts],
                "x:", color="#888780", label="hist. mean", lw=1.2)
        ax.set_xlabel("T (months)")
        if col == 0: ax.set_ylabel("Sharpe ratio")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Simulation — OOS R² and Sharpe: signal vs 1/K combination vs hist. mean",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_r2_bar(r2_df: pd.DataFrame, out: str = "r2_comparison.png"):
    """OOS R² bar chart per strategy, with hit-rate overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    labels  = r2_df.index.tolist()
    colors  = {
        "C0_hist_mean":    "#888780",
        "C1_equal_weight": "#1D9E75",
        "C2_r2_weighted":  "#0F6E56",
        "C3_pca":          "#7F77DD",
        "C4_adaptive":     "#BA7517",
    }
    bar_colors = [
        colors.get(l, "#378ADD") for l in labels
    ]
    x = np.arange(len(labels))
    ax1.bar(x, r2_df["r2_xs"] * 100, color=bar_colors, edgecolor="white", lw=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("OOS R²_XS (%)"); ax1.set_title("OOS cross-sectional R²")
    ax1.axhline(0, color="gray", lw=0.5, ls="--"); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, r2_df["hit_rate"] * 100, color=bar_colors, edgecolor="white", lw=0.5)
    ax2.axhline(50, color="gray", lw=0.5, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Hit rate (%)"); ax2.set_title("Sign accuracy (hit rate)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Statistical evaluation — OOS R² and hit rate by forecast strategy", fontsize=11)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_portfolio_comparison(
    metrics_list: list[dict], all_rets: dict[str, pd.Series],
    out_prefix: str = "portfolio"
):
    """Cumulative returns + Sharpe bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors
    for i, (label, ret) in enumerate(all_rets.items()):
        cum = (1 + ret).cumprod()
        ax1.plot(cum.index, cum.values, lw=1.5, color=colors[i % 10], label=label)
    ax1.set_title("Cumulative returns by forecast strategy")
    ax1.set_ylabel("Level")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    labels  = [m["label"] for m in metrics_list]
    sharpes = [m["sharpe"] for m in metrics_list]
    ax2.barh(labels, sharpes, color=colors[:len(labels)], edgecolor="white")
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_title("Sharpe ratio by forecast strategy")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Economic evaluation — portfolio performance", fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_economic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_prefix}_economic.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Predictor Combination for μ — BSc Thesis Experiment")
    print("=" * 70)

    # ── Layer 1: Simulation ──────────────────────────────────────────────────
    print("\n[1/4] Layer 1 — simulation (FF3 DGP, mirroring Hop 2017 section 4.1)...")
    print("      Running 500 Monte Carlo iterations per (N,T) cell...")
    print("      This takes a few minutes — reduce n_sims=50 for quick test.\n")

    sim_df = run_simulation(
        N_list  = [25, 50, 100],
        T_list  = [60, 120, 240],
        n_sims  = 50,      # set to 50 for quick test
        true_r2 = 0.01,
    )

    print("  Simulation summary (means across simulations):")
    sim_summary = sim_df.groupby(["N","T"])[
        ["r2_signal","r2_combo","r2_hist","sharpe_signal","sharpe_combo","sharpe_hist"]
    ].mean()
    print(sim_summary.round(4).to_string())
    plot_simulation_results(sim_df)

    print("\n  Key question: is r2_combo > r2_signal for high N/T?")
    for (N, T), row in sim_summary.iterrows():
        direction = "✓ combo > signal" if row["r2_combo"] > row["r2_signal"] else "✗ signal wins"
        print(f"    N={N:3d}, T={T:3d}: {direction}  "
              f"(Δ = {(row['r2_combo']-row['r2_signal'])*100:+.3f}%)")

    # ── Data download ────────────────────────────────────────────────────────
    print("\n[2/4] Downloading data and engineering features...")
    features        = fetch_and_engineer_features(TICKERS, START_DATE, END_DATE)
    monthly_returns = features["ret_1m"].unstack("ticker").sort_index()
    cov_dict        = get_shrunk_covariance(monthly_returns, lookback=LOOKBACK_COV)

    ret_pivot = features["ret_1m"].unstack("ticker")
    realized  = ret_pivot.shift(-1).stack().rename("realized")
    realized.index.names = ["date", "ticker"]

    # ── Compute single-predictor forecasts ───────────────────────────────────
    print("[3/4] Computing forecasts (single predictors + combinations)...")
    available_preds = {
        k: v for k, v in PREDICTOR_COLS.items()
        if v in features.columns
    }
    all_forecasts = compute_all_forecasts(features, available_preds, train_months=36)
    predictor_labels = [k for k in all_forecasts if k != "C0_hist_mean"]

    # Combination forecasts
    combos = build_combination_forecasts(all_forecasts, realized, predictor_labels)
    all_forecasts.update(combos)
    all_labels = list(all_forecasts.keys())
    print(f"  Forecast series available: {all_labels}")

    # ── Layer 2: Statistical evaluation ─────────────────────────────────────
    print("\n  Layer 2 — OOS R² and hit rate:")
    r2_df = compute_oos_r2_and_hitrate(all_forecasts, realized)
    print(r2_df[["r2_xs","hit_rate","n_months"]].round(4).to_string())
    plot_r2_bar(r2_df)

    # Diebold-Mariano: C1 vs C0 and C1 vs best single predictor
    best_single = max(
        predictor_labels,
        key=lambda k: r2_df.loc[k, "r2_xs"] if k in r2_df.index else -np.inf
    )
    print(f"\n  Best single predictor by OOS R²: {best_single}")

    if "C0_hist_mean" in all_forecasts and "C1_equal_weight" in all_forecasts:
        dm1 = diebold_mariano(all_forecasts["C0_hist_mean"],
                              all_forecasts["C1_equal_weight"],
                              realized, "C0 vs C1 (H0_stat)")
        print(f"\n  DM test (C0 hist mean vs C1 equal-weighted):")
        for k, v in dm1.items(): print(f"    {k:<35s} {v}")

    if best_single in all_forecasts and "C1_equal_weight" in all_forecasts:
        dm2 = diebold_mariano(all_forecasts[best_single],
                              all_forecasts["C1_equal_weight"],
                              realized, f"{best_single} vs C1")
        print(f"\n  DM test (best single vs C1 equal-weighted):")
        for k, v in dm2.items(): print(f"    {k:<35s} {v}")

    # ── Layer 3: Economic evaluation ─────────────────────────────────────────
    print("\n[4/4] Layer 3 — portfolio evaluation...")
    all_rets, all_wts, all_metrics = {}, {}, []

    for label, fc in all_forecasts.items():
        print(f"  Running MV backtest: {label}...")
        net, wts = run_mv_backtest(monthly_returns, fc, cov_dict)
        all_rets[label] = net
        all_wts[label]  = wts
        m = compute_portfolio_metrics(net, wts, label)
        ci_lo, ci_hi = bootstrap_sharpe_ci(net)
        m["sharpe_ci"] = f"[{ci_lo:.2f},{ci_hi:.2f}]"
        all_metrics.append(m)
        print(f"    Sharpe={m['sharpe']:.3f} {m['sharpe_ci']}  "
              f"Vol={m['ann_vol_pct']:.1f}%  CER={m['cer_pct']:.2f}%  "
              f"HHI={m['avg_hhi']:.4f}")

    # Jobson-Korkie tests
    print("\n  Jobson-Korkie Sharpe tests:")
    baseline_ret = all_rets.get("C0_hist_mean")
    c1_ret       = all_rets.get("C1_equal_weight")

    if baseline_ret is not None and c1_ret is not None:
        jk1 = jobson_korkie(c1_ret, baseline_ret, "C1 vs C0 (H0_econ_1)")
        print(f"\n  H0_econ_1: C1 Sharpe = C0 Sharpe?")
        for k, v in jk1.items(): print(f"    {k:<25s} {v}")

    if best_single in all_rets and c1_ret is not None:
        jk2 = jobson_korkie(c1_ret, all_rets[best_single],
                            f"C1 vs {best_single} (H0_econ_2)")
        print(f"\n  H0_econ_2: C1 Sharpe = best-single Sharpe?")
        for k, v in jk2.items(): print(f"    {k:<25s} {v}")

    # Summary table
    print("\n" + "═" * 70)
    print("SUMMARY TABLE")
    print("═" * 70)
    summary = pd.DataFrame(all_metrics).set_index("label")[
        ["sharpe","sharpe_ci","ann_vol_pct","cer_pct","max_dd_pct","avg_hhi","avg_turnover"]
    ]
    print(summary.to_string())

    # Figures
    print("\nGenerating figures...")
    plot_portfolio_comparison(all_metrics, all_rets)

    print("\n" + "═" * 70)
    print("INTERPRETATION GUIDE FOR THESIS WRITING")
    print("═" * 70)
    print("""
Layer 1 (simulation):
  If r2_combo > r2_signal for large N/T:
    → Combination reduces estimation noise in the forecast, paralleling
      DeMiguel et al.'s result in weight space. Write this as the
      simulation confirmation of your central analogy.
  If r2_combo ≤ r2_signal:
    → The noise predictors dilute the signal. Discuss optimal K selection
      (keep only positive-R² predictors → this is C2, not C1).

Layer 2 (statistical):
  DM test C0 vs C1: if p < 0.10, any forecast adds statistical value
  DM test best-single vs C1: if p < 0.10 and C1 has lower MSE,
    combination beats the best single predictor statistically.

Layer 3 (economic):
  JK test C0 vs C1: if p < 0.10, the forecast signal has economic value
    beyond the historical mean (the minimum bar).
  JK test C1 vs best-single: this is the thesis's core result.
    If C1 wins → forecast diversification replicates DeMiguel in μ space.
    If C1 loses → optimal predictor selection (C4) or R²-weighting (C2)
    is required, and simple equal-weighting is not sufficient.

Connection to Hop (for your related literature section):
  C3 (PCA combination) is the μ-space analog of Hop's subspace MV.
  Hop restricts the investment universe to the leading eigenvectors of Σ.
  C3 restricts the return forecast to the leading PC of predicted returns.
  Compare C3 vs C1 directly: structured combination vs naive equal-weight.
""")


if __name__ == "__main__":
    main()