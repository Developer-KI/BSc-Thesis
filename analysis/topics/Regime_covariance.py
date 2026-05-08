"""
Topic 4 — Regime-Aware Covariance Estimation
=============================================
Drop this file next to backtest.py and run it directly.

It adds one new covariance estimator and wires four strategies through
the existing backtest machinery:

  S1  Pooled Ledoit-Wolf (your current baseline)
  S2  Hard-switch: Σ_bull when bull, Σ_bear when bear
  S3  Soft-blend:  p·Σ_bull + (1−p)·Σ_bear  (p = rolling bear fraction)
  S4  VIX-threshold baseline (pooled LW, λ scaled by VIX, no SJM)

Evaluation has two layers:
  Layer 1 — covariance quality  (Mincer-Zarnowitz, Diebold-Mariano)
  Layer 2 — portfolio quality   (Sharpe, vol, drawdown, HHI, regime-split vol)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

import analysis.engine.backtest as bt
bt.PRIOR_TYPE = "historical_mean"
bt.RETURN_SMOOTHING = "shrink"
bt.SHRINK_FACTOR = 1.0


# ── re-use everything already in your project ──────────────────────────────
from analysis.engine.backtest import (
    RISK_AVERSION, RISK_FREE_RATE,
    TRAIN_WINDOW_MONTHS, SHRINK_FACTOR, PRIOR_TYPE, RETURN_SMOOTHING,
    get_spy_regime_labels,
    fetch_and_engineer_features,
    get_monthly_forecasts,
    process_forecasts,
    optimize_portfolio
)


# ═══════════════════════════════════════════════════════════════════════════
# TICKERS — keep identical to your existing backtest
# ═══════════════════════════════════════════════════════════════════════════
TICKERS = ['AAPL', 'ABT', 'ADBE', 'ADI', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'APH', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BMY', 'C', 'CAT', 'CB', 'CMCSA', 'CME', 'CMI', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DE', 'DHR', 'DIS', 'DUK', 'ETN', 'FDX', 'GD', 'GE', 'GILD', 'GLW', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU', 'ISRG', 'JCI', 'JNJ', 'JPM', 'KLAC', 'KO', 'LLY', 'LMT', 'LOW', 'MA', 'MAR', 'MCD', 'MCK', 'MDT', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PLD', 'PM', 'PNC', 'PWR', 'QCOM', 'SBUX', 'SCHW', 'SO', 'SPGI', 'SYK', 'T', 'TJX', 'TMO', 'TMUS', 'TXN', 'UNH', 'UNP', 'UPS', 'V', 'VZ', 'WDC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'XOM']
START_DATE     = "2011-01-01"
END_DATE       = "2026-01-01"
LOOKBACK_COV   = 12   # months of history for each covariance window
MIN_REGIME_OBS = 6    # minimum months in a regime state to compute LW
VIX_THRESHOLD  = 20.0 # for S4 VIX-baseline
BEAR_LAMBDA_MULT = 1.0 # risk-aversion multiplier in bear (used by S2/S3/S4)


# ═══════════════════════════════════════════════════════════════════════════
# NEW: REGIME-CONDITIONAL COVARIANCE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════

def _rolling_bear_fraction(monthly_regime: pd.Series, lookback: int) -> pd.Series:
    """
    At each month-end date, compute the fraction of days in the prior
    `lookback` months that were labeled bear (state == 1).
    Returns a monthly Series indexed the same as monthly_regime.
    """
    # monthly_regime is already resampled to month-end; treat each entry as 1 obs
    return monthly_regime.rolling(lookback, min_periods=1).mean()


def get_regime_conditional_covariance(
    monthly_returns: pd.DataFrame,
    monthly_regime: pd.Series,
    lookback: int = LOOKBACK_COV,
    min_obs: int = MIN_REGIME_OBS,
) -> dict[pd.Timestamp, dict]:
    """
    For each month t, partition the lookback window into bear months and
    bull months, fit a separate Ledoit-Wolf estimator on each, and return
    a blended covariance for both hard-switch (S2) and soft-blend (S3).

    Also returns the pooled (S1) covariance computed on the same window so
    all four covariance types are available from one function.

    Returns
    -------
    dict keyed by date, each value is a dict with:
        'tickers'      : list of tickers used (intersection of available data)
        'cov_pooled'   : (N,N) ndarray — standard Ledoit-Wolf on all months
        'cov_bull'     : (N,N) ndarray — LW on bull months (or pooled fallback)
        'cov_bear'     : (N,N) ndarray — LW on bear months (or pooled fallback)
        'bear_frac'    : float in [0,1] — fraction of bear months in window
        'n_bull'       : int — number of bull months used
        'n_bear'       : int — number of bear months used
    """
    dates = monthly_returns.index
    result = {}

    for i in range(lookback, len(dates)):
        end_date   = dates[i]
        start_date = dates[i - lookback]

        # Window of returns — drop any ticker with missing data
        window = (
            monthly_returns
            .loc[start_date:end_date]
            .dropna(axis=1, how="any")
        )
        if window.shape[0] < lookback or window.shape[1] < 2:
            continue

        # Align regime labels to this window
        regime_window = monthly_regime.reindex(window.index).ffill().fillna(0)

        bull_mask = (regime_window == 0)
        bear_mask = (regime_window == 1)
        n_bull = int(bull_mask.sum())
        n_bear = int(bear_mask.sum())
        bear_frac = n_bear / max(n_bull + n_bear, 1)

        tickers = list(window.columns)

        # ── Pooled LW (same as your existing baseline) ─────────────────────
        lw_pooled  = LedoitWolf().fit(window.values)
        cov_pooled = lw_pooled.covariance_

        # ── Regime-specific LW, with fallback to pooled when too few obs ───
        if n_bull >= min_obs:
            lw_bull  = LedoitWolf().fit(window[bull_mask].values)
            cov_bull = lw_bull.covariance_
        else:
            cov_bull = cov_pooled   # fallback: not enough bull data

        if n_bear >= min_obs:
            lw_bear  = LedoitWolf().fit(window[bear_mask].values)
            cov_bear = lw_bear.covariance_
        else:
            cov_bear = cov_pooled   # fallback: not enough bear data

        result[end_date] = {
            "tickers":    tickers,
            "cov_pooled": cov_pooled,
            "cov_bull":   cov_bull,
            "cov_bear":   cov_bear,
            "bear_frac":  bear_frac,
            "n_bull":     n_bull,
            "n_bear":     n_bear,
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST LOOP — accepts cov_mode to switch between the four strategies
# ═══════════════════════════════════════════════════════════════════════════

def run_regime_cov_backtest(
    monthly_returns:  pd.DataFrame,
    shrunk_forecasts: pd.Series,
    cov_dict:         dict,
    spy_labels:       pd.Series,
    vix_monthly:      pd.Series | None = None,
    cov_mode:         str = "pooled",  # "pooled" | "hard" | "soft" | "vix"
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Run the walk-forward backtest with one of four covariance modes.

    Parameters
    ----------
    cov_mode : str
        'pooled' — S1: standard Ledoit-Wolf (baseline)
        'hard'   — S2: select Σ_bull or Σ_bear based on current SJM label
        'soft'   — S3: blend Σ_bull and Σ_bear by rolling bear fraction
        'vix'    — S4: pooled LW + scale λ by VIX threshold (no regime Σ)

    Returns
    -------
    port_ret      : pd.Series  monthly portfolio returns
    bench_ret     : pd.Series  monthly 1/N returns
    weights_df    : pd.DataFrame  weights at each rebalance date
    cov_forecast  : pd.DataFrame  predicted portfolio variance per period
                    (used for Layer 1 Mincer-Zarnowitz evaluation)
    """
    dates         = sorted(shrunk_forecasts.index.get_level_values(0).unique())
    port_returns  = []
    bench_returns = []
    weights_records = []
    weights_dates = []
    cov_pred_records = []   # (date, predicted_port_var)
    prev_weights  = None

    monthly_regime = spy_labels.resample("ME").last()
    bear_frac_series = _rolling_bear_fraction(monthly_regime, lookback=LOOKBACK_COV)

    for i, date in enumerate(dates):
        try:
            mu_series = shrunk_forecasts.loc[date]
        except KeyError:
            continue

        # Find the most recent available covariance entry
        cov_date = date if date in cov_dict else max(
            (d for d in cov_dict if d <= date), default=None
        )
        if cov_date is None:
            continue

        entry       = cov_dict[cov_date]
        tickers_cov = entry["tickers"]

        common = mu_series.index.intersection(tickers_cov)
        if len(common) < 2:
            continue

        mu      = mu_series.loc[common].values
        idx_map = {t: j for j, t in enumerate(tickers_cov)}
        idx     = [idx_map[t] for t in common]

        # ── Select / blend covariance matrix ───────────────────────────────
        regime = int(monthly_regime.get(date, 0))

        if cov_mode == "pooled":
            cov = entry["cov_pooled"][np.ix_(idx, idx)]
            lam = RISK_AVERSION

        elif cov_mode == "hard":
            # Hard-switch: pick the state-specific matrix
            raw_cov = entry["cov_bear"] if regime == 1 else entry["cov_bull"]
            cov     = raw_cov[np.ix_(idx, idx)]
            lam     = RISK_AVERSION * (BEAR_LAMBDA_MULT if regime == 1 else 1.0)

        elif cov_mode == "soft":
            # Soft-blend: weight by rolling bear fraction
            p_bear  = float(bear_frac_series.get(date, entry["bear_frac"]))
            cov_raw = (
                p_bear       * entry["cov_bear"]
                + (1 - p_bear) * entry["cov_bull"]
            )
            cov = cov_raw[np.ix_(idx, idx)]
            lam = RISK_AVERSION * (1.0 + (BEAR_LAMBDA_MULT - 1.0) * p_bear)

        elif cov_mode == "vix":
            # VIX-threshold baseline: pooled Σ, λ scaled by VIX
            cov = entry["cov_pooled"][np.ix_(idx, idx)]
            if vix_monthly is not None:
                vix_val = float(vix_monthly.get(date, 15.0))
                lam = RISK_AVERSION * (BEAR_LAMBDA_MULT if vix_val > VIX_THRESHOLD else 1.0)
            else:
                lam = RISK_AVERSION
        else:
            raise ValueError(f"Unknown cov_mode: {cov_mode!r}")

        w_opt = optimize_portfolio(mu, cov, prev_weights, lambda_=lam)
        prev_weights = w_opt
        w_bench = np.ones(len(common)) / len(common)

        # Predicted portfolio variance for Layer 1 evaluation
        pred_port_var = float(w_opt @ cov @ w_opt)
        cov_pred_records.append({"date": date, "pred_port_var": pred_port_var})

        next_date = dates[i + 1] if i + 1 < len(dates) else None
        if next_date is not None and next_date in monthly_returns.index:
            rets_next = monthly_returns.loc[next_date][common].values
            port_returns.append(np.dot(w_opt,   rets_next))
            bench_returns.append(np.dot(w_bench, rets_next))
            weights_records.append(dict(zip(common, w_opt)))
            weights_dates.append(date)

    idx_slice   = dates[1 : len(port_returns) + 1]
    port_ret    = pd.Series(port_returns,  index=idx_slice)
    bench_ret   = pd.Series(bench_returns, index=idx_slice)
    weights_df  = pd.DataFrame(weights_records, index=weights_dates).fillna(0)
    cov_forecast = pd.DataFrame(cov_pred_records).set_index("date")
    return port_ret, bench_ret, weights_df, cov_forecast


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — COVARIANCE QUALITY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def layer1_covariance_quality(
    port_ret: pd.Series,
    cov_forecast: pd.DataFrame,
    label: str,
) -> dict:
    """
    Compare predicted portfolio variance to realized variance.

    Mincer-Zarnowitz (MZ) regression:
        realized_var_t = a + b * predicted_var_{t-1} + ε_t
    Ideal: a ≈ 0, b ≈ 1.

    Also returns MSE of variance forecast and the Pearson correlation.
    """
    from scipy import stats

    # Realized monthly variance: (return - mean)^2 approximation
    realized_var = port_ret.pow(2)   # using raw squared return as proxy

    # Align: predicted at t predicts realized at t+1
    pred  = cov_forecast["pred_port_var"].shift(1)
    common_idx = realized_var.index.intersection(pred.dropna().index)

    if len(common_idx) < 10:
        print(f"[{label}] Not enough aligned observations for Layer 1 evaluation.")
        return {}

    y = realized_var.loc[common_idx].values
    x = pred.loc[common_idx].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    mse   = np.mean((y - x) ** 2)
    corr  = np.corrcoef(x, y)[0, 1]

    return {
        "MZ intercept (a)":     round(intercept, 6),
        "MZ slope (b)":         round(slope, 4),
        "MZ R²":                round(r_value ** 2, 4),
        "MZ p-value (b=1 H0)":  round(p_value, 4),
        "Forecast MSE":         round(mse, 8),
        "Pred-Realized corr.":  round(corr, 4),
        "N observations":       len(common_idx),
    }


def diebold_mariano_test(
    port_ret: pd.Series,
    cov_forecast_s1: pd.DataFrame,
    cov_forecast_sx: pd.DataFrame,
    label_sx: str,
) -> dict:
    """
    Diebold-Mariano test: H0 = S1 and Sx have equal forecast accuracy.
    Uses squared error loss on portfolio variance.
    """
    from scipy import stats

    realized_var = port_ret.pow(2)
    pred_s1 = cov_forecast_s1["pred_port_var"].shift(1)
    pred_sx = cov_forecast_sx["pred_port_var"].shift(1)

    common_idx = (
        realized_var.index
        .intersection(pred_s1.dropna().index)
        .intersection(pred_sx.dropna().index)
    )
    if len(common_idx) < 10:
        return {}

    y   = realized_var.loc[common_idx].values
    e1  = (y - pred_s1.loc[common_idx].values) ** 2
    ex  = (y - pred_sx.loc[common_idx].values) ** 2
    d   = e1 - ex   # positive d means Sx is better (lower error)

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    n   = len(d)
    dm_stat = d.mean() / (d.std() / np.sqrt(n))
    p_val   = 2 * stats.t.sf(abs(dm_stat), df=n - 1)

    return {
        f"DM stat (S1 vs {label_sx})": round(dm_stat, 3),
        f"DM p-value":                 round(p_val, 4),
        f"Mean loss diff (S1 − {label_sx})": round(d.mean(), 8),
        f"Sx better (lower loss)?":    "yes" if d.mean() > 0 else "no",
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — PORTFOLIO QUALITY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def layer2_portfolio_quality(
    port_ret: pd.Series,
    weights_df: pd.DataFrame,
    monthly_regime: pd.Series,
    label: str,
) -> dict:
    """
    Extended portfolio metrics with regime-split realized volatility
    and portfolio concentration (Herfindahl-Hirschman Index).
    """
    monthly_rf = RISK_FREE_RATE / 12
    r          = port_ret.dropna()

    ann_return = r.mean() * 12
    ann_vol    = r.std()  * np.sqrt(12)
    sharpe     = (r.mean() - monthly_rf) / r.std() * np.sqrt(12)

    cumulative  = (1 + r).cumprod()
    running_max = cumulative.expanding().max()
    max_dd      = ((cumulative - running_max) / running_max).min()

    # Regime-split realized volatility
    regime_aligned = monthly_regime.reindex(r.index).ffill()
    bull_r = r[regime_aligned == 0]
    bear_r = r[regime_aligned == 1]
    vol_bull = bull_r.std() * np.sqrt(12) if len(bull_r) > 2 else np.nan
    vol_bear = bear_r.std() * np.sqrt(12) if len(bear_r) > 2 else np.nan

    # Portfolio concentration: average HHI across rebalance dates
    hhi_series = (weights_df ** 2).sum(axis=1)
    avg_hhi    = hhi_series.mean()

    # Average monthly turnover
    turnover_series = weights_df.diff().abs().sum(axis=1)
    avg_turnover    = turnover_series.mean()

    return {
        "Ann. Return (%)":       round(ann_return * 100, 2),
        "Ann. Volatility (%)":   round(ann_vol    * 100, 2),
        "Sharpe Ratio":          round(sharpe, 3),
        "Max Drawdown (%)":      round(max_dd   * 100, 2),
        "Realized vol — bull":   round(vol_bull  * 100, 2) if not np.isnan(vol_bull) else "n/a",
        "Realized vol — bear":   round(vol_bear  * 100, 2) if not np.isnan(vol_bear) else "n/a",
        "Vol ratio bear/bull":   round(vol_bear / vol_bull, 2) if (not np.isnan(vol_bull) and vol_bull > 0) else "n/a",
        "Avg HHI (concentration)": round(avg_hhi, 4),
        "Avg monthly turnover":  round(avg_turnover, 4),
        "N months":              len(r),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — wire everything together and print results
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Topic 4 — Regime-Aware Covariance Estimation")
    print("=" * 65)

    # ── Step 1: data + regime labels ────────────────────────────────────────
    print("\n[1/5] Downloading data and generating regime labels...")
    spy_labels = get_spy_regime_labels(start=START_DATE, train_duration=36)

    features = fetch_and_engineer_features(TICKERS, START_DATE, END_DATE)
    monthly_returns = (
        features["ret_1m"]
        .unstack("ticker")
        .sort_index()
    )

    # VIX (needed for S4)
    vix_px = yf.download("^VIX", start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)["Close"]
    if vix_px.index.tz is not None:
        vix_px.index = vix_px.index.tz_localize(None)
    vix_monthly = vix_px.resample("ME").last()

    # ── Step 2: forecasts (shared across all strategies) ────────────────────
    print("[2/5] Computing XGBoost walk-forward forecasts...")
    raw_forecasts   = get_monthly_forecasts(features, train_months=TRAIN_WINDOW_MONTHS)
    shrunk_forecasts = process_forecasts(
        raw_forecasts,
        mode=RETURN_SMOOTHING,
        prior_type=PRIOR_TYPE,
        shrink_factor=SHRINK_FACTOR,
    )

    # ── Step 3: compute all four covariance dictionaries ────────────────────
    print("[3/5] Estimating regime-conditional covariance matrices...")
    monthly_regime = spy_labels.resample("ME").last()
    cov_dict = get_regime_conditional_covariance(
        monthly_returns,
        monthly_regime,
        lookback=LOOKBACK_COV,
    )

    # ── Step 4: run backtests for S1 – S4 ───────────────────────────────────
    print("[4/5] Running four backtests...")
    strategies = {
        "S1 — Pooled LW (baseline)": "pooled",
        "S2 — Hard switch":          "hard",
        "S3 — Soft blend":           "soft",
        "S4 — VIX baseline":         "vix",
    }

    results = {}
    for name, mode in strategies.items():
        print(f"  Running {name}...")
        port_ret, bench_ret, weights_df, cov_forecast = run_regime_cov_backtest(
            monthly_returns  = monthly_returns,
            shrunk_forecasts = shrunk_forecasts,
            cov_dict         = cov_dict,
            spy_labels       = spy_labels,
            vix_monthly      = vix_monthly,
            cov_mode         = mode,
        )
        results[name] = {
            "port_ret":    port_ret,
            "bench_ret":   bench_ret,
            "weights_df":  weights_df,
            "cov_forecast": cov_forecast,
        }

    # ── Step 5: evaluate ────────────────────────────────────────────────────
    print("[5/5] Evaluating results...\n")

    # ── Layer 1 ─────────────────────────────────────────────────────────────
    print("─" * 65)
    print("LAYER 1 — COVARIANCE QUALITY (Mincer-Zarnowitz)")
    print("─" * 65)
    s1_cov_forecast = results["S1 — Pooled LW (baseline)"]["cov_forecast"]
    s1_port_ret     = results["S1 — Pooled LW (baseline)"]["port_ret"]

    for name, res in results.items():
        mz = layer1_covariance_quality(res["port_ret"], res["cov_forecast"], name)
        print(f"\n{name}")
        for k, v in mz.items():
            print(f"  {k:<35s} {v}")
        if name != "S1 — Pooled LW (baseline)":
            dm = diebold_mariano_test(
                s1_port_ret,
                s1_cov_forecast,
                res["cov_forecast"],
                label_sx=name,
            )
            for k, v in dm.items():
                print(f"  {k:<35s} {v}")

    # ── Layer 2 ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("LAYER 2 — PORTFOLIO QUALITY")
    print("─" * 65)
    for name, res in results.items():
        metrics = layer2_portfolio_quality(
            res["port_ret"],
            res["weights_df"],
            monthly_regime,
            label=name,
        )
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k:<35s} {v}")

    # ── Compact comparison table ─────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("SUMMARY TABLE")
    print("═" * 65)
    rows = []
    for name, res in results.items():
        m = layer2_portfolio_quality(
            res["port_ret"], res["weights_df"], monthly_regime, name
        )
        mz = layer1_covariance_quality(res["port_ret"], res["cov_forecast"], name)
        rows.append({
            "Strategy":        name,
            "Sharpe":          m.get("Sharpe Ratio"),
            "Ann. Vol (%)":    m.get("Ann. Volatility (%)"),
            "Max DD (%)":      m.get("Max Drawdown (%)"),
            "Vol bear/bull":   m.get("Vol ratio bear/bull"),
            "Avg HHI":         m.get("Avg HHI (concentration)"),
            "MZ slope (b)":    mz.get("MZ slope (b)"),
            "MZ R²":           mz.get("MZ R²"),
        })

    summary = pd.DataFrame(rows).set_index("Strategy")
    print(summary.to_string())

    print("\nDone. Interpret results using the framework in your thesis:")
    print("  H0: regime-conditional Σ does not reduce forecast variance error vs S1")
    print("  Reject if: DM p-value < 0.10 AND MZ slope(Sx) closer to 1.0 than S1")
    print("  Economic significance: Sharpe(Sx) − Sharpe(S1) with same μ")


if __name__ == "__main__":
    main()