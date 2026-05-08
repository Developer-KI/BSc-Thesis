"""
Combined Topic 1 + 2 — Ablation Study then Regime-Switching
============================================================
This is the thesis centrepiece. Two chapters, one script.

Chapter 1 — Ablation: "what works in the pipeline?"
    Build the full XGBoost+MV strategy one component at a time, measuring
    the MARGINAL contribution of each addition to Sharpe, vol, drawdown,
    and concentration (HHI). Isolates whether μ, Σ, or the optimizer
    is driving performance — and at what cost in portfolio volatility.

    A0  1/N equal weight                        (floor)
    A1  MV + historical mean μ + sample cov     (classical Markowitz)
    A2  MV + historical mean μ + Ledoit-Wolf Σ  (+ cov shrinkage)
    A3  MV + raw XGBoost μ + Ledoit-Wolf Σ      (+ ML signal, unfiltered)
    A4  MV + shrunk XGBoost μ + Ledoit-Wolf Σ   (+ forecast regularisation)
    A5  A4 + turnover penalty                   (full pipeline, no regime)
        ← this is your current active strategy

Chapter 2 — Regime-Switching: "can the SJM fix the concentration problem?"
    The ablation (Chapter 1) is expected to show that A5 earns its Sharpe
    at ~2× the volatility of A0. Chapter 2 tests four ways of using the SJM
    bear signal to de-risk in bear markets — WITHOUT touching the return
    forecasts, so the regime effect on Σ and w is cleanly isolated.

    R_VIX   A5 + scale λ by VIX>20            (no SJM — control)
    R1      A5 + scale λ×2 in SJM bear        (your existing regime code)
    R2      bull→full MV (A5), bear→min-var    (hard strategy switch)
    R3      soft blend: w = p·w_minvar+(1−p)·w_mv  (p = rolling bear frac)

    The key comparison: R2 vs R_VIX. If R2 beats R_VIX on Sharpe AND vol,
    the SJM is doing something the VIX cannot — that is the Chapter 2 finding.

Null hypotheses
    H0_1  No pipeline component improves Sharpe over A0 (1/N)
    H0_2  Regime switching does not reduce Ann. vol below A5
    H0_3  R2 (SJM) does not outperform R_VIX on Sharpe
    Tests  Paired t-test · bootstrap Sharpe CI · Jobson-Korkie test
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import yfinance as yf

from research.engine.backtest import (
    RISK_AVERSION, TURNOVER_PENALTY, L2_REG, RISK_FREE_RATE,
    TRAIN_WINDOW_MONTHS, LOOKBACK_COV,
    get_spy_regime_labels,
    fetch_and_engineer_features,
    get_monthly_forecasts,
    shrink_returns,
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
START_DATE      = "2013-01-01"
END_DATE        = "2024-11-30"
FIXED_ALPHA     = 0.25          # shrink factor matching current SHRINK_FACTOR
BEAR_LAMBDA_MULT = 2.0          # risk-aversion multiplier in bear
VIX_THRESHOLD   = 20.0
TC_BPS          = 10
SLIPPAGE_BPS    = 5
LOOKBACK_BEAR_FRAC = 12        # months window for rolling bear fraction
FEATURES = 5


# ═══════════════════════════════════════════════════════════════════════════
# COVARIANCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_sample_covariance(monthly_returns: pd.DataFrame, lookback: int = LOOKBACK_COV) -> dict:
    """Plain sample covariance — used only in A1."""
    cov_dict = {}
    dates = monthly_returns.index
    for i in range(lookback, len(dates)):
        end_date   = dates[i]
        start_date = dates[i - lookback]
        hist = monthly_returns.loc[start_date:end_date].dropna(axis=1, how="any")
        if len(hist) < lookback or hist.shape[1] < 2:
            continue
        cov_dict[end_date] = {"cov": hist.cov().values, "tickers": list(hist.columns)}
    return cov_dict


def _get_historical_mean_forecasts(monthly_returns: pd.DataFrame) -> pd.Series:
    """
    Expanding mean return per ticker — the no-ML μ used in A1, A2.
    At each month t, predict using the mean of all returns up to t-1.
    No look-ahead: expanding window, not in-sample mean.
    """
    records = []
    dates = monthly_returns.index
    for i in range(1, len(dates)):
        t = dates[i]
        hist = monthly_returns.iloc[:i]          # everything before t
        means = hist.mean()                       # expanding mean per ticker
        for ticker, val in means.items():
            records.append({"date": t, "ticker": ticker, "forecast": val})
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    return df["forecast"]


def _rolling_bear_fraction(monthly_regime: pd.Series, lookback: int) -> pd.Series:
    return monthly_regime.rolling(lookback, min_periods=1).mean()


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO OPTIMISERS
# ═══════════════════════════════════════════════════════════════════════════

def _mv_optimise(
    mu: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray | None = None,
    lambda_: float = RISK_AVERSION,
    turnover_penalty: float = 0.0,        # 0 = no turnover cost
    l2_reg: float = 0.0,                  # 0 = no l2
) -> np.ndarray:
    n    = len(mu)
    w_eq = np.ones(n) / n

    def objective(w):
        util = mu @ w - 0.5 * lambda_ * (w @ cov @ w)
        if prev_weights is not None and turnover_penalty > 0:
            util -= turnover_penalty * np.sum(np.abs(w - prev_weights))
        if l2_reg > 0:
            util -= l2_reg * np.sum((w - w_eq) ** 2)
        return -util

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds      = [(0, 1)] * n
    res = minimize(objective, w_eq, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    return res.x if res.success else w_eq


def _min_variance(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio (long-only)."""
    return _mv_optimise(np.zeros(len(cov)), cov, lambda_=1.0,
                        turnover_penalty=0.0, l2_reg=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURABLE BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_configurable_backtest(
    monthly_returns:  pd.DataFrame,
    forecasts:        pd.Series,            # (date, ticker) MultiIndex
    cov_dict:         dict,
    monthly_regime:   pd.Series,            # monthly SJM labels (0=bull,1=bear)
    vix_monthly:      pd.Series | None,
    *,
    use_turnover:     bool  = True,
    use_l2:           bool  = True,
    regime_mode:      str   = "none",       # "none"|"lambda"|"minvar"|"soft"|"vix"
    equal_weight:     bool  = False,        # A0: skip optimisation entirely
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Single backtest function that covers all nine strategies via flags.

    Returns
    -------
    net_returns   pd.Series   monthly net-of-cost portfolio returns
    weights_df    pd.DataFrame  portfolio weights at each decision date
    """
    dates         = sorted(forecasts.index.get_level_values(0).unique())
    port_returns  = []
    weights_list  = []
    weight_dates  = []
    prev_weights  = None

    bear_frac_series = _rolling_bear_fraction(monthly_regime, LOOKBACK_BEAR_FRAC)

    tp  = TURNOVER_PENALTY if use_turnover else 0.0
    l2  = L2_REG           if use_l2       else 0.0

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
        cov_full    = entry["cov"]
        tickers_cov = entry["tickers"]

        common = mu_series.index.intersection(tickers_cov)
        if len(common) < 2:
            continue

        mu      = mu_series.loc[common].values
        idx_map = {t: j for j, t in enumerate(tickers_cov)}
        idx     = [idx_map[t] for t in common]
        cov     = cov_full[np.ix_(idx, idx)]

        # ── Weights ─────────────────────────────────────────────────────────
        n = len(common)

        if equal_weight:
            w_opt = np.ones(n) / n

        elif regime_mode == "minvar":
            # R2: hard switch — full MV in bull, min-variance in bear
            regime = int(monthly_regime.get(date, 0))
            if regime == 1:   # bear
                w_opt = _min_variance(cov)
            else:              # bull
                w_opt = _mv_optimise(mu, cov, prev_weights,
                                     lambda_=RISK_AVERSION,
                                     turnover_penalty=tp, l2_reg=l2)

        elif regime_mode == "soft":
            # R3: soft blend between MV and min-var
            p_bear   = float(bear_frac_series.get(date, 0.0))
            w_mv     = _mv_optimise(mu, cov, prev_weights,
                                    lambda_=RISK_AVERSION,
                                    turnover_penalty=tp, l2_reg=l2)
            w_minvar = _min_variance(cov)
            w_opt    = p_bear * w_minvar + (1 - p_bear) * w_mv
            w_opt    = w_opt / w_opt.sum()   # re-normalise after blend

        elif regime_mode == "lambda":
            # R1: scale risk-aversion up in bear
            regime = int(monthly_regime.get(date, 0))
            lam    = RISK_AVERSION * (BEAR_LAMBDA_MULT if regime == 1 else 1.0)
            w_opt  = _mv_optimise(mu, cov, prev_weights,
                                  lambda_=lam, turnover_penalty=tp, l2_reg=l2)

        elif regime_mode == "vix":
            # R_VIX: scale λ by VIX threshold — no SJM
            vix_val = float(vix_monthly.get(date, 15.0)) if vix_monthly is not None else 15.0
            lam     = RISK_AVERSION * (BEAR_LAMBDA_MULT if vix_val > VIX_THRESHOLD else 1.0)
            w_opt   = _mv_optimise(mu, cov, prev_weights,
                                   lambda_=lam, turnover_penalty=tp, l2_reg=l2)

        else:
            # "none" — plain MV, no regime
            w_opt = _mv_optimise(mu, cov, prev_weights,
                                 lambda_=RISK_AVERSION,
                                 turnover_penalty=tp, l2_reg=l2)

        prev_weights = w_opt

        next_date = dates[i + 1] if i + 1 < len(dates) else None
        if next_date is not None and next_date in monthly_returns.index:
            rets = monthly_returns.loc[next_date][common].values
            port_returns.append(float(w_opt @ rets))
            weights_list.append(dict(zip(common, w_opt)))
            weight_dates.append(date)

    idx_slice  = dates[1 : len(port_returns) + 1]
    gross      = pd.Series(port_returns, index=idx_slice)
    weights_df = pd.DataFrame(weights_list, index=weight_dates).fillna(0)
    net, _     = apply_transaction_costs(gross, weights_df,
                                         tc_bps=TC_BPS, slippage_bps=SLIPPAGE_BPS)
    return net, weights_df


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS  (extended for thesis)
# ═══════════════════════════════════════════════════════════════════════════

def compute_full_metrics(
    ret:            pd.Series,
    weights_df:     pd.DataFrame,
    monthly_regime: pd.Series,
    label:          str,
) -> dict:
    r          = ret.dropna()
    monthly_rf = RISK_FREE_RATE / 12
    ann_ret    = r.mean() * 12
    ann_vol    = r.std()  * np.sqrt(12)
    sharpe     = (r.mean() - monthly_rf) / r.std() * np.sqrt(12)

    cum        = (1 + r).cumprod()
    dd         = (cum - cum.expanding().max()) / cum.expanding().max()
    max_dd     = dd.min()

    # Sortino
    downside   = r[r < monthly_rf] - monthly_rf
    down_std   = np.sqrt((downside**2).mean()) * np.sqrt(12) if len(downside) else np.nan
    sortino    = (ann_ret - RISK_FREE_RATE) / down_std if down_std and down_std > 0 else np.nan

    # CER  (Certainty-Equivalent Return)
    cer        = ann_ret - 0.5 * RISK_AVERSION * ann_vol**2

    # Regime-split realised vol
    reg        = monthly_regime.reindex(r.index).ffill()
    vol_bull   = r[reg == 0].std() * np.sqrt(12) if (reg == 0).sum() > 2 else np.nan
    vol_bear   = r[reg == 1].std() * np.sqrt(12) if (reg == 1).sum() > 2 else np.nan
    ret_bull   = r[reg == 0].mean() * 12         if (reg == 0).sum() > 2 else np.nan
    ret_bear   = r[reg == 1].mean() * 12         if (reg == 1).sum() > 2 else np.nan

    # Concentration (Herfindahl-Hirschman Index)
    hhi        = (weights_df**2).sum(axis=1).mean()

    # Average monthly turnover
    turnover   = weights_df.diff().abs().sum(axis=1).mean()

    return {
        "label":           label,
        "ann_ret":         ann_ret,
        "ann_vol":         ann_vol,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "cer":             cer,
        "max_dd":          max_dd,
        "vol_bull":        vol_bull,
        "vol_bear":        vol_bear,
        "ret_bull":        ret_bull,
        "ret_bear":        ret_bear,
        "hhi":             hhi,
        "avg_turnover":    turnover,
        "n_months":        len(r),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def paired_t_test(r_a: pd.Series, r_b: pd.Series, label: str) -> dict:
    """H0: mean(r_a − r_b) = 0."""
    common = r_a.index.intersection(r_b.index)
    diff   = r_a.loc[common] - r_b.loc[common]
    t, p   = stats.ttest_1samp(diff.dropna(), 0.0)
    return {
        "comparison":    label,
        "t_stat":        round(t, 3),
        "p_value":       round(p, 4),
        "mean_diff_bps": round(diff.mean() * 10_000, 1),
    }


def jobson_korkie_test(r_a: pd.Series, r_b: pd.Series, label: str) -> dict:
    """
    Jobson & Korkie (1981) test for equality of two Sharpe ratios,
    with the Memmel (2003) correction.
    H0: Sharpe(r_a) = Sharpe(r_b).
    """
    common = r_a.index.intersection(r_b.index)
    a, b   = r_a.loc[common].values, r_b.loc[common].values
    n      = len(a)
    rf     = RISK_FREE_RATE / 12

    mu_a, mu_b   = a.mean() - rf, b.mean() - rf
    sig_a, sig_b = a.std(), b.std()
    sr_a,  sr_b  = mu_a / sig_a * np.sqrt(12), mu_b / sig_b * np.sqrt(12)
    rho          = np.corrcoef(a, b)[0, 1]

    # Memmel (2003) standard error for the Sharpe difference
    theta  = (
        (2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2) - rho * sr_a * sr_b * (1 + rho**2) / 2)
        / n
    )
    se     = np.sqrt(theta) if theta > 0 else np.nan
    z_stat = (sr_a - sr_b) / se if se and se > 0 else np.nan
    p_val  = 2 * stats.norm.sf(abs(z_stat)) if not np.isnan(z_stat) else np.nan

    return {
        "comparison":  label,
        "Sharpe A":    round(sr_a, 3),
        "Sharpe B":    round(sr_b, 3),
        "JK z-stat":   round(z_stat, 3) if not np.isnan(z_stat) else "n/a",
        "JK p-value":  round(p_val, 4)  if not np.isnan(p_val)  else "n/a",
    }


def bootstrap_sharpe_ci(
    ret: pd.Series, n_boot: int = 2_000, seed: int = 42
) -> tuple[float, float]:
    rng  = np.random.default_rng(seed)
    r    = ret.dropna().values
    rf   = RISK_FREE_RATE / 12
    boot = [
        (rng.choice(r, len(r), replace=True).mean() - rf)
        / rng.choice(r, len(r), replace=True).std()
        * np.sqrt(12)
        for _ in range(n_boot)
    ]
    # use one resample per iteration properly
    boot = []
    for _ in range(n_boot):
        s = rng.choice(r, len(r), replace=True)
        boot.append((s.mean() - rf) / s.std() * np.sqrt(12))
    return round(np.percentile(boot, 2.5), 3), round(np.percentile(boot, 97.5), 3)


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _pct(x):
    return f"{x*100:+.2f}%" if not (x is None or (isinstance(x, float) and np.isnan(x))) else "n/a"

def _f(x, d=3):
    return f"{x:.{d}f}" if not (x is None or (isinstance(x, float) and np.isnan(x))) else "n/a"


def print_chapter1_table(metrics_list: list[dict], all_rets: dict[str, pd.Series]) -> None:
    print("\n" + "═" * 80)
    print("CHAPTER 1 — ABLATION: MARGINAL CONTRIBUTION TABLE")
    print("═" * 80)
    hdr = (
        f"{'Strategy':<36} {'Sharpe':>7} {'Ann.Vol':>8} {'Ann.Ret':>8} "
        f"{'MaxDD':>7} {'HHI':>6} {'Turnover':>9}"
    )
    print(hdr)
    print("─" * 80)
    for m in metrics_list:
        print(
            f"{m['label']:<36} {_f(m['sharpe']):>7} {_pct(m['ann_vol']):>8} "
            f"{_pct(m['ann_ret']):>8} {_pct(m['max_dd']):>7} "
            f"{_f(m['hhi'],4):>6} {_pct(m['avg_turnover']):>9}"
        )

    # Marginal Sharpe contribution
    print("\nMarginal Sharpe contribution (Δ Sharpe vs previous step):")
    for i in range(1, len(metrics_list)):
        prev, curr = metrics_list[i-1], metrics_list[i]
        delta = curr["sharpe"] - prev["sharpe"]
        print(f"  {prev['label']:<20} → {curr['label']:<20}  ΔSharpe = {delta:+.3f}")

    # Marginal vol contribution
    print("\nMarginal vol contribution (Δ Ann.Vol vs previous step):")
    for i in range(1, len(metrics_list)):
        prev, curr = metrics_list[i-1], metrics_list[i]
        delta = (curr["ann_vol"] - prev["ann_vol"]) * 100
        print(f"  {prev['label']:<20} → {curr['label']:<20}  ΔVol = {delta:+.2f}%")


def print_chapter2_table(metrics_list: list[dict]) -> None:
    print("\n" + "═" * 80)
    print("CHAPTER 2 — REGIME-SWITCHING: CONCENTRATION FIX")
    print("═" * 80)
    hdr = (
        f"{'Strategy':<28} {'Sharpe':>7} {'Ann.Vol':>8} {'MaxDD':>7} "
        f"{'Vol Bull':>9} {'Vol Bear':>9} {'HHI':>6}"
    )
    print(hdr)
    print("─" * 80)
    for m in metrics_list:
        print(
            f"{m['label']:<28} {_f(m['sharpe']):>7} {_pct(m['ann_vol']):>8} "
            f"{_pct(m['max_dd']):>7} {_pct(m['vol_bull']):>9} "
            f"{_pct(m['vol_bear']):>9} {_f(m['hhi'],4):>6}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "A0 — 1/N":                     "#888780",
    "A1 — classical Markowitz":     "#B4B2A9",
    "A2 — + Ledoit-Wolf Σ":         "#85B7EB",
    "A3 — + raw XGBoost μ":         "#378ADD",
    "A4 — + shrunk μ":              "#185FA5",
    "A5 — full pipeline":           "#0C447C",
    "R_VIX — control":              "#EF9F27",
    "R1 — λ scaling":               "#BA7517",
    "R2 — strategy switch":         "#1D9E75",
    "R3 — soft blend":              "#0F6E56",
}


def plot_cumulative(all_rets: dict[str, pd.Series], out: str = "topic12_cumulative.png") -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ablation_keys  = [k for k in all_rets if k.startswith("A")]
    regime_keys    = ["A5 — full pipeline"] + [k for k in all_rets if k.startswith("R")]

    for label, keys, ax, title in [
        ("Chapter 1", ablation_keys,  ax1, "Chapter 1 — Ablation: cumulative returns"),
        ("Chapter 2", regime_keys,    ax2, "Chapter 2 — Regime-switching: cumulative returns"),
    ]:
        for k in keys:
            r   = all_rets[k]
            cum = (1 + r).cumprod()
            ax.plot(cum.index, cum.values, lw=1.5,
                    color=COLORS.get(k, "gray"), label=k)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Cumulative return (level)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    ax2.set_xlabel("Date")
    fig.suptitle("Topic 1+2 — Ablation + Regime-Switching", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sharpe_vol_bars(
    metrics_list: list[dict],
    out: str = "topic12_sharpe_vol.png",
) -> None:
    labels  = [m["label"] for m in metrics_list]
    sharpes = [m["sharpe"] for m in metrics_list]
    vols    = [m["ann_vol"] * 100 for m in metrics_list]
    colors  = [COLORS.get(l, "gray") for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, sharpes, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_title("Sharpe ratio by strategy")
    ax1.axvline(4.5, color="gray", lw=0.8, ls="--")   # divider A5 / regime
    ax1.text(4.55, ax1.get_ylim()[1] * 0.95, "regime →", fontsize=7, color="gray")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, vols, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Ann. volatility (%)")
    ax2.set_title("Annualised volatility by strategy")
    ax2.axvline(4.5, color="gray", lw=0.8, ls="--")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Topic 1+2 — Sharpe and volatility across all strategies", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_hhi_concentration(
    metrics_list: list[dict],
    out: str = "topic12_hhi.png",
) -> None:
    """
    HHI (concentration) vs Sharpe scatter. The thesis argument:
    ML forecasts concentrate the portfolio (high HHI) without proportionally
    improving risk-adjusted returns. Regime-switching should move points
    toward lower HHI without losing Sharpe.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in metrics_list:
        color = COLORS.get(m["label"], "gray")
        ax.scatter(m["hhi"], m["sharpe"], s=90, color=color, zorder=3)
        ax.annotate(
            m["label"].split("—")[0].strip(),
            (m["hhi"], m["sharpe"]),
            textcoords="offset points", xytext=(5, 4),
            fontsize=8, color=color,
        )
    ax.set_xlabel("Avg HHI (portfolio concentration)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Concentration vs Sharpe — all strategies")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Combined Topic 1+2 — Ablation + Regime-Switching")
    print("=" * 80)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[1/6] Downloading data and regime labels...")
    spy_labels = get_spy_regime_labels(start=START_DATE, train_duration=36)

    vix_px = yf.download("^VIX", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)["Close"]
    if vix_px.index.tz is not None:
        vix_px.index = vix_px.index.tz_localize(None)
    vix_monthly = vix_px.resample("ME").last()

    features        = fetch_and_engineer_features(TICKERS, START_DATE, END_DATE)
    monthly_returns = features["ret_1m"].unstack("ticker").sort_index()
    monthly_regime  = spy_labels.resample("ME").last()

    # ── Covariance dictionaries ─────────────────────────────────────────────
    print("[2/6] Building covariance matrices...")
    lw_cov      = get_shrunk_covariance(monthly_returns, lookback=LOOKBACK_COV)
    sample_cov  = _get_sample_covariance(monthly_returns, lookback=LOOKBACK_COV)

    # ── Forecasts ───────────────────────────────────────────────────────────
    print("[3/6] Computing forecasts (historical mean + XGBoost)...")
    hist_mean_fc  = _get_historical_mean_forecasts(monthly_returns)
    raw_xgb_fc    = get_monthly_forecasts(features, train_months=TRAIN_WINDOW_MONTHS, PCA_count=FEATURES)
    shrunk_xgb_fc = shrink_returns(raw_xgb_fc, prior_type="cross_mean",
                                   shrink_factor=FIXED_ALPHA)

    # ── CHAPTER 1: Ablation ─────────────────────────────────────────────────
    print("\n[4/6] Chapter 1 — running ablation strategies A0–A5...")

    ablation_specs = [
        # (label,                 forecasts,      cov,        equal_w, use_to, use_l2, regime_mode)
        ("A0 — 1/N",              hist_mean_fc,   lw_cov,     True,   False, False, "none"),
        ("A1 — classical Markowitz", hist_mean_fc, sample_cov, False,  False, False, "none"),
        ("A2 — + Ledoit-Wolf Σ",  hist_mean_fc,   lw_cov,     False,  False, False, "none"),
        ("A3 — + raw XGBoost μ",  raw_xgb_fc,     lw_cov,     False,  False, False, "none"),
        ("A4 — + shrunk μ",       shrunk_xgb_fc,  lw_cov,     False,  False, False, "none"),
        ("A5 — full pipeline",    shrunk_xgb_fc,  lw_cov,     False,  True,  True,  "none"),
    ]

    all_rets:   dict[str, pd.Series]  = {}
    all_wts:    dict[str, pd.DataFrame] = {}
    ch1_metrics: list[dict]           = []

    for label, fc, cov_d, eq_w, to, l2, reg in ablation_specs:
        print(f"  {label}...")
        ret, wts = run_configurable_backtest(
            monthly_returns, fc, cov_d, monthly_regime, vix_monthly,
            use_turnover=to, use_l2=l2,
            regime_mode=reg, equal_weight=eq_w,
        )
        all_rets[label] = ret
        all_wts[label]  = wts
        m = compute_full_metrics(ret, wts, monthly_regime, label)
        ci_lo, ci_hi = bootstrap_sharpe_ci(ret)
        m["sharpe_ci"] = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        ch1_metrics.append(m)
        print(f"    Sharpe={m['sharpe']:.3f} {m['sharpe_ci']}  "
              f"Vol={m['ann_vol']*100:.1f}%  HHI={m['hhi']:.4f}")

    # ── CHAPTER 2: Regime-Switching ─────────────────────────────────────────
    print("\n[5/6] Chapter 2 — running regime-switching strategies...")

    regime_specs = [
        # (label,                 regime_mode)
        ("R_VIX — control",       "vix"),
        ("R1 — λ scaling",        "lambda"),
        ("R2 — strategy switch",  "minvar"),
        ("R3 — soft blend",       "soft"),
    ]

    ch2_metrics: list[dict] = [ch1_metrics[-1]]   # include A5 as reference

    for label, reg in regime_specs:
        print(f"  {label}...")
        ret, wts = run_configurable_backtest(
            monthly_returns, shrunk_xgb_fc, lw_cov, monthly_regime, vix_monthly,
            use_turnover=True, use_l2=True,
            regime_mode=reg, equal_weight=False,
        )
        all_rets[label] = ret
        all_wts[label]  = wts
        m = compute_full_metrics(ret, wts, monthly_regime, label)
        ci_lo, ci_hi = bootstrap_sharpe_ci(ret)
        m["sharpe_ci"] = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        ch2_metrics.append(m)
        print(f"    Sharpe={m['sharpe']:.3f} {m['sharpe_ci']}  "
              f"Vol={m['ann_vol']*100:.1f}%  "
              f"VolBull={m['vol_bull']*100:.1f}%  VolBear={m['vol_bear']*100:.1f}%")

    # ── Statistical tests ────────────────────────────────────────────────────
    print("\n[6/6] Statistical tests...")

    # H0_1: does A5 beat A0?
    print("\n── H0_1: A5 (full pipeline) vs A0 (1/N) ──")
    t1 = paired_t_test(all_rets["A5 — full pipeline"], all_rets["A0 — 1/N"],
                       "A5 vs A0")
    jk1 = jobson_korkie_test(all_rets["A5 — full pipeline"], all_rets["A0 — 1/N"],
                             "A5 vs A0")
    for d in [t1, jk1]:
        for k, v in d.items():
            print(f"  {k:<35s} {v}")

    # H0_2: does R2 reduce vol vs A5?
    print("\n── H0_2: R2 (strategy switch) vs A5 (full pipeline) ──")
    t2 = paired_t_test(all_rets["R2 — strategy switch"], all_rets["A5 — full pipeline"],
                       "R2 vs A5")
    jk2 = jobson_korkie_test(all_rets["R2 — strategy switch"], all_rets["A5 — full pipeline"],
                             "R2 vs A5")
    for d in [t2, jk2]:
        for k, v in d.items():
            print(f"  {k:<35s} {v}")

    # H0_3: does R2 (SJM) beat R_VIX?
    print("\n── H0_3: R2 (SJM switch) vs R_VIX (no SJM) ──")
    t3 = paired_t_test(all_rets["R2 — strategy switch"], all_rets["R_VIX — control"],
                       "R2 vs R_VIX")
    jk3 = jobson_korkie_test(all_rets["R2 — strategy switch"], all_rets["R_VIX — control"],
                             "R2 vs R_VIX")
    for d in [t3, jk3]:
        for k, v in d.items():
            print(f"  {k:<35s} {v}")

    # ── Print tables ─────────────────────────────────────────────────────────
    print_chapter1_table(ch1_metrics, all_rets)
    print_chapter2_table(ch2_metrics)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    all_metrics = ch1_metrics + ch2_metrics[1:]   # A5 not duplicated
    plot_cumulative(all_rets)
    plot_sharpe_vol_bars(all_metrics)
    plot_hhi_concentration(all_metrics)

    # ── Thesis interpretation guide ──────────────────────────────────────────
    print("\n" + "═" * 80)
    print("THESIS INTERPRETATION GUIDE")
    print("═" * 80)
    print("""
Chapter 1 — what to look for in the marginal contribution table:
  A0→A1: if ΔSharpe < 0, classical Markowitz destroys value vs 1/N
          (common finding — estimation error in μ and Σ hurts)
  A1→A2: if ΔSharpe > 0, Ledoit-Wolf adds value by taming Σ noise
  A2→A3: if ΔSharpe > 0 AND ΔVol spikes, raw XGBoost adds return
          but concentrates the portfolio (high HHI)
  A3→A4: if ΔSharpe > 0 and ΔVol falls, shrinkage tames the noise
  A4→A5: if ΔSharpe ≈ 0, turnover penalty reduces costs without
          hurting performance

Chapter 2 — what to look for in regime-switching results:
  R1 (λ scaling): if ΔVol < 0 and ΔSharpe ≈ 0 vs A5, risk-aversion
                   scaling de-risks without harming returns
  R2 (strategy switch): if VolBear(R2) < VolBear(A5), the SJM
                   successfully identifies bear states and switches to
                   min-variance — the 'fix' for the concentration problem
  R3 (soft blend): if Sharpe(R3) > Sharpe(R2), smooth transitions
                   add value over hard switches
  R_VIX vs R2:    if JK p-value > 0.10, you cannot reject that SJM
                   and VIX are equally good regime signals — that is
                   also a publishable finding (SJM not uniquely valuable)
""")


if __name__ == "__main__":
    main()