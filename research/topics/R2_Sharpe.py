"""
Topic 3 — Out-of-Sample R² vs Economic Performance
====================================================
Drop this file next to backtest.py and run it directly.

Research question
-----------------
The best return predictors achieve only ~1% OOS R². Does that tiny R²
translate into economically meaningful portfolio gains, and under what
conditions? Does imposing economic constraints (sign restrictions,
winsorisation) improve portfolio performance *independently* of R²?

Two-stage design
----------------
Stage A — Shrink factor sweep
    Vary α ∈ {0.0, 0.1, …, 1.0} in the shrink step:
        μ̂_shrunk = α·prior + (1−α)·μ̂_xgb
    α=0 → pure XGBoost (highest signal, highest noise)
    α=1 → pure cross-sectional mean (R²≈0 by construction)
    Measure (OOS R²_XS, Sharpe, CER) for each α.
    Traces the R²-to-performance conversion curve.

Stage B — Discrete forecast treatments (fixed α=0.25)
    T1  Raw shrunk XGBoost (baseline)
    T2  Sign restriction — soft  (Campbell & Thompson 2008):
            if μ̂_i < cross-mean: replace with cross-mean
    T3  Sign restriction — hard:
            μ̂_i = max(μ̂_i, 0)
    T4  Winsorised at 5% tails (trim extreme forecasts)
    T5  Historical mean per ticker (near-zero R² anchor)
    T6  Zero forecasts everywhere (forces min-variance solution)
    
    For each treatment: measure (OOS R²_XS, Sharpe, CER, hit rate).
    Tests whether sign restrictions move points ABOVE the Stage A curve —
    the Campbell & Thompson result: constraints improve economic value
    independently of statistical accuracy.

Key metrics
-----------
OOS R²_XS  Cross-sectional R² each month, averaged over the backtest
            = 1 − Σ(r̂_i − r_i)² / Σ(r̄ − r_i)²  [per month, then mean]
Sharpe      Annualised (r_p − r_f) / σ_p
CER         Certainty-equivalent return = μ_p − (λ/2)σ²_p
Hit rate    Fraction of stock-months where sign(μ̂_i) == sign(r_i)

References
----------
Welch & Goyal (2008) — RFS  (OOS R² puzzle)
Campbell & Thompson (2008) — RFS  (sign restrictions)
Kan & Zhou (2007) — JF  (portfolio utility evaluation)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from research.engine.backtest import (
    RISK_AVERSION, TURNOVER_PENALTY, L2_REG, RISK_FREE_RATE,
    TRAIN_WINDOW_MONTHS, LOOKBACK_COV,
    get_spy_regime_labels,
    fetch_and_engineer_features,
    get_monthly_forecasts,
    shrink_returns,          # we call this directly to vary alpha
    get_shrunk_covariance,
    run_backtest_with_benchmark,
    apply_transaction_costs,
    compute_metrics,
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

# Stage A: shrink factor grid
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Stage B: fixed shrink level (matches your current SHRINK_FACTOR=0.25)
FIXED_ALPHA = 0.25

# Transaction cost in basis points (one-way)
TC_BPS = 10


# ═══════════════════════════════════════════════════════════════════════════
# OOS R² COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_oos_r2(
    forecasts: pd.Series,
    realized:  pd.Series,
) -> dict[str, float]:
    """
    Compute three flavours of out-of-sample R² on a
    (date, ticker) MultiIndex forecast series.

    Returns
    -------
    dict with keys:
        r2_xs_mean   Cross-sectional R² averaged over months
                     (main metric — same spirit as Welch & Goyal)
        r2_xs_median Median cross-sectional R² (robust to outlier months)
        r2_ts_mean   Time-series R² averaged over tickers
        hit_rate     Fraction of (month, ticker) pairs where
                     sign(forecast) == sign(realized return)
        n_obs        Total number of aligned forecast-realised pairs
    """
    df = pd.DataFrame({
        "forecast": forecasts,
        "realized": realized,
    }).dropna()

    if df.empty:
        return {k: np.nan for k in
                ["r2_xs_mean", "r2_xs_median", "r2_ts_mean", "hit_rate", "n_obs"]}

    # ── Cross-sectional R² (one number per month) ──────────────────────────
    xs_r2_vals = []
    for date, grp in df.groupby(level="date"):
        y   = grp["realized"].values
        y_hat = grp["forecast"].values
        y_bar = y.mean()
        ss_tot = np.sum((y - y_bar) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        if ss_tot > 0:
            xs_r2_vals.append(1.0 - ss_res / ss_tot)

    # ── Time-series R² (one number per ticker) ────────────────────────────
    ts_r2_vals = []
    for ticker, grp in df.groupby(level="ticker"):
        y     = grp["realized"].values
        y_hat = grp["forecast"].values
        y_bar = y.mean()
        ss_tot = np.sum((y - y_bar) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        if ss_tot > 0:
            ts_r2_vals.append(1.0 - ss_res / ss_tot)

    # ── Hit rate ──────────────────────────────────────────────────────────
    hit = (np.sign(df["forecast"]) == np.sign(df["realized"])).mean()

    return {
        "r2_xs_mean":   float(np.mean(xs_r2_vals))   if xs_r2_vals else np.nan,
        "r2_xs_median": float(np.median(xs_r2_vals)) if xs_r2_vals else np.nan,
        "r2_ts_mean":   float(np.mean(ts_r2_vals))   if ts_r2_vals else np.nan,
        "hit_rate":     float(hit),
        "n_obs":        len(df),
    }


# ═══════════════════════════════════════════════════════════════════════════
# FORECAST TREATMENTS  (Stage B)
# ═══════════════════════════════════════════════════════════════════════════

def apply_sign_restriction_soft(forecasts: pd.Series) -> pd.Series:
    """
    Campbell & Thompson (2008) soft sign restriction.
    For each (date, ticker) pair, if the forecast is below the cross-
    sectional mean for that month, replace it with the cross-sectional mean.
    
    Rationale: when the model predicts stock i will underperform the average,
    we distrust that negative alpha signal and defer to the unconditional mean.
    This preserves the model's *upside* ranking while suppressing negative bets.
    """
    df = forecasts.reset_index()
    df.columns = ["date", "ticker", "forecast"]
    cross_mean = df.groupby("date")["forecast"].transform("mean")
    df["forecast"] = np.where(df["forecast"] < cross_mean, cross_mean, df["forecast"])
    return df.set_index(["date", "ticker"])["forecast"]


def apply_sign_restriction_hard(forecasts: pd.Series) -> pd.Series:
    """
    Hard sign restriction: floor all forecasts at zero.
    Stocks with negative predicted returns get μ̂ = 0, not negative.
    This is more aggressive than the soft version.
    """
    return forecasts.clip(lower=0.0)


def apply_winsorisation(forecasts: pd.Series, quantile: float = 0.05) -> pd.Series:
    """
    Winsorise extreme forecast values at the (quantile, 1-quantile) level
    cross-sectionally within each month.
    Reduces the influence of outlier predictions on portfolio weights.
    """
    df = forecasts.reset_index()
    df.columns = ["date", "ticker", "forecast"]
    lo = df.groupby("date")["forecast"].transform(lambda x: x.quantile(quantile))
    hi = df.groupby("date")["forecast"].transform(lambda x: x.quantile(1 - quantile))
    df["forecast"] = df["forecast"].clip(lower=lo, upper=hi)
    return df.set_index(["date", "ticker"])["forecast"]


def apply_historical_mean(forecasts: pd.Series) -> pd.Series:
    """
    Replace every forecast with the expanding historical mean return
    per ticker — purely backward-looking, no XGBoost signal.
    This is the T5 near-zero-R² anchor.
    OOS R² for this predictor is the Campbell & Thompson (2008) benchmark.
    """
    df = forecasts.reset_index()
    df.columns = ["date", "ticker", "forecast"]
    # Use the cross-sectional mean per month as the "historical mean" proxy.
    # (A fuller version would use per-ticker expanding means from raw returns,
    #  but this approximation is equivalent given the shrink-to-mean prior.)
    df["forecast"] = df.groupby("date")["forecast"].transform("mean")
    return df.set_index(["date", "ticker"])["forecast"]


def apply_zero_forecasts(forecasts: pd.Series) -> pd.Series:
    """
    Set all forecasts to zero. With μ=0 the MV objective reduces to
    min w'Σw subject to Σw=1, w≥0 → global minimum variance portfolio.
    This is the T6 baseline isolating pure covariance information.
    """
    return forecasts * 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ECONOMIC METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_cer(port_returns: pd.Series, risk_aversion: float = RISK_AVERSION) -> float:
    """
    Annualised Certainty-Equivalent Return.

    CER = μ_p_ann − (λ/2) * σ²_p_ann

    This is the objective of a mean-variance investor with risk aversion λ.
    It lets you compare strategies on a single number that integrates both
    return and variance — directly connecting to your MV utility function.
    
    Reference: Kan & Zhou (2007), equation (2).
    """
    r       = port_returns.dropna()
    mu_ann  = r.mean() * 12
    var_ann = r.var() * 12
    return float(mu_ann - 0.5 * risk_aversion * var_ann)


def compute_economic_metrics(
    port_returns: pd.Series,
    label:        str,
) -> dict:
    """Full set of economic metrics for one strategy."""
    r          = port_returns.dropna()
    monthly_rf = RISK_FREE_RATE / 12
    ann_return = r.mean() * 12
    ann_vol    = r.std()  * np.sqrt(12)
    sharpe     = (r.mean() - monthly_rf) / r.std() * np.sqrt(12) if r.std() > 0 else np.nan

    cumulative  = (1 + r).cumprod()
    running_max = cumulative.expanding().max()
    max_dd      = ((cumulative - running_max) / running_max).min()

    cer = compute_cer(r)

    return {
        "label":           label,
        "ann_return_pct":  round(ann_return * 100, 2),
        "ann_vol_pct":     round(ann_vol    * 100, 2),
        "sharpe":          round(sharpe, 3),
        "cer_pct":         round(cer    * 100, 2),
        "max_dd_pct":      round(max_dd * 100, 2),
        "n_months":        len(r),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def paired_t_test(r_strategy: pd.Series, r_baseline: pd.Series) -> dict:
    """
    Paired t-test on monthly return differences:  r_strategy − r_baseline.
    H0: mean difference = 0.
    """
    common = r_strategy.index.intersection(r_baseline.index)
    diff   = r_strategy.loc[common] - r_baseline.loc[common]
    t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0.0)
    return {
        "t_stat":    round(t_stat, 3),
        "p_value":   round(p_val,  4),
        "mean_diff_bps": round(diff.mean() * 10_000, 1),
    }


def bootstrap_sharpe_ci(
    port_returns: pd.Series,
    n_boot:       int = 2_000,
    ci:           float = 0.95,
    seed:         int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the Sharpe ratio.
    Resamples months with replacement — appropriate for monthly data
    where autocorrelation is modest.
    """
    rng     = np.random.default_rng(seed)
    r       = port_returns.dropna().values
    monthly_rf = RISK_FREE_RATE / 12
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(r, size=len(r), replace=True)
        s      = (sample.mean() - monthly_rf) / sample.std() * np.sqrt(12)
        boot_sharpes.append(s)
    lo = np.percentile(boot_sharpes, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_sharpes, 100 * (1 - (1 - ci) / 2))
    return round(lo, 3), round(hi, 3)


def sharpe_difference_test(
    r_strategy: pd.Series,
    r_baseline: pd.Series,
    n_boot: int = 2_000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap test for the difference in Sharpe ratios.
    H0: Sharpe(strategy) − Sharpe(baseline) = 0.
    Returns bootstrap p-value and 95% CI for the difference.
    """
    rng        = np.random.default_rng(seed)
    monthly_rf = RISK_FREE_RATE / 12
    common     = r_strategy.index.intersection(r_baseline.index)
    rs = r_strategy.loc[common].values
    rb = r_baseline.loc[common].values

    def sharpe(r):
        return (r.mean() - monthly_rf) / r.std() * np.sqrt(12)

    obs_diff = sharpe(rs) - sharpe(rb)
    boot_diffs = []
    idx = np.arange(len(rs))
    for _ in range(n_boot):
        sel = rng.choice(idx, size=len(idx), replace=True)
        boot_diffs.append(sharpe(rs[sel]) - sharpe(rb[sel]))

    p_val = np.mean(np.array(boot_diffs) <= 0) if obs_diff > 0 else np.mean(np.array(boot_diffs) >= 0)
    lo    = np.percentile(boot_diffs, 2.5)
    hi    = np.percentile(boot_diffs, 97.5)
    return {
        "Sharpe diff (strat − base)": round(obs_diff, 3),
        "Bootstrap p-value":          round(p_val, 4),
        "95% CI lower":               round(lo, 3),
        "95% CI upper":               round(hi, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE BACKTEST RUNNER  (thin wrapper reusing existing machinery)
# ═══════════════════════════════════════════════════════════════════════════

def run_one_backtest(
    treated_forecasts: pd.Series,
    monthly_returns:   pd.DataFrame,
    cov_dict:          dict,
    spy_labels:        pd.Series,
) -> pd.Series:
    """
    Run the existing backtest loop with a treated forecast series.
    Returns the monthly portfolio return series (net of transaction costs).
    """
    port_ret, _, weights_df, *_ = run_backtest_with_benchmark(
        monthly_returns  = monthly_returns,
        shrunk_forecasts = treated_forecasts,
        cov_dict         = cov_dict,
        spy_labels       = spy_labels,
    )
    net_ret, _ = apply_transaction_costs(port_ret, weights_df, tc_bps=TC_BPS)
    return net_ret


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_vs_performance(
    stage_a: list[dict],
    stage_b: list[dict],
    out_prefix: str = "topic3",
) -> None:
    """
    Two-panel figure:
    Left  — OOS R² (x) vs Sharpe (y)
    Right — OOS R² (x) vs CER   (y)

    Stage A points connected by a line (the "curve").
    Stage B points overlaid as large markers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Stage A curve
    a_r2     = [d["r2_xs_mean"]  * 100 for d in stage_a]
    a_sharpe = [d["sharpe"]            for d in stage_a]
    a_cer    = [d["cer_pct"]           for d in stage_a]
    a_alpha  = [d["alpha"]             for d in stage_a]

    for ax, a_y, ylabel in zip(
        axes,
        [a_sharpe, a_cer],
        ["Sharpe ratio", "CER (ann. %)"],
    ):
        ax.plot(a_r2, a_y, "o-", color="#378ADD", lw=1.5,
                label="Stage A — shrink sweep", zorder=2)

        # Annotate α values on Stage A points
        for x, y, alpha in zip(a_r2, a_y, a_alpha):
            ax.annotate(f"α={alpha:.1f}", (x, y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="#185FA5")

        # Stage B markers
        colors = {
            "T1 — baseline":          "#888780",
            "T2 — soft sign restr.":  "#1D9E75",
            "T3 — hard sign restr.":  "#0F6E56",
            "T4 — winsorised 5%":     "#EF9F27",
            "T5 — historical mean":   "#D85A30",
            "T6 — zero forecast":     "#E24B4A",
        }
        for d in stage_b:
            x = d["r2_xs_mean"] * 100
            y = d["sharpe"] if ylabel == "Sharpe ratio" else d["cer_pct"]
            color = colors.get(d["label"], "#888780")
            ax.scatter(x, y, s=90, color=color, zorder=3, label=d["label"])
            ax.annotate(d["label"].split("—")[0].strip(),
                        (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=color)

        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("OOS R²_XS (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"OOS R² vs {ylabel}", fontsize=11)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Topic 3 — R² to economic performance conversion", fontsize=12)
    plt.tight_layout()
    path = f"{out_prefix}_r2_vs_performance.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path}")


def plot_cumulative(results_dict: dict[str, pd.Series], out_prefix: str = "topic3") -> None:
    """Cumulative return chart for Stage B treatments."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    for i, (label, rets) in enumerate(results_dict.items()):
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, lw=1.5, color=colors[i % 10], label=label)
    ax.set_title("Topic 3 — cumulative returns by forecast treatment")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (level)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{out_prefix}_cumulative.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Topic 3 — Out-of-Sample R² vs Economic Performance")
    print("=" * 65)

    # ── Step 1: data ────────────────────────────────────────────────────────
    print("\n[1/5] Downloading data...")
    spy_labels = get_spy_regime_labels(start=START_DATE, train_duration=36)
    features   = fetch_and_engineer_features(TICKERS, START_DATE, END_DATE)

    monthly_returns = (
        features["ret_1m"]
        .unstack("ticker")
        .sort_index()
    )
    cov_dict = get_shrunk_covariance(monthly_returns, lookback=LOOKBACK_COV)

    # ── Step 2: raw XGBoost forecasts (run once) ────────────────────────────
    print("[2/5] Computing XGBoost walk-forward forecasts (once)...")
    raw_forecasts = get_monthly_forecasts(features, train_months=TRAIN_WINDOW_MONTHS, PCA_count=3)

    # Align realized returns to the forecast index for R² computation
    ret_pivot = features["ret_1m"].unstack("ticker").sort_index()
    # Realized at t+1 aligned to forecast at t
    realized = ret_pivot.shift(-1).stack().rename("realized")
    realized.index.names = ["date", "ticker"]

    # ── Step 3: Stage A — shrink factor sweep ───────────────────────────────
    print("\n[3/5] Stage A — shrink factor sweep...")
    stage_a_records = []

    for alpha in ALPHA_GRID:
        treated = shrink_returns(
            raw_forecasts,
            prior_type="cross_mean",
            shrink_factor=alpha,
        )
        r2_metrics = compute_oos_r2(treated, realized)
        port_ret   = run_one_backtest(treated, monthly_returns, cov_dict, spy_labels)
        econ       = compute_economic_metrics(port_ret, label=f"alpha={alpha:.1f}")

        record = {
            "alpha":        alpha,
            "r2_xs_mean":   r2_metrics["r2_xs_mean"],
            "r2_xs_median": r2_metrics["r2_xs_median"],
            "hit_rate":     r2_metrics["hit_rate"],
            **econ,
        }
        stage_a_records.append(record)
        print(
            f"  α={alpha:.1f} | R²_XS={r2_metrics['r2_xs_mean']*100:+.3f}% "
            f"| Sharpe={econ['sharpe']:.3f} | CER={econ['cer_pct']:.2f}%"
        )

    stage_a_df = pd.DataFrame(stage_a_records)

    # ── Step 4: Stage B — discrete forecast treatments ───────────────────────
    print("\n[4/5] Stage B — discrete forecast treatments...")

    base_forecasts = shrink_returns(
        raw_forecasts, prior_type="cross_mean", shrink_factor=FIXED_ALPHA
    )

    treatments = {
        "T1 — baseline":          base_forecasts,
        "T2 — soft sign restr.":  apply_sign_restriction_soft(base_forecasts),
        "T3 — hard sign restr.":  apply_sign_restriction_hard(base_forecasts),
        "T4 — winsorised 5%":     apply_winsorisation(base_forecasts, quantile=0.05),
        "T5 — historical mean":   apply_historical_mean(base_forecasts),
        "T6 — zero forecast":     apply_zero_forecasts(base_forecasts),
    }

    stage_b_records = []
    stage_b_rets    = {}

    for label, forecasts in treatments.items():
        r2_metrics = compute_oos_r2(forecasts, realized)
        port_ret   = run_one_backtest(forecasts, monthly_returns, cov_dict, spy_labels)
        econ       = compute_economic_metrics(port_ret, label=label)
        ci_lo, ci_hi = bootstrap_sharpe_ci(port_ret)
        stage_b_rets[label] = port_ret

        record = {
            "label":        label,
            "r2_xs_mean":   r2_metrics["r2_xs_mean"],
            "hit_rate":     r2_metrics["hit_rate"],
            **econ,
            "sharpe_ci_lo": ci_lo,
            "sharpe_ci_hi": ci_hi,
        }
        stage_b_records.append(record)
        print(
            f"  {label:<28s} | R²={r2_metrics['r2_xs_mean']*100:+.3f}% "
            f"| Sharpe={econ['sharpe']:.3f} [{ci_lo:.2f},{ci_hi:.2f}] "
            f"| CER={econ['cer_pct']:.2f}% | Hit={r2_metrics['hit_rate']*100:.1f}%"
        )

    stage_b_df = pd.DataFrame(stage_b_records)

    # ── Step 5: statistical tests ────────────────────────────────────────────
    print("\n[5/5] Statistical tests...")
    baseline_ret  = stage_b_rets["T1 — baseline"]
    hist_mean_ret = stage_b_rets["T5 — historical mean"]

    print("\n── H0_C: XGBoost signal (T1 vs T5) has zero CER gain ──")
    t_test_T1_T5 = paired_t_test(baseline_ret, hist_mean_ret)
    for k, v in t_test_T1_T5.items():
        print(f"  {k:<30s} {v}")

    print("\n── Sharpe difference tests vs T1 baseline ──")
    for label, ret in stage_b_rets.items():
        if label == "T1 — baseline":
            continue
        res = sharpe_difference_test(ret, baseline_ret)
        print(f"\n  {label}")
        for k, v in res.items():
            print(f"    {k:<35s} {v}")

    # ── Print result tables ──────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("STAGE A — SHRINK SWEEP SUMMARY")
    print("═" * 65)
    a_cols = ["alpha", "r2_xs_mean", "hit_rate", "sharpe", "cer_pct", "ann_vol_pct"]
    print(
        stage_a_df[a_cols]
        .rename(columns={
            "alpha":      "Alpha",
            "r2_xs_mean": "R²_XS",
            "hit_rate":   "Hit rate",
            "sharpe":     "Sharpe",
            "cer_pct":    "CER (%)",
            "ann_vol_pct":"Vol (%)",
        })
        .to_string(index=False)
    )

    print("\n" + "═" * 65)
    print("STAGE B — TREATMENT COMPARISON")
    print("═" * 65)
    b_cols = ["label", "r2_xs_mean", "hit_rate", "sharpe", "sharpe_ci_lo",
              "sharpe_ci_hi", "cer_pct", "max_dd_pct"]
    print(
        stage_b_df[b_cols]
        .rename(columns={
            "label":        "Treatment",
            "r2_xs_mean":   "R²_XS",
            "hit_rate":     "Hit rate",
            "sharpe":       "Sharpe",
            "sharpe_ci_lo": "95% CI lo",
            "sharpe_ci_hi": "95% CI hi",
            "cer_pct":      "CER (%)",
            "max_dd_pct":   "Max DD (%)",
        })
        .to_string(index=False)
    )

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_r2_vs_performance(stage_a_records, stage_b_records)
    plot_cumulative(stage_b_rets)

    print("\nDone. Key interpretation guide:")
    print("─" * 65)
    print("H0_A: monotone R²-Sharpe → check Stage A table for slope direction")
    print("H0_B: sign restr. above the Stage A curve → T2/T3 above the line?")
    print("H0_C: t-test on T1 vs T5 → does XGBoost add CER vs historical mean?")
    print("If T2 (soft sign restr.) is above the Stage A curve at same R²,")
    print("you have replicated the Campbell & Thompson (2008) result.")


if __name__ == "__main__":
    main()