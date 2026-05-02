"""
Regime detection for a custom active strategy using the SJM pipeline from
regimes.py. The "factor" is replaced by a user-supplied strategy ticker;
active return is computed as strategy_ret - mkt_ret throughout.

Configuration
-------------
Set STRATEGY_TICKER to the ticker that represents your active strategy and
MARKET_TICKER to the benchmark you trade against. All other settings mirror
regimes.py.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jumpmodels.sparse_jump import SparseJumpModel
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
from jumpmodels.utils import filter_date_range

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_TICKER = "MTUM"   # <- replace with your active strategy ticker
MARKET_TICKER   = "IVV"    # S&P 500 as the benchmark
START_DATE      = "2007-01-01"
TRAIN_END       = "2019-12-31"
JUMP_PENALTY    = 60.0
MAX_FEATS       = 5.0

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _ewma(s: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted moving average with the given span."""
    return s.ewm(span=span, adjust=False).mean()


def _wilder_ewma(s: pd.Series, window: int) -> pd.Series:
    """Wilder-style smoothing used by the classical RSI: alpha = 1/window."""
    return s.ewm(alpha=1.0 / window, adjust=False).mean()


# ---------------------------------------------------------------------------
# Factor-specific features
# ---------------------------------------------------------------------------

def active_return_ewma(active_ret: pd.Series, window: int) -> pd.Series:
    """EWMA of active returns over the given span."""
    return _ewma(active_ret, span=window).rename(f"r_factor_ewma_{window}")


def rsi(active_ret: pd.Series, window: int) -> pd.Series:
    """
    Relative Strength Index computed on the active-return series.

    RSI = 100 - 100 / (1 + RS),  RS = avg_up / avg_down,
    where the averages use Wilder smoothing (alpha = 1/window).
    """
    up = active_ret.clip(lower=0.0)
    down = (-active_ret).clip(lower=0.0)
    avg_up = _wilder_ewma(up, window)
    avg_down = _wilder_ewma(down, window)
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there were no down moves at all, RSI saturates at 100
    out = out.where(avg_down > 0, 100.0)
    return out.rename(f"RSI_{window}")


def stochastic_k(active_cum: pd.Series, window: int) -> pd.Series:
    """
    Stochastic Oscillator %K computed on the cumulative active-return series
    (which plays the role of an "active price" level).

        %K = 100 * (P_t - min(P) over w) / (max(P) over w - min(P) over w)
    """
    low = active_cum.rolling(window).min()
    high = active_cum.rolling(window).max()
    rng = (high - low).replace(0.0, np.nan)
    return (100.0 * (active_cum - low) / rng).rename(f"K_{window}")


def macd(active_cum: pd.Series, span_short: int, span_long: int) -> pd.Series:
    """
    MACD computed on the cumulative active-return level.

        MACD_{s,l} = (EMA_s(P) - EMA_l(P)) / EMA_l(P)

    The division by EMA_l keeps the indicator scale-free, which matters when
    the underlying level drifts over a long sample.
    """
    ema_s = _ewma(active_cum, span=span_short)
    ema_l = _ewma(active_cum, span=span_long)
    return ((ema_s - ema_l) / ema_l.replace(0.0, np.nan)).rename(
        f"MACD_{span_short}_{span_long}"
    )


def downside_deviation_log(active_ret: pd.Series, window: int,
                           eps: float = 1e-8) -> pd.Series:
    """
    log of the rolling downside deviation of the active returns.

        DD_w = sqrt( mean( min(r, 0)^2 ) over a window of size w )
    """
    neg2 = active_ret.clip(upper=0.0).pow(2)
    dd = np.sqrt(neg2.rolling(window).mean())
    return np.log(dd + eps).rename(f"log_DD_{window}")


def active_beta(factor_ret: pd.Series, mkt_ret: pd.Series,
                window: int) -> pd.Series:
    """
    Rolling beta of the factor return on the market return.

        beta_w = Cov_w(r_factor, r_mkt) / Var_w(r_mkt)
    """
    cov = factor_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    # If market variance is zero (constant returns), beta is undefined -> NaN
    beta = cov / var.replace(0.0, np.nan)
    return beta.rename(f"beta_{window}")


# ---------------------------------------------------------------------------
# Market-environment features
# ---------------------------------------------------------------------------

def market_return_ewma(mkt_ret: pd.Series, window: int = 21) -> pd.Series:
    """EWMA of market returns."""
    return _ewma(mkt_ret, span=window).rename(f"r_mkt_ewma_{window}")


def vix_feature(vix: pd.Series, window: int = 21) -> pd.Series:
    """log-VIX, first-differenced, then EWMA-smoothed."""
    log_diff = np.log(vix).diff()
    return _ewma(log_diff, span=window).rename(f"r_VIX_ewma_{window}")


def yield_diff_ewma(yld: pd.Series, window: int = 21,
                    name: str = "yld_diff_ewma") -> pd.Series:
    """First-differenced yield, EWMA-smoothed."""
    return _ewma(yld.diff(), span=window).rename(name)


# ---------------------------------------------------------------------------
# Full feature panel
# ---------------------------------------------------------------------------

# Default windows from the paper
FACTOR_WINDOWS_SHORT = (8, 21, 63)        # active return, RSI, %K
MACD_SPANS = ((8, 21), (21, 63))          # (short, long)
SINGLE_WINDOW = 21                        # DD, beta, market env.

def build_feature_panel(factor_ret: pd.Series,
                        mkt_ret: pd.Series,
                        vix: pd.Series,
                        yld_2y: pd.Series,
                        yld_10y_minus_2y: pd.Series,
                        min_obs: int = 100) -> pd.DataFrame:
    """
    Build the full SJM feature panel for one factor.

    Parameters
    ----------
    factor_ret : pd.Series
        Daily simple returns of the factor index.
    mkt_ret : pd.Series
        Daily simple returns of the market index, aligned to ``factor_ret``.
    vix : pd.Series
        Level of the VIX.
    yld_2y : pd.Series
        2-year Treasury yield, in percent or in decimal -- the unit is
        irrelevant because we take first differences and then standardize.
    yld_10y_minus_2y : pd.Series
        10-year minus 2-year yield (the slope of the curve).
    min_obs : int, optional
        Minimum number of rows required after dropping missing values.
        If fewer remain, an informative error is raised.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by date, with one column per feature.
    """
    factor_ret = factor_ret.astype(float)
    mkt_ret = mkt_ret.astype(float)

    # Active return and its cumulative "price" level
    active = (factor_ret - mkt_ret).rename("active_ret")
    active_cum = (1.0 + active).cumprod().rename("active_cum")

    cols: list[pd.Series] = []

    # Factor-specific
    for w in FACTOR_WINDOWS_SHORT:
        cols.append(active_return_ewma(active, w))
    for w in FACTOR_WINDOWS_SHORT:
        cols.append(rsi(active, w))
    for w in FACTOR_WINDOWS_SHORT:
        cols.append(stochastic_k(active_cum, w))
    for s, l in MACD_SPANS:
        cols.append(macd(active_cum, s, l))
    cols.append(downside_deviation_log(active, SINGLE_WINDOW))
    cols.append(active_beta(factor_ret, mkt_ret, SINGLE_WINDOW))

    # Market environment
    cols.append(market_return_ewma(mkt_ret, SINGLE_WINDOW))
    cols.append(vix_feature(vix, SINGLE_WINDOW))
    cols.append(yield_diff_ewma(yld_2y, SINGLE_WINDOW, name="2Y_diff_ewma"))
    cols.append(yield_diff_ewma(yld_10y_minus_2y, SINGLE_WINDOW,
                                name="10Y2Y_diff_ewma"))

    feats = pd.concat(cols, axis=1)

    # --- Robustness checks ---
    # Drop rows that are all NaN
    feats = feats.dropna(how='all')
    # Drop columns that are entirely NaN
    feats = feats.dropna(axis=1, how='all')
    # Now drop any remaining rows that contain any NaN
    feats = feats.dropna()

    if len(feats) < min_obs:
        raise ValueError(
            f"Feature panel has only {len(feats)} rows after dropping NaNs "
            f"(minimum required {min_obs}).\n"
            f"Check input series lengths:\n"
            f"  factor_ret: {len(factor_ret.dropna())} non-NaN\n"
            f"  mkt_ret:    {len(mkt_ret.dropna())} non-NaN\n"
            f"  vix:        {len(vix.dropna())} non-NaN\n"
            f"  yld_2y:     {len(yld_2y.dropna())} non-NaN\n"
            f"  spread:     {len(yld_10y_minus_2y.dropna())} non-NaN\n"
            f"Possible causes: short history, constant market returns, or missing yield/VIX data."
        )

    return feats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_strategy_data(strategy_ticker: str = STRATEGY_TICKER,
                       market_ticker: str = MARKET_TICKER,
                       start: str = START_DATE,
                       end: str | None = None) -> dict[str, pd.Series]:
    """
    Fetch daily returns for the strategy + market, VIX, and Treasury yields.
    The 'factor_ret' key is re-used throughout so that run_sjm() works without
    modification — it just holds strategy returns instead of a style-factor's.
    """
    import yfinance as yf
    import pandas_datareader.data as web

    px = yf.download(
        [strategy_ticker, market_ticker, "^VIX"],
        start=start, end=end,
        auto_adjust=True, progress=False,
    )["Close"]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)

    strategy_ret = px[strategy_ticker].pct_change()
    mkt_ret      = px[market_ticker].pct_change()
    vix          = px["^VIX"]

    yld = web.DataReader(["DGS2", "DGS10"], "fred", start, end)
    yld_2y           = yld["DGS2"]
    yld_10y_minus_2y = yld["DGS10"] - yld["DGS2"]

    idx = strategy_ret.dropna().index
    return {
        "factor_ret": strategy_ret.reindex(idx),   # strategy plays the factor role
        "mkt_ret":    mkt_ret.reindex(idx),
        "vix":        vix.reindex(idx).ffill(),
        "yld_2y":     yld_2y.reindex(idx).ffill(),
        "yld_10y_minus_2y": yld_10y_minus_2y.reindex(idx).ffill(),
    }

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_sjm(data: dict[str, pd.Series],
            train_end: str = "2019-12-31",
            jump_penalty: float = 60.0,
            max_feats: float = 5.0,
            random_state: int = 0):
    """
    Build features, fit the SJM on the training window, and run online
    inference on the rest of the sample.

    Returns a dict with the fitted model, feature matrices, and label series.
    """
    feats = build_feature_panel(
        factor_ret=data["factor_ret"],
        mkt_ret=data["mkt_ret"],
        vix=data["vix"],
        yld_2y=data["yld_2y"],
        yld_10y_minus_2y=data["yld_10y_minus_2y"],
    )

    X_train = filter_date_range(feats, end_date=train_end)
    X_test  = filter_date_range(feats, start_date=train_end)
    train_ret = data["factor_ret"].loc[X_train.index] - data["mkt_ret"].loc[X_train.index]

    # Clip + standardize, fitted on the training data only
    clipper = DataClipperStd(mul=3.0).fit(X_train)
    X_train_c = clipper.transform(X_train)
    X_test_c  = clipper.transform(X_test)

    scaler = StandardScalerPD().fit(X_train_c)
    X_train_s = scaler.transform(X_train_c)
    X_test_s  = scaler.transform(X_test_c)

    sjm = SparseJumpModel(
        n_components=2,
        max_feats=max_feats,
        jump_penalty=jump_penalty,
        cont=False,            # discrete SJM (matches the paper's setting)
        random_state=random_state,
        n_init_jm=10,
    )
    # Pass the active return so the states are sorted by cumulative return
    # -> state 0 = bull (high active return), state 1 = bear.
    sjm.fit(X_train_s, ret_ser=train_ret, sort_by="cumret")

    labels_train = pd.Series(sjm.labels_, index=X_train.index, name="state")
    labels_test  = sjm.predict_online(X_test_s)
    labels_test  = pd.Series(np.asarray(labels_test),
                             index=X_test_s.index, name="state")

    return {
        "model":   sjm,
        "feats":   feats,
        "X_train_s": X_train_s,
        "X_test_s":  X_test_s,
        "labels_train": labels_train,
        "labels_test":  labels_test,
        "feat_weights": pd.Series(sjm.feat_weights, index=feats.columns,
                                  name="feat_weight").sort_values(ascending=False),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_strategy_regimes(data: dict[str, pd.Series],
                          result: dict,
                          strategy_ticker: str = STRATEGY_TICKER,
                          market_ticker: str = MARKET_TICKER,
                          out_path: str = "strategy_regimes.png") -> str:
    """Plot cumulative strategy active return shaded by inferred regime."""
    active = (data["factor_ret"] - data["mkt_ret"]).reindex(result["feats"].index)
    cum = (1.0 + active).cumprod()

    labels = pd.concat([result["labels_train"], result["labels_test"]])
    labels = (labels[~labels.index.duplicated(keep="last")]
              .reindex(cum.index).ffill())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum.index, cum.values, color="black", lw=1.0,
            label=f"Cumulative active return ({strategy_ticker} vs {market_ticker})")

    # shade bear (state == 1) regions
    in_bear = labels.values == 1
    starts, ends, in_block = [], [], False
    for t, b in enumerate(in_bear):
        if b and not in_block:
            starts.append(cum.index[t]); in_block = True
        elif (not b) and in_block:
            ends.append(cum.index[t]); in_block = False
    if in_block:
        ends.append(cum.index[-1])
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color="tab:red", alpha=0.18, lw=0)

    split = result["labels_test"].index.min()
    ax.axvline(split, color="tab:blue", ls="--", lw=1, label="train / test split")

    ax.set_title(f"SJM regimes — {strategy_ticker} active return vs {market_ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative active return (level)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved regime plot to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data for strategy={STRATEGY_TICKER}, market={MARKET_TICKER} ...")
    data = load_strategy_data()

    print("Running SJM ...")
    result = run_sjm(data,
                     train_end=TRAIN_END,
                     jump_penalty=JUMP_PENALTY,
                     max_feats=MAX_FEATS)

    print("\n=== Feature-weight ranking ===")
    print(result["feat_weights"].round(4).to_string())

    print("\n=== State centers (selected features only) ===")
    centers = pd.DataFrame(result["model"].centers_,
                           columns=result["feats"].columns,
                           index=["state_0 (bull)", "state_1 (bear)"])
    nonzero = result["feat_weights"][result["feat_weights"] > 1e-6].index
    print(centers[nonzero].T.round(3).to_string())

    print("\n=== Per-state annualised stats (training set) ===")
    stats = pd.DataFrame({
        "ret_ann": result["model"].ret_,
        "vol_ann": result["model"].vol_,
    }, index=["state_0 (bull)", "state_1 (bear)"])
    print(stats.round(4).to_string())

    plot_strategy_regimes(data, result)


if __name__ == "__main__":
    main()