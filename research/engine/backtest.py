from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter

from research.engine.regimes import run_sjm

# ===========================
# CONFIGURATION
# ===========================
RISK_AVERSION         = 2.5
REGIME_RISK_AVERSION_MULT = 2
TURNOVER_PENALTY      = 0.01
L2_REG                = 0.05
LOOKBACK_COV          = 6
RISK_FREE_RATE        = 0.04
TRAIN_WINDOW_MONTHS   = 36
SHRINK_FACTOR         = 0.25
PRIOR_TYPE            = 'cross_mean'
RETURN_SMOOTHING      = 'both'
KALMAN_TRANSITION_COV = 0.01
KALMAN_OBSERVATION_COV = 0.05


# ===========================
# SPY REGIME DETECTION
# ===========================
def get_spy_regime_labels(
    start: str = "2013-01-01",
    train_duration: int = 36,
    jump_penalty: float = 60.0,
    max_feats: float = 5.0,
) -> pd.Series:
    """
    Fit a SJM on SPY absolute returns and return a daily bull/bear label Series.
    mkt_ret is set to zero so all active-return features become absolute SPY features.
    Returns a pd.Series indexed by date with values 0=bull, 1=bear.
    """
    import pandas_datareader.data as web

    px = yf.download(["SPY", "^VIX"], start=start, auto_adjust=True, progress=False)["Close"]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    spy_ret = px["SPY"].pct_change()
    vix = px["^VIX"]

    yld = web.DataReader(["DGS2", "DGS10"], "fred", start, None)
    yld_2y = yld["DGS2"]
    spread = yld["DGS10"] - yld["DGS2"]

    idx = spy_ret.dropna().index
    data = {
        "factor_ret": spy_ret.reindex(idx),
        "mkt_ret":    pd.Series(0.0, index=idx),
        "vix":        vix.reindex(idx).ffill(),
        "yld_2y":     yld_2y.reindex(idx).ffill(),
        "yld_10y_minus_2y": spread.reindex(idx).ffill(),
    }

    date_obj = pd.to_datetime(start)
    end_date = date_obj + pd.DateOffset(months=train_duration)

    result = run_sjm(data, train_end=end_date.strftime("%Y-%m-%d"), jump_penalty=jump_penalty, max_feats=max_feats)
    labels = pd.concat([result["labels_train"], result["labels_test"]])
    labels = labels[~labels.index.duplicated(keep="last")]
    return labels


# ===========================
# FEATURE ENGINEERING
# ===========================
def _compute_days_in_regime(labels: pd.Series) -> pd.Series:
    counts, count, prev = [], 0, None
    for val in labels:
        count = 1 if val != prev else count + 1
        prev = val
        counts.append(count)
    return pd.Series(counts, index=labels.index)


def fetch_and_engineer_features(tickers, start, end):
    """
    Download price/volume data and compute characteristics.
    Returns a DataFrame with MultiIndex (date, ticker).
    """
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)
    adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})
    volume    = pd.DataFrame({ticker: data[ticker]['Volume']    for ticker in tickers})

    daily_returns  = adj_close.pct_change().dropna(how='all')
    monthly_close  = adj_close.resample('ME').last()
    monthly_volume = volume.resample('ME').sum()

    all_features = []
    for ticker in tickers:
        prices           = monthly_close[ticker]
        vol_monthly      = monthly_volume[ticker]
        daily_ret_ticker = daily_returns[ticker].dropna()

        df = pd.DataFrame(index=prices.index)
        df['ticker'] = ticker
        df['price']  = prices
        df['volume'] = vol_monthly
        df['month'] = df.index.month

        # Momentum
        df['mom_3m']  = prices / prices.shift(3)  - 1
        df['mom_6m']  = prices / prices.shift(6)  - 1
        df['mom_12m'] = prices / prices.shift(12) - 1

        # Volatility (annualised from 21-day rolling daily returns)
        daily_vol   = daily_ret_ticker.rolling(21).std() * np.sqrt(252)
        monthly_vol = daily_vol.resample('ME').last()
        df['vol_21d'] = monthly_vol.reindex(df.index).ffill()

        # Idiosyncratic volatility (CAPM residual vol proxy)
        market_daily = daily_returns.mean(axis=1).dropna()
        if len(daily_ret_ticker) > 21:
            common = daily_ret_ticker.index.intersection(market_daily.index)
            if len(common) > 21:
                resid_vol, resid_dates = [], []
                for date in common[20:]:
                    slice_ret = daily_ret_ticker.loc[date - pd.Timedelta(days=30):date]
                    slice_mkt = market_daily.loc[date - pd.Timedelta(days=30):date]
                    if len(slice_ret) >= 21:
                        _, resid = np.polyfit(slice_mkt, slice_ret, 1)
                        resid_vol.append(np.std(slice_ret - resid))
                        resid_dates.append(date)
                if resid_dates:
                    resid_vol_series = pd.Series(resid_vol, index=resid_dates)
                    monthly_resid    = resid_vol_series.resample('ME').last()
                    df['idiosyncratic_vol'] = monthly_resid.reindex(df.index).ffill()

        # Max daily return in past month
        monthly_max = daily_ret_ticker.rolling(21).max().resample('ME').last()
        df['max_daily_ret'] = monthly_max.reindex(df.index).ffill()

        # Volume trend (12-month change)
        df['volume_trend'] = vol_monthly / vol_monthly.shift(12) - 1

        # Amihud illiquidity proxy
        dollar_volume = volume[ticker] * adj_close[ticker]
        amihud        = (daily_ret_ticker.abs() / dollar_volume).rolling(21).mean()
        df['amihud_illiq'] = amihud.resample('ME').last().reindex(df.index).ffill()

        # Rolling aggregates
        for col in ['mom_6m', 'vol_21d']:
            if col in df.columns:
                for window in [3, 6, 12]:
                    df[f'{col}_mean_{window}m'] = df[col].rolling(window).mean()
                    df[f'{col}_std_{window}m']  = df[col].rolling(window).std()

        df.reset_index(inplace=True)
        all_features.append(df)

    features   = pd.concat(all_features, ignore_index=True)
    ticker_col = features['ticker'].copy()
    features   = features.groupby('ticker').ffill()
    features['ticker'] = ticker_col.values
    features.dropna(inplace=True)

    # Only monthly return, no target yet
    features['ret_1m'] = features.groupby('ticker')['price'].pct_change()
    features.dropna(subset=['ret_1m'], inplace=True)

    date_col = next((c for c in features.columns if c.lower() in ('date', 'datetime')), None)
    if date_col is None:
        raise ValueError("Could not find a date column after reset_index().")

    features = features.set_index([date_col, 'ticker'])
    features.index.names = ['date', 'ticker']
    return features


# ===========================
# WALK-FORWARD XGBOOST FORECASTS
# ===========================
def get_monthly_forecasts(features, train_months=36, PCA_added=True, PCA_count=5):
    """
    Walk‑forward XGBoost forecasts with no lookahead.
    For each month `t`, trains on features of months [t-train_months, t-1]
    with targets = return of the month *following* each feature month.
    Predicts using features of month `t` to forecast return of month `t+1`.
    """
    features = features.copy()
    # Ensure ret_1m exists
    if 'ret_1m' not in features.columns:
        raise ValueError("Features must contain 'ret_1m' (monthly return).")

    dates = sorted(features.index.get_level_values('date').unique())
    if len(dates) < train_months + 2:   # need at least train_months + 1 future month
        raise ValueError("Not enough months for rolling training.")

    # Create a pivot of returns for easy shifting
    ret_pivot = features['ret_1m'].unstack('ticker')

    # Prepare features (excluding return, price, volume, month dummy later)
    exclude_cols = ['price', 'volume', 'ret_1m']   # no 'target'
    features_dummies = pd.get_dummies(features, columns=['month'], prefix='month', drop_first=True)
    feature_cols = [c for c in features_dummies.columns if c not in exclude_cols + ['month']]

    preds_list = []

    # Loop: for each month where we have features and a next-month return
    for i in range(train_months, len(dates) - 1):
        # Training months: from i-train_months to i-1
        train_start = dates[i - train_months]
        train_end   = dates[i - 1]          # last month whose features we can use for training
        # Forecast month (features we use now)
        forecast_month = dates[i]
        # Target month (return we want to predict)
        target_month   = dates[i + 1]

        # ----- Build training set -----
        # Filter feature rows for training dates
        train_mask = (
            (features_dummies.index.get_level_values('date') >= train_start) &
            (features_dummies.index.get_level_values('date') <= train_end)
        )
        X_train_df = features_dummies[train_mask]

        # Align targets: for each training date d, target = ret_1m at date d+1
        # We'll merge X_train_df with a shifted version of ret_pivot
        # Shift ret_pivot backward by one month so that index d holds return of d+1
        target_next = ret_pivot.shift(-1).stack().rename('target')
        # Merge on (date, ticker)
        train_merged = X_train_df.reset_index().merge(
            target_next.reset_index(), on=['date', 'ticker'], how='inner'
        ).set_index(['date', 'ticker'])
        # Drop rows where target is NaN (e.g., the last month of the whole series)
        train_merged.dropna(subset=['target'], inplace=True)

        if len(train_merged) == 0:
            continue

        X_train = train_merged[feature_cols]
        y_train = train_merged['target']

        # ----- Train model -----
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)

        if PCA_added:
            pca = PCA(n_components=PCA_count).fit(X_train_scaled)
            X_train_aug = pca.transform(X_train_scaled)
        else:
            X_train_aug = X_train_scaled

        model = xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.5,
            reg_lambda=1.0, reg_alpha=0.1, random_state=42,
        )
        model.fit(X_train_aug, y_train)

        # ----- Predict for forecast_month -----
        test_mask = features_dummies.index.get_level_values('date') == forecast_month
        X_test_df = features_dummies[test_mask]
        if len(X_test_df) == 0:
            continue

        X_test = X_test_df[feature_cols]
        X_test_scaled = scaler.transform(X_test.values)
        if PCA_added:
            X_test_aug = pca.transform(X_test_scaled)
        else:
            X_test_aug = X_test_scaled

        preds = model.predict(X_test_aug)
        pred_series = pd.Series(preds, index=X_test_df.index, name='xgboost_pred')
        preds_list.append(pred_series)

    if not preds_list:
        return pd.Series(dtype=float)
    return pd.concat(preds_list)


# ===========================
# RETURN SMOOTHING
# ===========================
def shrink_returns(forecasts, prior_type='cross_mean', shrink_factor=0.5):
    df = forecasts.reset_index()
    df.columns = ['date', 'ticker', 'raw_forecast']
    if prior_type == 'zero':
        prior = 0
    elif prior_type == 'cross_mean':
        prior = df.groupby('date')['raw_forecast'].transform('mean')
    elif prior_type == 'historical_mean':
        prior = df['raw_forecast'].mean()
    else:
        raise ValueError(f"Unknown prior_type: {prior_type!r}")
    df['shrunk_forecast'] = shrink_factor * prior + (1 - shrink_factor) * df['raw_forecast']
    return df.set_index(['date', 'ticker'])['shrunk_forecast']


def kalman_filter_returns(
    forecasts,
    transition_cov=KALMAN_TRANSITION_COV,
    observation_cov=KALMAN_OBSERVATION_COV,
):
    df = forecasts.reset_index()
    df.columns = ['date', 'ticker', 'raw_forecast']
    filtered = []
    for _, grp in df.groupby('ticker'):
        grp = grp.sort_values('date').copy()
        obs = grp['raw_forecast'].values
        kf  = KalmanFilter(
            transition_matrices=[[1]],
            observation_matrices=[[1]],
            initial_state_mean=[[obs[0]]],
            initial_state_covariance=[[1]],
            observation_covariance=[[observation_cov]],
            transition_covariance=[[transition_cov]],
        )
        state_means, _ = kf.filter(obs.reshape(-1, 1))
        grp['filtered_forecast'] = state_means.flatten()
        filtered.append(grp)
    result = pd.concat(filtered)
    return result.set_index(['date', 'ticker'])['filtered_forecast']


def process_forecasts(
    forecasts,
    mode=RETURN_SMOOTHING,
    prior_type=PRIOR_TYPE,
    shrink_factor=SHRINK_FACTOR,
):
    if mode == 'shrink':
        return shrink_returns(forecasts, prior_type=prior_type, shrink_factor=shrink_factor)
    elif mode == 'kalman':
        return kalman_filter_returns(forecasts)
    elif mode == 'both':
        shrunk = shrink_returns(forecasts, prior_type=prior_type, shrink_factor=shrink_factor)
        return kalman_filter_returns(shrunk)
    else:
        raise ValueError(f"Unknown RETURN_SMOOTHING mode: {mode!r}. Use 'shrink', 'kalman', or 'both'.")


# ===========================
# COVARIANCE SHRINKAGE
# ===========================
def get_shrunk_covariance(monthly_returns, lookback=12):
    cov_dict = {}
    dates = monthly_returns.index
    for i in range(lookback, len(dates)):
        end_date   = dates[i]
        start_date = dates[i - lookback]
        hist       = monthly_returns.loc[start_date:end_date].dropna(axis=1, how='any')
        if len(hist) < lookback or hist.shape[1] < 2:
            continue
        lw = LedoitWolf().fit(hist)
        cov_dict[end_date] = {'cov': lw.covariance_, 'tickers': list(hist.columns)}
    return cov_dict


# ===========================
# MEAN-VARIANCE OPTIMIZATION
# ===========================
def optimize_portfolio(
    mu,
    cov,
    prev_weights=None,
    lambda_=None,
    turnover_penalty=None,
    l2_reg=None,
):
    if lambda_ is None:
        lambda_ = RISK_AVERSION
    if turnover_penalty is None:
        turnover_penalty = TURNOVER_PENALTY
    if l2_reg is None:
        l2_reg = L2_REG
    n    = len(mu)
    w_eq = np.ones(n) / n

    def objective(w):
        utility = mu @ w - 0.5 * lambda_ * (w @ cov @ w)
        if prev_weights is not None:
            utility -= turnover_penalty * np.sum(np.abs(w - prev_weights))
        utility -= l2_reg * np.sum((w - w_eq) ** 2)
        return -utility

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds      = [(0, 1)] * n
    result      = minimize(objective, w_eq, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else w_eq


# ===========================
# BACKTEST LOOP
# ===========================
def run_backtest_with_benchmark(monthly_returns, shrunk_forecasts, cov_dict, spy_labels=None):
    dates          = sorted(shrunk_forecasts.index.get_level_values(0).unique())
    port_returns   = []
    bench_returns  = []
    weights_records = []
    weights_dates  = []
    prev_weights   = None

    monthly_regime = spy_labels.resample("ME").last() if spy_labels is not None else None

    for i, date in enumerate(dates):
        try:
            mu_series = shrunk_forecasts.loc[date]
        except KeyError:
            continue

        cov_date = date if date in cov_dict else max(
            (d for d in cov_dict if d <= date), default=None
        )
        if cov_date is None:
            continue
        cov_info    = cov_dict[cov_date]
        cov         = cov_info['cov']
        tickers_cov = cov_info['tickers']

        common = mu_series.index.intersection(tickers_cov)
        if len(common) < 2:
            continue

        mu      = mu_series.loc[common].values
        idx_map = {t: j for j, t in enumerate(tickers_cov)}
        idx     = [idx_map[t] for t in common]
        cov_sub = cov[np.ix_(idx, idx)]

        regime       = int(monthly_regime.get(date, 0)) if monthly_regime is not None else 0
        scaled_lambda = RISK_AVERSION * (REGIME_RISK_AVERSION_MULT if regime == 1 else 1.0)
        w_opt        = optimize_portfolio(mu, cov_sub, prev_weights, lambda_=scaled_lambda)
        prev_weights = w_opt
        w_bench      = np.ones(len(common)) / len(common)

        next_date = dates[i + 1] if i + 1 < len(dates) else None
        if next_date is not None and next_date in monthly_returns.index:
            rets_next = monthly_returns.loc[next_date][common].values
            port_returns.append(np.dot(w_opt,   rets_next))
            bench_returns.append(np.dot(w_bench, rets_next))
            weights_records.append(dict(zip(common, w_opt)))
            weights_dates.append(date)

    idx_slice   = dates[1:len(port_returns) + 1]
    port_ret    = pd.Series(port_returns,  index=idx_slice)
    bench_ret   = pd.Series(bench_returns, index=idx_slice)
    weights_df  = pd.DataFrame(weights_records, index=weights_dates).fillna(0)
    return port_ret, bench_ret, weights_df


# ===========================
# TRANSACTION COST APPLICATION
# ===========================
def apply_transaction_costs(gross_returns, weights_df, tc_bps=10, slippage_bps=5):
    """
    Deduct transaction costs + slippage on one-way turnover from gross returns.

    Turnover_k = Σ |w_k − w_{k-1}|.  Initial purchase = Σ |w_0|.
    Cost drag_k = (tc_bps + slippage_bps) / 10_000 * turnover_k

    Parameters
    ----------
    gross_returns : pd.Series  monthly gross returns (len == len(weights_df))
    weights_df    : pd.DataFrame  weights indexed by decision date
    tc_bps        : one-way transaction cost in basis points
    slippage_bps  : one-way slippage in basis points

    Returns
    -------
    net_returns : pd.Series  gross_returns minus cost drag, same index
    turnover    : pd.Series  one-way turnover per period, same index
    """
    total_cost = (tc_bps + slippage_bps) / 10_000
    w = weights_df.fillna(0)
    to = w.diff().abs().sum(axis=1)
    to.iloc[0] = w.iloc[0].abs().sum()
    to_vals = to.values[:len(gross_returns)]
    turnover = pd.Series(to_vals, index=gross_returns.index)
    net_returns = gross_returns - turnover * total_cost
    return net_returns, turnover


# ===========================
# PERFORMANCE METRICS
# ===========================
def compute_metrics(returns_series, name, benchmark_series=None):
    if len(returns_series) == 0:
        return {name: {}}

    monthly_rf = RISK_FREE_RATE / 12
    r          = returns_series.dropna()

    ann_return = r.mean() * 12
    cum_ret    = (1 + r).prod() - 1
    ann_vol    = r.std() * np.sqrt(12)

    cumulative  = (1 + r).cumprod()
    running_max = cumulative.expanding().max()
    drawdown    = (cumulative - running_max) / running_max
    max_dd      = drawdown.min()

    dd_lengths, count = [], 0
    for in_dd in (drawdown < 0):
        if in_dd:
            count += 1
        elif count:
            dd_lengths.append(count)
            count = 0
    if count:
        dd_lengths.append(count)
    avg_dd_dur = np.mean(dd_lengths) if dd_lengths else 0
    max_dd_dur = max(dd_lengths)     if dd_lengths else 0

    sharpe = (r.mean() - monthly_rf) / r.std() * np.sqrt(12)

    downside     = r[r < monthly_rf] - monthly_rf
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(12) if len(downside) else np.nan
    sortino      = (ann_return - RISK_FREE_RATE) / downside_std if (downside_std and downside_std > 0) else np.nan
    calmar       = ann_return / abs(max_dd) if max_dd else np.nan

    gains      = (r[r > monthly_rf] - monthly_rf).sum()
    losses_sum = (monthly_rf - r[r < monthly_rf]).sum()
    omega      = gains / losses_sum if losses_sum > 0 else np.nan

    var_95    = np.percentile(r, 5)
    cvar_95   = r[r <= var_95].mean() if (r <= var_95).any() else np.nan
    p95       = np.percentile(r, 95)
    tail_ratio = abs(p95) / abs(var_95) if var_95 != 0 else np.nan

    skewness    = r.skew()
    excess_kurt = r.kurt()

    hit_rate   = (r > 0).mean()
    wins       = r[r > 0]
    losses_neg = r[r < 0]
    avg_gain   = wins.mean()       if len(wins)       else np.nan
    avg_loss   = losses_neg.mean() if len(losses_neg) else np.nan
    win_loss   = abs(avg_gain / avg_loss) if (avg_loss and avg_loss != 0) else np.nan
    best_month  = r.max()
    worst_month = r.min()

    beta = alpha = ir = treynor = tracking_error = np.nan
    if benchmark_series is not None:
        common = r.index.intersection(benchmark_series.dropna().index)
        if len(common) > 12:
            rb, rp = benchmark_series.loc[common], r.loc[common]
            var_b  = np.var(rb)
            if var_b > 0:
                beta           = np.cov(rp, rb)[0, 1] / var_b
                alpha          = (rp.mean() - monthly_rf) * 12 - beta * (rb.mean() - monthly_rf) * 12
                excess         = rp - rb
                tracking_error = excess.std() * np.sqrt(12)
                ir             = excess.mean() * 12 / tracking_error if tracking_error > 0 else np.nan
                treynor        = (rp.mean() - monthly_rf) * 12 / beta if beta != 0 else np.nan

    return {
        name: {
            "Ann. Return (%)":      ann_return * 100,
            "Cumul. Return (%)":    cum_ret    * 100,
            "Ann. Volatility (%)":  ann_vol    * 100,
            "Max Drawdown (%)":     max_dd     * 100,
            "Avg DD Duration (mo)": avg_dd_dur,
            "Max DD Duration (mo)": max_dd_dur,
            "VaR 95% (%)":          var_95     * 100,
            "CVaR 95% (%)":         cvar_95    * 100,
            "Sharpe Ratio":         sharpe,
            "Sortino Ratio":        sortino,
            "Calmar Ratio":         calmar,
            "Omega Ratio":          omega,
            "Tail Ratio":           tail_ratio,
            "Beta":                 beta,
            "Alpha (ann. %)":       alpha          * 100 if not np.isnan(alpha)          else np.nan,
            "Treynor Ratio":        treynor,
            "Information Ratio":    ir,
            "Tracking Error (%)":   tracking_error * 100 if not np.isnan(tracking_error) else np.nan,
            "Skewness":             skewness,
            "Excess Kurtosis":      excess_kurt,
            "Hit Rate (%)":         hit_rate    * 100,
            "Best Month (%)":       best_month  * 100,
            "Worst Month (%)":      worst_month * 100,
            "Avg Gain (%)":         avg_gain    * 100 if not np.isnan(avg_gain) else np.nan,
            "Avg Loss (%)":         avg_loss    * 100 if not np.isnan(avg_loss) else np.nan,
            "Win/Loss Ratio":       win_loss,
        }
    }
