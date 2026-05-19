# hrp_vs_ew_equity_curve_fixed_with_drawdown.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt import risk_models
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'JPM', 'JNJ', 'V', 'PG']
START_DATE = '2013-01-01'
END_DATE   = '2025-12-31'
LOOKBACK = 252                  # days for covariance estimation (1 year)
REBALANCE_FREQ = 'ME'           # Month-end rebalancing

# -------------------------------------------------------------------
# 2. Download and prepare data
# -------------------------------------------------------------------
print("Downloading data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                   auto_adjust=True, group_by='ticker')

if isinstance(data.columns, pd.MultiIndex):
    data = data.xs('Close', axis=1, level=1)

data = data.dropna(axis=1, how='all')
print(f"Data shape: {data.shape}")

returns = data.pct_change().dropna(how='all')

# -------------------------------------------------------------------
# 3. HRP backtest function (corrected, no lookahead bias)
# -------------------------------------------------------------------
def backtest_hrp(returns, start_date, end_date, lookback, rebalance_freq='ME'):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if rebalance_freq == 'Q':
        rebal_dates = pd.date_range(start=start_ts, end=end_ts, freq='Q')
    elif rebalance_freq == 'ME':
        rebal_dates = pd.date_range(start=start_ts, end=end_ts, freq='ME')
    elif rebalance_freq == 'Y':
        rebal_dates = pd.date_range(start=start_ts, end=end_ts, freq='Y')
    else:
        raise ValueError("rebalance_freq must be 'Q', 'ME', or 'Y'")

    rebal_dates = [d for d in rebal_dates if d in returns.index]
    if not rebal_dates:
        raise ValueError("No rebalancing dates found.")

    portfolio_returns = []
    all_weights = []

    for i, rebal_date in enumerate(rebal_dates):
        start_lookback = rebal_date - pd.Timedelta(days=lookback)
        hist_returns = returns.loc[start_lookback:rebal_date].dropna()

        if len(hist_returns) < 10:
            print(f"⚠️  Not enough data at {rebal_date.date()}, skipping.")
            continue

        try:
            cov_matrix = risk_models.sample_cov(hist_returns)
        except Exception as e:
            print(f"⚠️  Covariance estimation failed at {rebal_date.date()}: {e}")
            continue

        hrp = HRPOpt(returns=hist_returns, cov_matrix=cov_matrix)
        weights = hrp.optimize()
        weights_series = pd.Series(weights, index=hist_returns.columns)

        if i + 1 < len(rebal_dates):
            next_rebal = rebal_dates[i + 1]
            start_hold = rebal_date + pd.Timedelta(days=1)
            idx_start = returns.index.searchsorted(start_hold)
            if idx_start >= len(returns.index):
                continue
            hold_start_date = returns.index[idx_start]

            idx_end = returns.index.searchsorted(next_rebal)
            if idx_end == 0:
                continue
            hold_end_date = returns.index[idx_end - 1]

            period_returns = returns.loc[hold_start_date:hold_end_date][weights_series.index]
        else:
            start_hold = rebal_date + pd.Timedelta(days=1)
            idx_start = returns.index.searchsorted(start_hold)
            if idx_start >= len(returns.index):
                continue
            hold_start_date = returns.index[idx_start]
            period_returns = returns.loc[hold_start_date:end_ts][weights_series.index]

        period_port_returns = (period_returns * weights_series).sum(axis=1)
        portfolio_returns.append(period_port_returns)
        all_weights.append(weights_series)

    if not portfolio_returns:
        raise RuntimeError("No valid periods for HRP backtest.")

    portfolio_returns = pd.concat(portfolio_returns)
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]
    return portfolio_returns.sort_index()


def equal_weight_returns(returns, start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    ew_weights = np.ones(len(returns.columns)) / len(returns.columns)
    return returns.loc[start_ts:end_ts].dot(ew_weights)


def compute_drawdown(equity_curve):
    """Compute drawdown series from a cumulative equity curve."""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown

# -------------------------------------------------------------------
# 4. Run backtests
# -------------------------------------------------------------------
print("\nComputing HRP portfolio...")
hrp_returns = backtest_hrp(returns, START_DATE, END_DATE, LOOKBACK, REBALANCE_FREQ)

print("Computing Equal Weight portfolio...")
ew_returns = equal_weight_returns(returns, START_DATE, END_DATE)

# -------------------------------------------------------------------
# 5. Equity curves and drawdowns
# -------------------------------------------------------------------
hrp_equity = (1 + hrp_returns).cumprod()
ew_equity  = (1 + ew_returns).cumprod()

hrp_drawdown = compute_drawdown(hrp_equity)
ew_drawdown  = compute_drawdown(ew_equity)

# -------------------------------------------------------------------
# 6. Performance metrics
# -------------------------------------------------------------------
def calc_metrics(returns):
    annual_factor = 252
    ann_return = returns.mean() * annual_factor
    ann_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    # Additional: max drawdown
    equity = (1 + returns).cumprod()
    dd = compute_drawdown(equity)
    max_dd = dd.min()
    return ann_return, ann_vol, sharpe, max_dd

hrp_ann_ret, hrp_ann_vol, hrp_sharpe, hrp_max_dd = calc_metrics(hrp_returns)
ew_ann_ret,  ew_ann_vol,  ew_sharpe,  ew_max_dd  = calc_metrics(ew_returns)

# -------------------------------------------------------------------
# 7. Plotting: Equity curves (top) + Drawdowns (bottom)
# -------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# --- HRP Equity (top left) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(hrp_equity.index, hrp_equity.values, label='HRP', color='blue', linewidth=2)
ax1.set_title('HRP – Equity Curve')
ax1.set_ylabel('Cumulative Return')
ax1.grid(alpha=0.3)
ax1.legend()
text1 = f'Ann. Return: {hrp_ann_ret:.2%}\nAnn. Vol: {hrp_ann_vol:.2%}\nSharpe: {hrp_sharpe:.2f}\nMax DD: {hrp_max_dd:.2%}'
ax1.text(0.05, 0.95, text1, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- EW Equity (top right) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ew_equity.index, ew_equity.values, label='Equal Weight', color='green', linewidth=2)
ax2.set_title('Equal Weight – Equity Curve')
ax2.set_ylabel('Cumulative Return')
ax2.grid(alpha=0.3)
ax2.legend()
text2 = f'Ann. Return: {ew_ann_ret:.2%}\nAnn. Vol: {ew_ann_vol:.2%}\nSharpe: {ew_sharpe:.2f}\nMax DD: {ew_max_dd:.2%}'
ax2.text(0.05, 0.95, text2, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- HRP Drawdown (bottom left) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(hrp_drawdown.index, 0, hrp_drawdown.values, color='red', alpha=0.4, label='Drawdown')
ax3.plot(hrp_drawdown.index, hrp_drawdown.values, color='darkred', linewidth=1)
ax3.set_title('HRP – Drawdown')
ax3.set_xlabel('Date')
ax3.set_ylabel('Drawdown')
ax3.grid(alpha=0.3)
ax3.legend()

# --- EW Drawdown (bottom right) ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.fill_between(ew_drawdown.index, 0, ew_drawdown.values, color='red', alpha=0.4, label='Drawdown')
ax4.plot(ew_drawdown.index, ew_drawdown.values, color='darkred', linewidth=1)
ax4.set_title('Equal Weight – Drawdown')
ax4.set_xlabel('Date')
ax4.set_ylabel('Drawdown')
ax4.grid(alpha=0.3)
ax4.legend()

plt.suptitle('HRP vs. Equal Weight – Equity Curves & Drawdowns', fontsize=16, y=0.98)
plt.savefig('hrp_vs_ew_equity_drawdown.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Done! Comparison saved as 'hrp_vs_ew_equity_drawdown.png'")