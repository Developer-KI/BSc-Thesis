import sys
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from pathlib import Path
# Need to cite to use in published work
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plotting import (
    plot_cumulative_returns, plot_drawdown,
    plot_rolling_sharpe, plot_sharpe_bar, plot_performance_summary,
)

# --- ENVIRONMENT SETUP ---
warnings.filterwarnings('ignore')


# --- DATA LOADING & SYNC ---
START_DATE = "2021-06-01"
END_DATE = "2022-06-01"
PM_cov_window = 66
RUN_IDENTIFIER = "MV_Standard_vs_Robust"

Constituents = pd.read_excel(r'./data/test_assets.xlsx').set_index('Date')
Data_Assets = Constituents.pct_change(1).fillna(0).astype(float)
Assets_Names = Data_Assets.columns.values
Data_Assets.index = pd.to_datetime(Data_Assets.index)

if START_DATE != "":
    Data_Assets = Data_Assets.loc[START_DATE:]

actual_start = Data_Assets.index[0].strftime('%Y-%m-%d')
actual_end   = END_DATE if END_DATE != "" else Data_Assets.index[-1].strftime('%Y-%m-%d')
RESULTS_DIR  = Path(__file__).resolve().parent.parent / 'results' / f'{actual_start}_to_{actual_end}_{RUN_IDENTIFIER}'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}")

dates_series = Data_Assets.index[PM_cov_window::22]

# Storage
weights_history = {m: pd.DataFrame() for m in ['MinVol', 'MaxSharpe', 'Robust_MinVol', 'Robust_MaxSharpe', '1/N']}
is_metrics = []

# --- CORE BACKTEST LOOP ---
for rebalance_date in tqdm(dates_series, desc="MV vs Robust Optimization"):
    ii = Data_Assets.index.get_loc(rebalance_date)
    lookback_returns = Data_Assets.iloc[ii-PM_cov_window:ii]
    lookback_prices = Constituents.loc[lookback_returns.index]
    
    # 1. Standard Efficient Frontier (Sample Cov)
    mu, S_risk = expected_returns.mean_historical_return(lookback_prices), risk_models.sample_cov(lookback_prices)
    ef = EfficientFrontier(mu, S_risk); w_minvol = pd.Series(ef.min_volatility())
    ef = EfficientFrontier(mu, S_risk); w_maxsharpe = pd.Series(ef.max_sharpe())

    # 2. Robust: Ledoit-Wolf + L2 Regularization
    S_robust = risk_models.CovarianceShrinkage(lookback_prices).ledoit_wolf()

    ef_robust = EfficientFrontier(mu, S_robust)
    ef_robust.add_objective(objective_functions.L2_reg, gamma=0.1)
    w_robust_min = pd.Series(ef_robust.min_volatility())

    ef_robust = EfficientFrontier(mu, S_robust)
    ef_robust.add_objective(objective_functions.L2_reg, gamma=0.1)
    w_robust_max = pd.Series(ef_robust.max_sharpe())

    # 3. Benchmark 1/N
    w_equal = pd.Series(1/len(Assets_Names), index=Assets_Names)

    # Weights Sync
    model_list = [w_minvol, w_maxsharpe, w_robust_min, w_robust_max, w_equal]
    for name, w in zip(weights_history.keys(), model_list):
        temp_w = pd.Series(w).reindex(Assets_Names).fillna(0)
        temp_w['Date'] = rebalance_date
        weights_history[name] = pd.concat([weights_history[name], temp_w.to_frame().T], ignore_index=True)
        ret_is = lookback_returns.dot(temp_w[Assets_Names])
        is_metrics.append({'Date': rebalance_date, 'Strategy': name, 'Sharpe': (ret_is.mean()/ret_is.std())*np.sqrt(252)})

# --- GENERATE OUTPUTS ---
oos_returns = pd.DataFrame()
for i in range(len(Data_Assets)):
    dt = Data_Assets.index[i]
    daily = {'Date': dt}
    for name in weights_history:
        v_w = weights_history[name][weights_history[name]['Date'] <= dt]
        if not v_w.empty:
            daily[name] = Data_Assets.iloc[i].dot(v_w.iloc[-1][Assets_Names].astype(float))
    if len(daily) > 1:
        oos_returns = pd.concat([oos_returns, pd.DataFrame([daily])], ignore_index=True)

oos_returns.set_index('Date', inplace=True)

plot_cumulative_returns(oos_returns, title=f"OOS Returns - {RUN_IDENTIFIER}", save_path=str(RESULTS_DIR / "OOS_Cumulative_Returns.png"))
plot_drawdown(oos_returns, title=f"OOS Drawdown - {RUN_IDENTIFIER}", save_path=str(RESULTS_DIR / "OOS_Drawdown.png"))
plot_rolling_sharpe(oos_returns, title=f"OOS Rolling Sharpe - {RUN_IDENTIFIER}", save_path=str(RESULTS_DIR / "OOS_Rolling_Sharpe.png"))

is_summary = pd.DataFrame(is_metrics).groupby('Strategy')['Sharpe'].mean()
plot_sharpe_bar(is_summary, title="Avg In-Sample Sharpe", save_path=str(RESULTS_DIR / "IS_Sharpe_Comparison.png"))

cum_returns = (1 + oos_returns).cumprod()
max_dd = ((cum_returns / cum_returns.cummax()) - 1).min() * 100
stats = pd.DataFrame({
    'Ann. Return (%)': oos_returns.mean() * 252 * 100,
    'Ann. Vol (%)': oos_returns.std() * np.sqrt(252) * 100,
    'Sharpe Ratio': (oos_returns.mean() * 252) / (oos_returns.std() * np.sqrt(252)),
    'Max Drawdown (%)': max_dd,
    'Train Sharpe Ratio': is_summary,
})

plot_performance_summary(stats[['Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe Ratio', 'Max Drawdown (%)']], save_path=str(RESULTS_DIR / "Performance_Summary_Panel.png"))
stats.to_excel(str(RESULTS_DIR / "Performance_Summary.xlsx"))
print(stats)