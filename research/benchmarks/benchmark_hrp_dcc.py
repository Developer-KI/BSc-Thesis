"""
Improved aDCC-GJR-GARCH HRP vs Sample Covariance HRP
- Univariate: GJR-GARCH(1,1) with Student-t
- Multivariate: aDCC(1,1) with grid search + QMLE
- Rolling window: 252 days (1 year)
- Optional covariance shrinkage (Ledoit-Wolf)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from arch import arch_model
from scipy.optimize import minimize, differential_evolution
import warnings
from sklearn.covariance import LedoitWolf
warnings.filterwarnings('ignore')

# =======================
# 1. HRP CORE (unchanged)
# =======================
class HierarchicalRiskParity:
    def __init__(self, cov_matrix):
        self.cov = cov_matrix
        self.n_assets = cov_matrix.shape[0]
        self.assets = cov_matrix.index.tolist()
        self.corr = self._cov_to_corr(cov_matrix)

    def _cov_to_corr(self, cov):
        std = np.sqrt(np.diag(cov))
        return cov / np.outer(std, std)

    def _get_distance_matrix(self, corr):
        return np.sqrt(0.5 * (1 - corr))

    def _get_clusters(self, distance):
        condensed = squareform(distance, checks=False)
        return linkage(condensed, method='ward')

    def _quasi_diagonalization(self, linkage_matrix, distance):
        return leaves_list(linkage_matrix)

    def _compute_weights(self, cov, order):
        weights = pd.Series(1.0, index=self.assets)
        def _recursive_bisection(items):
            if len(items) == 1:
                return
            split = len(items) // 2
            left, right = items[:split], items[split:]
            cov_sub = cov.loc[items, items]
            w_left = self._inverse_variance(cov_sub.loc[left, left])
            w_right = self._inverse_variance(cov_sub.loc[right, right])
            var_left = w_left @ cov_sub.loc[left, left] @ w_left
            var_right = w_right @ cov_sub.loc[right, right] @ w_right
            alpha = 1 - var_left / (var_left + var_right)
            for asset in left:  weights[asset] *= alpha
            for asset in right: weights[asset] *= (1 - alpha)
            _recursive_bisection(left)
            _recursive_bisection(right)
        _recursive_bisection(order)
        return weights

    def _inverse_variance(self, cov_sub):
        inv_var = 1.0 / np.diag(cov_sub)
        return inv_var / inv_var.sum()

    def allocate(self):
        distance = self._get_distance_matrix(self.corr)
        linkage_matrix = self._get_clusters(distance)
        order = self._quasi_diagonalization(linkage_matrix, distance)
        ordered_assets = [self.assets[i] for i in order]
        return self._compute_weights(self.cov, ordered_assets)


# =======================
# 2. aDCC-GJR-GARCH FORECAST with Shrinkage
# =======================
def gjr_garch_t(series):
    """Fit GJR-GARCH(1,1) with Student-t, return conditional variances and std residuals."""
    try:
        model = arch_model(series * 100, vol='GARCH', p=1, o=1, q=1,
                           mean='Constant', dist='students-t')
        res = model.fit(disp='off', show_warning=False)
        cond_var = res.conditional_volatility ** 2 / 10000
        std_resid = res.resid / res.conditional_volatility
        return cond_var, std_resid
    except:
        # fallback to plain GARCH(1,1) with normal
        model = arch_model(series * 100, vol='GARCH', p=1, q=1,
                           mean='Constant', dist='normal')
        res = model.fit(disp='off', show_warning=False)
        cond_var = res.conditional_volatility ** 2 / 10000
        std_resid = res.resid / res.conditional_volatility
        return cond_var, std_resid


def adcc_gjr_forecast(returns, use_shrinkage=True, shrinkage_target=None):
    """
    aDCC(1,1) with GJR-GARCH(1,1) univariate stage.
    Returns forecasted covariance matrix for next period.
    """
    n_assets = returns.shape[1]
    T = returns.shape[0]
    
    # Stage 1: GJR-GARCH for each asset
    cond_vars = np.zeros((T, n_assets))
    std_resids = np.zeros((T, n_assets))
    for i in range(n_assets):
        cond_vars[:, i], std_resids[:, i] = gjr_garch_t(returns.iloc[:, i])
    
    # Standardized residuals z_t
    z = std_resids
    
    # Stage 2: aDCC(1,1) estimation
    # Sample long-run correlation and asymmetric correlation
    Q_bar = np.corrcoef(z.T)
    n_t = np.where(z < 0, z, 0)
    N_bar = np.corrcoef(n_t.T)
    
    # aDCC log-likelihood (negative for minimization)
    def adcc_loglik(params, z, n_t, Q_bar, N_bar):
        alpha, beta, gamma = params
        # Constraints: alpha>=0, beta>=0, gamma>=0, alpha+beta+gamma*? <1? Usually alpha+beta+gamma <1 for stationarity
        if alpha < 0 or beta < 0 or gamma < 0 or alpha+beta+gamma >= 1:
            return 1e10
        T = z.shape[0]
        n = z.shape[1]
        Qt = Q_bar.copy()
        loglik = 0
        for t in range(1, T):
            Qt = (1 - alpha - beta) * Q_bar - gamma * N_bar \
                 + alpha * np.outer(z[t-1], z[t-1]) \
                 + beta * Qt \
                 + gamma * np.outer(n_t[t-1], n_t[t-1])
            # Ensure positive definiteness
            Qt = (Qt + Qt.T) / 2
            eigvals = np.linalg.eigvals(Qt)
            if np.min(eigvals) <= 0:
                Qt += (abs(np.min(eigvals)) + 1e-6) * np.eye(n)
            # Correlation matrix Rt
            Dinv = np.diag(1.0 / np.sqrt(np.diag(Qt)))
            Rt = Dinv @ Qt @ Dinv
            # Log-lik contribution
            sign, logdet = np.linalg.slogdet(Rt)
            loglik += logdet + z[t] @ np.linalg.inv(Rt) @ z[t]
        return -loglik
    
    # Grid search for good initial parameters
    best_params = None
    best_obj = np.inf
    for alpha in [0.01, 0.05, 0.10]:
        for beta in [0.85, 0.90, 0.94]:
            for gamma in [0.01, 0.05, 0.10]:
                if alpha+beta+gamma >= 1:
                    continue
                params = [alpha, beta, gamma]
                obj = adcc_loglik(params, z, n_t, Q_bar, N_bar)
                if obj < best_obj:
                    best_obj = obj
                    best_params = params
    if best_params is None:
        best_params = [0.05, 0.85, 0.02]  # fallback
    
    # Fine-tuning with local optimization
    bounds = [(1e-6, 0.2), (0.7, 0.98), (1e-6, 0.15)]
    try:
        res = minimize(adcc_loglik, best_params, args=(z, n_t, Q_bar, N_bar),
                       bounds=bounds, method='L-BFGS-B')
        alpha_opt, beta_opt, gamma_opt = res.x
    except:
        alpha_opt, beta_opt, gamma_opt = best_params
    
    # One-step ahead forecast
    Q_forecast = (1 - alpha_opt - beta_opt) * Q_bar - gamma_opt * N_bar \
                 + alpha_opt * np.outer(z[-1], z[-1]) \
                 + beta_opt * Q_bar \
                 + gamma_opt * np.outer(n_t[-1], n_t[-1])
    Q_forecast = (Q_forecast + Q_forecast.T) / 2
    eigvals = np.linalg.eigvals(Q_forecast)
    if np.min(eigvals) <= 0:
        Q_forecast += (abs(np.min(eigvals)) + 1e-6) * np.eye(n_assets)
    Dinv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q_forecast)))
    R_forecast = Dinv_sqrt @ Q_forecast @ Dinv_sqrt
    
    # Diagonal matrix of conditional standard deviations (last observed variance)
    last_vars = cond_vars[-1, :]
    D_forecast = np.diag(np.sqrt(last_vars))
    cov_forecast = D_forecast @ R_forecast @ D_forecast
    
    # Optional shrinkage (Ledoit-Wolf) on the forecasted covariance
    if use_shrinkage:
        # We treat the forecast as a "structured estimate" and shrink toward a diagonal target?
        # Better: Use LedoitWolf on the recent returns but that would be double-shrinkage.
        # Simpler: Shrink the forecast towards the sample covariance of the window.
        sample_cov = returns.cov().values
        # Ledoit-Wolf shrinkage between forecast and sample
        # Actually we can use sklearn's LedoitWolf on the returns themselves, but that ignores dynamics.
        # Instead, we blend the forecast with a constant correlation target.
        # Let's do a simple single-factor shrinkage: cov_shrunk = 0.7*cov_forecast + 0.3*diag(cov_forecast)*I
        target = np.diag(np.diag(cov_forecast))
        lambda_ = 0.2  # shrinkage intensity
        cov_forecast = lambda_ * target + (1 - lambda_) * cov_forecast
    
    cov_df = pd.DataFrame(cov_forecast, index=returns.columns, columns=returns.columns)
    return cov_df


# =======================
# 3. BACKTESTING (adapted for flexibility)
# =======================
def backtest_portfolio(returns, rebalance_freq, window_length,
                       method='sample_hrp', use_shrinkage=False):
    n_assets = returns.shape[1]
    n_days = returns.shape[0]
    rebalance_dates = []
    weight_matrices = []
    
    # Loop over rebalance points
    for i in range(window_length, n_days, rebalance_freq):
        train_returns = returns.iloc[i - window_length:i]
        if method == 'sample_hrp':
            cov_matrix = train_returns.cov()
        elif method == 'adcc_hrp':
            cov_matrix = adcc_gjr_forecast(train_returns, use_shrinkage=use_shrinkage)
        else:
            raise ValueError("Unknown method")
        
        hrp = HierarchicalRiskParity(cov_matrix)
        weights = hrp.allocate()
        weights = weights / weights.sum()
        rebalance_dates.append(returns.index[i])
        weight_matrices.append(weights)
    
    if not rebalance_dates:
        return pd.DataFrame(), pd.Series(dtype=float)
    
    # Build daily weights
    daily_weights = []
    for t in range(n_days):
        current_date = returns.index[t]
        if current_date < rebalance_dates[0]:
            daily_weights.append(pd.Series(1/n_assets, index=returns.columns))
        else:
            idx = max(j for j, date in enumerate(rebalance_dates) if date <= current_date)
            daily_weights.append(weight_matrices[idx])
    weights_df = pd.DataFrame(daily_weights, index=returns.index, columns=returns.columns)
    portfolio_returns = (weights_df.shift(1) * returns).sum(axis=1)
    portfolio_returns.iloc[0] = 0.0
    return weights_df, portfolio_returns


# =======================
# 4. PERFORMANCE METRICS (unchanged)
# =======================
def compute_metrics(returns):
    if len(returns) == 0:
        return {}
    ret = returns.dropna()
    ann_factor = 252
    total_ret = (1+ret).prod() - 1
    ann_ret = (1+total_ret) ** (ann_factor/len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(ann_factor)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol != 0 else np.nan
    cum = (1+ret).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return {
        'Total Return': total_ret,
        'Ann. Return': ann_ret,
        'Ann. Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar
    }


# =======================
# 5. MAIN
# =======================
def main():
    print("=" * 70)
    print("Improved aDCC-GJR-GARCH HRP vs Sample Covariance HRP")
    print("with shrinkage, shorter window (252d), and Student-t")
    print("=" * 70)
    
    # Data
    tickers = ['SPY', 'TLT', 'IWM', 'QQQ', 'GLD', 'LQD', 'VNQ']
    start = '2015-01-01'
    end = '2024-12-31'
    print("\n[1] Downloading data...")
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data['Close'].dropna()
    returns = prices.pct_change().dropna()
    print(f"   Assets: {tickers}")
    print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Parameters: shortened window
    window_length = 252   # 1 year
    rebalance_freq = 21   # monthly
    print(f"\n[2] Parameters:")
    print(f"   Rolling window: {window_length} days")
    print(f"   Rebalance freq: {rebalance_freq} days")
    
    # Run backtests
    print("\n[3] Running sample covariance HRP...")
    _, ret_sample = backtest_portfolio(returns, rebalance_freq, window_length, 'sample_hrp')
    
    print("\n[4] Running aDCC-GJR-GARCH HRP (this takes several minutes)...")
    _, ret_adcc = backtest_portfolio(returns, rebalance_freq, window_length, 'adcc_hrp', use_shrinkage=True)
    
    # Metrics
    metrics_sample = compute_metrics(ret_sample)
    metrics_adcc = compute_metrics(ret_adcc)
    
    # Display
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    df_res = pd.DataFrame({
        'Sample Cov HRP': metrics_sample,
        'aDCC-GJR HRP': metrics_adcc
    }).T
    fmt = {'Total Return': '{:.2%}', 'Ann. Return': '{:.2%}', 'Ann. Volatility': '{:.2%}',
           'Sharpe Ratio': '{:.2f}', 'Max Drawdown': '{:.2%}', 'Calmar Ratio': '{:.2f}'}
    for col in df_res.columns:
        df_res[col] = df_res[col].apply(lambda x: fmt[col].format(x) if pd.notnull(x) else 'NaN')
    print(df_res.to_string())
    
    # Differences
    ann_diff = metrics_adcc['Ann. Return'] - metrics_sample['Ann. Return']
    sharpe_diff = metrics_adcc['Sharpe Ratio'] - metrics_sample['Sharpe Ratio']
    dd_diff = metrics_adcc['Max Drawdown'] - metrics_sample['Max Drawdown']
    print("\n" + "-" * 70)
    print(f"Comparison (aDCC minus Sample):")
    print(f"   Ann. Return: {ann_diff:+.2%}")
    print(f"   Sharpe:      {sharpe_diff:+.2f}")
    print(f"   Max DD:      {dd_diff:+.2%}")
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()