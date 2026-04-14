import numpy as np
import pandas as pd
import tensorflow as tf2
import warnings
import matplotlib.pyplot as plt
import os
import optuna
from tqdm import tqdm
from scipy.spatial import distance
from numpy import linalg as la
from scipy.cluster.hierarchy import linkage, dendrogram
# Need to cite to use in published work
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# --- ENVIRONMENT SETUP ---
tf = tf2.compat.v1
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore')
real_type = tf.float32

if not os.path.exists('./results'):
    os.makedirs('./results')

# --- MATHEMATICAL UTILITIES ---

def nearestPD(A):
    """Finds the nearest positive-definite matrix for numerical stability"""
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3): return A3
    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def isPD(B):
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def get_quasi_diag(link):
    """Recursive clustering sort for hierarchical allocation"""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_cluster_var(cov, c_items):
    """Calculates cluster variance based on Inverse Variance Portfolio"""
    cov_ = cov.iloc[c_items, c_items]
    ivp = 1. / np.diag(cov_)
    ivp /= ivp.sum()
    w_ = ivp.reshape(-1, 1)
    return np.dot(np.dot(w_.T, cov_.values), w_)[0, 0]

def get_rec_bipart(cov, sort_ix):
    """Lopez de Prado Recursive Bisection (Parity weight allocation)"""
    w = pd.Series(1.0, index=sort_ix) 
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[int(j):int(k)] for i in c_items for j, k in 
                   ((0, len(i)/2), (len(i)/2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0, c_items1 = c_items[i], c_items[i+1]
            v0, v1 = get_cluster_var(cov, c_items0), get_cluster_var(cov, c_items1)
            alpha = 1 - v0 / (v0 + v1)
            w.loc[c_items0] *= alpha
            w.loc[c_items1] *= (1 - alpha)
    return w

# --- NEURAL NETWORK ARCHITECTURE (PDE & AAD) ---
def vanilla_net(input_dim, hidden_units, hidden_layers, seed):
    tf.set_random_seed(seed)
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)
    ws, bs, zs = [None], [None], [xs]
    ws.append(tf.get_variable("w1", [input_dim, hidden_units], initializer=tf.variance_scaling_initializer()))
    bs.append(tf.get_variable("b1", [hidden_units], initializer=tf.zeros_initializer()))
    zs.append(tf.matmul(zs[0], ws[1]) + bs[1])
    for l in range(1, hidden_layers):
        ws.append(tf.get_variable(f"w{l+1}", [hidden_units, hidden_units], initializer=tf.variance_scaling_initializer()))
        bs.append(tf.get_variable(f"b{l+1}", [hidden_units], initializer=tf.zeros_initializer()))
        zs.append(tf.nn.softplus(zs[l]) @ ws[l+1] + bs[l+1])
    ws.append(tf.get_variable(f"w{hidden_layers+1}", [hidden_units, 1], initializer=tf.variance_scaling_initializer()))
    bs.append(tf.get_variable(f"b{hidden_layers+1}", [1], initializer=tf.zeros_initializer()))
    zs.append(tf.nn.softplus(zs[hidden_layers]) @ ws[hidden_layers+1] + bs[hidden_layers+1])
    return xs, (ws, bs), zs, zs[-1]

def backprop(weights_and_biases, zs):
    """AAD logic to extract partial derivatives (Sensitivities)"""
    ws, _ = weights_and_biases
    L = len(zs) - 1
    zbar = tf.ones_like(zs[L])
    for l in range(L - 1, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l+1])) * tf.nn.sigmoid(zs[l])
    return zbar @ tf.transpose(ws[1])

def objective(trial, X_data, Y_data):
    """Optuna objective function for architecture search."""
    tf.reset_default_graph()
    h_layers = trial.suggest_int('hidden_layers', 1, 3)
    h_units = trial.suggest_int('hidden_units', 10, 40)
    inputs, _, _, outputs = vanilla_net(X_data.shape[1], h_units, h_layers, 42)
    labels_ph = tf.placeholder(real_type, [None, 1])
    loss = tf.losses.mean_squared_error(labels_ph, outputs)
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(30): sess.run(train_op, feed_dict={inputs: X_data, labels_ph: Y_data})
        return sess.run(loss, feed_dict={inputs: X_data, labels_ph: Y_data})

# --- DATA LOADING & SYNC ---
START_DATE = ""
END_DATE = ""

# Lookabck period for all strategies
PM_cov_window = 66

# HSP Strategy Parameters
n_corr, N_Drivers_Selection = 125, 12

Drivers = pd.read_excel(r'./data/Drivers_no_SB_Sectors.xlsx').set_index("Date")
Constituents = pd.read_excel(r'./data/Assets_SPX.xlsx').set_index('Date')

Data_Assets = Constituents.pct_change(1).fillna(0).astype(float)
Assets_Names = Data_Assets.columns.values
Drivers.index = pd.to_datetime(Drivers.index, format='%Y%m%d')
Data_Assets.index = pd.to_datetime(Data_Assets.index)
Data_Drivers_shift0 = pd.merge(Drivers, Data_Assets, left_index=True, right_index=True)

if START_DATE != "":
    Data_Drivers_shift0 = Data_Drivers_shift0.loc[START_DATE:]

dates_series = Data_Drivers_shift0.index[n_corr::22]

# Storage
weights_history = {m: pd.DataFrame() for m in ['HSP', 'HRP', 'MinVol', 'MaxSharpe', 'Max_Util', 'Robust_MinVol', 'Robust_MaxSharpe', 'Robust_Max_Util', '1/N']}
is_metrics = []
last_hsp_link, last_hrp_link = None, None

# Seatch Depth
TRIALS = 8
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CORE BACKTEST LOOP ---
for rebalance_date in tqdm(dates_series, desc="Monthly Optimization"):
    ii = Data_Drivers_shift0.index.get_loc(rebalance_date)
    lookback_returns = Data_Assets.iloc[ii-PM_cov_window:ii]
    lookback_prices = Constituents.iloc[ii-PM_cov_window:ii]
    cov = lookback_returns.cov()
    
    # Create Sensitivity Matrix
    driver_candidates = Drivers.columns
    corr_block = Data_Drivers_shift0.iloc[ii-n_corr:ii].corr().fillna(0)
    vote_counts = pd.Series(0, index=driver_candidates)
    
    for asset in Assets_Names:
        asset_corrs = corr_block[asset].loc[driver_candidates].abs()
        top_for_asset = asset_corrs.nlargest(N_Drivers_Selection).index
        vote_counts[top_for_asset] += 1
    
    selected_drivers = vote_counts.nlargest(N_Drivers_Selection).index.tolist()
    
    sens_results = []
    for asset in Assets_Names:
        # Prepare Data
        X_drivers = Data_Drivers_shift0[selected_drivers].iloc[ii-100:ii].values
        X_lag = Data_Assets[asset].iloc[ii-101:ii-1].values.reshape(-1, 1)
        X_pde, Y = np.hstack([X_drivers, X_lag]), Data_Assets[asset].iloc[ii-100:ii].values.reshape(-1, 1)
        
        # Bayesian Search
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, X_pde, Y), n_trials=TRIALS)
        
        # Train Best Model & Extract AAD
        tf.reset_default_graph()
        inputs, wb, ln, outputs = vanilla_net(X_pde.shape[1], study.best_params['hidden_units'], study.best_params['hidden_layers'], 42)
        greeks_op = backprop(wb, ln)
        labels_ph = tf.placeholder(real_type, [None, 1])
        train_op = tf.train.AdamOptimizer(0.01).minimize(tf.losses.mean_squared_error(labels_ph, outputs))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(50): sess.run(train_op, feed_dict={inputs: X_pde, labels_ph: Y})
            sens_vals = sess.run(greeks_op, feed_dict={inputs: X_pde})
            sens_results.append(np.append(np.nan_to_num(np.mean(sens_vals, axis=0))[:N_Drivers_Selection], asset))

    # 1. HSP 
    df_sens = pd.DataFrame(sens_results, columns=selected_drivers + ["Asset"]).set_index("Asset").astype(float).reindex(Assets_Names)
    hsp_link = linkage(nearestPD(distance.cdist(df_sens.values, df_sens.values, 'euclidean')), 'single')
    last_hsp_link = hsp_link
    w_hsp = get_rec_bipart(cov, get_quasi_diag(hsp_link))
    w_hsp.index = [Assets_Names[i] for i in w_hsp.index]

    # 2. HRP
    hrp_link = linkage(lookback_returns.corr(), 'single')
    last_hrp_link = hrp_link
    w_hrp = get_rec_bipart(cov, get_quasi_diag(hrp_link))
    w_hrp.index = [Assets_Names[i] for i in w_hrp.index]

    # 3. Efficient Frontier
    mu, S_risk = expected_returns.mean_historical_return(lookback_prices), risk_models.sample_cov(lookback_prices)
    ef = EfficientFrontier(mu, S_risk); w_minvol = pd.Series(ef.min_volatility())
    ef = EfficientFrontier(mu, S_risk); w_maxsharpe = pd.Series(ef.max_sharpe())

    # 4. Benchmark 1/N
    w_equal = pd.Series(1/len(Assets_Names), index=Assets_Names)

    # 5. Robust Ledoit-Wolf + L2 Regularization
    S_robust = risk_models.CovarianceShrinkage(lookback_prices).ledoit_wolf()

    ef_robust = EfficientFrontier(mu, S_robust)
    ef_robust.add_objective(objective_functions.L2_reg, gamma=0.1)
    w_robust_min = pd.Series(ef_robust.min_volatility())

    ef_robust = EfficientFrontier(mu, S_robust)
    ef_robust.add_objective(objective_functions.L2_reg, gamma=0.1)
    w_robust_max = pd.Series(ef_robust.max_sharpe())

    # 6. Maximization Quadratic utility
    ef = EfficientFrontier(mu, S_risk); w_max_qu = pd.Series(ef.max_quadratic_utility)
    ef_robust = EfficientFrontier(mu, S_robust)
    ef_robust.add_objective(objective_functions.L2_reg, gamma=0.1)
    w_robust_max_qu = pd.Series(ef_robust.max_quadratic_utility())

    # Weights Sync
    model_list = [w_hsp, w_hrp, w_minvol, w_maxsharpe, w_max_qu, w_robust_min, w_robust_max, w_robust_max_qu, w_equal]
    for name, w in zip(weights_history.keys(), model_list):
        temp_w = pd.Series(w).reindex(Assets_Names).fillna(0)
        temp_w['Date'] = rebalance_date
        weights_history[name] = pd.concat([weights_history[name], temp_w.to_frame().T], ignore_index=True)
        ret_is = lookback_returns.dot(temp_w[Assets_Names])
        is_metrics.append({'Date': rebalance_date, 'Strategy': name, 'Sharpe': (ret_is.mean()/ret_is.std())*np.sqrt(252)})

# --- GENERATE OUTPUTS ---
# Dendrogram Plots
print("Saving Dendrograms...")
for link, name in zip([last_hsp_link, last_hrp_link], ["HSP (Sensitivity)", "HRP (Correlation)"]):
    plt.figure(figsize=(10, 6))
    dendrogram(link, labels=Assets_Names, leaf_rotation=90)
    plt.title(f"{name} Hierarchy - Last Window")
    plt.tight_layout()
    plt.savefig(f'./results/{name.split()[0]}_Dendrogram_{START_DATE}.png')
    plt.close()

# Out-of-Sample Performance
oos_returns = pd.DataFrame()
for i in range(len(Data_Assets)):
    dt = Data_Assets.index[i]
    daily = {'Date': dt}
    for name in weights_history:
        v_w = weights_history[name][weights_history[name]['Date'] <= dt]
        if not v_w.empty: daily[name] = Data_Assets.iloc[i].dot(v_w.iloc[-1][Assets_Names].astype(float))
    if len(daily) > 1: oos_returns = pd.concat([oos_returns, pd.DataFrame([daily])], ignore_index=True)

oos_returns.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
(1 + oos_returns).cumprod().plot(ax=plt.gca(), title="OOS Cumulative Returns Strategy Comparison")
plt.grid(True); plt.savefig(f'./results/OOS_Performance_{START_DATE}.png'); plt.close()

# In-Sample Metrics Bar Chart
is_summary = pd.DataFrame(is_metrics).groupby('Strategy')['Sharpe'].mean()
plt.figure(figsize=(10, 5))
is_summary.plot(kind='bar', color='teal', title="Average In-Sample Sharpe (Model Calibration)")
plt.ylabel("Sharpe Ratio")
plt.savefig(f'./results/IS_Comparison_{START_DATE}.png'); plt.close()

# Final Comparison Table
stats = pd.DataFrame({
    'Ann. Return (%)': oos_returns.mean() * 252 * 100,
    'Ann. Vol (%)': oos_returns.std() * np.sqrt(252) * 100,
    'Sharpe Ratio': (oos_returns.mean() * 252) / (oos_returns.std() * np.sqrt(252)),
    'Train Sharpe Ratio': is_summary
})
stats.to_excel(f'./results/Performance_Summary_{START_DATE}.xlsx')

print("\n--- Final Metrics Summary ---")
print(stats)
print("\nProcess Complete. Check the './results' directory for plots and the metrics spreadsheet")