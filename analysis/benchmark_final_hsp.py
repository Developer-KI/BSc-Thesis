import numpy as np
import pandas as pd
import tensorflow as tf2
import warnings
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance
from numpy import linalg as la
from scipy.cluster.hierarchy import linkage
from hmmlearn import hmm
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# --- CONFIGURATION ---
UNIVERSE_SIZE = 10
USE_SPY_ONLY_REGIME = True  
INDEX_TICKER = 'SPY'        
N_REB = 30              
M_GAP = 10              
HMM_WINDOW = 252        
EST_WINDOW = 125        
N_DRIVERS = 12          
TRIALS = 5              

tf = tf2.compat.v1
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore')
real_type = tf.float32
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- MATH UTILITIES ---
def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3): return A3
    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0]); k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def isPD(B):
    try:
        _ = la.cholesky(B); return True
    except la.LinAlgError: return False

def get_quasi_diag(link):
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
    cov_ = cov.iloc[c_items, c_items]
    ivp = 1. / (np.diag(cov_) + 1e-9)
    ivp /= ivp.sum()
    w_ = ivp.reshape(-1, 1)
    return np.dot(np.dot(w_.T, cov_.values), w_)[0, 0]

def get_rec_bipart(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix) 
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[int(j):int(k)] for i in c_items for j, k in 
                   ((0, len(i)/2), (len(i)/2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0, c_items1 = c_items[i], c_items[i+1]
            v0, v1 = get_cluster_var(cov, c_items0), get_cluster_var(cov, c_items1)
            alpha = 1 - v0 / (v0 + v1 + 1e-12)
            w.loc[c_items0] *= alpha
            w.loc[c_items1] *= (1 - alpha)
    return w

# --- HSP NEURAL NET ---
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
    ws, _ = weights_and_biases
    L = len(zs) - 1
    zbar = tf.ones_like(zs[L])
    for l in range(L - 1, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l+1])) * tf.nn.sigmoid(zs[l])
    return zbar @ tf.transpose(ws[1])

def objective(trial, X_data, Y_data):
    tf.reset_default_graph()
    h_layers = trial.suggest_int('hidden_layers', 1, 2)
    h_units = trial.suggest_int('hidden_units', 10, 30)
    inputs, _, _, outputs = vanilla_net(X_data.shape[1], h_units, h_layers, 42)
    labels_ph = tf.placeholder(real_type, [None, 1])
    train_op = tf.train.AdamOptimizer(0.01).minimize(tf.losses.mean_squared_error(labels_ph, outputs))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(20): sess.run(train_op, feed_dict={inputs: X_data, labels_ph: Y_data})
        return sess.run(tf.losses.mean_squared_error(labels_ph, outputs), feed_dict={inputs: X_data, labels_ph: Y_data})

# --- DATA ---
def load_data(universe_size: int = 20):
    drivers = pd.read_excel(r'./data/sensitivity_drivers.xlsx')
    assets = pd.read_excel(r'./data/assets.xlsx')
    drivers['Date'] = pd.to_datetime(drivers['Date'].astype(str))
    assets['Date'] = pd.to_datetime(assets['Date'].astype(str))
    drivers = drivers.set_index("Date")
    assets = assets.set_index("Date").iloc[:, :universe_size]
    returns = assets.pct_change().dropna()
    full_df = pd.merge(drivers, returns, left_index=True, right_index=True, how='inner')
    return full_df[returns.columns], full_df[drivers.columns], assets.reindex(full_df.index)

# --- BACKTEST ENGINE ---
rets, drivers, prices = load_data(UNIVERSE_SIZE)

if rets is not None:
    asset_names = rets.columns.tolist()
    if len(rets) <= HMM_WINDOW + 50:
        HMM_WINDOW = int(len(rets) * 0.5)
    
    start_idx = HMM_WINDOW + 20
    history = {s: [] for s in ['Active_Strategy', 'HRP_Bench', 'Robust_Bench', '1/N_Bench']}
    weights_history = []  # Store Active weights to visualize
    regime_track, reb_dates = [], []
    last_reb_idx, current_regime = start_idx, 0
    curr_weights = {s: pd.Series(1/len(asset_names), index=asset_names) for s in history.keys()}

    for i in tqdm(range(start_idx, len(rets)), desc="Backtesting"):
        days_since = i - last_reb_idx
        
        # 1. HMM Regime Detection
        if USE_SPY_ONLY_REGIME:
            hmm_raw = rets[INDEX_TICKER].iloc[i-HMM_WINDOW:i].values.reshape(-1, 1)
        else:
            hmm_raw = rets.iloc[i-HMM_WINDOW:i].mean(axis=1).values.reshape(-1, 1)
        
        hmm_input = hmm_raw * 100 
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, tol=0.1, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(hmm_input)
        
        high_vol_state = np.argmax([model.covars_[s][0][0] for s in range(2)])
        new_regime = 1 if model.predict(hmm_input)[-1] == high_vol_state else 0
        
        # 2. Trigger Logic
        reg_change = (new_regime != current_regime)
        do_reb = (days_since >= N_REB) or (reg_change and days_since >= M_GAP)
        
        if do_reb:
            look_rets = rets.iloc[i-EST_WINDOW:i]
            look_prices = prices.iloc[i-EST_WINDOW:i]
            cov, mu = look_rets.cov(), expected_returns.mean_historical_return(look_prices)
            
            # --- HSP ---
            corr_block = pd.concat([drivers.iloc[i-EST_WINDOW:i], look_rets], axis=1).corr()
            votes = pd.Series(0, index=drivers.columns)
            for a in asset_names:
                votes[corr_block[a].loc[drivers.columns].abs().nlargest(N_DRIVERS).index] += 1
            sel_drv = votes.nlargest(N_DRIVERS).index.tolist()
            
            sens_res = []
            for a in asset_names:
                X = np.hstack([drivers[sel_drv].iloc[i-100:i].values, rets[a].iloc[i-101:i-1].values.reshape(-1,1)])
                Y = rets[a].iloc[i-100:i].values.reshape(-1,1)
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda t: objective(t, X, Y), n_trials=TRIALS)
                tf.reset_default_graph()
                inp, wb, ln, out = vanilla_net(X.shape[1], study.best_params['hidden_units'], study.best_params['hidden_layers'], 42)
                train_op = tf.train.AdamOptimizer(0.01).minimize(tf.losses.mean_squared_error(tf.placeholder(real_type, [None,1]), out))
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for _ in range(30): sess.run(train_op, feed_dict={inp: X, "Placeholder_1:0": Y})
                    sens_res.append(np.nan_to_num(np.mean(sess.run(backprop(wb, ln), feed_dict={inp: X}), axis=0))[:N_DRIVERS])
            
            hsp_link = linkage(nearestPD(distance.cdist(sens_res, sens_res, 'euclidean')), 'single')
            w_hsp = get_rec_bipart(cov, get_quasi_diag(hsp_link)).reindex(asset_names).fillna(0)

            # --- Robust Quadratic Utility ---
            S_rob = risk_models.CovarianceShrinkage(look_prices).ledoit_wolf()
            try:
                ef = EfficientFrontier(mu, S_rob)
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                w_robust = pd.Series(ef.max_quadratic_utility(risk_aversion=2.0)).reindex(asset_names).fillna(0)
            except:
                w_robust = pd.Series(1/len(asset_names), index=asset_names)

            # --- HRP Fixed ---
            hrp_link = linkage(look_rets.corr(), 'single')
            w_hrp = get_rec_bipart(cov, get_quasi_diag(hrp_link))
            w_hrp.index = [asset_names[j] for j in w_hrp.index]
            w_hrp = w_hrp.reindex(asset_names).fillna(0)

            # --- Switch Proper Update ---
            curr_weights['Robust_Bench'] = w_robust
            curr_weights['HRP_Bench'] = w_hrp
            curr_weights['1/N_Bench'] = pd.Series(1/len(asset_names), index=asset_names)
            
            # This is where the switch activates
            if new_regime == 1:
                curr_weights['Active_Strategy'] = w_hsp
            else:
                curr_weights['Active_Strategy'] = w_robust
            
            current_regime, last_reb_idx = new_regime, i
            reb_dates.append(rets.index[i])

        # Track history
        for s in history.keys(): history[s].append((rets.iloc[i] * curr_weights[s]).sum())
        
        # Track daily weights for Active Strategy
        w_step = curr_weights['Active_Strategy'].copy()
        w_step['Date'] = rets.index[i]
        weights_history.append(w_step)
        
        regime_track.append(current_regime)

    # --- PLOTTING ---
    res_df = pd.DataFrame(history, index=rets.index[start_idx:])
    reg_s = pd.Series(regime_track, index=res_df.index)
    w_df = pd.DataFrame(weights_history).set_index('Date')
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    
    # Plot 1: Cumulative Returns
    cum_rets = (1 + res_df).cumprod()
    cum_rets.plot(ax=axes[0], lw=2)
    axes[0].fill_between(reg_s.index, 0, cum_rets.max().max(), where=reg_s==1, color='red', alpha=0.1, label='High Vol (HSP)')
    axes[0].set_title("OOS Cumulative Returns with Regime Shading", fontsize=14)
    axes[0].legend()

    # Plot 2: Active Strategy Weights Evolution
    w_df.clip(lower=0).plot.area(ax=axes[1], stacked=True, alpha=0.7, cmap='tab20')
    axes[1].set_title("Active Strategy Weight Allocation (HSP vs Robust Switch)", fontsize=14)
    axes[1].set_ylabel("Weight (%)")
    axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Plot 3: Rolling Sharpe
    rs = (res_df.rolling(126).mean() / res_df.rolling(126).std()) * np.sqrt(252)
    rs.plot(ax=axes[2], title="Rolling 6-Month Sharpe Ratio")
    axes[2].fill_between(reg_s.index, rs.min().min(), rs.max().max(), where=reg_s==1, color='red', alpha=0.1)
    
    plt.tight_layout()
    plt.show()

    print("\n--- PERFORMANCE SUMMARY ---")
    stats = pd.DataFrame({
        'Ann. Return (%)': res_df.mean() * 252 * 100,
        'Ann. Vol (%)': res_df.std() * np.sqrt(252) * 100,
        'Sharpe Ratio': (res_df.mean() * 252) / (res_df.std() * np.sqrt(252)),
        'Max DD (%)': ((1+res_df).cumprod() / (1+res_df).cumprod().cummax() - 1).min() * 100
    }).T
    print(stats.round(3))