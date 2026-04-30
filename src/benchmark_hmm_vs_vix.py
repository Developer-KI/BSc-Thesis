"""
Comparison of HMM vs VIX Threshold for Volatility Regime Detection (SPY)
Focus: Regime assignment quality, not trading performance
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Data Fetching & Preparation
# -------------------------------
print("Fetching SPY and VIX data...")
start_date = '2005-01-01'
end_date = '2024-12-31'

spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

df = pd.DataFrame(index=spy.index)
df['spy_close'] = spy['Close']
df['spy_return'] = df['spy_close'].pct_change().dropna() * 100  # percentage
df['vix_close'] = vix['Close']
df = df.dropna()

# Forward realized volatility (5-day) for evaluation
df['realized_vol_5d'] = df['spy_return'].rolling(5).std() * np.sqrt(252)

print(f"Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} obs)")

# -------------------------------
# 2. VIX Threshold Model
# -------------------------------
vix_threshold = 20
df['vix_regime'] = (df['vix_close'] > vix_threshold).astype(int)

# -------------------------------
# 3. Hidden Markov Model (HMM)
# -------------------------------
print("\nFitting 2-state HMM on SPY returns...")
X = df['spy_return'].values.reshape(-1, 1)

model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X)

hidden_states = model.predict(X)
state_probs = model.predict_proba(X)

# Identify high-vol state (higher variance of returns)
state_0_var = np.var(df['spy_return'][hidden_states == 0])
state_1_var = np.var(df['spy_return'][hidden_states == 1])

if state_0_var > state_1_var:
    high_vol_state = 0
    low_vol_state = 1
else:
    high_vol_state = 1
    low_vol_state = 0

df['hmm_regime'] = np.where(hidden_states == high_vol_state, 1, 0)
df['hmm_prob_high'] = state_probs[:, high_vol_state]

transmat = model.transmat_
print(f"Transition Matrix:\n{transmat}")
print(f"High Vol Persistence: {transmat[high_vol_state, high_vol_state]:.2%}")
print(f"Low Vol Persistence:  {transmat[low_vol_state, low_vol_state]:.2%}")

# -------------------------------
# 4. Evaluation Metrics
# -------------------------------
print("\n" + "="*50)
print("REGIME DETECTION QUALITY METRICS")
print("="*50)

vix_shift = df['vix_regime'].diff().fillna(0) != 0
vix_persistence = 1 - (vix_shift.sum() / len(df))
print(f"\nPersistence (VIX): {vix_persistence:.2%}")

hmm_vol_high = df[df['hmm_regime'] == 1]['spy_return'].std() * np.sqrt(252)
hmm_vol_low  = df[df['hmm_regime'] == 0]['spy_return'].std() * np.sqrt(252)
hmm_vol_ratio = hmm_vol_high / hmm_vol_low

vix_vol_high = df[df['vix_regime'] == 1]['spy_return'].std() * np.sqrt(252)
vix_vol_low  = df[df['vix_regime'] == 0]['spy_return'].std() * np.sqrt(252)
vix_vol_ratio = vix_vol_high / vix_vol_low

print(f"\nAnnualized Volatility by Regime:")
print(f"  HMM: High={hmm_vol_high:.2f}%, Low={hmm_vol_low:.2f}%, Ratio={hmm_vol_ratio:.2f}x")
print(f"  VIX: High={vix_vol_high:.2f}%, Low={vix_vol_low:.2f}%, Ratio={vix_vol_ratio:.2f}x")

corr_hmm_vix = df['hmm_prob_high'].corr(df['vix_close'])
print(f"\nCorrelation (HMM High-Vol Prob vs VIX Level): {corr_hmm_vix:.3f}")

corr_hmm_fwd = df['hmm_prob_high'].corr(df['realized_vol_5d'])
corr_vix_fwd = df['vix_regime'].corr(df['realized_vol_5d'])
print(f"\nCorrelation with Forward 5-Day Realized Vol:")
print(f"  HMM Prob: {corr_hmm_fwd:.3f}")
print(f"  VIX Regime: {corr_vix_fwd:.3f}")

agreement = (df['hmm_regime'] == df['vix_regime']).mean()
kappa = cohen_kappa_score(df['hmm_regime'], df['vix_regime'])
print(f"\nAgreement (HMM vs VIX): {agreement:.2%}")
print(f"Cohen's Kappa: {kappa:.3f}")

hmm_switches = (df['hmm_regime'].diff().fillna(0) != 0).sum()
vix_switches = (df['vix_regime'].diff().fillna(0) != 0).sum()
print(f"\nNumber of Regime Switches:")
print(f"  HMM: {hmm_switches} (avg days per regime: {len(df)/hmm_switches:.1f})")
print(f"  VIX: {vix_switches} (avg days per regime: {len(df)/vix_switches:.1f})")

# -------------------------------
# 5. Visualizations (Pure Matplotlib)
# -------------------------------
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-darkgrid')

# Focus on recent period for clarity (last 3 years)
plot_df = df[-252*3:].copy()

fig = plt.figure(figsize=(16, 14))

# Plot 1: SPY Price with Regime Shading
ax1 = plt.subplot(4, 1, 1)
ax1.plot(plot_df.index, plot_df['spy_close'], color='black', linewidth=1.5, label='SPY')
# HMM high vol shading (red)
hmm_high = plot_df[plot_df['hmm_regime'] == 1]
for i, idx in enumerate(hmm_high.index):
    ax1.axvspan(idx, idx + pd.Timedelta(days=1), alpha=0.2, color='red',
                label='HMM High Vol' if i == 0 else "")
# VIX high vol shading (blue)
vix_high = plot_df[plot_df['vix_regime'] == 1]
for i, idx in enumerate(vix_high.index):
    ax1.axvspan(idx, idx + pd.Timedelta(days=1), alpha=0.1, color='blue',
                label='VIX High Vol' if i == 0 else "")
ax1.set_ylabel('SPY Price')
ax1.set_title('SPY with Regime Detection Overlay (Last 3 Years)')
ax1.legend(loc='upper left')

# Plot 2: VIX with Threshold
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
ax2.plot(plot_df.index, plot_df['vix_close'], color='purple', linewidth=1.2)
ax2.axhline(y=vix_threshold, color='red', linestyle='--', label=f'Threshold = {vix_threshold}')
ax2.set_ylabel('VIX')
ax2.legend(loc='upper left')
ax2.set_title('VIX Index')

# Plot 3: HMM Probability vs VIX
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.plot(plot_df.index, plot_df['hmm_prob_high'], color='green', linewidth=1.2, label='HMM High-Vol Probability')
ax3.set_ylabel('Probability')
ax3.set_ylim(0, 1)
ax3.legend(loc='upper left')
ax3_twin = ax3.twinx()
ax3_twin.plot(plot_df.index, plot_df['vix_close'], color='purple', alpha=0.4, linewidth=0.8, label='VIX')
ax3_twin.set_ylabel('VIX', color='purple')
ax3_twin.legend(loc='upper right')
ax3.set_title(f'HMM High-Vol Probability vs VIX (Correlation: {corr_hmm_vix:.3f})')

# Plot 4: Binary Regime Comparison
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.fill_between(plot_df.index, 0, plot_df['hmm_regime'], step='pre', alpha=0.6, color='red', label='HMM')
ax4.fill_between(plot_df.index, 0, plot_df['vix_regime']*0.9, step='pre', alpha=0.4, color='blue', label='VIX')
ax4.set_ylim(0, 1.1)
ax4.set_ylabel('Regime (1=High Vol)')
ax4.set_xlabel('Date')
ax4.legend(loc='upper left')
ax4.set_title('Binary Regime Comparison')

plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap (Matplotlib imshow)
fig2, ax = plt.subplots(figsize=(6,5))
cm = confusion_matrix(df['vix_regime'], df['hmm_regime'], labels=[0,1])
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(2), yticks=np.arange(2),
       xticklabels=['Low Vol', 'High Vol'],
       yticklabels=['Low Vol', 'High Vol'],
       xlabel='HMM Predicted Regime',
       ylabel='VIX Threshold Regime',
       title=f'Confusion Matrix (Agreement: {agreement:.1%}, Kappa: {kappa:.3f})')
# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()

# Boxplot of returns by regime (Matplotlib boxplot)
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

# HMM boxplot
hmm_low_returns = df[df['hmm_regime'] == 0]['spy_return'].dropna()
hmm_high_returns = df[df['hmm_regime'] == 1]['spy_return'].dropna()
axes[0].boxplot([hmm_low_returns, hmm_high_returns], labels=['Low Vol', 'High Vol'],
                patch_artist=True, boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                capprops=dict(color='black'), flierprops=dict(marker='o', markersize=2))
axes[0].set_title('HMM: Return Distribution by Regime')
axes[0].set_ylabel('Daily Return (%)')

# VIX boxplot
vix_low_returns = df[df['vix_regime'] == 0]['spy_return'].dropna()
vix_high_returns = df[df['vix_regime'] == 1]['spy_return'].dropna()
axes[1].boxplot([vix_low_returns, vix_high_returns], labels=['Low Vol', 'High Vol'],
                patch_artist=True, boxprops=dict(facecolor='lightcoral'),
                medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                capprops=dict(color='black'), flierprops=dict(marker='o', markersize=2))
axes[1].set_title('VIX Threshold: Return Distribution by Regime')
axes[1].set_ylabel('Daily Return (%)')

plt.tight_layout()
plt.show()

# Summary Table
print("\n" + "="*50)
print("SUMMARY: HMM vs VIX THRESHOLD")
print("="*50)
summary = pd.DataFrame({
    'Metric': [
        'High Vol Persistence',
        'Low Vol Persistence',
        'Volatility Ratio (High/Low)',
        'Corr with VIX Level',
        'Corr with Fwd Realized Vol',
        'Number of Switches',
        'Avg Days per Regime'
    ],
    'HMM': [
        f"{transmat[high_vol_state, high_vol_state]:.2%}",
        f"{transmat[low_vol_state, low_vol_state]:.2%}",
        f"{hmm_vol_ratio:.2f}x",
        f"{corr_hmm_vix:.3f}",
        f"{corr_hmm_fwd:.3f}",
        f"{hmm_switches}",
        f"{len(df)/hmm_switches:.1f}"
    ],
    'VIX Threshold': [
        f"{vix_persistence:.2%}",
        f"{vix_persistence:.2%}",
        f"{vix_vol_ratio:.2f}x",
        "1.000 (by def)",
        f"{corr_vix_fwd:.3f}",
        f"{vix_switches}",
        f"{len(df)/vix_switches:.1f}"
    ]
})
print(summary.to_string(index=False))

print("\nScript completed.")