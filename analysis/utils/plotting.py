# Plotting tools for BSc Thesis – portfolio strategy analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram
from scipy import stats as sp_stats
from typing import List, Optional

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#E91E63",
]


def _save(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_dendrogram(
    link,
    labels: List[str],
    title: str = "Hierarchy Dendrogram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a scipy linkage dendrogram.

    Parameters
    ----------
    link      : scipy linkage matrix
    labels    : asset ticker labels
    title     : plot title
    save_path : if given, saves the figure to this path
    """
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.18), 6))
    dendrogram(link, labels=labels, leaf_rotation=90, ax=ax, color_threshold=0)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    title: str = "Cumulative Returns",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots cumulative wealth index (1 + r).cumprod() for each strategy column.

    Parameters
    ----------
    returns_df : DataFrame of period returns, one column per strategy
    """
    cum = (1 + returns_df).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(cum.columns):
        ax.plot(cum.index, cum[col],
                label=col, color=_PALETTE[i % len(_PALETTE)], linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    return _save(fig, save_path)


def plot_drawdown(
    returns_df: pd.DataFrame,
    title: str = "Drawdown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the underwater (drawdown) curve for each strategy column.
    """
    cum = (1 + returns_df).cumprod()
    drawdown = (cum / cum.cummax()) - 1
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(drawdown.columns):
        ax.fill_between(drawdown.index, drawdown[col], 0,
                        alpha=0.25, color=_PALETTE[i % len(_PALETTE)])
        ax.plot(drawdown.index, drawdown[col],
                label=col, color=_PALETTE[i % len(_PALETTE)], linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_rolling_sharpe(
    returns_df: pd.DataFrame,
    window: int = 63,
    title: str = "Rolling Sharpe Ratio",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the annualised rolling Sharpe ratio (window trading days).

    Parameters
    ----------
    window : rolling window in trading days (default 63 ≈ 1 quarter)
    """
    rolling_sharpe = (
        returns_df.rolling(window).mean() / returns_df.rolling(window).std()
    ) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(rolling_sharpe.columns):
        ax.plot(rolling_sharpe.index, rolling_sharpe[col],
                label=col, color=_PALETTE[i % len(_PALETTE)], linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title}  (window = {window} days)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 63,
    title: str = "Rolling Volatility",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Annualised rolling volatility for each strategy column.

    Parameters
    ----------
    window : rolling window in periods (default 63 ≈ 1 quarter for daily data;
             use 12 for monthly data)
    """
    rolling_vol = returns_df.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(rolling_vol.columns):
        ax.plot(rolling_vol.index, rolling_vol[col],
                label=col, color=_PALETTE[i % len(_PALETTE)], linewidth=1.2)
    ax.set_title(f"{title}  (window = {window} periods)")
    ax.set_ylabel("Annualised Volatility")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_sharpe_bar(
    sharpe_series: pd.Series,
    title: str = "Average Sharpe Ratio by Strategy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of Sharpe ratios, coloured blue (positive) / red (negative).

    Parameters
    ----------
    sharpe_series : Series indexed by strategy name
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_PALETTE[0] if v >= 0 else _PALETTE[1] for v in sharpe_series]
    sharpe_series.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xticklabels(sharpe_series.index, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_performance_summary(
    stats_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2×2 panel: Ann. Return, Ann. Vol, Sharpe Ratio, Max Drawdown per strategy.

    Parameters
    ----------
    stats_df : DataFrame with any subset of columns:
               'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe Ratio', 'Max Drawdown (%)'
    """
    metrics = [c for c in
               ["Ann. Return (%)", "Ann. Vol (%)", "Sharpe Ratio", "Max Drawdown (%)"]
               if c in stats_df.columns]
    n = len(metrics)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.5))
    axes = np.array(axes).flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = stats_df[metric]
        bar_colors = [_PALETTE[0] if v >= 0 else _PALETTE[1] for v in vals]
        vals.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
        ax.set_title(metric)
        ax.set_xticklabels(vals.index, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Strategy Performance Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Annotated correlation heatmap with a diverging colour scale centred at 0.
    Annotations are shown only when n ≤ 30 to avoid clutter.
    """
    corr = returns_df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(6, n * 0.4)))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr.values, cmap="RdYlGn", norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=6, color="black")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_weights_heatmap(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap where rows = assets, columns = rebalance dates, colour = weight.

    Parameters
    ----------
    weights_df : DataFrame indexed by date, columns are asset tickers
    """
    fig, ax = plt.subplots(figsize=(max(10, len(weights_df) * 0.25),
                                    max(5, len(weights_df.columns) * 0.18)))
    im = ax.imshow(weights_df.values.T, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=weights_df.values.max())
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Weight")
    ax.set_yticks(range(len(weights_df.columns)))
    ax.set_yticklabels(weights_df.columns, fontsize=7)
    step = max(1, len(weights_df) // 8)
    ax.set_xticks(range(0, len(weights_df), step))
    ax.set_xticklabels(
        [str(weights_df.index[i])[:10] for i in range(0, len(weights_df), step)],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_regime_detection(
    returns: pd.Series,
    regime_labels,
    title: str = "Market Regime Detection",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel regime visualisation.

    Top panel    – cumulative wealth curve with coloured background shading
                   for each detected regime.
    Bottom panel – discrete regime-state timeline (coloured horizontal bands).

    Parameters
    ----------
    returns       : return series (daily or monthly), pd.Series with DatetimeIndex
    regime_labels : array-like of regime names ('Bull', 'Bear', 'Crab', …) or
                    integers; must be the same length as returns
    """
    if not isinstance(regime_labels, pd.Series):
        regime_labels = pd.Series(regime_labels, index=returns.index)
    else:
        regime_labels = regime_labels.reindex(returns.index)

    # Canonical ordering and colours; anything else falls back to _PALETTE
    _REGIME_COLORS = {
        "Bull":    "#4CAF50",
        "Crab":    "#9E9E9E",
        "Bear":    "#F44336",
        "Neutral": "#FF9800",
    }
    # Preserve intuitive ordering (best → worst) where labels are recognised
    ordered = [r for r in ("Bull", "Crab", "Neutral", "Bear")
               if r in regime_labels.values]
    for r in regime_labels.dropna().unique():
        if r not in ordered:
            ordered.append(r)

    regime_color = {}
    for i, r in enumerate(ordered):
        regime_color[r] = _REGIME_COLORS.get(str(r), _PALETTE[i % len(_PALETTE)])

    cum = (1 + returns).cumprod()
    dates = returns.index

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Panel 1: cumulative return with shaded regime periods ─────────────────
    ax1.plot(dates, cum.values, color="black", linewidth=1.5, zorder=3)
    for regime in ordered:
        mask = (regime_labels == regime).values
        ax1.fill_between(
            dates, cum.values.flatten(), 0,
            where=mask, color=regime_color[regime], alpha=0.22,
            label=str(regime),
        )
    ax1.set_ylabel("Growth of $1")
    ax1.set_title(title, fontsize=13)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax1.legend(loc="upper left", fontsize=9)

    # ── Panel 2: discrete regime-state timeline ───────────────────────────────
    regime_to_int = {r: i for i, r in enumerate(ordered)}
    regime_int = regime_labels.map(regime_to_int)
    ax2.step(dates, regime_int.values, where="post",
             color="black", linewidth=0.8, zorder=3)
    for regime in ordered:
        mask = (regime_labels == regime).values
        ax2.fill_between(
            dates, 0, 1,
            where=mask,
            transform=ax2.get_xaxis_transform(),
            color=regime_color[regime], alpha=0.40,
        )
    ax2.set_yticks(list(regime_to_int.values()))
    ax2.set_yticklabels(list(regime_to_int.keys()), fontsize=9)
    ax2.set_ylabel("Regime")

    fig.tight_layout()
    return _save(fig, save_path)


def plot_weights_area(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weight Allocation Over Time",
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked area chart of weight allocation over time.
    The top_n assets by average weight are shown individually;
    the remainder are grouped as 'Other'.

    Parameters
    ----------
    weights_df : DataFrame indexed by date, columns are asset tickers
    top_n      : max individual tickers to display
    """
    top_assets = weights_df.mean().sort_values(ascending=False).head(top_n).index.tolist()
    plot_df = weights_df[top_assets].copy()
    other = weights_df.drop(columns=top_assets).sum(axis=1)
    if other.max() > 1e-6:
        plot_df["Other"] = other

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        plot_df.index, plot_df.values.T,
        labels=plot_df.columns,
        colors=plt.cm.tab20.colors[: len(plot_df.columns)],
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=1)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_monthly_returns_heatmap(
    returns_series: pd.Series,
    title: str = "Monthly Returns Heatmap",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Calendar heatmap: rows = years, columns = months, cell colour = return.
    Annual return is annotated to the right of each row.
    """
    r = returns_series.copy()
    r.index = pd.to_datetime(r.index)
    df = r.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(values="ret", index="year", columns="month")
    pivot.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in pivot.columns]

    ann_ret = (1 + r).resample("YE").prod() - 1
    ann_ret.index = ann_ret.index.year

    flat = pivot.values[~np.isnan(pivot.values)]
    abs_max = max(abs(flat).max(), 0.001)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5 + 1)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02,
                 format=mtick.PercentFormatter(xmax=1, decimals=1))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.12 else "white")
        yr = pivot.index[i]
        if yr in ann_ret.index:
            ax.text(len(pivot.columns) + 0.15, i,
                    f"  {ann_ret[yr]*100:.1f}%",
                    ha="left", va="center", fontsize=8,
                    color="#4CAF50" if ann_ret[yr] >= 0 else "#F44336",
                    fontweight="bold")

    ax.set_xlim(-0.5, len(pivot.columns) - 0.5 + 1.2)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_return_distribution(
    returns_df: pd.DataFrame,
    title: str = "Return Distribution",
    bins: int = 35,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram + KDE + normal overlay for each strategy with key stats annotated.
    """
    n = len(returns_df.columns)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(returns_df.columns):
        ax = axes_flat[i]
        r = returns_df[col].dropna()
        color = _PALETTE[i % len(_PALETTE)]

        ax.hist(r, bins=bins, density=True, color=color, alpha=0.35, edgecolor="white")
        x_range = np.linspace(r.min() * 1.3, r.max() * 1.3, 300)
        kde = sp_stats.gaussian_kde(r)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2, label="KDE")

        mu_n, sig_n = r.mean(), r.std()
        ax.plot(x_range, sp_stats.norm.pdf(x_range, mu_n, sig_n),
                color="black", linewidth=1.2, linestyle="--", alpha=0.65, label="Normal")

        ax.axvline(0, color="black", linewidth=0.7, linestyle=":")
        ax.axvline(mu_n, color=color, linewidth=1.2, linestyle="--", alpha=0.9)

        skew_val = sp_stats.skew(r)
        kurt_val = sp_stats.kurtosis(r)
        var95    = np.percentile(r, 5)
        ax.text(0.03, 0.97,
                f"Mean: {mu_n*100:.2f}%\nStd:  {sig_n*100:.2f}%\n"
                f"Skew: {skew_val:.2f}\nKurt: {kurt_val:.2f}\n"
                f"VaR95: {var95*100:.2f}%",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Monthly Return")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_annual_returns(
    returns_df: pd.DataFrame,
    title: str = "Annual Returns by Strategy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart of calendar-year returns for each strategy.
    """
    annual = (1 + returns_df).resample("YE").prod() - 1
    annual.index = annual.index.year

    n_strats = len(annual.columns)
    bar_width = 0.8 / n_strats
    x = np.arange(len(annual.index))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(annual.columns):
        offsets = x + i * bar_width - 0.4 + bar_width / 2
        vals = annual[col].values
        colors = [_PALETTE[i % len(_PALETTE)] if v >= 0 else _PALETTE[1] for v in vals]
        ax.bar(offsets, vals, width=bar_width * 0.92,
               color=colors, label=col, alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_risk_return_scatter(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    title: str = "Risk-Return Profile",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Annualised return vs annualised volatility with iso-Sharpe reference lines.
    """
    ann_ret = returns_df.mean() * 12
    ann_vol = returns_df.std() * np.sqrt(12)

    fig, ax = plt.subplots(figsize=(9, 7))

    vol_range = np.linspace(0, ann_vol.max() * 1.35, 300)
    for sr in [0, 0.5, 1.0, 1.5, 2.0]:
        ret_line = risk_free_rate + sr * vol_range
        ax.plot(vol_range, ret_line, color="grey", linewidth=0.7,
                linestyle="--", alpha=0.45)
        mid = len(vol_range) // 2
        ax.text(vol_range[mid], ret_line[mid] + 0.004,
                f"SR={sr}", fontsize=7, color="grey", va="bottom")

    for i, col in enumerate(returns_df.columns):
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter(ann_vol[col], ann_ret[col], s=140, color=color,
                   zorder=5, edgecolors="white", linewidths=1.3)
        ax.annotate(col, (ann_vol[col], ann_ret[col]),
                    textcoords="offset points", xytext=(9, 4),
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(risk_free_rate, color="black", linewidth=0.8, linestyle=":",
               label=f"Risk-Free ({risk_free_rate:.0%})")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_rolling_beta(
    returns_df: pd.DataFrame,
    benchmark_col: str,
    window: int = 12,
    title: str = "Rolling Beta vs Benchmark",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Rolling beta of each strategy vs benchmark_col over a rolling window.
    """
    bench = returns_df[benchmark_col]
    strategies = [c for c in returns_df.columns if c != benchmark_col]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(strategies):
        strat = returns_df[col]
        rolling_cov  = strat.rolling(window).cov(bench)
        rolling_var  = bench.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        ax.plot(rolling_beta.index, rolling_beta,
                label=col, color=_PALETTE[i % len(_PALETTE)], linewidth=1.3)

    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="β=1")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(f"{title}  (window={window} mo)")
    ax.set_ylabel("Beta")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_metrics_table(
    metrics_df: pd.DataFrame,
    title: str = "Comprehensive Performance Metrics",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a metrics DataFrame (strategies as rows, metrics as columns) as a
    colour-coded table.  Green = best in column, red = worst.
    """
    higher_is_better = {
        "Ann. Return (%)": True,  "Cumul. Return (%)": True,
        "Sharpe Ratio": True,     "Sortino Ratio": True,
        "Calmar Ratio": True,     "Omega Ratio": True,
        "Tail Ratio": True,       "Hit Rate (%)": True,
        "Best Month (%)": True,   "Avg Gain (%)": True,
        "Win/Loss Ratio": True,   "Alpha (ann. %)": True,
        "Information Ratio": True,"Treynor Ratio": True,
        "Ann. Volatility (%)": False, "Max Drawdown (%)": False,
        "Avg DD Duration (mo)": False,"Max DD Duration (mo)": False,
        "VaR 95% (%)": False,     "CVaR 95% (%)": False,
        "Tracking Error (%)": False,  "Worst Month (%)": False,
        "Avg Loss (%)": False,
    }

    df = metrics_df.T
    n_rows, n_cols = df.shape
    fig_h = max(5, n_rows * 0.40 + 1)
    fig_w = max(10, n_cols * 2.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_labels = list(df.columns)
    row_labels  = list(df.index)
    cell_text   = [
        [f"{v:.3f}" if pd.notna(v) else "—" for v in df.loc[row]]
        for row in row_labels
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)

    for ri, metric in enumerate(row_labels):
        vals = [df.loc[metric, c] for c in col_labels]
        numeric = [v for v in vals if pd.notna(v)]
        if not numeric:
            continue
        good = higher_is_better.get(metric, True)
        best_val  = max(numeric) if good else min(numeric)
        worst_val = min(numeric) if good else max(numeric)
        for ci, v in enumerate(vals):
            if pd.isna(v):
                continue
            cell = table[ri + 1, ci]
            if v == best_val:
                cell.set_facecolor("#C8E6C9")
            elif v == worst_val:
                cell.set_facecolor("#FFCDD2")

    for ci in range(len(col_labels)):
        table[0, ci].set_facecolor("#37474F")
        table[0, ci].set_text_props(color="white", fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return _save(fig, save_path)
