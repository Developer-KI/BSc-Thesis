# Plotting tools for BSc Thesis – portfolio strategy analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram
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


# ── 1. Dendrogram ─────────────────────────────────────────────────────────────
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


# ── 2. Cumulative Returns ─────────────────────────────────────────────────────
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


# ── 3. Drawdown ───────────────────────────────────────────────────────────────
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


# ── 4. Rolling Sharpe ─────────────────────────────────────────────────────────
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


# ── 5. Sharpe Bar Chart ───────────────────────────────────────────────────────
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


# ── 6. Performance Summary (2×2 panel) ────────────────────────────────────────
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


# ── 7. Correlation Heatmap ────────────────────────────────────────────────────
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


# ── 8. Weights Heatmap ────────────────────────────────────────────────────────
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


# ── 9. Stacked Area Weights ───────────────────────────────────────────────────
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
