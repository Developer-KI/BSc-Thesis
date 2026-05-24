"""
run_crisis.py  –  Crisis-period analysis of HMVA backtest results.

Loads daily portfolio returns from results/backtest/daily_excess_returns.csv
and analyses risk-adjusted performance across defined market regimes.

Crisis periods analysed:
  2002-01-01 → 2002-10-31    Dot-com trough
  2007-10-01 → 2009-03-31    Global Financial Crisis (GFC)
  2020-01-15 → 2020-04-30    COVID-19 crash
  2022-01-01 → 2022-12-31    Rate-hike cycle

Calm periods (for contrast):
  2003-01-01 → 2007-09-30    Pre-GFC bull market
  2009-04-01 → 2020-01-14    Post-GFC bull market
  2020-05-01 → 2021-12-31    Post-COVID rebound

Outputs in results/crisis/:
  period_metrics.csv           per-period, per-strategy metrics
  rolling_sharpe.png           63-day rolling Sharpe with crisis shading
  crisis_equity.png            equity curves for each crisis window (2×2 grid)
  period_sharpe_heatmap.png    heatmap: Sharpe by period × strategy
  monthly_heatmap_HMVA.png     calendar heatmap of HMVA monthly returns
  regime_summary.csv           crisis vs calm aggregated Sharpe comparison
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import utils.plotting as _plt

_PALETTE = _plt._PALETTE


# ── configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = "results"

STRATEGIES = ["HMVA", "HMVA-mv", "HRP", "MVO", "GMV", "EW", "SPY-K"]

CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
    "Dot-com trough\n(2002)":       ("2002-01-01", "2002-10-31"),
    "GFC\n(2007–2009)":             ("2007-10-01", "2009-03-31"),
    "COVID-19 crash\n(2020 Q1)":    ("2020-01-15", "2020-04-30"),
    "Rate-hike cycle\n(2022)":      ("2022-01-01", "2022-12-31"),
}

CALM_PERIODS: Dict[str, Tuple[str, str]] = {
    "Pre-GFC bull\n(2003–2007)":     ("2003-01-01", "2007-09-30"),
    "Post-GFC bull\n(2009–2020)":    ("2009-04-01", "2020-01-14"),
    "Post-COVID rebound\n(2020–21)": ("2020-05-01", "2021-12-31"),
}

_CRISIS_SHADE = "#fce4e4"


# ── helpers ───────────────────────────────────────────────────────────────────

def _col_color(col: str, columns) -> str:
    idx = list(columns).index(col) if col in columns else 0
    return _PALETTE[idx % len(_PALETTE)]


def _shade_crises(ax) -> None:
    for label, (s, e) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=_CRISIS_SHADE, alpha=0.65, zorder=0,
                   label=f"Crisis: {label.splitlines()[0]}")


def _compute_period_metrics(daily: pd.DataFrame,
                             start: str,
                             end: str) -> pd.DataFrame:
    sub = daily.loc[start:end].dropna(how="all")
    rows: Dict = {}
    for col in daily.columns:
        r = sub[col].dropna()
        if len(r) < 5:
            rows[col] = dict(AnnReturn=np.nan, AnnVol=np.nan, Sharpe=np.nan,
                             MaxDD=np.nan, Calmar=np.nan, NDays=len(r))
            continue
        ann_ret = float((1 + r).prod() ** (252.0 / len(r)) - 1)
        ann_vol = float(r.std() * np.sqrt(252))
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum     = (1 + r).cumprod()
        max_dd  = float((cum / cum.cummax() - 1).min())
        calmar  = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
        rows[col] = dict(AnnReturn=ann_ret, AnnVol=ann_vol, Sharpe=sharpe,
                         MaxDD=max_dd, Calmar=calmar, NDays=len(r))
    return pd.DataFrame(rows).T


def _load_daily(results_dir: str) -> pd.DataFrame:
    path = f"{results_dir}/backtest/daily_excess_returns.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run run_backtest.py first."
        )
    daily = pd.read_csv(path, index_col=0, parse_dates=True)
    present = [c for c in STRATEGIES if c in daily.columns]
    if not present:
        raise ValueError(f"No recognised strategy columns in {path}. "
                         f"Found: {list(daily.columns)}")
    return daily[present]


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_rolling_sharpe(daily: pd.DataFrame, outdir: str) -> None:
    window = 63
    roll = daily.rolling(window).apply(
        lambda r: (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
        raw=True,
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_crises(ax)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    for col in daily.columns:
        lw = 2.2 if col == "HMVA" else 1.2
        ax.plot(roll.index, roll[col],
                color=_col_color(col, daily.columns), lw=lw, label=col)
    ax.set_title(f"{window}-day rolling Sharpe ratio by strategy  (CRSP 2002–2024)")
    ax.set_ylabel("Rolling Sharpe (annualised)")
    ax.set_ylim(-4, 5)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{outdir}/rolling_sharpe.png", bbox_inches="tight")
    plt.close(fig)


def plot_crisis_equity(daily: pd.DataFrame, outdir: str) -> None:
    periods = list(CRISIS_PERIODS.items())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Equity curves during crisis periods", fontsize=13)
    for ax, (label, (s, e)) in zip(axes.flat, periods):
        sub = daily.loc[s:e].dropna(how="all")
        if sub.empty:
            ax.set_visible(False)
            continue
        for col in sub.columns:
            cum = (1 + sub[col]).cumprod()
            lw  = 2.2 if col == "HMVA" else 1.2
            ax.plot(cum.index, cum.values,
                    color=_col_color(col, daily.columns), lw=lw, label=col)
        ax.axhline(1, color="black", lw=0.5, ls="--")
        ax.set_title(label.replace("\n", "  "), fontsize=10)
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{outdir}/crisis_equity.png", bbox_inches="tight")
    plt.close(fig)


def plot_period_sharpe_heatmap(all_metrics: pd.DataFrame, outdir: str) -> None:
    pivot = all_metrics.pivot(index="period", columns="strategy", values="Sharpe")
    crisis_names = [k.replace("\n", " ") for k in CRISIS_PERIODS]
    calm_names   = [k.replace("\n", " ") for k in CALM_PERIODS]
    row_order    = crisis_names + calm_names
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    col_order = [c for c in STRATEGIES if c in pivot.columns]
    pivot = pivot[col_order]

    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = max(abs(vals).max(), 0.1) if len(vals) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, len(pivot) * 0.7 + 1.5))
    im = ax.imshow(pivot.values.astype(float), aspect="auto",
                   cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    n_crisis = sum(1 for k in crisis_names if k in pivot.index.tolist())
    ax.axhline(n_crisis - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, label="Annualised Sharpe ratio")
    ax.set_title("Sharpe ratio by market regime and strategy", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/period_sharpe_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(daily: pd.DataFrame, strategy: str, outdir: str) -> None:
    if strategy not in daily.columns:
        return
    monthly = (1 + daily[strategy].dropna()).resample("ME").prod() - 1
    _plt.plot_monthly_returns_heatmap(
        monthly,
        title=f"Monthly returns: {strategy}  (CRSP 2002–2024)",
        save_path=f"{outdir}/monthly_heatmap_{strategy.replace('-', '_')}.png",
    )
    plt.close("all")


def plot_annual_returns(daily: pd.DataFrame, outdir: str) -> None:
    annual = (1 + daily).resample("YE").prod() - 1
    cols   = [c for c in STRATEGIES if c in annual.columns]
    width  = 0.8 / len(cols)
    x      = np.arange(len(annual))

    fig, ax = plt.subplots(figsize=(14, 5))
    years = list(annual.index.year)
    for start, end in CRISIS_PERIODS.values():
        s_year, e_year = pd.Timestamp(start).year, pd.Timestamp(end).year
        xs = [i for i, y in enumerate(years) if s_year <= y <= e_year]
        if xs:
            ax.axvspan(min(xs) - 0.5, max(xs) + 0.5,
                       color=_CRISIS_SHADE, alpha=0.4, zorder=0)
    for i, col in enumerate(cols):
        offset = (i - len(cols) / 2 + 0.5) * width
        ax.bar(x + offset, annual[col], width=width * 0.92,
               label=col, color=_col_color(col, daily.columns),
               edgecolor="white", alpha=0.85)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index.year, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Annual return")
    ax.set_title("Annual returns by strategy  (CRSP 2002–2024)")
    ax.legend(ncol=len(cols), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/annual_returns.png", bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="HMVA crisis-period analysis")
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--out",         default="results/crisis_analysis")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    print(f"[crisis] loading daily returns from {args.results_dir}/backtest/ ...")
    daily = _load_daily(args.results_dir)
    print(f"[crisis] loaded {len(daily)} days, strategies: {list(daily.columns)}")
    print(f"[crisis] period: {daily.index[0].date()} → {daily.index[-1].date()}")

    # ── compute metrics per period ────────────────────────────────────────────
    all_rows = []
    all_periods: Dict[str, Tuple[str, str]] = {**CRISIS_PERIODS, **CALM_PERIODS}

    for label, (s, e) in all_periods.items():
        regime = "crisis" if label in CRISIS_PERIODS else "calm"
        m = _compute_period_metrics(daily, s, e)
        for strat in m.index:
            row = {"period": label.replace("\n", " "), "regime": regime,
                   "strategy": strat, "start": s, "end": e}
            row.update(m.loc[strat].to_dict())
            all_rows.append(row)

    period_metrics = pd.DataFrame(all_rows)
    period_metrics.to_csv(f"{args.out}/period_metrics.csv", index=False)

    # ── regime summary (crisis vs calm) ───────────────────────────────────────
    regime_rows = []
    for strat in daily.columns:
        sub = period_metrics[period_metrics["strategy"] == strat]
        for regime in ["crisis", "calm"]:
            vals = sub[sub["regime"] == regime]["Sharpe"].dropna()
            if len(vals):
                regime_rows.append({
                    "strategy":    strat,
                    "regime":      regime,
                    "mean_sharpe": float(vals.mean()),
                    "n_periods":   len(vals),
                })
    regime_summary = pd.DataFrame(regime_rows)
    regime_summary.to_csv(f"{args.out}/regime_summary.csv", index=False)

    print("\n=== Sharpe by regime ===")
    pivot_regime = regime_summary.pivot(index="strategy", columns="regime",
                                        values="mean_sharpe")
    if "crisis" in pivot_regime.columns and "calm" in pivot_regime.columns:
        pivot_regime["crisis_to_calm"] = (
            pivot_regime["crisis"] / pivot_regime["calm"].abs()
        )
    print(pivot_regime.round(3).to_string())

    print("\n=== Period metrics (Sharpe) ===")
    pivot_sharpe = period_metrics.pivot_table(
        index="period", columns="strategy", values="Sharpe"
    )
    print(pivot_sharpe.round(3).to_string())

    # ── generate plots ────────────────────────────────────────────────────────
    print("\n[crisis] generating plots ...")
    plot_rolling_sharpe(daily, args.out)
    plot_crisis_equity(daily, args.out)
    plot_period_sharpe_heatmap(period_metrics, args.out)
    plot_annual_returns(daily, args.out)

    for strat in ["HMVA", "EW"]:
        if strat in daily.columns:
            plot_monthly_heatmap(daily, strat, args.out)

    print(f"\n[crisis] done → {args.out}/")


if __name__ == "__main__":
    main()
