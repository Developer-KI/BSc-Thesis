from __future__ import annotations
import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from pandas_datareader.data import DataReader
import utils.plotting as _plt
import utils.backtest as L

_PALETTE = _plt._PALETTE


# ── configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
SUBFOLDER   = "base"

STRATEGIES = ["HMVA", "HMVA-mv", "HRP-E", "MHRP-EK", "MVO-EK", "GMV-EK", "EW", "SPY-100"]

# Pretty names for NBER recessions, keyed by YYYY-MM of the first month USREC=1 in FRED
# (FRED month-start labels lag the NBER peak month by one month)
_RECESSION_NAMES: Dict[str, str] = {
    "1990-08": "1990 Recession",
    "2001-04": "Dot Com Bubble",
    "2008-01": "Global Financial Crisis",
    "2020-03": "COVID-19 Recession",
}

_CRISIS_SHADE  = "#fce4e4"
_ROLLING_WINDOW = 126


# ── FRED recession loader ─────────────────────────────────────────────────────

def _fetch_recession_periods(index: pd.DatetimeIndex) -> Dict[str, Tuple[str, str]]:
    """
    Download NBER recession indicator (USREC) from FRED and return
    {label: (start, end)} for every recession that overlaps the index.
    """
    try:
        usrec = DataReader(
            "USREC", "fred",
            index.min() - pd.DateOffset(months=2),
            index.max() + pd.DateOffset(months=2),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download USREC from FRED: {exc}\n"
            "Check your internet connection or pandas-datareader installation."
        ) from exc

    # Align monthly series to the data's daily index
    rec = (
        usrec["USREC"]
        .resample("D").ffill()
        .reindex(index, method="ffill")
        .fillna(0)
        .astype(int)
    )

    periods: Dict[str, Tuple[str, str]] = {}
    in_rec     = False
    start_date = None

    for date, val in rec.items():
        if val == 1 and not in_rec:
            in_rec     = True
            start_date = date
        elif val == 0 and in_rec:
            in_rec    = False
            end_date  = rec.index[rec.index.get_loc(date) - 1]
            key       = start_date.strftime("%Y-%m")
            label     = _RECESSION_NAMES.get(key, f"Recession {key}")
            periods[label] = (start_date.strftime("%Y-%m-%d"),
                              end_date.strftime("%Y-%m-%d"))

    # Recession extending to the end of the data window
    if in_rec and start_date is not None:
        key   = start_date.strftime("%Y-%m")
        label = _RECESSION_NAMES.get(key, f"Recession {key}")
        periods[label] = (start_date.strftime("%Y-%m-%d"),
                          rec.index[-1].strftime("%Y-%m-%d"))

    return periods


# ── helpers ───────────────────────────────────────────────────────────────────

def _calm_daily(daily: pd.DataFrame,
                recession_periods: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """Return daily rows that fall outside every recession period."""
    mask = pd.Series(True, index=daily.index)
    for s, e in recession_periods.values():
        mask.loc[s:e] = False
    return daily[mask]


def _calm_periods_label(index: pd.DatetimeIndex) -> Dict[str, Tuple[str, str]]:
    """Single 'Calm' entry spanning the full data range (rows filtered via _calm_daily)."""
    return {"Calm": (index.min().strftime("%Y-%m-%d"), index.max().strftime("%Y-%m-%d"))}


def _col_color(col: str, columns) -> str:
    idx = list(columns).index(col) if col in columns else 0
    return _PALETTE[idx % len(_PALETTE)]


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
    path = f"{results_dir}/backtest/{SUBFOLDER}/daily_excess_returns.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Run run_backtest.py first."
        )
    daily   = pd.read_csv(path, index_col=0, parse_dates=True)
    present = [c for c in STRATEGIES if c in daily.columns]
    if not present:
        raise ValueError(f"No recognised strategy columns in {path}. "
                         f"Found: {list(daily.columns)}")
    return daily[present]


def _grid_shape(n: int) -> Tuple[int, int, Tuple[float, float]]:
    """Return (nrows, ncols, figsize) for n subplots.

    1–3 panels: single row (wide).
    4   panels: 2×2.
    5+  panels: ceil-sqrt columns, variable rows.
    """
    if n <= 3:
        return 1, n, (5.5 * n, 4.8)
    if n == 4:
        return 2, 2, (11.0, 9.0)
    import math
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols, (5.5 * ncols, 4.8 * nrows)


# ── statistical tests per subperiod ──────────────────────────────────────────

_TEST_BASES = ("HMVA",)


def _period_sharpe_tests(
    daily: pd.DataFrame,
    calm_periods: Dict[str, Tuple[str, str]],
    recession_periods: Dict[str, Tuple[str, str]],
    n_boot: int = 5000,
) -> pd.DataFrame:
    """
    LW block-bootstrap Sharpe tests for each subperiod.

    Tests the calm period vs all strategies, then the combined recession pool.
    BH FDR correction applied within each (period, base) group.
    """
    def _sr_ann(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 2:
            return float("nan")
        ann_ret = float((1 + r).prod() ** (252.0 / len(r)) - 1)
        ann_vol = float(r.std(ddof=1) * np.sqrt(252))
        return ann_ret / ann_vol if ann_vol > 0 else float("nan")

    cd          = _calm_daily(daily, recession_periods)
    all_periods = {**recession_periods, **calm_periods}
    rows        = []

    # Build combined recession DataFrame upfront
    rec_chunks     = [daily.loc[s:e].dropna(how="all")
                      for _, (s, e) in recession_periods.items()]
    combined_rec   = (pd.concat([c for c in rec_chunks if not c.empty])
                      .sort_index())

    for label, (s, e) in all_periods.items():
        regime = "crisis" if label in recession_periods else "calm"
        if regime == "crisis":
            continue  # individual recession periods handled via combined block below
        source = cd
        sub    = source.loc[s:e].dropna(how="all")
        if len(sub) < 10:
            continue
        block = max(5, min(21, len(sub) // 10))

        for base in _TEST_BASES:
            if base not in sub.columns:
                continue
            r_base     = sub[base].dropna()
            sr_base    = _sr_ann(r_base)
            group_rows = []
            for other in sub.columns:
                if other == base:
                    continue
                r_other = sub[other].dropna()
                if len(r_other) < 10:
                    continue
                diff, p, ci_lo, ci_hi = L.lw_sharpe_test(
                    r_base, r_other, n_boot=n_boot, block=block
                )
                group_rows.append({
                    "period":       label,
                    "regime":       regime,
                    "base":         base,
                    "strategy":     other,
                    "Sharpe_base":  sr_base,
                    "Sharpe_other": _sr_ann(r_other),
                    "Sharpe_diff":  diff,
                    "CI_lo_95":     ci_lo,
                    "CI_hi_95":     ci_hi,
                    "LW_p":         p,
                })
            if not group_rows:
                continue
            bh = L.benjamini_hochberg(
                {r["strategy"]: r["LW_p"] for r in group_rows}
            )
            for r in group_rows:
                r["LW_p_adj"]  = float(bh.loc[r["strategy"], "pval_adj"])
                r["LW_reject"] = bool(bh.loc[r["strategy"], "reject"])
            rows.extend(group_rows)

    # ── combined "All Recessions" block ───────────────────────────────────────
    if len(combined_rec) >= 10:
        block = max(5, min(21, len(combined_rec) // 10))
        for base in _TEST_BASES:
            if base not in combined_rec.columns:
                continue
            r_base     = combined_rec[base].dropna()
            sr_base    = _sr_ann(r_base)
            group_rows = []
            for other in combined_rec.columns:
                if other == base:
                    continue
                r_other = combined_rec[other].dropna()
                if len(r_other) < 10:
                    continue
                diff, p, ci_lo, ci_hi = L.lw_sharpe_test(
                    r_base, r_other, n_boot=n_boot, block=block
                )
                group_rows.append({
                    "period":       "Recessions",
                    "regime":       "crisis",
                    "base":         base,
                    "strategy":     other,
                    "Sharpe_base":  sr_base,
                    "Sharpe_other": _sr_ann(r_other),
                    "Sharpe_diff":  diff,
                    "CI_lo_95":     ci_lo,
                    "CI_hi_95":     ci_hi,
                    "LW_p":         p,
                })
            if group_rows:
                bh = L.benjamini_hochberg(
                    {r["strategy"]: r["LW_p"] for r in group_rows}
                )
                for r in group_rows:
                    r["LW_p_adj"]  = float(bh.loc[r["strategy"], "pval_adj"])
                    r["LW_reject"] = bool(bh.loc[r["strategy"], "reject"])
                rows.extend(group_rows)

    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def _rolling_sharpe(daily: pd.DataFrame, window: int = _ROLLING_WINDOW) -> pd.DataFrame:
    roll_ret = daily.rolling(window).mean() * 252
    roll_vol = daily.rolling(window).std() * np.sqrt(252)
    return roll_ret / roll_vol.replace(0, np.nan)


_FOCUS = {"HMVA", "HMVA-mv"}


def _draw_rolling_sharpe_lines(ax, rs: pd.DataFrame, daily_cols) -> None:
    others = [c for c in rs.columns if c not in _FOCUS]
    for col in others:
        ax.plot(rs.index, rs[col], color=_col_color(col, daily_cols),
                lw=1.0, alpha=0.45, label=col)
    for col in [c for c in rs.columns if c in _FOCUS]:
        ax.plot(rs.index, rs[col], color=_col_color(col, daily_cols),
                lw=2.4, alpha=1.0, label=col, zorder=5)


def _style_sharpe_ax(ax) -> None:
    ax.axhline(0,  color="black", lw=0.8, ls="-",  zorder=4)
    ax.axhline(1,  color="grey",  lw=0.7, ls="--", alpha=0.6, zorder=3)
    ax.axhline(-1, color="grey",  lw=0.7, ls="--", alpha=0.6, zorder=3)
    ax.yaxis.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Rolling Sharpe")


def plot_rolling_sharpe_full(daily: pd.DataFrame,
                              outdir: str,
                              recession_periods: Dict[str, Tuple[str, str]]) -> None:
    rs = _rolling_sharpe(daily)
    fig, ax = plt.subplots(figsize=(14, 5))

    for s, e in recession_periods.values():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=_CRISIS_SHADE, alpha=0.55, zorder=0)

    _draw_rolling_sharpe_lines(ax, rs, daily.columns)
    _style_sharpe_ax(ax)
    ax.set_title(f"Rolling {_ROLLING_WINDOW}-day Sharpe ratio", fontsize=12)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, ncol=len(rs.columns), loc="upper left",
              framealpha=0.85, edgecolor="lightgrey")

    ax.set_ylim(ax.get_ylim())
    ybot = ax.get_ylim()[0]
    for label, (s, e) in recession_periods.items():
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        mid    = ts + (te - ts) / 2
        ax.text(mid, ybot, label, ha="center", va="bottom",
                fontsize=7.5, color="#8b0000",
                bbox=dict(fc="white", alpha=0.6, ec="none", pad=1))

    fig.tight_layout()
    fig.savefig(f"{outdir}/rolling_sharpe_full.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_crisis_equity(daily: pd.DataFrame,
                       outdir: str,
                       recession_periods: Dict[str, Tuple[str, str]]) -> None:
    periods             = list(recession_periods.items())
    n                   = len(periods)
    nrows, ncols, fsize = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=fsize, squeeze=False)
    fig.suptitle("Equity curves during NBER recession periods", fontsize=13)

    for idx, (label, (s, e)) in enumerate(periods):
        ax  = axes[idx // ncols][idx % ncols]
        sub = daily.loc[s:e].dropna(how="all")
        if sub.empty:
            ax.set_visible(False)
            continue
        others = [c for c in sub.columns if c not in _FOCUS]
        focus  = [c for c in sub.columns if c in _FOCUS]
        for col in others:
            cum = (1 + sub[col]).cumprod()
            ax.plot(cum.index, cum.values,
                    color=_col_color(col, daily.columns), lw=1.0, alpha=0.45, label=col)
        for col in focus:
            cum = (1 + sub[col]).cumprod()
            ax.plot(cum.index, cum.values,
                    color=_col_color(col, daily.columns), lw=2.4, alpha=1.0, label=col, zorder=5)
        ax.axhline(1, color="black", lw=0.5, ls="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)

    # hide any unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{outdir}/crisis_equity.png", bbox_inches="tight")
    plt.close(fig)


def plot_period_sharpe_heatmap(all_metrics: pd.DataFrame,
                                outdir: str,
                                calm_periods: Dict[str, Tuple[str, str]],
                                recession_periods: Dict[str, Tuple[str, str]]) -> None:
    pivot        = all_metrics.pivot(index="period", columns="strategy", values="Sharpe")
    crisis_names = list(recession_periods.keys())
    calm_names   = list(calm_periods.keys())
    row_order    = crisis_names + calm_names
    pivot        = pivot.reindex([r for r in row_order if r in pivot.index])
    col_order    = [c for c in STRATEGIES if c in pivot.columns]
    pivot        = pivot[col_order]

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
    if 0 < n_crisis < len(pivot):
        ax.axhline(n_crisis - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, label="Annualised Sharpe ratio")
    ax.set_title("Sharpe ratio by market regime and strategy", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/period_sharpe_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_crisis_rolling_sharpe(daily: pd.DataFrame,
                                outdir: str,
                                recession_periods: Dict[str, Tuple[str, str]]) -> None:
    rs        = _rolling_sharpe(daily)
    periods             = list(recession_periods.items())
    n                   = len(periods)
    nrows, ncols, fsize = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=fsize, squeeze=False)
    fig.suptitle(f"Rolling {_ROLLING_WINDOW}-day Sharpe during NBER recessions", fontsize=13)

    for idx, (label, (s, e)) in enumerate(periods):
        ax  = axes[idx // ncols][idx % ncols]
        sub = rs.loc[s:e].dropna(how="all")
        if sub.empty:
            ax.set_visible(False)
            continue
        others = [c for c in sub.columns if c not in _FOCUS]
        focus  = [c for c in sub.columns if c in _FOCUS]
        for col in others:
            ax.plot(sub.index, sub[col], color=_col_color(col, daily.columns),
                    lw=1.0, alpha=0.45, label=col)
        for col in focus:
            ax.plot(sub.index, sub[col], color=_col_color(col, daily.columns),
                    lw=2.4, alpha=1.0, label=col, zorder=5)
        ax.axhline(0,  color="black", lw=0.8, ls="-")
        ax.axhline(1,  color="grey",  lw=0.7, ls="--", alpha=0.6)
        ax.axhline(-1, color="grey",  lw=0.7, ls="--", alpha=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Rolling Sharpe")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{outdir}/crisis_rolling_sharpe.png", bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns(daily: pd.DataFrame,
                        outdir: str,
                        recession_periods: Dict[str, Tuple[str, str]]) -> None:
    annual = (1 + daily).resample("YE").prod() - 1
    cols   = [c for c in STRATEGIES if c in annual.columns]
    width  = 0.8 / len(cols)
    x      = np.arange(len(annual))
    years  = list(annual.index.year)

    fig, ax = plt.subplots(figsize=(14, 5))
    for s, e in recession_periods.values():
        s_year = pd.Timestamp(s).year
        e_year = pd.Timestamp(e).year
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
    ax.set_title("Annual returns by strategy (shaded = NBER recession years)")
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

    # ── fetch NBER recession periods from FRED ─────────────────────────────────
    print("[crisis] fetching NBER recession dates from FRED ...")
    recession_periods = _fetch_recession_periods(daily.index)
    if not recession_periods:
        print("[crisis] WARNING: no recession periods found in data window!")
    else:
        for label, (s, e) in recession_periods.items():
            print(f"  {label}: {s} → {e}")

    # ── compute metrics per period ─────────────────────────────────────────────
    calm_periods = _calm_periods_label(daily.index)
    cd           = _calm_daily(daily, recession_periods)
    all_rows     = []
    all_periods  = {**recession_periods, **calm_periods}

    for label, (s, e) in all_periods.items():
        regime = "crisis" if label in recession_periods else "calm"
        source = daily if regime == "crisis" else cd
        m      = _compute_period_metrics(source, s, e)
        for strat in m.index:
            row = {"period": label, "regime": regime,
                   "strategy": strat, "start": s, "end": e}
            row.update(m.loc[strat].to_dict())
            all_rows.append(row)

    period_metrics = pd.DataFrame(all_rows)
    period_metrics.to_csv(f"{args.out}/period_metrics.csv", index=False)

    # ── regime summary (recession vs calm) ────────────────────────────────────
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

    # ── subperiod statistical tests ────────────────────────────────────────────
    print("\n[crisis] running subperiod Sharpe tests ...")
    test_df = _period_sharpe_tests(daily, calm_periods, recession_periods)
    test_df.to_csv(f"{args.out}/subperiod_sharpe_tests.csv", index=False)

    for base in _TEST_BASES:
        sub = test_df[test_df["base"] == base]
        if sub.empty:
            continue
        print(f"\n=== Subperiod LW Sharpe tests — base: {base} ===")
        print(sub[["period", "strategy", "Sharpe_base", "Sharpe_other",
                   "Sharpe_diff", "CI_lo_95", "CI_hi_95",
                   "LW_p", "LW_p_adj", "LW_reject"]].round(4).to_string(index=False))

    # ── generate plots ─────────────────────────────────────────────────────────
    print("\n[crisis] generating plots ...")
    plot_crisis_equity(daily, args.out, recession_periods)
    plot_crisis_rolling_sharpe(daily, args.out, recession_periods)
    plot_rolling_sharpe_full(daily, args.out, recession_periods)
    plot_period_sharpe_heatmap(period_metrics, args.out, calm_periods, recession_periods)
    plot_annual_returns(daily, args.out, recession_periods)

    print(f"\n[crisis] done → {args.out}/")


if __name__ == "__main__":
    main()
