from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import utils.backtest as L
import utils.data as C
import utils.plotting as _plt


# -----------------------------------------------------------------------------
# Defaults (override via CLI)
# -----------------------------------------------------------------------------
DATA_CSV = "./data/stock_daily_returns.csv"
CONSTITUENTS_CSV = "./data/constiuents.csv"
PERMNO_LIST_TXT = "./data/unique_ids.txt"

# CRSP CIZ format defaults
PERMNO_COL = "PERMNO"
DATE_COL = "DlyCalDt"
PRICE_COL = "DlyClose"
RET_COL = "DlyRet"

# Sample period
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"

# Default grid
DEFAULT_LOOKBACKS = "63, 126, 252, 504"
DEFAULT_HISTORY = 504
DEFAULT_COSTS = "0, 2, 5, 10, 20"
DEFAULT_REBALANCE = 21
DEFAULT_TOP_K = 100

RISK_FREE_BPS = 0.0

# -----------------------------------------------------------------------------
# Strategy filtering
# -----------------------------------------------------------------------------

def filter_strategies(strategies: dict,
                      keep: List[str]) -> dict:
    """
    Trim the strategy dict to the requested subset, always preserving the
    EW baseline (it's the reference for ΔSharpe).
    """
    keep_set = set(keep)
    keep_set.add("EW")            # always required
    return {k: v for k, v in strategies.items() if k in keep_set}


# -----------------------------------------------------------------------------
# Grid runner
# -----------------------------------------------------------------------------

def run_grid(returns_wide: pd.DataFrame,
             universe_fn: C.UniverseFn,
             lookbacks: List[int],
             costs: List[float],
             rebalance: int,
             strategies_keep: List[str],
             rf_daily: pd.Series = None,
             market_cap_wide=None
             ) -> pd.DataFrame:
    rows = []
    total = len(lookbacks) * len(costs)
    cnt = 0
    grid_t0 = time.time()
    max_lookback = DEFAULT_HISTORY  # all cells start after the longest lookback

    for lb in lookbacks:
        for cost in costs:
            cnt += 1
            cell_t0 = time.time()
            print("\n" + "-" * 72)
            print(f"[grid {cnt}/{total}]  lookback={lb}  cost_bps={cost}")
            print("-" * 72)

            strategies = L.make_crsp_strategies(market_cap_wide=market_cap_wide)
            strategies = filter_strategies(strategies, strategies_keep)
            print(f"  strategies in this cell: {list(strategies.keys())}")

            try:
                daily, weights = L.backtest_pit(
                    returns_wide, universe_fn, strategies,
                    lookback=lb, rebalance=rebalance,
                    min_history_days=max_lookback,
                    cost_bps=cost, rf_daily=rf_daily,
                    verbose=True)
                metrics = L.compute_metrics_pit(daily, weights)

                base_sr = metrics.loc["EW", "Sharpe"] \
                    if "EW" in metrics.index else np.nan
                base_so = metrics.loc["EW", "Sortino"] \
                    if "EW" in metrics.index and "Sortino" in metrics.columns \
                    else np.nan
                for strat, m in metrics.iterrows():
                    rows.append({
                        "lookback": lb, "cost_bps": cost,
                        "strategy": strat,
                        "ann_return":  m["AnnReturn"],
                        "ann_vol":     m["AnnVol"],
                        "sharpe":      m["Sharpe"],
                        "sortino":     m.get("Sortino",  np.nan),
                        "omega":       m.get("Omega",    np.nan),
                        "max_dd":      m["MaxDD"],
                        "calmar":      m["Calmar"],
                        "var95":       m.get("VaR95",    np.nan),
                        "cvar95":      m.get("CVaR95",   np.nan),
                        "hit_rate":    m.get("HitRate",  np.nan),
                        "turnover":    m["Turnover"],
                        "sharpe_minus_base":  m["Sharpe"] - base_sr,
                        "sortino_minus_base": m.get("Sortino", np.nan) - base_so,
                    })
            except Exception as e:
                print(f"  ! cell failed: {type(e).__name__}: {e}")
                rows.append({
                    "lookback": lb, "cost_bps": cost,
                    "strategy": "FAILED", "error": str(e),
                })

            cell_dt = time.time() - cell_t0
            grid_dt = time.time() - grid_t0
            eta = grid_dt / cnt * (total - cnt)
            print(f"  cell completed in {cell_dt:.0f}s.  "
                  f"Grid elapsed {grid_dt/60:.1f}min, "
                  f"ETA {eta/60:.1f}min")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_heatmaps(df: pd.DataFrame, outdir: str) -> None:
    """Heatmaps per strategy (lookback × cost_bps). Produces ΔSharpe and ΔSortino grids."""
    os.makedirs(outdir, exist_ok=True)
    sns.set_style("white")

    df = df[df["strategy"] != "FAILED"].copy()
    if df.empty:
        return

    _metrics = [
        ("sharpe_minus_base",  "ΔSharpe vs EW",  "sharpe"),
        ("sortino_minus_base", "ΔSortino vs EW", "sortino"),
    ]

    strategies = sorted(s for s in df["strategy"].unique() if s != "EW")
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        safe = strat.replace("-", "_").replace("/", "_")
        for col, label, fname_tag in _metrics:
            if col not in df.columns or sub[col].isna().all():
                continue
            try:
                pivot = sub.set_index(["lookback", "cost_bps"])[col].unstack("cost_bps")
            except Exception:
                continue
            if pivot.empty:
                continue
            _, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                        center=0, ax=ax, cbar_kws={"label": label})
            ax.set_title(f"{strat}: {label}")
            ax.set_xlabel("cost_bps")
            ax.set_ylabel("lookback")
            plt.tight_layout()
            plt.savefig(f"{outdir}/heatmap_{fname_tag}_{safe}.png", dpi=120)
            plt.close()


def make_summary_table_plot(summary: pd.DataFrame, outdir: str) -> None:
    """Colour-coded metrics table of the robustness summary using plotting.py."""
    if _plt is None:
        return
    _rename = {
        "mean_sharpe":   "Sharpe Ratio",
        "mean_sortino":  "Sortino Ratio",
        "mean_calmar":   "Calmar Ratio",
        "mean_max_dd":   "Max Drawdown (%)",
        "mean_var95":    "VaR 95% (%)",
        "mean_hit_rate": "Hit Rate (%)",
        "mean_turnover": "Turnover",
        "mean_dSharpe":  "ΔSharpe vs EW",
        "mean_dSortino": "ΔSortino vs EW",
    }
    cols = [c for c in _rename if c in summary.columns]
    tbl = summary[cols].rename(columns=_rename)
    _plt.plot_metrics_table(
        tbl,
        title="Robustness Summary: Mean Performance Metrics by Strategy",
        save_path=f"{outdir}/robustness_metrics_table.png",
    )
    plt.close("all")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Cost Robustness of Strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="Keep only top K PERMNOs by market cap each rebalance")
    p.add_argument("--lookbacks", default=DEFAULT_LOOKBACKS,
                   help="comma-separated, e.g. 126,252,504")
    p.add_argument("--costs", default=DEFAULT_COSTS,
                   help="comma-separated cost_bps levels, e.g. 0,2,5,10,20")
    p.add_argument("--rebalance", type=int, default=DEFAULT_REBALANCE,
                   help="rebalance frequency in trading days (fixed)")
    p.add_argument("--strategies",
                   default="EW,HMVA",
                   help="comma-separated strategy names to include")
    p.add_argument("--out", default="results/cost_robustness")
    args = p.parse_args(argv)

    np.random.seed(42)
    os.makedirs(args.out, exist_ok=True)

    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    costs = [float(x) for x in args.costs.split(",") if x.strip()]
    strategies_keep = [x.strip() for x in args.strategies.split(",") if x.strip()]

    print("=" * 72)
    print(" CRSP S&P 500 robustness sweep")
    print("=" * 72)
    print(f"  lookbacks  : {lookbacks}")
    print(f"  costs_bps  : {costs}")
    print(f"  rebalance  : {args.rebalance}")
    print(f"  top_k      : {args.top_k}")
    print(f"  strategies : {strategies_keep}")
    print(f"  total cells: {len(lookbacks) * len(costs)}")

    # -- 1.  Load CRSP returns ONCE -------------------------------------
    permno_subset = None
    if os.path.exists(PERMNO_LIST_TXT):
        with open(PERMNO_LIST_TXT) as f:
            permno_subset = [int(x.strip()) for x in f if x.strip()]
        print(f"  PERMNO subset: {len(permno_subset)} ids from "
              f"{PERMNO_LIST_TXT}")

    print("\n[main] Loading CRSP returns ...")
    returns_wide = C.load_crsp_returns(
        args.data,
        permno_subset=permno_subset,
        start_date=args.start,
        end_date=args.end,
        price_col=PRICE_COL,
        date_col=DATE_COL,
        permno_col=PERMNO_COL,
        ret_col=RET_COL,
    )

    universe_fn = C.make_universe_fn(
        args.constituents,
        market_cap_csv=args.data if args.top_k else None,
        top_k=args.top_k,
    )

    # rf = 0 for the CRSP run (see run_crsp.py for justification)
    daily_rate = (1 + RISK_FREE_BPS)**(1/365) - 1
    rf_daily = pd.Series(daily_rate, index=returns_wide.index)

    cap_wide = universe_fn._cap_wide

    # -- 2.  Run the sweep ----------------------------------------------
    df = run_grid(
        returns_wide, universe_fn,
        lookbacks=lookbacks,
        costs=costs,
        rebalance=args.rebalance,
        strategies_keep=strategies_keep,
        rf_daily=rf_daily,
        market_cap_wide=cap_wide
    )

    long_path = f"{args.out}/robustness_long.csv"
    df.to_csv(long_path, index=False)
    print(f"\n[done] {len(df)} rows -> {long_path}")

    # -- 3.  Summary table ----------------------------------------------
    df_clean = df[df["strategy"] != "FAILED"].copy()
    if not df_clean.empty:
        summary = (df_clean.groupby("strategy")
                            .agg(mean_dSharpe=("sharpe_minus_base", "mean"),
                                 median_dSharpe=("sharpe_minus_base", "median"),
                                 pct_positive=("sharpe_minus_base",
                                               lambda s: (s > 0).mean()),
                                 mean_dSortino=("sortino_minus_base", "mean"),
                                 mean_sharpe=("sharpe", "mean"),
                                 mean_sortino=("sortino", "mean"),
                                 mean_calmar=("calmar", "mean"),
                                 mean_max_dd=("max_dd", "mean"),
                                 mean_var95=("var95", "mean"),
                                 mean_hit_rate=("hit_rate", "mean"),
                                 mean_turnover=("turnover", "mean"),
                                 n_cells=("sharpe", "count")))
        print("\n=== Summary across robustness cells ===")
        print(summary.round(4).to_string())
        summary.to_csv(f"{args.out}/robustness_summary.csv")

        # -- 4.  Plots --------------------------------------------------
        make_heatmaps(df_clean, args.out)
        make_summary_table_plot(summary, args.out)
        print(f"[done] plots in {args.out}/")
    else:
        print("[warn] no successful cells; nothing to summarise.")


if __name__ == "__main__":
    main()