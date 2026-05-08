"""
=============================================================================
run_robustness.py - CRSP point-in-time robustness sweep
=============================================================================

Sweeps the empirical experiment over a configurable grid of:

    lookback    in {126, 252, 504}                (0.5 / 1 / 2 years)
    cost_bps    in {0, 2, 5, 10}                  (transaction cost levels)
    linkage     in {single, average, ward}        (HRP linkage method, averaged)

Rebalance frequency is fixed (default = 21 days / monthly).

This is the *companion* to run_crsp.py: run_crsp.py answers
"do the advanced estimators help?" at the default settings; this script
answers "is that conclusion robust to the parameter choices?"

Defaults are tuned for ~2-3h on a modern laptop:
    lookbacks  = 126, 252, 504
    costs      = 0, 2, 5, 10
    linkages   = single, average, ward
        => 3 x 4 x 3 = 36 cells

Speed-up tricks:
    * The CRSP file is read ONCE at startup and shared across all cells.
    * --no-apoet skips POET-CV (the runtime bottleneck): ~5x faster.
    * --strategies filters the strategy set; e.g. drop slow MinVar-NLS.

Outputs in results/crsp_robustness/:
    robustness_long.csv      one row per (cell, strategy)
    robustness_summary.csv   aggregated across cells per strategy
    heatmap_*.png            per-strategy Sharpe-vs-base heatmaps (lookback x cost)
=============================================================================
"""

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

import analysis.hrp_lib as L
import analysis.crsp_data as C


# -----------------------------------------------------------------------------
# Defaults (override via CLI)
# -----------------------------------------------------------------------------
DATA_CSV = "./data/universe/stock_daily_returns.csv"
CONSTITUENTS_CSV = "./data/universe/constiuents.csv"
PERMNO_LIST_TXT = "./data/unique_ids.txt"

# CRSP CIZ format defaults
PRICE_COL = "DlyClose"
DATE_COL = "DlyCalDt"
PERMNO_COL = "PERMNO"
RET_COL = "DlyRet"

# Sample period
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"

# Default grid
DEFAULT_LOOKBACKS = "126,252,504"
DEFAULT_COSTS = "0,2,5,10"
DEFAULT_LINKAGES = "single,average,ward"
DEFAULT_REBALANCE = 21
DEFAULT_TOP_K = 100


# -----------------------------------------------------------------------------
# Strategy filtering
# -----------------------------------------------------------------------------

def filter_strategies(strategies: dict,
                      keep: List[str],
                      drop_apoet: bool = False) -> dict:
    """
    Trim the strategy dict to the requested subset, always preserving the
    HRP-Sample baseline (it's the reference for ΔSharpe).
    """
    keep_set = set(keep)
    keep_set.add("HRP-Sample")            # always required
    if drop_apoet:
        keep_set.discard("HRP-PoetCV")
    return {k: v for k, v in strategies.items() if k in keep_set}


# -----------------------------------------------------------------------------
# Grid runner
# -----------------------------------------------------------------------------

def run_grid(returns_wide: pd.DataFrame,
             universe_fn: C.UniverseFn,
             lookbacks: List[int],
             costs: List[float],
             linkages: List[str],
             rebalance: int,
             strategies_keep: List[str],
             drop_apoet: bool = False,
             rf_daily: pd.Series = None,
             ) -> pd.DataFrame:
    rows = []
    total = len(lookbacks) * len(costs) * len(linkages)
    cnt = 0
    grid_t0 = time.time()

    for lb in lookbacks:
        for cost in costs:
            for lnk in linkages:
                cnt += 1
                cell_t0 = time.time()
                print("\n" + "-" * 72)
                print(f"[grid {cnt}/{total}]  lookback={lb}  cost_bps={cost}  "
                      f"linkage={lnk}")
                print("-" * 72)

                strategies = L.make_crsp_strategies(linkage_method=lnk)
                strategies = filter_strategies(strategies, strategies_keep,
                                               drop_apoet=drop_apoet)
                print(f"  strategies in this cell: {list(strategies.keys())}")

                try:
                    daily, weights = L.backtest_pit(
                        returns_wide, universe_fn, strategies,
                        lookback=lb, rebalance=rebalance,
                        cost_bps=cost, rf_daily=rf_daily,
                        verbose=True)
                    metrics = L.compute_metrics_pit(daily, weights)

                    base_sr = metrics.loc["HRP-Sample", "Sharpe"] \
                        if "HRP-Sample" in metrics.index else np.nan
                    for strat, m in metrics.iterrows():
                        rows.append({
                            "lookback": lb, "cost_bps": cost, "linkage": lnk,
                            "strategy": strat,
                            "ann_return": m["AnnReturn"],
                            "ann_vol": m["AnnVol"],
                            "sharpe": m["Sharpe"],
                            "max_dd": m["MaxDD"],
                            "calmar": m["Calmar"],
                            "turnover": m["Turnover"],
                            "sharpe_minus_base": m["Sharpe"] - base_sr,
                        })
                except Exception as e:
                    print(f"  ! cell failed: {type(e).__name__}: {e}")
                    rows.append({
                        "lookback": lb, "cost_bps": cost, "linkage": lnk,
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
    """One ΔSharpe-vs-base heatmap per strategy (lookback × cost_bps), averaged over linkage."""
    os.makedirs(outdir, exist_ok=True)
    sns.set_style("white")

    df = df[df["strategy"] != "FAILED"].copy()
    if df.empty:
        return

    strategies = sorted(s for s in df["strategy"].unique()
                        if s != "HRP-Sample")
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        try:
            pivot = (sub.groupby(["lookback", "cost_bps"])["sharpe_minus_base"]
                        .mean().unstack("cost_bps"))
        except Exception:
            continue
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, ax=ax, cbar_kws={"label": "ΔSharpe vs HRP-Sample"})
        ax.set_title(f"{strat}: Sharpe − HRP-Sample\n"
                     f"averaged over linkage methods")
        ax.set_xlabel("cost_bps")
        ax.set_ylabel("lookback")
        plt.tight_layout()
        safe = strat.replace("-", "_").replace("/", "_")
        plt.savefig(f"{outdir}/heatmap_{safe}.png", dpi=120)
        plt.close()


def linkage_sensitivity_plot(df: pd.DataFrame, outdir: str) -> None:
    """How much does linkage matter, holding lookback × cost_bps fixed?"""
    df = df[df["strategy"] != "FAILED"].copy()
    if df.empty:
        return
    sub = df[df["strategy"].isin(
        ["HRP-Sample", "HRP-LW", "HRP-NLS", "HRP-POET", "HRP-PoetCV"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=sub, x="strategy", y="sharpe", hue="linkage", ax=ax)
    ax.set_title("Sharpe distribution across (lookback × cost_bps) cells "
                 "by linkage method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/linkage_sensitivity.png", dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="HRP × covariance robustness sweep on CRSP S&P 500",
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
                   help="comma-separated cost_bps levels, e.g. 0,2,5,10")
    p.add_argument("--rebalance", type=int, default=DEFAULT_REBALANCE,
                   help="rebalance frequency in trading days (fixed)")
    p.add_argument("--linkages", default=DEFAULT_LINKAGES,
                   help="comma-separated, e.g. single,average,ward")
    p.add_argument("--strategies",
                   default="HRP-Sample,HRP-LW,HRP-NLS,HRP-POET,HRP-POETRY,"
                           "MHRP,MHRP-LW,MHRP-NLS,EW",
                   help="comma-separated strategy names to include")
    p.add_argument("--out", default="results/crsp_robustness")
    args = p.parse_args(argv)

    np.random.seed(42)
    os.makedirs(args.out, exist_ok=True)

    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    costs = [float(x) for x in args.costs.split(",") if x.strip()]
    linkages = [x.strip() for x in args.linkages.split(",") if x.strip()]
    strategies_keep = [x.strip() for x in args.strategies.split(",") if x.strip()]

    print("=" * 72)
    print(" CRSP S&P 500 robustness sweep")
    print("=" * 72)
    print(f"  lookbacks  : {lookbacks}")
    print(f"  costs_bps  : {costs}")
    print(f"  rebalance  : {args.rebalance}")
    print(f"  linkages   : {linkages}  (averaged in heatmaps)")
    print(f"  top_k      : {args.top_k}")
    print(f"  strategies : {strategies_keep}"
          f"{'  (no-apoet)' if args.no_apoet else ''}")
    print(f"  total cells: {len(lookbacks) * len(costs) * len(linkages)}")

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
    rf_daily = pd.Series(0.0, index=returns_wide.index)

    # -- 2.  Run the sweep ----------------------------------------------
    df = run_grid(
        returns_wide, universe_fn,
        lookbacks=lookbacks,
        costs=costs,
        linkages=linkages,
        rebalance=args.rebalance,
        strategies_keep=strategies_keep,
        drop_apoet=False,
        rf_daily=rf_daily,
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
                                 mean_sharpe=("sharpe", "mean"),
                                 mean_turnover=("turnover", "mean"),
                                 n_cells=("sharpe", "count")))
        print("\n=== Summary across robustness cells ===")
        print(summary.round(4).to_string())
        summary.to_csv(f"{args.out}/robustness_summary.csv")

        # -- 4.  Plots --------------------------------------------------
        make_heatmaps(df_clean, args.out)
        linkage_sensitivity_plot(df_clean, args.out)
        print(f"[done] plots in {args.out}/")
    else:
        print("[warn] no successful cells; nothing to summarise.")


if __name__ == "__main__":
    main()