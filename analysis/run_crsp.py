"""
=============================================================================
run_crsp.py - Point-in-time CRSP S&P 500 experiment
=============================================================================

Runs the HRP x ariance comparison on the *true* time-varying S&P 500
universe using CRSP daily data, swept over three lookback configurations:

    lookback = 252  trading days  (~ 1 year)   -> N/T ~ 2.0   (p > n)
    lookback = 504  trading days  (~ 2 years)  -> N/T ~ 1.0   (p ~ n)
    lookback = 756  trading days  (~ 3 years)  -> N/T ~ 0.66  (p < n)

These three regimes are exactly where the advanced estimators are
designed to operate, and where the differences in their behaviour
should be most visible.  This is the headline experiment of the thesis.

Inputs (override via CLI arguments or the constants below):
    DATA_CSV          : path to CRSP daily file (long format)
    CONSTITUENTS_CSV  : path to S&P 500 constituents (range format)
    PRICE_COL, DATE_COL, PERMNO_COL : adjust if your column names differ
    START / END       : restrict to this date range

Outputs in results/crsp_lb{LB}/:
    metrics.csv                # headline metrics table
    statistical_tests.csv      # DM + LW Sharpe, Holm + BH adjusted
    daily_excess_returns.csv
    universe_sizes.csv         # diagnostic: N_t at each rebalance
    *.png                      # equity / drawdown / sharpe / turnover / etc.

A combined cross-lookback summary is written to results/crsp_summary.csv.
=============================================================================
"""

from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hrp_lib as L
import crsp_data as C


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

# Sample period for the experiment
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"
TOP_K = 100

# Lookbacks to sweep: 1m, 3m, 6m, 1y, 2y
LOOKBACKS = (126, 252, 504)
REBALANCE = 21

# Portfolio frinction
RISK_FREE = 0.0
COST_BPS = 0.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def riskfree_proxy(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Hard-coded constant rf = 0 for the CRSP run.

    Reason: the user has CRSP returns; pulling ^IRX from yfinance for a
    historical CRSP run risks future-data leakage if the dates do not line
    up cleanly.  Better to use a constant 0 here (so we report excess vs.
    cash = 0) and document it.  Replace with a proper FRED DGS3MO series
    or CRSP T-bill file when available.
    """
    daily_rate = (1 + RISK_FREE)**(1/365) - 1

    return pd.Series(daily_rate, index=dates)


def run_one_lookback(returns_wide: pd.DataFrame,
                     universe_fn: C.UniverseFn,
                     lookback: int,
                     min_history: int,
                     rebalance: int,
                     cost_bps: float,
                     outdir: str,
                     market_cap_wide=None) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print(f" CRSP S&P 500  -  lookback = {lookback} (N/T ~ {TOP_K/lookback:.2f})")
    print("=" * 72)
    os.makedirs(outdir, exist_ok=True)

    rf = riskfree_proxy(returns_wide.index)
    strategies = L.make_crsp_strategies(linkage_method="single",
                                        market_cap_wide=market_cap_wide)



    daily, weights = L.backtest_pit(returns_wide, universe_fn, strategies,
                                    lookback=lookback, rebalance=rebalance,
                                    cost_bps=cost_bps, rf_daily=rf, min_history_days=min_history)

    daily.to_csv(f"{outdir}/daily_excess_returns.csv")

    metrics = L.compute_metrics_pit(daily, weights)
    print("\n=== Performance summary ===")
    print(metrics.round(4).to_string())
    metrics.to_csv(f"{outdir}/metrics.csv")

    # Tests vs EW (1/N baseline)
    print("\n=== Tests vs EW (1/N) ===")
    base = daily["EW"]
    rows = []
    for s in daily.columns:
        if s == "EW":
            continue
        dm_stat, dm_p = L.diebold_mariano(daily[s], base, h=21)
        lw_diff, lw_p = L.lw_sharpe_test(daily[s], base, n_boot=2000, block=21)
        rows.append({"Strategy": s, "DM_stat": dm_stat, "DM_p": dm_p,
                     "Sharpe_diff": lw_diff, "LW_p": lw_p})
    test_df = pd.DataFrame(rows)
    test_df["DM_p_holm"] = L.adjust_pvalues(test_df["DM_p"].values, "holm")
    test_df["LW_p_holm"] = L.adjust_pvalues(test_df["LW_p"].values, "holm")
    test_df["DM_p_bh"] = L.adjust_pvalues(test_df["DM_p"].values, "bh")
    test_df["LW_p_bh"] = L.adjust_pvalues(test_df["LW_p"].values, "bh")
    print(test_df.round(4).to_string(index=False))
    test_df.to_csv(f"{outdir}/statistical_tests.csv", index=False)

    # plots
    suffix = f"(CRSP lookback={lookback})"
    L.plot_equity_and_drawdown(daily, outdir, title_suffix=suffix)
    L.plot_metric_bars(metrics, outdir, title_suffix=suffix)

    metrics["lookback"] = lookback
    metrics.index.name = "strategy"
    return metrics.reset_index()


def plot_summary(summary_long: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pivot = summary_long.pivot(index="strategy", columns="lookback",
                               values="Sharpe")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot.bar(ax=ax, edgecolor="black")
    ax.set_title("Sharpe ratio by lookback (CRSP S&P 500)")
    ax.set_ylabel("Sharpe")
    ax.legend(title="lookback")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sharpe_by_lookback.png", dpi=120)
    plt.close()

    pivot_to = summary_long.pivot(index="strategy", columns="lookback",
                                  values="Turnover")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_to.plot.bar(ax=ax, edgecolor="black", color=["#3a6ea5", "#7e9bb8", "#c0d0e0"])
    ax.set_title("Average turnover per rebalance, by lookback")
    ax.set_ylabel("Sum |Δw|")
    ax.legend(title="lookback")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{outdir}/turnover_by_lookback.png", dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description="HRP x CRSP S&P 500 experiment")
    p.add_argument("--data", default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--top-k", type=int, default=TOP_K,
                   help="Keep only top K PERMNOs by market cap each rebalance")
    p.add_argument("--lookbacks", default=",".join(str(x) for x in LOOKBACKS),
                   help="comma-separated, e.g. 252,504,756")
    p.add_argument("--rebalance", type=int, default=REBALANCE)
    p.add_argument("--cost-bps", type=float, default=COST_BPS)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    np.random.seed(42)
    lookbacks = [int(x) for x in args.lookbacks.split(",")]

    # restrict permno_subset to those in unique_ids.txt
    permno_subset = None
    if os.path.exists(PERMNO_LIST_TXT):
        with open(PERMNO_LIST_TXT) as f:
            permno_subset = [int(line.strip()) for line in f if line.strip()]
        print(f"[main] restricting to {len(permno_subset)} PERMNOs from "
              f"{PERMNO_LIST_TXT}")

    # 1. load CRSP returns
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

    # 2. load constituents
    universe_fn = C.make_universe_fn(
        args.constituents,
        market_cap_csv=args.data if args.top_k else None,
        top_k=args.top_k
    )
    cap_wide = universe_fn._cap_wide  # already loaded; None when top_k=0

    # 3. sweep lookbacks
    summary_pieces = []
    for lb in lookbacks:
        outdir = f"{args.out}/crsp_lb{lb}"
        df = run_one_lookback(returns_wide, universe_fn,
                              lookback=lb, rebalance=args.rebalance,
                              cost_bps=args.cost_bps, outdir=outdir,
                              min_history=max(lookbacks),
                              market_cap_wide=cap_wide)
        summary_pieces.append(df)

    summary_long = pd.concat(summary_pieces, ignore_index=True)
    summary_long.to_csv(f"{args.out}/crsp_summary.csv", index=False)
    plot_summary(summary_long, args.out)
    print(f"\n[done] cross-lookback summary at {args.out}/crsp_summary.csv")


if __name__ == "__main__":
    main()