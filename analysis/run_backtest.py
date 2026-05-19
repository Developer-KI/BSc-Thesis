from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
import pandas as pd

import utils.strategy as L
import utils.data as C


# -----------------------------------------------------------------------------
# Defaults (override via CLI)
# -----------------------------------------------------------------------------
DATA_CSV = "./data/stock_daily_returns.csv"
CONSTITUENTS_CSV = "./data/constiuents.csv"
PERMNO_LIST_TXT = "./data/unique_ids.txt"

PERMNO_COL = "PERMNO"
DATE_COL = "DlyCalDt"
PRICE_COL = "DlyClose"
RET_COL = "DlyRet"

# How to switch between them and why do they perform differently. Whats the root cause?
#1980 - 2002-06 min vol vb tree wins absolutely
#2000 - 2025 max sharpe vb tree wins absolutely

START_DATE = "2000-01-01"
END_DATE = "2025-01-01"
TOP_K = 100

LOOKBACK = 504
REBALANCE = 21

RISK_FREE_BPS = 0.00
COST_BPS = 0.00


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def riskfree_proxy(dates: pd.DatetimeIndex) -> pd.Series:
    daily_rate = (1 + RISK_FREE_BPS)**(1/365) - 1
    return pd.Series(daily_rate, index=dates)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description="HRP x CRSP S&P 500 experiment")
    p.add_argument("--data", default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--rebalance", type=int, default=REBALANCE)
    p.add_argument("--cost-bps", type=float, default=COST_BPS)
    p.add_argument("--out", default="results/backtest")
    args = p.parse_args(argv)

    np.random.seed(42)
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    permno_subset = None
    if os.path.exists(PERMNO_LIST_TXT):
        with open(PERMNO_LIST_TXT) as f:
            permno_subset = [int(line.strip()) for line in f if line.strip()]
        print(f"[main] restricting to {len(permno_subset)} PERMNOs from {PERMNO_LIST_TXT}")

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
        top_k=args.top_k
    )
    cap_wide = universe_fn._cap_wide

    print("\n" + "=" * 72)
    print(f" CRSP S&P 500  -  lookback = {args.lookback}")
    print("=" * 72)

    rf = riskfree_proxy(returns_wide.index)
    strategies = L.make_crsp_strategies(market_cap_wide=cap_wide)

    daily, weights = L.backtest_pit(returns_wide, universe_fn, strategies,
                                    lookback=args.lookback, rebalance=args.rebalance,
                                    cost_bps=args.cost_bps, rf_daily=rf,
                                    min_history_days=args.lookback)

    daily.to_csv(f"{outdir}/daily_excess_returns.csv")

    metrics = L.compute_metrics_pit(daily, weights)
    print("\n=== Performance summary ===")
    print(metrics.round(4).to_string())
    metrics.to_csv(f"{outdir}/metrics.csv")

    print("\n=== Tests HMVA vs others ===")
    base = daily["HMVA"]
    rows = []
    for s in daily.columns:
        if s == "HMVA":
            continue
        lw_diff, lw_p = L.lw_sharpe_test(base, daily[s], n_boot=2000, block=21)
        rows.append({"Strategy": s, "Sharpe_diff": lw_diff, "LW_p": lw_p})
    test_df = pd.DataFrame(rows)
    test_df["LW_p_holm"] = L.adjust_pvalues(test_df["LW_p"].values, "holm")
    test_df["LW_p_bh"] = L.adjust_pvalues(test_df["LW_p"].values, "bh")
    print(test_df.round(4).to_string(index=False))
    test_df.to_csv(f"{outdir}/statistical_tests.csv", index=False)

    L.plot_backtest_results(daily, weights, metrics, outdir,
                            title_suffix=f"(CRSP lookback={args.lookback})")

    L.plot_holdings_concentration(weights, outdir,
                                  title_suffix=f"(CRSP lookback={args.lookback})")

    print(f"\n[done] results at {outdir}/")


if __name__ == "__main__":
    main()
