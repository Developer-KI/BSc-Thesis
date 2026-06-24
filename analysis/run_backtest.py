from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
import pandas as pd

import utils.backtest as L
import utils.data as C


# -----------------------------------------------------------------------------
# Defaults (override via CLI)
# -----------------------------------------------------------------------------
DATA_CSV = "./data/stock_daily_returns.csv"
CONSTITUENTS_CSV = "./data/constiuents.csv"
PERMNO_LIST_TXT = "./data/unique_ids.txt"

SUBFOLDER_OUTPUT = "full"

PERMNO_COL = "PERMNO"
DATE_COL = "DlyCalDt"
PRICE_COL = "DlyClose"
RET_COL = "DlyRet"

START_DATE = "2000-01-01"
END_DATE = "2025-01-01"
TOP_K = 100

LOOKBACK = 126
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
    p = argparse.ArgumentParser(description="CRSP S&P 500 experiment")
    p.add_argument("--data", default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--rebalance", type=int, default=REBALANCE)
    p.add_argument("--cost-bps", type=float, default=COST_BPS)
    p.add_argument("--out", default=f"results/backtest/{SUBFOLDER_OUTPUT}")
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
    
    if "mean" in SUBFOLDER_OUTPUT:
        print("\n=== Tests HMVA vs others (excl. HMVA-mv) ===")
        base_hmva = daily["HMVA"]
        rows = []
        for s in daily.columns:
            if s in ("HMVA", "HMVA-mv"):
                continue
            lw_diff, lw_p, ci_lo, ci_hi = L.lw_sharpe_test(base_hmva, daily[s], n_boot=2000, block=21)
            rows.append({
                "Strategy":    s,
                "Sharpe_diff": lw_diff,
                "CI_lo_95":    ci_lo,
                "CI_hi_95":    ci_hi,
                "LW_p":        lw_p,
            })
        test_df = pd.DataFrame(rows).set_index("Strategy")
        bh_lw = L.benjamini_hochberg(test_df["LW_p"].to_dict())
        test_df["LW_p_adj"]  = bh_lw["pval_adj"]
        test_df["LW_reject"] = bh_lw["reject"]
        test_df = test_df.reset_index()
        print(test_df.round(4).to_string(index=False))
        test_df.to_csv(f"{outdir}/statistical_tests_hmva.csv", index=False)

    if "var" in SUBFOLDER_OUTPUT:
        print("\n=== Tests HMVA-mv vs others (excl. HMVA) ===")
        base_mv = daily["HMVA-mv"]
        rows = []
        for s in daily.columns:
            if s in ("HMVA-mv", "HMVA"):
                continue
            lw_diff, lw_p, ci_lo, ci_hi = L.lw_sharpe_test(base_mv, daily[s], n_boot=2000, block=21)
            rows.append({
                "Strategy":    s,
                "Sharpe_diff": lw_diff,
                "CI_lo_95":    ci_lo,
                "CI_hi_95":    ci_hi,
                "LW_p":        lw_p,
            })
        test_mv_df = pd.DataFrame(rows).set_index("Strategy")
        bh_lw_mv = L.benjamini_hochberg(test_mv_df["LW_p"].to_dict())
        test_mv_df["LW_p_adj"]  = bh_lw_mv["pval_adj"]
        test_mv_df["LW_reject"] = bh_lw_mv["reject"]
        test_mv_df = test_mv_df.reset_index()
        print(test_mv_df.round(4).to_string(index=False))
        test_mv_df.to_csv(f"{outdir}/statistical_tests_hmva_mv.csv", index=False)

    L.plot_backtest_results(daily, weights, metrics, outdir)

    L.plot_holdings_concentration(weights, outdir)

    print(f"\n[done] results at {outdir}/")


if __name__ == "__main__":
    main()
