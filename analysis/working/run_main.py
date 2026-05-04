"""
=============================================================================
run_main.py - Main empirical experiment
=============================================================================

Runs the full backtest on TWO universes back-to-back:

    (a) ETF universe       (~30 assets, real-world sanity check)
    (b) Large-cap stocks   (~100 assets, the high-dim regime where the
                            advanced estimators have something to fix)

Each run produces:
    - performance metrics CSV
    - statistical tests CSV (DM and LW2008 Sharpe test, with Holm + BH adj)
    - equity curve, drawdown, Sharpe / turnover / drawdown bar charts,
      last-rebalance weight heatmap, adaptive POET (K, C) trace.

Risk-free rate: ^IRX (13-week T-bill).
Transaction cost: 2 bps per unit weight changed at rebalance (configurable).
=============================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import hrp_lib as L


def run_one_universe(name: str,
                     tickers: list,
                     start: str,
                     end: str,
                     lookback: int,
                     rebalance: int,
                     cost_bps: float,
                     outdir: str) -> None:
    """End-to-end pipeline for one (universe, period) combination."""

    print("\n" + "=" * 72)
    print(f" Universe: {name}")
    print("=" * 72)

    os.makedirs(outdir, exist_ok=True)

    # --- 1. data ---------------------------------------------------------
    returns = L.get_returns(tickers, start=start, end=end)
    rf = L.get_riskfree(returns.index)

    # --- 2. backtest -----------------------------------------------------
    strategies, apoet = L.make_default_strategies(linkage_method="single")
    daily, weights = L.backtest(returns, strategies,
                                lookback=lookback, rebalance=rebalance,
                                cost_bps=cost_bps, rf_daily=rf)

    daily.to_csv(f"{outdir}/daily_excess_returns.csv")

    # --- 3. metrics ------------------------------------------------------
    metrics = L.compute_metrics(daily, weights)
    print("\n=== Performance summary (excess returns over ^IRX) ===")
    print(metrics.round(4).to_string())
    metrics.to_csv(f"{outdir}/metrics.csv")

    # --- 4. statistical tests vs HRP-Sample ------------------------------
    print("\n=== Tests vs HRP-Sample ===")
    base = daily["HRP-Sample"]
    rows = []
    for s in daily.columns:
        if s == "HRP-Sample":
            continue
        dm_stat, dm_p = L.diebold_mariano(daily[s], base, h=21)
        lw_diff, lw_p = L.lw_sharpe_test(daily[s], base, n_boot=2000, block=21)
        rows.append({"Strategy": s,
                     "DM_stat": dm_stat, "DM_p": dm_p,
                     "Sharpe_diff": lw_diff, "LW_p": lw_p})
    test_df = pd.DataFrame(rows)

    # multiple-testing correction across the 8 comparisons
    test_df["DM_p_holm"]  = L.adjust_pvalues(test_df["DM_p"].values,  "holm")
    test_df["LW_p_holm"]  = L.adjust_pvalues(test_df["LW_p"].values,  "holm")
    test_df["DM_p_bh"]    = L.adjust_pvalues(test_df["DM_p"].values,  "bh")
    test_df["LW_p_bh"]    = L.adjust_pvalues(test_df["LW_p"].values,  "bh")
    print(test_df.round(4).to_string(index=False))
    test_df.to_csv(f"{outdir}/statistical_tests.csv", index=False)

    # --- 5. plots --------------------------------------------------------
    suffix = f"({name})"
    L.plot_equity_and_drawdown(daily, outdir, title_suffix=suffix)
    L.plot_metric_bars(metrics, outdir, title_suffix=suffix)
    L.plot_weights_heatmap(weights, outdir)

    # adaptive POET diagnostic
    if apoet.history:
        hist = pd.DataFrame(apoet.history)
        hist.to_csv(f"{outdir}/apoet_history.csv", index_label="rebal")
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax[0].plot(hist["K"], marker="o", lw=1)
        ax[0].set_ylabel("K* (chosen)")
        ax[0].set_title("Adaptive POET cross-validation choices")
        ax[1].plot(hist["C"], marker="s", color="darkorange", lw=1)
        ax[1].set_ylabel("C* (chosen)")
        ax[1].set_xlabel("Rebalance index")
        plt.tight_layout()
        plt.savefig(f"{outdir}/apoet_history.png", dpi=120)
        plt.close()

    print(f"\n[done] All outputs written to {outdir}/")


def main() -> None:
    np.random.seed(42)

    # --- run (a): ETF universe (low-dim sanity check) -------------------
    run_one_universe(
        name="ETFs",
        tickers=L.ETF_UNIVERSE,
        start="2015-01-01",
        end="2025-01-01",
        lookback=504,
        rebalance=21,
        cost_bps=2.0,
        outdir="results/etfs",
    )

    # --- run (b): high-dim stock universe -------------------------------
    run_one_universe(
        name="Stocks",
        tickers=L.STOCKS_UNIVERSE,
        start="2015-01-01",
        end="2025-01-01",
        lookback=504,
        rebalance=21,
        cost_bps=2.0,
        outdir="results/stocks",
    )


if __name__ == "__main__":
    main()