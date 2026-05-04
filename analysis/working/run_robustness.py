"""
=============================================================================
run_robustness.py - 2 x 2 x 2 robustness matrix
=============================================================================

Sweeps over:
    lookback   in {252, 504, 756}      (1 / 2 / 3 years)
    rebalance  in {5, 21, 63}          (weekly / monthly / quarterly)
    linkage    in {single, average, ward}

For each cell we record the 9 strategies' Sharpe, ann. vol, max DD,
turnover, and Sharpe difference vs. HRP-Sample.

Default universe is the ETF panel because (i) it is fast and (ii) the
robustness story is about parameter sensitivity, not the dimensional
regime - the dimensional regime is the role of run_main.py.

Results land in results/robustness/  as a long DataFrame plus heatmaps.
=============================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import hrp_lib as L


def run_grid(returns: pd.DataFrame,
             rf: pd.Series,
             cost_bps: float = 2.0) -> pd.DataFrame:
    rows = []
    grid_lb  = [252, 504, 756]
    grid_rb  = [5, 21, 63]
    grid_lnk = ["single", "average", "ward"]

    total = len(grid_lb) * len(grid_rb) * len(grid_lnk)
    cnt = 0
    for lb in grid_lb:
        for rb in grid_rb:
            for lnk in grid_lnk:
                cnt += 1
                print(f"\n[grid {cnt}/{total}] lookback={lb} rebal={rb} link={lnk}")
                strategies, _ = L.make_default_strategies(linkage_method=lnk)
                try:
                    daily, weights = L.backtest(returns, strategies,
                                                lookback=lb, rebalance=rb,
                                                cost_bps=cost_bps, rf_daily=rf)
                    metrics = L.compute_metrics(daily, weights)
                    base_sr = metrics.loc["HRP-Sample", "Sharpe"]
                    for strat, m in metrics.iterrows():
                        rows.append({
                            "lookback": lb, "rebalance": rb, "linkage": lnk,
                            "strategy": strat,
                            "sharpe": m["Sharpe"],
                            "ann_vol": m["AnnVol"],
                            "max_dd": m["MaxDD"],
                            "turnover": m["Turnover"],
                            "sharpe_minus_base": m["Sharpe"] - base_sr,
                        })
                except Exception as e:
                    print(f"   ! cell failed: {e}")
                    continue
    return pd.DataFrame(rows)


def make_heatmaps(df: pd.DataFrame, outdir: str,
                  strategies_to_plot=("HRP-LW", "HRP-NLS", "HRP-POET",
                                      "HRP-PoetCV", "MinVar-NLS", "EW")) -> None:
    os.makedirs(outdir, exist_ok=True)
    sns.set_style("white")
    for strat in strategies_to_plot:
        sub = df[df["strategy"] == strat]
        if sub.empty:
            continue
        # average across linkage to get a 2-D (lookback x rebalance) heatmap
        pivot = (sub.groupby(["lookback", "rebalance"])["sharpe_minus_base"]
                    .mean().unstack("rebalance"))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, ax=ax, cbar_kws={"label": "ΔSharpe"})
        ax.set_title(f"{strat}: Sharpe minus HRP-Sample\n"
                     f"(averaged over linkage methods)")
        plt.tight_layout()
        plt.savefig(f"{outdir}/heatmap_{strat.replace('-', '_')}.png", dpi=120)
        plt.close()


def main() -> None:
    np.random.seed(42)
    outdir = "results/robustness"
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 72)
    print(" Robustness: lookback x rebalance x linkage  (ETF universe)")
    print("=" * 72)

    returns = L.get_returns(L.ETF_UNIVERSE,
                            start="2015-01-01", end="2025-01-01")
    rf = L.get_riskfree(returns.index)

    df = run_grid(returns, rf, cost_bps=2.0)
    df.to_csv(f"{outdir}/robustness_long.csv", index=False)
    print(f"\n[done] {len(df)} rows -> {outdir}/robustness_long.csv")

    # summary: average ΔSharpe per strategy across all 27 cells
    summary = (df.groupby("strategy")
                 .agg(mean_dSharpe=("sharpe_minus_base", "mean"),
                      median_dSharpe=("sharpe_minus_base", "median"),
                      pct_positive=("sharpe_minus_base",
                                    lambda s: (s > 0).mean()),
                      mean_sharpe=("sharpe", "mean"),
                      mean_turnover=("turnover", "mean")))
    print("\n=== Summary across 27 robustness cells ===")
    print(summary.round(4).to_string())
    summary.to_csv(f"{outdir}/robustness_summary.csv")

    make_heatmaps(df, outdir)


if __name__ == "__main__":
    main()