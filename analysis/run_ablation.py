"""
run_ablation.py — HMVA performance attribution study.

Component ablation (fixed lookback=126): all HMVA variants in one backtest call.
Produces: attribution table, grouped bar chart, LW significance tests.

HMVA components ablated
-----------------------
  return_signal — BL-trend mu used for Sharpe bisection vs inverse-variance
  ewma          — exponential pseudo-returns (halflife=21) before NLS estimation
  K            — Kalman-filter weight smoother
  nls           — nonlinear shrinkage vs plain sample covariance
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.backtest as L
import utils.data as C

# ── constants ─────────────────────────────────────────────────────────────────
DATA_CSV         = "./data/stock_daily_returns.csv"
CONSTITUENTS_CSV = "./data/constiuents.csv"
PERMNO_LIST_TXT  = "./data/unique_ids.txt"
PERMNO_COL       = "PERMNO"
DATE_COL         = "DlyCalDt"
PRICE_COL        = "DlyClose"
RET_COL          = "DlyRet"

START_DATE    = "2000-01-01"
END_DATE      = "2025-01-01"
TOP_K         = 100
REBALANCE     = 21
COST_BPS      = 0.0
N_BOOT        = 2000
BASE_LOOKBACK = 126


# ── strategy builders ─────────────────────────────────────────────────────────

def build_ablation_strategies() -> Dict[str, tuple]:
    """
    All HMVA variants. Grouped as:
      – One-at-a-time removal from full HMVA
      – Covariance quality ablation
    """
    strats: Dict[str, tuple] = {}

    # HMVA — full: vb_tree, Sharpe bisect, EWMA+NLS, K
    strats["HMVA"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="sharpe", ewma_halflife=21, kf_tp=True,
    )

    # ── One-at-a-time removals from full HMVA ────────────────────────────────
    strats["HMVA-noK"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="sharpe", ewma_halflife=21, kf_tp=False,
    )

    strats["HMVA-noE"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="sharpe", ewma_halflife=None, kf_tp=True,
    )

    strats["HMVA-noK-noE"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="sharpe", ewma_halflife=None, kf_tp=False,
    )

    # Remove return signal (= HMVA-mv): vb_tree, vol bisect, EWMA+NLS, KF
    strats["HMVA-mv"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="vol", ewma_halflife=21, kf_tp=True,
    )

    strats["HMVA-mv-noK"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="vol", ewma_halflife=21, kf_tp=False,
    )

    strats["HMVA-mv-noE"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="vol", ewma_halflife=None, kf_tp=True,
    )

    # Bare vb_tree + NLS, no extras
    strats["HMVA-mv-noK-noE"] = L.vol_bl_strategy(
        L.cov_nonlinear_shrink,
        bisect_method="vol", ewma_halflife=None, kf_tp=False,
    )

    return strats


# ── component description table ───────────────────────────────────────────────

COMPONENT_FLAGS = {
    # name               : (return_signal, ewma, kf, nls)
    "HMVA":                (True,  True,  True,  True ),
    "HMVA-noK":           (True,  True,  False, True ),
    "HMVA-noE":            (True,  False, True,  True ),
    "HMVA-noK-noE":       (True,  False, False, True ),
    "HMVA-mv":             (False, True,  True,  True ),
    "HMVA-mv-noK":        (False, True,  False, True ),
    "HMVA-mv-noE":         (False, False, True,  True ),
    "HMVA-mv-noK-noE":    (False, False, False, True ),
}


def _flag(b: bool) -> str:
    return "✓" if b else "✗"


# ── backtest helpers ──────────────────────────────────────────────────────────

def _run_pit(
    returns_wide: pd.DataFrame,
    universe_fn,
    strategies: Dict[str, tuple],
    lookback: int,
    rf: pd.Series,
) -> tuple[pd.DataFrame, Dict, pd.DataFrame]:
    daily, weights = L.backtest_pit(
        returns_wide, universe_fn, strategies,
        lookback=lookback, rebalance=REBALANCE,
        cost_bps=COST_BPS, rf_daily=rf,
        min_history_days=lookback,
        verbose=True,
    )
    metrics = L.compute_metrics_pit(daily, weights)
    return daily, weights, metrics


def run_lw_tests(daily: pd.DataFrame, reference: str = "HMVA") -> pd.DataFrame:
    """LW block-bootstrap Sharpe tests: each strategy vs reference."""
    base = daily[reference]
    rows = []
    for s in daily.columns:
        if s == reference:
            continue
        diff, p, ci_lo, ci_hi = L.lw_sharpe_test(
            base, daily[s], n_boot=N_BOOT, block=REBALANCE,
        )
        rows.append({"strategy": s, "sharpe_diff": diff,
                     "lw_p": p, "ci_lo_95": ci_lo, "ci_hi_95": ci_hi})
    df = pd.DataFrame(rows).set_index("strategy")
    if not df.empty:
        bh = L.benjamini_hochberg(df["lw_p"].to_dict())
        df["lw_p_adj"] = bh["pval_adj"].values
        df["reject"]   = bh["reject"].values
    return df


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_ablation_table(metrics: pd.DataFrame, tests: pd.DataFrame, outdir: str) -> None:
    """Grouped bar chart: Sharpe for every ablation config, coloured by group."""
    order = [k for k in [
        "HMVA", "HMVA-noK", "HMVA-noE", "HMVA-noK-noE",
        "HMVA-mv", "HMVA-mv-noK", "HMVA-mv-noE", "HMVA-mv-noK-noE",
    ] if k in metrics.index]

    sharpes     = [metrics.loc[k, "Sharpe"] for k in order]
    hmva_sharpe = metrics.loc["HMVA", "Sharpe"] if "HMVA" in metrics.index else 0.0

    def _color(k: str) -> str:
        return "#E65100" if "-mv" in k else "#1976D2"

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.75), 4.5))
    bars = ax.bar(range(len(order)), sharpes,
                  color=[_color(k) for k in order], edgecolor="white", width=0.65)
    ax.axhline(hmva_sharpe, color="#1976D2", linewidth=1.4, linestyle="--")

    for bar, v, k in zip(bars, sharpes, order):
        reject = tests.loc[k, "reject"] if k in tests.index and "reject" in tests.columns else None
        sig = " *" if reject else ""
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}{sig}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("HMVA ablation — Sharpe by configuration")
    ax.set_ylim(0, max(sharpes) * 1.22)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1976D2", label="Sharpe bisect (return signal)"),
        Patch(color="#E65100", label="Vol bisect (no return signal)"),
    ], loc="lower right", fontsize=8)
    ax.grid(axis="y", linewidth=0.4, linestyle="--")
    plt.tight_layout()
    path = os.path.join(outdir, "ablation_sharpe_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── summary printing ──────────────────────────────────────────────────────────

def print_attribution_table(metrics: pd.DataFrame, tests: pd.DataFrame) -> None:
    hmva_sharpe = metrics.loc["HMVA", "Sharpe"] if "HMVA" in metrics.index else float("nan")

    col_w = 22
    header = (
        f"{'Config':<{col_w}} "
        f"{'Signal':^7} {'EWMA':^7} {'K':^7} {'NLS':^7}  "
        f"{'Sharpe':>8} {'ΔSharpe':>9} {'LW-p':>8} {'Reject':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    order = [
        "HMVA", "HMVA-noK", "HMVA-noE", "HMVA-noK-noE",
        None,
        "HMVA-mv", "HMVA-mv-noK", "HMVA-mv-noE", "HMVA-mv-noK-noE",
    ]

    for k in order:
        if k is None:
            print("-" * len(header))
            continue
        if k not in metrics.index:
            continue
        sh    = metrics.loc[k, "Sharpe"]
        delta = sh - hmva_sharpe
        flags = COMPONENT_FLAGS.get(k, (False,) * 5)
        f_str = "  ".join(_flag(f) for f in flags)

        p_str = rej_str = ""
        if k in tests.index:
            p_str   = f"{tests.loc[k, 'lw_p_adj']:.3f}" if "lw_p_adj" in tests.columns else ""
            rej_str = _flag(tests.loc[k, "reject"])      if "reject"   in tests.columns else ""

        print(f"{k:<{col_w}} {f_str}  {sh:>8.4f} {delta:>+9.4f} {p_str:>8} {rej_str:>7}")

    print("=" * len(header))
    print("  Signal=BL return mu (Sharpe bisect), EWMA=exp. pseudo-returns (hl=21d),")
    print("  K=Kalman smoother, NLS=nonlinear shrinkage")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description="HMVA performance attribution ablation study")
    p.add_argument("--data",         default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start",        default=START_DATE)
    p.add_argument("--end",          default=END_DATE)
    p.add_argument("--top-k",        type=int, default=TOP_K)
    p.add_argument("--out",          default="results/ablation")
    args = p.parse_args(argv)

    np.random.seed(42)
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    permno_subset = None
    if os.path.exists(PERMNO_LIST_TXT):
        with open(PERMNO_LIST_TXT) as f:
            permno_subset = [int(x.strip()) for x in f if x.strip()]
        print(f"[main] restricting to {len(permno_subset)} PERMNOs")

    print("[main] loading returns …")
    returns_wide = C.load_crsp_returns(
        args.data, permno_subset=permno_subset,
        start_date=args.start, end_date=args.end,
        price_col=PRICE_COL, date_col=DATE_COL,
        permno_col=PERMNO_COL, ret_col=RET_COL,
    )

    print("[main] building universe …")
    universe_fn = C.make_universe_fn(
        args.constituents,
        market_cap_csv=args.data if args.top_k else None,
        top_k=args.top_k,
    )
    rf = pd.Series(0.0, index=returns_wide.index)

    print(f"\n{'='*64}")
    print(f" Component ablation  (lookback={BASE_LOOKBACK})")
    print(f"{'='*64}")

    ablation_strats = build_ablation_strategies()
    daily, _, metrics = _run_pit(
        returns_wide, universe_fn, ablation_strats, BASE_LOOKBACK, rf,
    )

    print("\n=== LW Sharpe tests (each config vs HMVA) ===")
    tests = run_lw_tests(daily, reference="HMVA")

    print_attribution_table(metrics, tests)

    metrics.to_csv(os.path.join(outdir, "ablation_metrics.csv"))
    tests.to_csv(os.path.join(outdir, "ablation_tests.csv"))
    daily.to_csv(os.path.join(outdir, "ablation_daily_returns.csv"))
    print(f"\nSaved CSVs to {outdir}/")

    print("\n[plots] …")
    plot_ablation_table(metrics, tests, outdir)

    print(f"\n[done] results at {outdir}/")


if __name__ == "__main__":
    main()
