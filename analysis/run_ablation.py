"""
run_ablation.py  –  Stage-by-stage HMVA ablation study.

Incrementally adds each HMVA component over a standard HRP baseline and
measures the marginal Sharpe contribution of each stage:

  Stage 0   HRP           sample cov | correlation tree | variance bisection
  +S1+S2    +NLS+EWMA     NLS+EWMA  | correlation tree | variance bisection
  +S4       +VBTree        NLS+EWMA  | vol-balanced tree (fixed λ) | equal split
  +S3+S5    +BL+Sharpe     NLS+EWMA  | vol-balanced tree (fixed λ) | Sharpe bisection
  +S6       HMVA           NLS+EWMA  | vol-balanced tree (fixed λ) | Sharpe bisection | L2+L1

Outputs in results/ablation/:
  metrics.csv          full performance metrics per stage
  sharpe_delta.csv     cumulative and marginal Sharpe gains
  equity_curves.png
  sharpe_bars.png      Sharpe bars with marginal Δ annotations
  waterfall.png        waterfall chart of marginal Sharpe contributions
  drawdowns.png        drawdown curves per stage
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.strategy as L
import utils.data as C


# ── defaults ─────────────────────────────────────────────────────────────────
DATA_CSV         = "./data/stock_daily_returns.csv"
CONSTITUENTS_CSV = "./data/constiuents.csv"
PERMNO_LIST_TXT  = "./data/unique_ids.txt"
PERMNO_COL = "PERMNO"
DATE_COL   = "DlyCalDt"
PRICE_COL  = "DlyClose"
RET_COL    = "DlyRet"

START_DATE = "2000-01-01"
END_DATE   = "2025-01-01"
TOP_K      = 100
LOOKBACK   = 504
DEFAULT_HISTORY = 504
REBALANCE  = 21
COST_BPS   = 0.0

# HMVA baseline hyperparameters
EWMA_HL       = 21
LAM_COV       = 0.25

ORDERED_STAGES = ["HRP", "+NLS+EWMA", "+VBTree", "+BL+Sharpe", "HMVA"]
STAGE_LABELS   = [
    "HRP\n(baseline)",
    "+NLS+EWMA\n(Stages 1+2)",
    "+VBTree\n(Stage 4)",
    "+BL+Sharpe\n(Stages 3+5)",
    "+KF Smoother\n(Stage 6)",
]
COLORS = ["#6c757d", "#3a7abf", "#f0a500", "#e05c2a", "#1a7a4a"]


# ── strategy builders ─────────────────────────────────────────────────────────

def _make_vb_fixed_alloc():
    """Vol-balanced top-down tree with fixed lam_cov and vol bisection (mu = 0)."""
    def alloc_fn(cov: np.ndarray) -> np.ndarray:
        N = cov.shape[0]
        return L.vol_hrp_bl_weights(
            cov, np.zeros(N), rf=0.0,
            lam_cov=LAM_COV, bisect_method="vol",
        )
    return alloc_fn


def make_ablation_strategies() -> L.StrategyMap:
    """
    Return a dict of (cov_fn, alloc_fn) pairs, one per ablation step.
    Each entry adds exactly one HMVA component over the previous.
    """
    # BL returns + Sharpe bisection
    s_bl_cov, s_bl_alloc = L.vol_hrp_bl_strategy(
        L.cov_nonlinear_shrink,
        ewma_halflife=EWMA_HL,
        lam_cov=LAM_COV,
        kf_tp=False
    )
    # Add KF weight smoother
    hmva_cov, hmva_alloc = L.vol_hrp_bl_strategy(
        L.cov_nonlinear_shrink,
        ewma_halflife=EWMA_HL,
        lam_cov=LAM_COV,
        kf_tp=True,
    )

    return {
        "HRP":        (L.cov_sample,  lambda c: L.hrp_weights(c)),
        "+NLS+EWMA":  (L.cov_ewa_nls, lambda c: L.hrp_weights(c)),
        "+VBTree":    (L.cov_ewa_nls, _make_vb_fixed_alloc()),
        "+BL+Sharpe": (s_bl_cov,      s_bl_alloc),
        "HMVA":       (hmva_cov,      hmva_alloc),
        "EW":         (L.cov_sample,  L.equal_weights),
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def _crisis_spans():
    """Return list of (start, end) for shading."""
    return [
        ("2002-01-01", "2002-10-31"),
        ("2007-10-01", "2009-03-31"),
        ("2020-01-15", "2020-04-30"),
        ("2022-01-01", "2022-12-31"),
    ]


def plot_equity(daily: pd.DataFrame, outdir: str) -> None:
    cum = (1 + daily[ORDERED_STAGES + ["EW"]]).cumprod()
    fig, ax = plt.subplots(figsize=(13, 6))
    for start, end in _crisis_spans():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="#f0e0e0", alpha=0.6, zorder=0)
    ax.axhline(1, color="black", lw=0.4, ls="--")
    for i, col in enumerate(ORDERED_STAGES):
        lw = 2.0 if col == "HMVA" else 1.0
        ax.plot(cum.index, cum[col], lw=lw, color=COLORS[i],
                label=STAGE_LABELS[i].replace("\n", " "))
    ax.plot(cum.index, cum["EW"], color="gray", lw=1.0, ls="--", label="EW (1/N)")
    ax.set_title(f"Ablation: cumulative return by HMVA stage  (CRSP 2002–2024, T={LOOKBACK})")
    ax.set_ylabel("Growth of $1")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{outdir}/equity_curves.png", dpi=150)
    plt.close()


def plot_drawdowns(daily: pd.DataFrame, outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for start, end in _crisis_spans():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="#f0e0e0", alpha=0.6, zorder=0)
    for i, col in enumerate(ORDERED_STAGES):
        cum = (1 + daily[col]).cumprod()
        dd  = cum / cum.cummax() - 1
        lw  = 2.0 if col == "HMVA" else 1.0
        ax.plot(dd.index, dd * 100, lw=lw, color=COLORS[i],
                label=STAGE_LABELS[i].replace("\n", " "))
    ax.plot(daily.index,
            ((1 + daily["EW"]).cumprod() / (1 + daily["EW"]).cumprod().cummax() - 1) * 100,
            color="gray", lw=1.0, ls="--", label="EW (1/N)")
    ax.set_title("Drawdown by HMVA stage")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{outdir}/drawdowns.png", dpi=150)
    plt.close()


def plot_sharpe_bars(metrics: pd.DataFrame, outdir: str) -> None:
    sharpes = [float(metrics.loc[s, "Sharpe"]) for s in ORDERED_STAGES]
    deltas  = [np.nan] + [sharpes[i] - sharpes[i - 1] for i in range(1, len(sharpes))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(ORDERED_STAGES)), sharpes,
                  color=COLORS, edgecolor="black", lw=0.7, zorder=3)

    for bar, d in zip(bars, deltas):
        if not np.isnan(d):
            sign = "+" if d >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{sign}{d:.3f}", ha="center", va="bottom", fontsize=9)

    ew_sr = float(metrics.loc["EW", "Sharpe"])
    ax.axhline(ew_sr, color="gray", ls="--", lw=1.2, label=f"EW = {ew_sr:.3f}")
    ax.set_xticks(range(len(ORDERED_STAGES)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=8.5)
    ax.set_ylabel("Sharpe ratio (annualised)")
    ax.set_title(f"Sharpe ratio by HMVA stage with marginal ΔSharpe  (T={LOOKBACK})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(sharpes) * 1.18)
    ax.yaxis.grid(True, lw=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{outdir}/sharpe_bars.png", dpi=150)
    plt.close()


def plot_waterfall(sharpe_delta: pd.DataFrame, outdir: str) -> None:
    labels   = list(sharpe_delta["stage"])
    marginal = list(sharpe_delta["marginal_delta"])
    running  = 0.0
    bottoms, heights, cols = [], [], []
    for i, m in enumerate(marginal):
        bottoms.append(0.0 if i == 0 else running)
        heights.append(m)
        cols.append(COLORS[0] if i == 0 else ("#1a7a4a" if m >= 0 else "#c0392b"))
        running += m

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (b, h, c) in enumerate(zip(bottoms, heights, cols)):
        ax.bar(i, h, bottom=b, color=c, edgecolor="black", lw=0.7, zorder=3)
    ax.axhline(0, color="black", lw=0.5)

    for i, (b, h) in enumerate(zip(bottoms, heights)):
        sign = "+" if h >= 0 else ""
        ax.text(i, b + h + 0.003,
                f"{sign}{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=8.5)
    ax.set_ylabel("Sharpe contribution")
    ax.set_title(f"Marginal Sharpe contribution per HMVA stage  (T={LOOKBACK})")
    ax.yaxis.grid(True, lw=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{outdir}/waterfall.png", dpi=150)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="HMVA stage-by-stage ablation")
    p.add_argument("--data",         default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start",        default=START_DATE)
    p.add_argument("--end",          default=END_DATE)
    p.add_argument("--top-k",        type=int,   default=TOP_K)
    p.add_argument("--lookback",     type=int,   default=LOOKBACK)
    p.add_argument("--cost-bps",     type=float, default=COST_BPS)
    p.add_argument("--out",          default="results/ablation")
    args = p.parse_args(argv)

    np.random.seed(42)
    os.makedirs(args.out, exist_ok=True)

    permno_subset = None
    if os.path.exists(PERMNO_LIST_TXT):
        with open(PERMNO_LIST_TXT) as f:
            permno_subset = [int(line.strip()) for line in f if line.strip()]
        print(f"[ablation] restricting to {len(permno_subset)} PERMNOs")

    print("[ablation] loading CRSP returns ...")
    returns_wide = C.load_crsp_returns(
        args.data, permno_subset=permno_subset,
        start_date=args.start, end_date=args.end,
        price_col=PRICE_COL, date_col=DATE_COL,
        permno_col=PERMNO_COL, ret_col=RET_COL,
    )
    universe_fn = C.make_universe_fn(
        args.constituents,
        market_cap_csv=args.data if args.top_k else None,
        top_k=args.top_k,
    )

    strategies = make_ablation_strategies()
    rf = pd.Series(0.0, index=returns_wide.index)

    print(f"\n[ablation] running {len(strategies)} strategies, lookback={args.lookback} ...")
    daily, weights = L.backtest_pit(
        returns_wide, universe_fn, strategies,
        lookback=args.lookback, rebalance=REBALANCE,
        cost_bps=args.cost_bps, rf_daily=rf,
        min_history_days=DEFAULT_HISTORY,
    )
    daily.to_csv(f"{args.out}/daily_returns.csv")

    metrics = L.compute_metrics_pit(daily, weights)
    print("\n=== Ablation performance table ===")
    cols_show = ["AnnReturn", "AnnVol", "Sharpe", "MaxDD", "Calmar", "Sortino", "Turnover"]
    print(metrics[cols_show].round(4).to_string())
    metrics.to_csv(f"{args.out}/metrics.csv")

    # marginal Sharpe decomposition
    rows = []
    prev_sr = 0.0
    hrp_sr  = float(metrics.loc["HRP", "Sharpe"])
    for name, label in zip(ORDERED_STAGES, STAGE_LABELS):
        sr     = float(metrics.loc[name, "Sharpe"])
        margin = sr - prev_sr
        rows.append({
            "stage":            name,
            "label":            label.replace("\n", " "),
            "sharpe":           sr,
            "marginal_delta":   margin,
            "cumulative_delta": sr - hrp_sr,
        })
        prev_sr = sr
    sharpe_delta = pd.DataFrame(rows)
    sharpe_delta.to_csv(f"{args.out}/sharpe_delta.csv", index=False)

    print("\n=== Marginal Sharpe contributions ===")
    print(sharpe_delta[["stage", "sharpe", "marginal_delta", "cumulative_delta"]]
          .round(4).to_string(index=False))

    plot_equity(daily, args.out)
    plot_drawdowns(daily, args.out)
    plot_sharpe_bars(metrics, args.out)
    plot_waterfall(sharpe_delta, args.out)

    print(f"\n[ablation] done → {args.out}/")


if __name__ == "__main__":
    main()
