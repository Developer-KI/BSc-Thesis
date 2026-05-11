"""
run_param_sweep.py  –  HMVA hyperparameter sensitivity via block-bootstrap Monte Carlo.

For each hyperparameter, fixes all others at their thesis defaults and sweeps one
value at a time (one-at-a-time / OAT design).  For each configuration:

  1. Runs a full point-in-time backtest on the CRSP S&P 500 universe.
  2. Computes the annualised Sharpe ratio vs EW baseline.
  3. Uses a circular block-bootstrap (B=500 by default) to estimate the 95%
     confidence interval of the Sharpe ratio — the Monte Carlo component that
     quantifies sampling uncertainty for each parameter value.

The result is one sensitivity plot per parameter (Sharpe ± 95% CI vs value),
which provides empirical motivation for the chosen thesis defaults.

Parameters swept and their grids:
  ewma_halflife    [5, 10, 14, 21, 30, 42, 63]       half-life of EWMA (trading days)
  weight_reg       [0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]   L2 blend (ρ_L2)
  turnover_penalty [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]   L1 soft-threshold (τ_L1)
  lam_scale        [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]              slope on CV_vol for λ_eff
  lam_corr         [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]   slope on avg_corr for λ_eff

Note: delta (BL risk-aversion) is intentionally excluded.  With adaptive_tau=True
(tau = 1/T ≈ 0.004 at lookback=252), the BL posterior is dominated by the momentum
views and delta's effect on the equilibrium prior is negligible at typical lookbacks.

Outputs in results/param_sweep/:
  sensitivity.csv              full results (param, value, sharpe, ci_low, ci_high, ...)
  sensitivity_{param}.png      sensitivity curve per parameter
  sensitivity_overview.png     all parameters in one figure (grid of subplots)
  defaults_summary.csv         performance at the thesis default configuration

CLI flags:
  --n-boot N      bootstrap replications per config  (default 500; use 100 for speed)
  --lookback T    estimation window in trading days   (default 252)
  --start / --end sample period
  --params p1,p2  restrict sweep to these parameters only
  --quick         alias for --n-boot 100 --start 2010-01-01
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.strategy as L
import utils.data as C


# ── data ──────────────────────────────────────────────────────────────────────
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
REBALANCE  = 21

# ── thesis defaults ────────────────────────────────────────────────────────────
DEFAULTS = dict(
    ewma_halflife    = 21,
    weight_reg       = 0.10,
    turnover_penalty = 0.05,
    lam_scale        = 0.5,
    lam_corr         = 0.2,
)

# ── parameter grids ────────────────────────────────────────────────────────────
PARAM_GRIDS: Dict[str, List] = {
    "ewma_halflife":    [5, 10, 14, 21, 30, 42, 63],
    "weight_reg":       [0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30],
    "turnover_penalty": [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "lam_scale":        [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "lam_corr":         [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.60],
}

PARAM_LABELS = {
    "ewma_halflife":    "EWMA half-life $h$ (days)",
    "weight_reg":       "L2 blend $\\rho_{L2}$",
    "turnover_penalty": "L1 soft-threshold $\\tau_{L1}$",
    "lam_scale":        "Regime scale $\\lambda_{\\mathrm{scale}}$",
    "lam_corr":         "Regime corr-slope $\\lambda_{\\mathrm{corr}}$",
}

N_BOOT_DEFAULT = 500
BLOCK          = 21   # block length for circular bootstrap (≈ 1 month)


# ── bootstrap helpers ──────────────────────────────────────────────────────────

def _sharpe(r: np.ndarray) -> float:
    """Annualised Sharpe from a daily return array."""
    std = r.std(ddof=1)
    if std < 1e-12:
        return np.nan
    return float(r.mean() / std * np.sqrt(252))


def _bootstrap_sharpe_ci(r: pd.Series,
                          n_boot: int = N_BOOT_DEFAULT,
                          block: int = BLOCK,
                          alpha: float = 0.05,
                          seed: int = 42) -> Tuple[float, float, float]:
    """
    Circular block bootstrap for the Sharpe ratio.

    Returns (sharpe_observed, ci_low, ci_high).
    Block bootstrap preserves autocorrelation structure of daily returns.
    """
    arr = r.dropna().values
    n   = len(arr)
    if n < 2 * block:
        s = _sharpe(arr)
        return s, np.nan, np.nan

    rng      = np.random.default_rng(seed)
    obs_sr   = _sharpe(arr)
    n_blocks = int(np.ceil(n / block))
    boot_sr  = np.empty(n_boot)

    for b in range(n_boot):
        # circular: start indices drawn uniformly from [0, n)
        starts = rng.integers(0, n, n_blocks)
        idx    = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        boot_sr[b] = _sharpe(arr[idx])

    ci_lo = float(np.nanpercentile(boot_sr, 100 * alpha / 2))
    ci_hi = float(np.nanpercentile(boot_sr, 100 * (1 - alpha / 2)))
    return obs_sr, ci_lo, ci_hi


# ── backtest runner ────────────────────────────────────────────────────────────

def _run_config(
    returns_wide: pd.DataFrame,
    universe_fn,
    lookback: int,
    n_boot: int,
    **hmva_kwargs,
) -> Dict:
    """
    Run HMVA + EW backtest for one hyperparameter configuration and return
    Sharpe summary dict including bootstrap CI.
    """
    cov_fn, alloc_fn = L.vol_hrp_bl_strategy(
        L.cov_nonlinear_shrink,
        regime_lam=True,
        **hmva_kwargs,
    )
    strategies = {
        "HMVA": (cov_fn, alloc_fn),
        "EW":   (L.cov_sample, L.equal_weights),
    }
    rf = pd.Series(0.0, index=returns_wide.index)

    daily, _ = L.backtest_pit(
        returns_wide, universe_fn, strategies,
        lookback=lookback, rebalance=REBALANCE,
        cost_bps=0.0, rf_daily=rf,
        min_history_days=lookback,
        verbose=False,
    )

    r_hmva = daily["HMVA"].dropna()
    r_ew   = daily["EW"].dropna()

    sharpe, ci_lo, ci_hi = _bootstrap_sharpe_ci(r_hmva, n_boot=n_boot)
    ew_sr  = _sharpe(r_ew.values)

    ann_ret = float((1 + r_hmva).prod() ** (252.0 / len(r_hmva)) - 1) if len(r_hmva) else np.nan
    ann_vol = float(r_hmva.std() * np.sqrt(252)) if len(r_hmva) else np.nan
    cum     = (1 + r_hmva).cumprod()
    max_dd  = float((cum / cum.cummax() - 1).min()) if len(r_hmva) else np.nan

    return dict(
        sharpe=sharpe, ci_lo=ci_lo, ci_hi=ci_hi,
        delta_sharpe=sharpe - ew_sr,
        ew_sharpe=ew_sr,
        ann_ret=ann_ret, ann_vol=ann_vol, max_dd=max_dd,
        n_days=len(r_hmva),
    )


# ── sensitivity plots ──────────────────────────────────────────────────────────

def _sensitivity_plot_one(df_param: pd.DataFrame,
                           param: str,
                           default_val,
                           ax: plt.Axes,
                           title: bool = True) -> None:
    """Plot one sensitivity curve (Sharpe ± CI vs parameter value) on `ax`."""
    x     = df_param["value"].values.astype(float)
    y     = df_param["sharpe"].values.astype(float)
    ci_lo = df_param["ci_lo"].values.astype(float)
    ci_hi = df_param["ci_hi"].values.astype(float)

    ax.fill_between(x, ci_lo, ci_hi, alpha=0.25, color="#3a7abf", label="95% bootstrap CI")
    ax.plot(x, y, color="#3a7abf", lw=2, marker="o", ms=5, zorder=3, label="Sharpe")

    # mark default value
    def_y = df_param.loc[df_param["value"] == default_val, "sharpe"]
    if not def_y.empty:
        ax.axvline(default_val, color="#c0392b", ls="--", lw=1.2, alpha=0.8,
                   label=f"Default = {default_val}")
        ax.scatter([default_val], [float(def_y.values[0])],
                   color="#c0392b", s=60, zorder=5)

    # EW baseline
    ew_vals = df_param["ew_sharpe"].dropna()
    if len(ew_vals):
        ax.axhline(float(ew_vals.mean()), color="gray", ls=":", lw=1.0, label="EW Sharpe")

    ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=9)
    ax.set_ylabel("Sharpe ratio" if ax.get_ylabel() == "" else ax.get_ylabel(), fontsize=9)
    if title:
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.yaxis.grid(True, lw=0.4)


def plot_sensitivity_one(df_param: pd.DataFrame,
                          param: str,
                          default_val,
                          outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    _sensitivity_plot_one(df_param, param, default_val, ax, title=False)
    ax.set_title(f"Sensitivity: {PARAM_LABELS.get(param, param)}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{outdir}/sensitivity_{param}.png", dpi=150)
    plt.close()


def plot_overview(results: pd.DataFrame, outdir: str) -> None:
    """All sensitivity curves in one figure (grid of subplots)."""
    params  = list(results["param"].unique())
    n       = len(params)
    ncols   = 3
    nrows   = int(np.ceil(n / ncols))
    # squeeze=False guarantees axes is always 2D, avoiding edge-case handling
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              squeeze=False)
    fig.suptitle("HMVA hyperparameter sensitivity\n(Sharpe ± 95% bootstrap CI, CRSP 2002–2024)",
                 fontsize=12)

    ax_list = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax, param in zip(ax_list, params):
        sub = results[results["param"] == param].copy().sort_values("value")
        _sensitivity_plot_one(sub, param, DEFAULTS.get(param), ax, title=True)

    for ax in ax_list[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/sensitivity_overview.png", dpi=150)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="HMVA hyperparameter sensitivity sweep")
    p.add_argument("--data",         default=DATA_CSV)
    p.add_argument("--constituents", default=CONSTITUENTS_CSV)
    p.add_argument("--start",        default=START_DATE)
    p.add_argument("--end",          default=END_DATE)
    p.add_argument("--top-k",        type=int,   default=TOP_K)
    p.add_argument("--lookback",     type=int,   default=LOOKBACK)
    p.add_argument("--n-boot",       type=int,   default=N_BOOT_DEFAULT,
                   help="Bootstrap replications per config (default 500)")
    p.add_argument("--params",       default=None,
                   help="Comma-separated subset of parameters to sweep, e.g. delta,tau")
    p.add_argument("--quick",        action="store_true",
                   help="Shorthand for --n-boot 100 --start 2010-01-01")
    p.add_argument("--out",          default="results/param_robustness")
    args = p.parse_args(argv)

    if args.quick:
        args.n_boot  = 100
        args.start   = max(args.start, "2010-01-01")
        print("[sweep] --quick mode: n_boot=100, start=2010-01-01")

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(42)

    params_to_sweep = list(PARAM_GRIDS.keys())
    if args.params:
        params_to_sweep = [p.strip() for p in args.params.split(",")
                           if p.strip() in PARAM_GRIDS]
        if not params_to_sweep:
            raise ValueError(f"No valid params in --params. "
                             f"Available: {list(PARAM_GRIDS.keys())}")

    # ── load data once ────────────────────────────────────────────────────────
    permno_subset = None
    if os.path.exists(PERMNO_LIST_TXT):
        with open(PERMNO_LIST_TXT) as f:
            permno_subset = [int(line.strip()) for line in f if line.strip()]
        print(f"[sweep] restricting to {len(permno_subset)} PERMNOs")

    print("[sweep] loading CRSP returns (this may take a minute) ...")
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
    print(f"[sweep] data loaded: {returns_wide.shape}, period "
          f"{returns_wide.index[0].date()} → {returns_wide.index[-1].date()}")

    # ── run default configuration first ───────────────────────────────────────
    print(f"\n[sweep] running default configuration ...")
    t0 = time.time()
    default_result = _run_config(
        returns_wide, universe_fn, args.lookback, args.n_boot,
        **DEFAULTS,
    )
    default_result.update(DEFAULTS)
    pd.DataFrame([default_result]).to_csv(f"{args.out}/defaults_summary.csv", index=False)
    print(f"[sweep] default Sharpe = {default_result['sharpe']:.3f} "
          f"[{default_result['ci_lo']:.3f}, {default_result['ci_hi']:.3f}]  "
          f"({time.time()-t0:.0f}s)")

    # ── OAT sweep ─────────────────────────────────────────────────────────────
    total_configs = sum(len(PARAM_GRIDS[p]) for p in params_to_sweep)
    print(f"\n[sweep] sweeping {len(params_to_sweep)} parameters × grids "
          f"= {total_configs} total configurations")
    print(f"[sweep] bootstrap B={args.n_boot}, block={BLOCK} days\n")

    all_rows: List[Dict] = []
    config_num = 0

    for param in params_to_sweep:
        grid = PARAM_GRIDS[param]
        print(f"── {param:20s}  ({len(grid)} values) ──")
        for val in grid:
            config_num += 1
            kwargs = {**DEFAULTS, param: val}
            t_start = time.time()
            try:
                result = _run_config(
                    returns_wide, universe_fn, args.lookback, args.n_boot,
                    **kwargs,
                )
            except Exception as e:
                print(f"  [{config_num}/{total_configs}] {param}={val} ERROR: {e}")
                result = dict(sharpe=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                              delta_sharpe=np.nan, ew_sharpe=np.nan,
                              ann_ret=np.nan, ann_vol=np.nan,
                              max_dd=np.nan, n_days=0)

            elapsed = time.time() - t_start
            row = {"param": param, "value": val, **result}
            all_rows.append(row)

            is_default = (val == DEFAULTS.get(param))
            tag = " ◄ default" if is_default else ""
            print(f"  [{config_num:3d}/{total_configs}] {param}={val:<8}  "
                  f"Sharpe={result['sharpe']:.3f}  "
                  f"CI=[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]  "
                  f"Δvs.EW={result['delta_sharpe']:.3f}  "
                  f"({elapsed:.0f}s){tag}")

            # save incrementally so partial runs are recoverable
            results_so_far = pd.DataFrame(all_rows)
            results_so_far.to_csv(f"{args.out}/sensitivity.csv", index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(f"{args.out}/sensitivity.csv", index=False)
    print(f"\n[sweep] results saved to {args.out}/sensitivity.csv")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("[sweep] generating sensitivity plots ...")
    for param in params_to_sweep:
        sub = results[results["param"] == param].sort_values("value")
        if sub["sharpe"].notna().any():
            plot_sensitivity_one(sub, param, DEFAULTS.get(param), args.out)

    plot_overview(results, args.out)

    print(f"\n[sweep] done → {args.out}/")
    print(f"[sweep] overview plot: {args.out}/sensitivity_overview.png")


if __name__ == "__main__":
    main()
