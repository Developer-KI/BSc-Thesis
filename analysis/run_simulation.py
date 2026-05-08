"""
==============================================================================
run_simulation.py - Comprehensive two-regime simulation over N/T scenarios
==============================================================================

Sweeps over N/T ratios from 0.5 to 2.0 (p > n, p ≈ n, p < n) for two
theoretically distinct data-generating processes:

  - factor_sparse : low-rank common factors + banded sparse residual.
                    POET designed to excel here; NLS competitive; LW weakest.
  - dispersed_eigs : dense covariance with power-law eigenvalue decay.
                    NLS should dominate LW because it shrinks each eigenvalue
                    individually; POET has no structure to exploit.

For each (regime, N/T) cell we simulate n_reps = 50 draws and compute:
  - Min-variance portfolio variance under the true covariance Σ_true.
  - Frobenius loss (reported but not used for inference).
  - Relative improvement over Sample and over Ledoit–Wolf linear shrinkage.

Paired Wilcoxon tests (one-sided) ask:
  Family 1: NLS and POET better than Sample?  (always true)
  Family 2: NLS and POET better than LW?      (the more interesting test)

Outputs are stored in results/simulation/ and include heatmaps of relative
improvement over LW, summary tables, and detailed test results.

The simulation replicates the empirical design of run_robustness.py but in
a clean synthetic environment with known ground truth.
==============================================================================
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

import analysis.hrp_lib as L


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base number of time periods (252 trading days ≈ 1 year, 504 ≈ 2 years)
DEFAULT_N = 100

# T/N ratios to sweep 
DEFAULT_T_N_RATIOS = "0.63, 1.26, 2.52, 5.04"

# Data-generating regimes
REGIMES = ["dispersed_eigs", "factor_sparse"]

# Number of replications per cell
N_REPS = 50

# -----------------------------------------------------------------------------
# Helper: run one cell (regime, N/T) for all replications
# -----------------------------------------------------------------------------

def run_cell(regime: str,
             n_ratio: float,
             N: int,
             n_reps: int,
             base_seed: int = 0) -> pd.DataFrame:
    """
    Run n_reps simulations for a given regime and N/T ratio.

    Returns a DataFrame with columns:
        rep, estimator, minvar_true_var, frobenius,
        mv_relative_to_sample, mv_relative_to_lw
    """
    T = int(round(n_ratio * N))
    if N < 2:
        raise ValueError(f"N must be at least 2, got {N} for ratio {n_ratio}")

    estimators = {
        "Sample": L.cov_sample,
        "LW": L.cov_linear_shrink,
        "NLS": L.cov_nonlinear_shrink,
        "POET": L.cov_poet_cv,
    }

    rows = []
    for rep in range(n_reps):
        seed = base_seed + rep * 17 + abs(hash(regime)) % 10000
        X, Sigma_true = L.simulate_returns(T, N, regime=regime, seed=seed)

        # Baseline losses for relative improvement
        base_sample = L.evaluate_sigma(estimators["Sample"](X), Sigma_true)
        base_lw = L.evaluate_sigma(estimators["LW"](X), Sigma_true)

        for name, fn in estimators.items():
            try:
                Sigma_hat = fn(X)
                ev = L.evaluate_sigma(Sigma_hat, Sigma_true)

                rel_sample = (ev["minvar_true_var"] - base_sample["minvar_true_var"]) \
                             / base_sample["minvar_true_var"]
                rel_lw = (ev["minvar_true_var"] - base_lw["minvar_true_var"]) \
                         / base_lw["minvar_true_var"]

                rows.append({
                    "rep": rep,
                    "estimator": name,
                    "minvar_true_var": ev["minvar_true_var"],
                    "frobenius": ev["frobenius"],
                    "mv_relative_to_sample": rel_sample,
                    "mv_relative_to_lw": rel_lw
                })
            except Exception as e:
                # If one estimator fails, record NaN (rare, but robust)
                rows.append({
                    "rep": rep,
                    "estimator": name,
                    "minvar_true_var": np.nan,
                    "frobenius": np.nan,
                    "mv_relative_to_sample": np.nan,
                    "mv_relative_to_lw": np.nan
                })
                print(f"  [warn] {regime} N={N} rep={rep} {name} failed: {e}")

    df = pd.DataFrame(rows)
    df["regime"] = regime
    df["N"] = N
    df["T"] = T
    df["N_T_ratio"] = n_ratio
    return df


# -----------------------------------------------------------------------------
# Statistical tests (paired Wilcoxon, one-sided)
# -----------------------------------------------------------------------------

def paired_wilcoxon_one_sided(df_cell: pd.DataFrame,
                              baseline: str,
                              challengers: List[str]) -> pd.DataFrame:
    """
    For a given cell (regime, N/T), perform one-sided Wilcoxon signed-rank test
    for each challenger vs baseline: H1: challenger has LOWER minvar_true_var.
    Returns DataFrame with challenger, statistic, p_value.
    """
    base_data = df_cell[df_cell["estimator"] == baseline].sort_values("rep")["minvar_true_var"].values
    results = []
    for chal in challengers:
        chal_data = df_cell[df_cell["estimator"] == chal].sort_values("rep")["minvar_true_var"].values
        # Drop pairs where either is NaN
        mask = ~(np.isnan(base_data) | np.isnan(chal_data))
        if mask.sum() < 2:
            stat, p = np.nan, np.nan
        else:
            stat, p = wilcoxon(chal_data[mask], base_data[mask], alternative="less")
        results.append({"challenger": chal, "statistic": stat, "p_value": p})
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Plotting: heatmaps of mean relative improvement over LW
# -----------------------------------------------------------------------------

def plot_relative_improvement_heatmaps(summary_df: pd.DataFrame,
                                       outdir: str) -> None:
    """
    For each estimator (except Sample and LW), create a heatmap of
    mean(mv_relative_to_lw) across regimes and N/T ratios.
    Negative values = better than LW.
    """
    os.makedirs(outdir, exist_ok=True)
    estimators = [e for e in ["NLS", "POET"] if e in summary_df["estimator"].unique()]

    for est in estimators:
        sub = summary_df[summary_df["estimator"] == est].copy()
        # Use the correct column name: 'mean_rel_lw' (from the aggregation)
        if "mean_rel_lw" not in sub.columns:
            print(f"Warning: 'mean_rel_lw' not found in summary for {est}, skipping heatmap.")
            continue
        pivot = sub.pivot(index="regime", columns="N_T_ratio", values="mean_rel_lw")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, cbar_kws={"label": "Mean (MVP var_rel to LW)"})
        ax.set_title(f"{est}: Min-variance variance relative to LW\n"
                     f"(negative = better than LW)")
        plt.tight_layout()
        plt.savefig(f"{outdir}/heatmap_rel_to_lw_{est}.png", dpi=120)
        plt.close()


def plot_bar_improvement_over_lw(results_df: pd.DataFrame,
                                 outdir: str) -> None:
    """
    Bar chart per regime showing mean relative improvement of NLS and Poet over LW.
    With 95% bootstrap confidence intervals. Uses the passed results DataFrame.
    """
    os.makedirs(outdir, exist_ok=True)
    sns.set_style("whitegrid")

    # Filter to the two challengers
    long_df = results_df[results_df["estimator"].isin(["NLS", "POET"])].copy()
    if long_df.empty:
        print("No NLS/POET data for bar plot.")
        return

    regimes = long_df["regime"].unique()
    ratios = sorted(long_df["N_T_ratio"].unique())
    width = 0.35
    x = np.arange(len(ratios))

    for regime in regimes:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, est in enumerate(["NLS", "POET"]):
            sub = long_df[(long_df["regime"] == regime) & (long_df["estimator"] == est)]
            means = []
            cis_lo = []
            cis_hi = []
            for r in ratios:
                vals = sub[sub["N_T_ratio"] == r]["mv_relative_to_lw"].dropna().values
                if len(vals) < 2:
                    means.append(np.nan)
                    cis_lo.append(np.nan)
                    cis_hi.append(np.nan)
                else:
                    boot_means = [np.mean(np.random.choice(vals, len(vals), replace=True))
                                  for _ in range(2000)]
                    means.append(np.mean(vals))
                    cis_lo.append(np.percentile(boot_means, 2.5))
                    cis_hi.append(np.percentile(boot_means, 97.5))
            ax.bar(x + i*width - width/2, means, width,
                   yerr=[np.array(means)-np.array(cis_lo), np.array(cis_hi)-np.array(means)],
                   capsize=3, label=est, error_kw={"elinewidth": 1, "capsize": 2})
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.1f}" for r in ratios])
        ax.set_xlabel("N/T ratio")
        ax.set_ylabel("Min‑variance variance relative to LW")
        ax.set_title(f"Regime: {regime}\n(negative = better than LW)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/bar_rel_to_lw_{regime}.png", dpi=120)
        plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Comprehensive simulation over T/N ratios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", type=int, default=DEFAULT_N,
                   help="Number of time periods (e.g., 252 or 504)")
    p.add_argument("--n-reps", type=int, default=N_REPS,
                   help="Number of replications per cell")
    p.add_argument("--n-t-ratios", type=str, default=DEFAULT_T_N_RATIOS,
                   help="Comma-separated N/T ratios, e.g. 2, 3, 4")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed")
    p.add_argument("--out", type=str, default="results/simulation",
                   help="Output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)

    n_ratios = [float(x.strip()) for x in args.n_t_ratios.split(",")]

    print("=" * 72)
    print(" COMPREHENSIVE SIMULATION STUDY")
    print("=" * 72)
    print(f"  N = {args.N} assets")
    print(f"  N/T ratios = {n_ratios}")
    print(f"  Regimes = {REGIMES}")
    print(f"  Replications per cell = {args.n_reps}")
    print(f"  Output directory = {args.out}")

    # -------------------------------------------------------------------------
    # Run all cells
    # -------------------------------------------------------------------------
    all_dfs = []
    total_cells = len(REGIMES) * len(n_ratios)
    cell_idx = 0
    start_time = time.time()

    for regime in REGIMES:
        for n_ratio in n_ratios:
            cell_idx += 1
            print(f"\n[cell {cell_idx}/{total_cells}] regime={regime}, N/T={n_ratio:.2f}")
            t0 = time.time()
            df_cell = run_cell(regime, n_ratio, args.N, args.n_reps, base_seed=args.seed)
            all_dfs.append(df_cell)
            elapsed = time.time() - t0
            print(f"  completed in {elapsed:.1f}s")
            # Save intermediate results periodically
            if cell_idx % 2 == 0:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined.to_csv(f"{args.out}/sim_results_temp.csv", index=False)

    # -------------------------------------------------------------------------
    # Combine and save final results
    # -------------------------------------------------------------------------
    results = pd.concat(all_dfs, ignore_index=True)
    results.to_csv(f"{args.out}/sim_results.csv", index=False)
    print(f"\nFull results saved to {args.out}/sim_results.csv")

    # -------------------------------------------------------------------------
    # Aggregate summary
    # -------------------------------------------------------------------------
    summary = (results.groupby(["regime", "N_T_ratio", "estimator"])
                     .agg(mean_mv=("minvar_true_var", "mean"),
                          std_mv=("minvar_true_var", "std"),
                          mean_rel_sample=("mv_relative_to_sample", "mean"),
                          mean_rel_lw=("mv_relative_to_lw", "mean"),
                          n_valid=("minvar_true_var", "count"))
                     .reset_index())
    summary.to_csv(f"{args.out}/sim_summary.csv", index=False)
    print("\n=== Summary (mean min‑var variance) ===")
    print(summary.pivot_table(index="regime", columns=["N_T_ratio", "estimator"],
                              values="mean_mv").round(5))

    # -------------------------------------------------------------------------
    # Statistical tests vs Sample and vs LW
    # -------------------------------------------------------------------------
    test_vs_sample = []
    test_vs_lw = []

    for regime in REGIMES:
        for n_ratio in n_ratios:
            sub = results[(results["regime"] == regime) & (results["N_T_ratio"] == n_ratio)]
            if sub.empty:
                continue
            # vs Sample
            res_s = paired_wilcoxon_one_sided(sub, "Sample", ["NLS", "POET"])
            res_s["regime"] = regime
            res_s["N_T_ratio"] = n_ratio
            test_vs_sample.append(res_s)
            # vs LW
            res_l = paired_wilcoxon_one_sided(sub, "LW", ["NLS", "POET"])
            res_l["regime"] = regime
            res_l["N_T_ratio"] = n_ratio
            test_vs_lw.append(res_l)

    df_test_sample = pd.concat(test_vs_sample, ignore_index=True)
    df_test_lw = pd.concat(test_vs_lw, ignore_index=True)
    df_test_sample.to_csv(f"{args.out}/sim_paired_tests_vs_sample.csv", index=False)
    df_test_lw.to_csv(f"{args.out}/sim_paired_tests_vs_lw.csv", index=False)

    print("\n=== Paired Wilcoxon (one‑sided) – challenger better than Sample ===")
    print(df_test_sample.pivot_table(index=["regime", "N_T_ratio"], columns="challenger",
                                     values="p_value").round(4))
    print("\n=== Paired Wilcoxon (one‑sided) – challenger better than LW ===")
    print(df_test_lw.pivot_table(index=["regime", "N_T_ratio"], columns="challenger",
                                 values="p_value").round(4))

    # -------------------------------------------------------------------------
    # Plots (using corrected functions)
    # -------------------------------------------------------------------------
    plot_relative_improvement_heatmaps(summary, args.out)
    plot_bar_improvement_over_lw(results, args.out)

    # Also use the simulation plotting function from hrp_lib (optional)
    try:
        L.plot_simulation_results(results, args.out)
        print("Generated additional plots from hrp_lib.plot_simulation_results")
    except Exception as e:
        print(f"hrp_lib.plot_simulation_results failed: {e}")

    total_time = time.time() - start_time
    print(f"\n[done] Total runtime: {total_time/60:.1f} minutes.")
    print(f"All outputs in {args.out}/")


if __name__ == "__main__":
    main()