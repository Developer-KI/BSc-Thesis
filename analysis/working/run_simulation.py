"""
=============================================================================
run_simulation.py - Synthetic factor-model simulation study
=============================================================================

Generates returns from a strict factor model with known true covariance
under three regimes:

    sharp    - few factors, big eigenvalue gap     (POET should win)
    medium   - moderate factor structure
    diffuse  - many small factors, no clear gap    (NLS should win)

For each regime and each replication we compare four covariance estimators
(sample / LW / NLS / POET) on:

    1. Frobenius distance to the true covariance.
    2. Variance of the implied minimum-variance portfolio under the *true*
       covariance.  This is the "portfolio loss" of Engle-Ledoit-Wolf
       (2019) - the metric that actually matters for portfolio choice.

This is the "conditions under which X is preferred" result examiners
appreciate: it gives the empirical numbers in run_main.py a theoretical
anchor.

Outputs in results/simulation/:
    sim_results.csv        -- long DataFrame (structure x rep x estimator)
    sim_summary.csv        -- mean / SD per (structure, estimator)
    sim_frobenius.png      -- boxplot
    sim_minvar_true_var.png -- boxplot
=============================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

import hrp_lib as L


def main() -> None:
    np.random.seed(0)
    outdir = "results/simulation"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print(" Simulation: factor-structure regime sweep")
    print("=" * 72)

    df = L.run_simulation_study(
        T=126, N=100, K_true=3,
        structures=("sharp", "medium", "diffuse"),
        n_reps=300, seed0=0,
    )
    df.to_csv(f"{outdir}/sim_results.csv", index=False)

    summary = (df.groupby(["structure", "estimator"])
                 .agg(mean_frob=("frobenius", "mean"),
                      sd_frob=("frobenius", "std"),
                      mean_mv=("minvar_true_var", "mean"),
                      sd_mv=("minvar_true_var", "std"))
                 .round(4))
    print("\n=== Simulation summary (mean and SD across replications) ===")
    print(summary.to_string())
    summary.to_csv(f"{outdir}/sim_summary.csv")

    # winner per cell: the estimator with the lowest mean MV variance
    winners = (df.groupby(["structure", "estimator"])["minvar_true_var"]
                 .mean()
                 .reset_index()
                 .sort_values(["structure", "minvar_true_var"]))
    print("\n=== Ranking by Min-Var portfolio variance (lower is better) ===")
    print(winners.round(5).to_string(index=False))

    L.plot_simulation_results(df, outdir)
    print(f"\n[done] simulation outputs in {outdir}/")


if __name__ == "__main__":
    main()
