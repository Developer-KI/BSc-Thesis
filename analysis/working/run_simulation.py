"""
=============================================================================
run_simulation.py - Synthetic regime sweep
=============================================================================

Three theoretically distinct regimes:

    factor_sparse  : low-rank common factor + truly sparse (banded) residual.
                     The regime POET was DESIGNED for. POET should win.
    toeplitz       : AR(1) Toeplitz covariance. No factor structure, dense.
                     POET has nothing to extract; NLS should win.
    identity_like  : near-identity covariance, weak uniform correlation.
                     LW (which shrinks toward identity) should be best.

For each (regime, replication, estimator) we record:

    1. Frobenius distance to the true covariance.
    2. Variance of the implied minimum-variance portfolio under the *true*
       covariance.  This is the portfolio-relevant loss (Engle-Ledoit-Wolf
       2019).
    3. Relative improvement vs. Sample baseline:
            rel = (loss_X - loss_Sample) / loss_Sample
       Negative means improvement.  This is what the headline figure plots,
       with bootstrap 95% CIs.

Why this is the "conditions under which X is preferred" anchor:
The three regimes are constructed so that *each* of {LW, NLS, POET} has a
regime where it should be the best estimator, theoretically.  Verifying
that pattern empirically gives the CRSP results in run_crsp.py a clear
theoretical foundation.

Outputs in results/simulation/:
    sim_results.csv             # long DataFrame (regime x rep x estimator)
    sim_summary.csv             # mean and SD per (regime, estimator)
    sim_paired_tests.csv        # paired-rep p-value of (X minus Sample) per regime
    sim_minvar_relative.png     # *** headline: relative MV-loss vs Sample ***
    sim_minvar_true_var.png     # raw MV variance boxplots (appendix)
    sim_frobenius.png           # Frobenius boxplots (appendix, with caveat)
=============================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

import hrp_lib as L


def main() -> None:
    np.random.seed(0)
    outdir = "results/simulation"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print(" Simulation: theoretical-regime sweep")
    print("=" * 72)
    print("  T=300, N=300  (N/T ~ 1) -- moderate high-dim regime")

    # Add a param sweep N/T 0.25 to 0.5 to 1 to 2
    df = L.run_simulation_study(
        T=300, N=300,
        regimes=("factor_sparse", "toeplitz", "identity_like"),
        n_reps=300, seed0=42,
    )
    df.to_csv(f"{outdir}/sim_results.csv", index=False)

    summary = (df.groupby(["regime", "estimator"])
                 .agg(mean_frob=("frobenius", "mean"),
                      sd_frob=("frobenius", "std"),
                      mean_mv=("minvar_true_var", "mean"),
                      sd_mv=("minvar_true_var", "std"),
                      mean_mv_rel=("mv_relative", "mean"),
                      median_mv_rel=("mv_relative", "median"))
                 .round(4))
    print("\n=== Simulation summary ===")
    print(summary.to_string())
    summary.to_csv(f"{outdir}/sim_summary.csv")

    # Paired Wilcoxon tests: H0: X has same MV variance as Sample;
    # alternative: X has LOWER MV variance (i.e. X is better).
    print("\n=== Paired Wilcoxon tests vs Sample "
          "(H1: X better than Sample) ===")
    rows = []
    for regime in df["regime"].unique():
        sub_sample = df[(df["regime"] == regime) &
                        (df["estimator"] == "Sample")] \
                     .sort_values("rep")["minvar_true_var"].values
        for est in ["LW", "NLS", "POET"]:
            sub_est = df[(df["regime"] == regime) &
                         (df["estimator"] == est)] \
                      .sort_values("rep")["minvar_true_var"].values
            try:
                stat, p = stats.wilcoxon(sub_est, sub_sample,
                                         alternative="less")
            except ValueError:
                stat, p = float("nan"), float("nan")
            mean_rel = ((sub_est - sub_sample) / sub_sample).mean()
            rows.append({"regime": regime, "estimator": est,
                         "mean_rel_improvement": mean_rel,
                         "wilcoxon_stat": stat,
                         "p_X_better": p})
    tests = pd.DataFrame(rows)
    print(tests.round(4).to_string(index=False))
    tests.to_csv(f"{outdir}/sim_paired_tests.csv", index=False)

    winners = (df.groupby(["regime", "estimator"])["minvar_true_var"]
                 .mean()
                 .reset_index()
                 .sort_values(["regime", "minvar_true_var"]))
    print("\n=== Ranking by mean Min-Var portfolio variance "
          "(lower is better) ===")
    print(winners.round(5).to_string(index=False))

    L.plot_simulation_results(df, outdir)
    print(f"\n[done] simulation outputs in {outdir}/")
    print("\nHeadline figure: sim_minvar_relative.png")
    print("(negative bars = improvement over Sample baseline)")


if __name__ == "__main__":
    main()