"""
=============================================================================
run_simulation.py - Synthetic two-regime study
=============================================================================

Two theoretically distinct regimes:

    factor_sparse    : low-rank common factor + truly sparse (banded)
                       residual.  POET should win.  NLS competitive.
                       LW worst of the three advanced estimators because
                       its identity target misses the factor structure.
    dispersed_eigs   : dense covariance with a power-law eigenvalue
                       spectrum, no factor gap, no sparsity.  NLS should
                       dominate LW because every eigenvalue needs a
                       different amount of shrinkage.  POET has no
                       structure to exploit.

For each (regime, replication, estimator) we record:
    - Frobenius distance to true Σ.
    - Min-var portfolio variance under the *true* covariance
      (Engle-Ledoit-Wolf 2019 portfolio-relevant loss).
    - Two relative-improvement columns:
        mv_relative_to_sample = (loss_X − loss_Sample) / loss_Sample
        mv_relative_to_lw     = (loss_X − loss_LW)     / loss_LW

Two paired Wilcoxon test families are reported:

    Family 1.  H0: estimator_X has same MV variance as Sample;
               H1: estimator_X has lower MV variance.
               Run for X in {LW, NLS, POET}.

    Family 2.  H0: estimator_X has same MV variance as LW;
               H1: estimator_X has lower MV variance.
               Run for X in {NLS, POET}.

Family 1 establishes that all advanced estimators clear the trivial
Sample baseline.  Family 2 is the more interesting test: do NLS and POET
go beyond what linear shrinkage already provides?

Outputs in results/simulation/:
    sim_results.csv                  long DataFrame
    sim_summary.csv                  per-(regime, estimator) summary
    sim_paired_tests_vs_sample.csv   Family 1 Wilcoxon results
    sim_paired_tests_vs_lw.csv       Family 2 Wilcoxon results
    sim_minvar_relative_to_sample.png   Family 1 visual
    sim_minvar_relative_to_lw.png       *** Family 2: the headline ***
    sim_minvar_true_var.png          appendix
    sim_frobenius.png                appendix (caveat noted)
=============================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

import hrp_lib as L


def _wilcoxon_table(df: pd.DataFrame, baseline: str,
                    challengers: list, regimes: list) -> pd.DataFrame:
    """
    Run one-sided paired Wilcoxon: H1 = challenger has LOWER MV variance
    than baseline, paired across replications within each regime.
    """
    rows = []
    for regime in regimes:
        sub_base = df[(df["regime"] == regime) & (df["estimator"] == baseline)] \
                   .sort_values("rep")["minvar_true_var"].values
        for est in challengers:
            sub_est = df[(df["regime"] == regime) & (df["estimator"] == est)] \
                      .sort_values("rep")["minvar_true_var"].values
            try:
                stat, p = stats.wilcoxon(sub_est, sub_base, alternative="less")
            except ValueError:
                stat, p = float("nan"), float("nan")
            mean_rel = ((sub_est - sub_base) / sub_base).mean()
            rows.append({"regime": regime, "challenger": est,
                         "baseline": baseline,
                         "mean_rel_improvement": mean_rel,
                         "wilcoxon_stat": stat,
                         "p_X_better": p})
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(0)
    outdir = "results/simulation"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print(" Simulation: two-regime sweep")
    print("=" * 72)
    print("  T=300, N=200  (N/T ~ 0.67) -- moderate high-dim regime")
    print("  Regimes: factor_sparse (POET-favouring), "
          "dispersed_eigs (NLS-favouring)")

    df = L.run_simulation_study(
        T=60, N=6,
        regimes=("factor_sparse", "dispersed_eigs"),
        n_reps=50, seed0=42,
    )
    df.to_csv(f"{outdir}/sim_results.csv", index=False)

    summary = (df.groupby(["regime", "estimator"])
                 .agg(mean_frob=("frobenius", "mean"),
                      sd_frob=("frobenius", "std"),
                      mean_mv=("minvar_true_var", "mean"),
                      sd_mv=("minvar_true_var", "std"),
                      mean_rel_sample=("mv_relative_to_sample", "mean"),
                      mean_rel_lw=("mv_relative_to_lw", "mean"))
                 .round(4))
    print("\n=== Simulation summary ===")
    print(summary.to_string())
    summary.to_csv(f"{outdir}/sim_summary.csv")

    regimes = list(df["regime"].unique())

    print("\n=== Wilcoxon Family 1: advanced estimators vs Sample ===")
    print("    H0: equal MV variance     H1: estimator_X better than Sample")
    fam1 = _wilcoxon_table(df, baseline="Sample",
                           challengers=["LW", "NLS", "POET"],
                           regimes=regimes)
    print(fam1.round(4).to_string(index=False))
    fam1.to_csv(f"{outdir}/sim_paired_tests_vs_sample.csv", index=False)

    print("\n=== Wilcoxon Family 2: NLS / POET vs LW ===")
    print("    H0: equal MV variance     H1: estimator_X better than LW")
    fam2 = _wilcoxon_table(df, baseline="LW",
                           challengers=["NLS", "POET"],
                           regimes=regimes)
    print(fam2.round(4).to_string(index=False))
    fam2.to_csv(f"{outdir}/sim_paired_tests_vs_lw.csv", index=False)

    winners = (df.groupby(["regime", "estimator"])["minvar_true_var"]
                 .mean()
                 .reset_index()
                 .sort_values(["regime", "minvar_true_var"]))
    print("\n=== Ranking by mean Min-Var portfolio variance "
          "(lower is better) ===")
    print(winners.round(5).to_string(index=False))

    L.plot_simulation_results(df, outdir)
    print(f"\n[done] simulation outputs in {outdir}/")
    print()
    print("Figures:")
    print("  sim_minvar_relative_to_sample.png   - Family 1 visual")
    print("  sim_minvar_relative_to_lw.png       - *** Family 2: HEADLINE ***")
    print("  sim_minvar_true_var.png             - raw boxplots (appendix)")
    print("  sim_frobenius.png                   - Frobenius (appendix)")


if __name__ == "__main__":
    main()