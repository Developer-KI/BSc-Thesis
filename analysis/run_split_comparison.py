from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 15,
})
import seaborn as sns

from typing import List

import utils.backtest as L


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_split_cost(cov: np.ndarray, A: List[int], B: List[int]) -> float:
    """Evaluate _vb_merge_cost for an arbitrary (A, B) partition."""
    if len(A) == 0 or len(B) == 0:
        return np.inf
    sA    = float(cov[np.ix_(A, A)].sum())
    sB    = float(cov[np.ix_(B, B)].sum())
    cross = float(cov[np.ix_(A, B)].sum())
    return L._vb_merge_cost(cross, sA, sB)


def _random_factor_cov(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Factor-model covariance matrix:  Sigma = B B' + diag(sigma^2)
    K = max(1, n//4) factors with random loadings; diagonal noise U[0.2, 1.5].
    """
    K   = max(1, n // 4)
    B   = rng.standard_normal((n, K))
    cov = B @ B.T + np.diag(rng.uniform(0.2, 1.5, n))
    return L._ensure_pd(cov)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_comparison(
    cluster_sizes: List[int],
    n_reps: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each cluster size n draw n_reps random covariance matrices and
    compare heuristic vs brute-force split quality.

    Returns a long-form DataFrame with columns:
        n, rep, score_bf, score_h, approx_ratio, exact_match
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for n in cluster_sizes:
        indices = list(range(n))
        for rep in range(n_reps):
            cov = _random_factor_cov(n, rng)

            A_bf, B_bf = L._vb_split_bruteforce(cov, indices)
            score_bf   = _eval_split_cost(cov, A_bf, B_bf)

            A_h,  B_h  = L._vb_split_heuristic(cov, indices)
            score_h    = _eval_split_cost(cov, A_h,  B_h)

            if score_bf > 1e-12:
                ratio = float(np.clip(score_h / score_bf, 1.0, 5.0))
            elif score_bf < -1e-12:
                ratio = float(np.clip(score_h / score_bf, 0.0, 1.0))
            else:
                ratio = 1.0

            exact = (
                frozenset(map(frozenset, [A_bf, B_bf])) ==
                frozenset(map(frozenset, [A_h,  B_h ]))
            )

            rows.append(dict(
                n=n, rep=rep,
                score_bf=score_bf, score_h=score_h,
                approx_ratio=ratio, exact_match=int(exact),
            ))

        print(f"  n={n:2d}  done  ({n_reps} reps)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    n_vals = sorted(df["n"].unique())

    _, ax = plt.subplots(figsize=(7, 4.2))
    med = df.groupby("n")["approx_ratio"].median()
    p95 = df.groupby("n")["approx_ratio"].quantile(0.95)
    p99 = df.groupby("n")["approx_ratio"].quantile(0.99)
    ax.fill_between(n_vals, 1.0, p99.loc[n_vals],
                    alpha=0.10, color="steelblue", label="99th pct")
    ax.fill_between(n_vals, 1.0, p95.loc[n_vals],
                    alpha=0.25, color="steelblue", label="95th pct")
    ax.plot(n_vals, med.loc[n_vals], "o-",
            color="steelblue", lw=1.8, ms=5, label="Median")
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="Optimal  (ratio = 1)")
    ax.set_xlabel("Cluster size  n")
    ax.set_ylabel("Score(heuristic) / Score(brute-force)")
    ax.set_title("Heuristic approximation quality")
    ax.set_xticks(n_vals)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "approx_ratio.png", dpi=130)
    plt.close()

    _, ax = plt.subplots(figsize=(7, 4.2))
    match_rate = df.groupby("n")["exact_match"].mean() * 100
    ax.plot(n_vals, match_rate.loc[n_vals], "s-",
            color="steelblue", lw=1.6, ms=6)
    ax.set_xlabel("Cluster size  n")
    ax.set_ylabel("Exact match rate  (%)")
    ax.set_title("Fraction of trials where heuristic = brute-force optimum")
    ax.set_ylim(0, 105)
    ax.set_xticks(n_vals)
    plt.tight_layout()
    plt.savefig(outdir / "exact_match_rate.png", dpi=130)
    plt.close()

    _, ax = plt.subplots(figsize=(5.5, 4.8))
    sub = df.sample(min(3000, len(df)), random_state=0)
    ax.scatter(sub["score_bf"], sub["score_h"],
               alpha=0.12, s=7, color="steelblue", rasterized=True)
    lo = min(sub["score_bf"].min(), sub["score_h"].min())
    hi = max(sub["score_bf"].max(), sub["score_h"].max())
    pad = (hi - lo) * 0.03
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=0.9, label="y = x  (optimal)")
    ax.set_xlabel("Brute-force objective")
    ax.set_ylabel("Heuristic objective")
    ax.set_title("Objective values: heuristic vs. exhaustive")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "score_scatter.png", dpi=130)
    plt.close()

    print(f"[plot] 3 figures saved to  {outdir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    CLUSTER_SIZES = [12, 14, 16, 18, 20]
    N_REPS        = 100
    OUTDIR        = Path(__file__).resolve().parent.parent / "results" / "split_comparison"

    print("=== HMVA heuristic vs. brute-force split comparison ===")
    print(f"  objective      : Corr(EW_L,EW_R)")
    print(f"  cluster sizes  : {CLUSTER_SIZES}")
    print(f"  reps per size  : {N_REPS}")
    print()

    df = run_comparison(CLUSTER_SIZES, N_REPS, seed=42)

    summary = (
        df.groupby("n")
        .agg(
            median_ratio=("approx_ratio", "median"),
            p95_ratio   =("approx_ratio", lambda x: x.quantile(0.95)),
            max_ratio   =("approx_ratio", "max"),
            match_pct   =("exact_match",  lambda x: x.mean() * 100),
        )
        .reset_index()
    )

    print("\n=== Approximation quality summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    df.to_csv(OUTDIR / "raw.csv", index=False)

    plot_results(df, OUTDIR)

    print(f"\n[done]  All outputs in  {OUTDIR}/")


if __name__ == "__main__":
    main()
