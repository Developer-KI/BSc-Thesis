# BSc Thesis Plan & Content Overview

**Topic:** Regime-Switching Portfolio Selection — An HMM-Based Framework for Dynamic Allocation Between Ledoit-Wolf Mean-Variance and Hierarchical Risk-Based Methods

---

## 0. Working Title (pick one)

1. *Regime-Conditional Portfolio Selection: An HMM-Based Switching Framework Between Shrinkage Mean-Variance and Hierarchical Risk Allocation*
2. *Does Regime Detection Justify Method Selection? An Empirical Study of HMM-Driven Switching Between Ledoit-Wolf Markowitz and Hierarchical Risk Parity*
3. *When to Optimize and When to Diversify: A Regime-Switching Approach to Portfolio Construction*

Recommendation: **(2)** — the most defensible and academically conservative. It frames the thesis as an empirical question with a clear null hypothesis.

---

## 1. Research Question & Hypotheses

### Primary Research Question
*Does a Hidden Markov Model used as a discrete regime-switching selector — allocating between a Ledoit-Wolf-shrunk Mean-Variance optimizer in calm regimes and a Hierarchical Risk Parity (or HSP) allocator in turbulent regimes — deliver superior risk-adjusted out-of-sample performance compared to each standalone method and to simpler regime-detection baselines, after realistic transaction costs?*

### Testable Hypotheses

- **H1 (Main):** The HMM switching strategy achieves higher risk-adjusted returns (Sharpe ratio) than either standalone strategy, after transaction costs, statistically significant at the 5% level.
- **H2 (Value of complexity):** The HMM selector outperforms a naive VIX/realized-volatility threshold selector.
- **H3 (Mechanism):** Performance gains are concentrated around regime-transition periods, particularly during drawdowns.
- **H4 (Robustness):** Results hold across (a) asset universes (equity-only, multi-asset), (b) rebalancing frequencies (weekly, monthly), and (c) transaction cost assumptions (0, 5, 10, 25 bps).

### Null Results Are Acceptable
Even if H1–H4 are rejected, the thesis remains valuable: a rigorous demonstration that HMM-based switching does not improve on simpler methods would itself be a defensible contribution.

---

## 2. Positioning & Contribution

### Gap Filled
No existing paper implements HMM as a *discrete selector* between Ledoit-Wolf Mean-Variance and Hierarchical Risk Parity / HSP. Adjacent work uses:
- HMMs with *single* allocators adapted to regime (Nystrup et al., 2018, 2020)
- Deep RL selectors over *multiple HRP variants* (Millea & Edalat, 2023)
- HMMs to rotate between *factor models* (Wang, Lin & Mikhelson, 2020)
- Analytical MV with Markov switching (Zhou & Yin, 2003)

### Stated Contribution
1. An empirically validated regime-switching *meta-strategy* that combines two families of portfolio construction methods rather than adapting one.
2. An ablation isolating the value of (a) HMM vs. volatility-threshold regime detection, (b) Ledoit-Wolf vs. sample covariance, and (c) HRP vs. HSP as the risk-based allocator.
3. A public, reproducible backtesting pipeline.

---

## 3. Timeline: 16-Week Plan

Assumes 3–4 months of active work. Buffer weeks built in for when (not if) something breaks.

### Phase 1 — Foundation (Weeks 1–2)

**Week 1:**
- Finalize literature review (target: ~40 references, of which ~15 "core").
- Confirm thesis topic and research questions with supervisor.
- Set up Zotero/Mendeley, Git repository, LaTeX or Typst document skeleton.
- Read and take notes on: López de Prado (2016), Raffinot (2017, 2018), Ledoit & Wolf (2004), Nystrup et al. (2018), Ciciretti & Bucci (2022), DeMiguel, Garlappi & Uppal (2009).

**Week 2:**
- Write the **Introduction** and **Literature Review** first drafts (yes, before code — they anchor your scope).
- Obtain written supervisor sign-off on research question and scope.
- Select asset universes and data sources. Confirm data access.
- Draft a *pre-registration document* (below) and commit it to Git with a timestamp. This protects you from being accused of data-snooping later.

### Phase 2 — Infrastructure (Weeks 3–5)

*This phase is the single biggest predictor of thesis quality. Do not skip.*

**Week 3:**
- Download and clean all price data. Handle survivorship bias, corporate actions, delistings.
- Build a `DataLoader` class with strict point-in-time guarantees (no future data leaks).
- Compute and store: adjusted returns, rolling volatility, VIX series, realized covariance windows.

**Week 4:**
- Build the **walk-forward backtesting engine** with:
  - Rolling expanding-window re-estimation
  - Transaction cost model (linear cost ∝ turnover)
  - Rebalance logic with configurable frequency
  - Output: time series of weights, turnover, costs, portfolio returns
- Unit tests: verify 1/N baseline produces identical results to analytical benchmark.

**Week 5:**
- Implement and unit-test the 1/N and equal-risk-contribution benchmark strategies.
- Produce first sanity-check plots. Fix anything weird before proceeding.
- **Milestone: reproducible, version-controlled pipeline runs end-to-end on a toy strategy.**

### Phase 3 — Model Implementation (Weeks 6–8)

**Week 6: Covariance + Markowitz**
- Implement sample covariance, Ledoit-Wolf shrinkage covariance.
- Implement Mean-Variance optimizer (minimum variance and maximum Sharpe variants). Use `cvxpy` or `scipy.optimize`.
- Add long-only constraint (standard in this literature) and max-weight cap (e.g., 20%) for numerical stability.
- Run MV + LW on the full pipeline. Store results.

**Week 7: Hierarchical Risk Methods**
- Implement HRP (López de Prado 2016). Check against `PyPortfolioOpt` or `riskfolio-lib` implementation.
- Implement HSP as the secondary variant (see citation in reading list — verify primary reference with supervisor).
- Compare weights against reference implementation on identical input. Discrepancy > 1% means a bug.

**Week 8: HMM**
- Fit 2-state Gaussian HMM on features (options: VIX changes, realized vol, return + vol stack).
- **Critical:** Use online filtering (forward algorithm / Viterbi) with rolling re-estimation. Never use Baum-Welch smoothed states for backtest-period assignment.
- Run EM from multiple seeds; keep the highest-likelihood fit.
- Sanity-check: do inferred regimes align visually with known crisis periods (2008, 2020, 2022)?

### Phase 4 — Experiments & Analysis (Weeks 9–11)

**Week 9: Main Experiment**
- Run the switching strategy end-to-end on the primary asset universe.
- Run all baselines: 1/N, LW-MV, sample-cov MV, HRP, HSP, VIX-threshold switcher.
- Produce the main performance table (returns, vol, Sharpe, Sortino, max drawdown, turnover, net Sharpe after TC).

**Week 10: Ablations & Robustness**
- Ablation 1: sample covariance vs. Ledoit-Wolf (isolate shrinkage effect).
- Ablation 2: HRP vs. HSP (isolate hierarchical allocator choice).
- Ablation 3: HMM vs. VIX-threshold selector (isolate regime-detection method).
- Sensitivity grid: rebalance frequency ∈ {weekly, monthly, quarterly}; TC ∈ {0, 5, 10, 25 bps}; HMM states ∈ {2, 3}.
- Statistical tests: Ledoit-Wolf (2008) robust Sharpe ratio test; bootstrap confidence intervals.

**Week 11: Mechanism Analysis**
- Regime-conditional performance attribution: how much of the outperformance comes from each regime?
- Drawdown analysis: behavior during 2008, 2018, 2020, 2022.
- Turnover decomposition: how costly are regime switches?
- **Milestone: all empirical results frozen. No more model changes after this point.**

### Phase 5 — Writing (Weeks 12–14)

**Week 12:**
- Write **Methodology** (Chapter 3) — this is the most technical chapter; do it when the implementation is fresh.
- Write **Data** (Chapter 4) and **Experimental Design** (Chapter 5).

**Week 13:**
- Write **Results** (Chapter 6). Tables and figures first, prose second.
- Redraft **Introduction** and **Literature Review** with the benefit of knowing your actual results.

**Week 14:**
- Write **Discussion** (Chapter 7) and **Conclusion** (Chapter 8).
- Write **Abstract** last (it is a summary of the finished thesis, not a prediction).
- Supervisor review of full first draft.

### Phase 6 — Revision & Submission (Weeks 15–16)

**Week 15:**
- Incorporate supervisor feedback.
- Polish figures, typeset equations, check citation consistency.
- Proofread twice, at least once on paper.
- Verify code is reproducible from a clean clone.

**Week 16:**
- Final supervisor sign-off.
- Format according to institutional requirements.
- Submit.
- Back up everything. Twice.

---

## 4. Pre-Registration Document

Commit this to Git at end of Week 2. It binds you to choices made *before* seeing results, which is your strongest defense against accusations of p-hacking.

- **Asset universes:** (specify tickers and date ranges)
- **Train / out-of-sample split:** (specify exact dates)
- **Rebalance frequency (primary):** Monthly
- **HMM states:** 2 (primary), 3 (robustness)
- **HMM features:** VIX levels + VIX log-changes
- **Re-estimation window:** Expanding, re-fit every 12 months
- **Covariance lookback:** 252 trading days
- **Transaction cost (primary):** 10 bps per unit turnover
- **Max weight per asset:** 20%
- **Short selling:** Disallowed
- **Primary performance metric:** Net-of-cost Sharpe ratio
- **Significance test:** Ledoit-Wolf (2008) robust Sharpe difference test, α = 0.05

---

## 5. Full Thesis Content Outline

Target length: 40–60 pages main text plus appendices. Below is the chapter breakdown with target page counts and specific content.

### Front Matter
- Title page, declaration, acknowledgments
- **Abstract** (~250 words): problem, method, result, implication
- Table of contents, list of figures, list of tables, list of abbreviations

### Chapter 1: Introduction (4–6 pages)
1.1 Motivation — the limits of Modern Portfolio Theory in practice
1.2 The regime-switching perspective — why markets are not stationary
1.3 Research gap — combining HMM selection with two distinct allocator families
1.4 Research question and hypotheses
1.5 Contributions (bulleted)
1.6 Thesis structure

### Chapter 2: Literature Review (8–10 pages)
2.1 Classical portfolio theory — Markowitz (1952), Sharpe (1966)
2.2 The estimation problem — Michaud (1989), Best & Grauer (1991), DeMiguel et al. (2009)
2.3 Robust covariance estimation — Ledoit & Wolf (2003, 2004, 2020)
2.4 Hierarchical methods — López de Prado (2016), Raffinot (2017, 2018), Lohre et al. (2020)
2.5 Regime detection — Hamilton (1989), Ang & Bekaert (2002), Guidolin & Timmermann (2007), Nystrup et al. (2018)
2.6 Regime-switching in portfolio construction — Costa & Kwon (2019), Ciciretti & Bucci (2022), Wang et al. (2020), Millea & Edalat (2023)
2.7 Synthesis and identification of the research gap

### Chapter 3: Methodology (10–12 pages)
3.1 Notation and portfolio problem setup
3.2 Mean-Variance optimization with Ledoit-Wolf shrinkage
  3.2.1 The sample covariance problem
  3.2.2 Shrinkage estimator and optimal intensity
  3.2.3 Optimization problem and constraints
3.3 Hierarchical Risk Parity
  3.3.1 Distance metric and hierarchical clustering
  3.3.2 Quasi-diagonalization
  3.3.3 Recursive bisection
3.4 Hierarchical Sensitivity Parity (HSP) — variant used for robustness
3.5 Hidden Markov Model for regime detection
  3.5.1 Model specification and emission distributions
  3.5.2 Feature selection
  3.5.3 Estimation via Expectation-Maximization
  3.5.4 Online state inference — critical discussion of filtering vs. smoothing
  3.5.5 Rolling re-estimation protocol
3.6 The switching framework
  3.6.1 Regime classification rule
  3.6.2 Allocator assignment per regime
  3.6.3 Rebalancing logic and cost handling
3.7 Benchmark strategies
  3.7.1 1/N equal-weight
  3.7.2 Standalone LW-MV and HRP
  3.7.3 VIX-threshold selector
  3.7.4 Sample-covariance MV (for shrinkage ablation)

### Chapter 4: Data (3–4 pages)
4.1 Asset universes
  4.1.1 Primary: S&P 500 sector ETFs
  4.1.2 Secondary (multi-asset): SPY, AGG, GLD, DBC, VNQ, EFA, IEF
4.2 Sample period and walk-forward split
4.3 Data sources and cleaning
4.4 Handling corporate actions, delistings, survivorship bias
4.5 VIX and auxiliary series
4.6 Descriptive statistics and stylized facts

### Chapter 5: Experimental Design (3–4 pages)
5.1 Walk-forward backtesting protocol
5.2 Rebalancing and re-estimation schedules
5.3 Transaction cost model
5.4 Performance metrics — annualized return, volatility, Sharpe, Sortino, Calmar, max drawdown, turnover
5.5 Statistical tests — Ledoit-Wolf robust Sharpe test, stationary bootstrap, Deflated Sharpe Ratio (Bailey & López de Prado, 2014) if space permits
5.6 Ablation and robustness grid
5.7 Reproducibility statement — Git repository link, environment pinning

### Chapter 6: Results (10–12 pages)
6.1 Main performance comparison — master table
6.2 Equity curves and drawdown plots
6.3 Regime identification — inferred regime time series with overlay of historical crises
6.4 Performance conditional on regime
6.5 Ablation 1: Ledoit-Wolf vs. sample covariance
6.6 Ablation 2: HRP vs. HSP
6.7 Ablation 3: HMM vs. VIX-threshold regime detection
6.8 Sensitivity to transaction costs, rebalance frequency, HMM states
6.9 Turnover and implementation costs
6.10 Statistical significance of Sharpe differences

### Chapter 7: Discussion (4–6 pages)
7.1 Do the results support the hypotheses?
7.2 When and why does the switch add value? Mechanism discussion.
7.3 Comparison with published results in the literature
7.4 Practical implications for portfolio managers
7.5 Limitations — regime lag, HMM instability, asset universe scope, backtest-only evidence, lookback sensitivity
7.6 Threats to validity

### Chapter 8: Conclusion (2–3 pages)
8.1 Summary of findings
8.2 Contributions restated
8.3 Directions for future research — more states, alternative regime detection (Bayesian change-point, MSGARCH), tail-aware hierarchical methods, live-trading evaluation

### References (APA or whatever your institution requires; aim for 40–60 entries)

### Appendices
- A: Derivation of Ledoit-Wolf optimal shrinkage intensity
- B: HRP pseudocode
- C: HMM EM pseudocode
- D: Full sensitivity grid tables
- E: Supplementary figures
- F: Code listing of the switching logic
- G: Reproducibility guide and repository structure

---

## 6. Required Reading List (Prioritized)

### Must Read (Week 1)
1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
2. López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *Journal of Portfolio Management*.
3. Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix. *Journal of Portfolio Management*.
4. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus Naive Diversification. *Review of Financial Studies*.
5. Raffinot, T. (2017). Hierarchical Clustering-Based Asset Allocation. *Journal of Portfolio Management*. [Plus the primary HSP reference — confirm exact citation with supervisor; candidates include Lohre et al. (2020) on hierarchical risk parity with tail dependencies, or the specific HSP paper you are drawing the method from.]
6. Nystrup, P., Madsen, H., & Lindström, E. (2018). Dynamic Allocation or Diversification: A Regime-Based Approach to Multiple Assets.

### Must Read (Weeks 2–3)
7. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*.
8. Ang, A., & Bekaert, G. (2002). International Asset Allocation With Regime Shifts.
9. Guidolin, M., & Timmermann, A. (2007). Asset Allocation under Multivariate Regime Switching.
10. Ciciretti, V., & Bucci, A. (2022). Hierarchical Risk Parity and Minimum Variance Portfolio Design on NYSE.
11. Ledoit, O., & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio.
12. Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.

### Read Before Writing Methodology
13. Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models.
14. López de Prado, M. (2018). *Advances in Financial Machine Learning* — Chapter 16.
15. Michaud, R. O. (1989). The Markowitz Optimization Enigma.

---

## 7. Tools & Stack (Suggested)

- **Language:** Python 3.11+
- **Core libraries:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `cvxpy`
- **HMM:** `hmmlearn` (well-tested; avoid rolling your own for the thesis)
- **Portfolio:** `riskfolio-lib` or `PyPortfolioOpt` for reference implementations; your own code for the main pipeline
- **Plotting:** `matplotlib` + `seaborn`
- **Data:** `yfinance` (free, acceptable for a BSc) or Refinitiv / Bloomberg if available through your institution
- **Writing:** LaTeX with Overleaf, or Typst
- **Version control:** Git + GitHub (private repo); tag weekly snapshots
- **Reproducibility:** `requirements.txt` pinned; a `Makefile` with `make all` regenerating every figure

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Look-ahead bias in HMM | High | Fatal | Rolling re-estimation; filter-only state inference; document protocol |
| Transaction costs kill alpha | High | High | Model from day 1; report net-of-cost results |
| HMM fails to converge or flips states | Medium | Medium | Multiple random seeds; lock state labels by mean volatility |
| Over-fitting to chosen period | Medium | High | Pre-register choices; test on secondary universe |
| Scope creep | High | High | Hard-freeze scope at Week 2; extensions go to Future Work |
| Supervisor changes expectations mid-thesis | Medium | High | Weekly written updates; lock scope in writing |
| Data access problems | Low | High | Confirm access Week 1; have free-tier fallback |
| Code bugs discovered during writing | Medium | High | Comprehensive unit tests; freeze code at Week 11 |

---

## 9. Success Criteria

### Minimum (passing)
- Clean walk-forward backtest on one asset universe
- HMM switching vs. 1/N, LW-MV, HRP comparison
- Transaction costs included
- Coherent 40-page writeup

### Target (good grade)
- Two asset universes
- All ablations (LW vs. sample-cov; HRP vs. HSP; HMM vs. VIX)
- Statistical significance testing
- Mechanism analysis
- 50-page writeup with strong discussion

### Stretch (top grade)
- Three asset universes (adds e.g., single-name equities)
- MSGARCH or Bayesian change-point as additional regime detector
- Deflated Sharpe Ratio computation
- Public GitHub repo with documentation, regenerates all figures with `make all`
- A finding that is non-obvious and clearly stated (positive or negative)

---

## 10. Weekly Checkpoint Routine

Every Friday, do the following and send to supervisor:

1. What I did this week (bullets)
2. What I will do next week (bullets)
3. Blockers
4. One figure or result, if available

This routine protects you: it creates a paper trail of progress and a running log of decisions. It also forces you to ship something every week.

---

## 11. Final Advice

- **Build infrastructure before models.** A bulletproof backtester is worth more than a clever HMM.
- **Write every week.** The thesis is what is graded, not the code. Start writing in Week 2, not Week 12.
- **Defend the null.** If your HMM switcher does not beat the simpler baseline, that is a valid and interesting finding. Frame it that way from the start.
- **Ablate everything.** Every upgrade over plain Markowitz+HRP needs a row in the results table showing what it bought you.
- **Freeze scope at Week 2 and freeze code at Week 11.** Everything else is Future Work.
