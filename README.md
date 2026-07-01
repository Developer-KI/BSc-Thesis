# Building Mean-Variance Portfolios That Perform Out-of-Sample

_BSc Econometrics and Data Science Thesis — University of Amsterdam_

_Author: Kiril Ivanov · Supervisor: Dr. Sanders Barendse_

---

## Project Layout

```
analysis/
├── utils/
│   ├── backtest.py          # HMVA pipeline, all strategy constructors, statistical tests
│   ├── data.py              # CRSP loading, point-in-time universe (UniverseFn)
│   └── plotting.py          # visualisation helpers
├── run_backtest.py          # main strategy comparison
├── run_cost_robustness.py   # transaction-cost robustness sweep
├── run_crisis_analysis.py   # NBER regime / sub-period analysis
└── run_split_comparison.py  # heuristic vs brute-force split accuracy simulation
thesis/
├── thesis.tex               # LaTeX thesis source
└── references.bib           # bibliography
data/                        # CRSP files
results/
├── backtest/                # metrics, statistical_tests, equity/DD plots
├── cost_robustness/         # robustness tables, Sharpe heatmaps
├── crisis_analysis/         # period_metrics, regime_summary, rolling plots
└── split_comparison/        # summary.csv, raw.csv, accuracy plots
requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
cd analysis
```

### Main strategy comparison

```bash
python run_backtest.py
```

Outputs: `results/backtest/metrics.csv`, `statistical_tests.csv`, equity/drawdown/Sharpe plots.

### Transaction-cost robustness sweep

```bash
python run_cost_robustness.py
```

Outputs: `results/cost_robustness/robustness_long.csv`, `robustness_summary.csv`, Sharpe heatmaps.

### Regime/crisis-period analysis

```bash
python run_crisis_analysis.py
```

Outputs: `results/crisis_analysis/period_metrics.csv`, `regime_summary.csv`, rolling Sharpe plots.

### Split criterion accuracy simulation

```bash
python run_split_comparison.py
```

Outputs: `results/split_comparison/summary.csv`, `raw.csv`, approximation-ratio plots.

---
