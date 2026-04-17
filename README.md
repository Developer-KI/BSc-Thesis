# BSc_Thesis

Topic: Optimal portfolio strategies for a large universe of assets

## Data

Daily close/return data over 5 years

Sectors: Bloomberg data from 100+ macro indecies
Assets: Yahoo Finance data for almost all (492/503) current 500 SPY constituients

## Methodology

### Methodology HSP:

Common drivers are selected as the top \( K \) (hyperparameter 1) with the highest common correlation with respect to all portfolio constituents, using correlation time series of daily returns and a time window \( W*{CD} \) (hyperparameter 2). With this common driver selection as inputs and one portfolio constituent as output, we train several feed‑forward networks for the prediction task. We use a time window \( W*{NN} \) (hyperparameter 3) for many architectures (varying numbers of layers and neurons) and select the most accurate one using Baysean Serch with a fixed number of maximum trials \( T \) (hyperparameter 4). We repeat this process for the rest of the portfolio constituents using the same common drivers.

Once the dynamics are approximated, for each previous optimal architecture (for one portfolio constituent), we compute the sensitivities of each constituent with respect to the common drivers using Automatic Adjoint Differentiation (AAD) on the feed‑forward networks, applied to the training set. Each sensitivity is a function of the training set, and we average it. Each constituent is assigned a vector of average sensitivity values with respect to the same common drivers, which is used for embedding. A distance matrix is computed for all constituents using these average sensitivity values as coordinates.

This matrix is used for portfolio optimization, first by finding the nearest positive semi‑definite neighbor matrix with numerical methods, then applying hierarchical clustering to that neighbor matrix, where hierarchies based on sensitivities are recorded. Finally, the positive semi‑definite neighbor of the sensitivity matrix is sorted according to these hierarchies, and weights are computed based on the hierarchical partitions and clusters' covariance matrices.

### Methodology HRP:

Here we use the covariance matrix for portfolio optimization, first by finding the nearest positive semi‑definite neighbor matrix with numerical methods if needed, then applying hierarchical clustering to that neighbor matrix, where hierarchies based on risk are recorded. Finally, the positive semi‑definite neighbor of the covariance matrix is sorted according to these hierarchies, and weights are computed based on the hierarchical partitions and clusters' covariance matrices.

### Methodology Markowitz

Here we use the historical data and the pyportfolioopt package to estimate a robust solution (using the Leodot Wolf cov methods and/or Black Litterman returns) and a base markowitz optimization

### Methodology Benchmark

I will also include a simple evenly split portfolio a market cap portfolio (SPY) to benchmark against all candidates for comparison

## Estimation Times

One iteration is one rebalance of the portfolio (mounthly):
Expected time for full backtest of X years = Time per rebalance (25s for 15 test assets/Xs for full 493 assets) x [X Years / 12 rebalances]
