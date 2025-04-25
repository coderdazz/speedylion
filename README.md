# SpeedyLions

**SpeedyLions** is a powerful Python library designed to accelerate quantitative research and investment analysis in finance. Built with performance in mind, SpeedyLions provides a set of tools to calculate key performance metrics, measure risks, and manage portfolio analysis more efficiently. The library supports key functions for financial research, from risk metrics to portfolio optimization, and is optimized to handle large datasets with minimal overhead.

## Key Features

- **Performance Metrics**: Calculate various financial performance metrics like annualized return, volatility, Sharpe ratio, maximum drawdown, value-at-risk (VaR), and expected shortfall.
- **Risk Analysis**: Provides tools for risk management, including Ledoit-Wolf shrinkage for covariance estimation, tracking error, risk contributions, and downside deviation.
- **Portfolio Statistics**: Comprehensive portfolio analysis including risk-return trade-offs, Sharpe ratios, and effective number of constituents (ENC).
- **Rolling and Forward Returns**: Create rolling cumulative returns and forward returns for different window lengths with adjustable lags.
- **Transaction Costs**: Includes functionality to account for transaction costs in strategy performance metrics like Sharpe ratio.

## Installation

You can install SpeedyLions using pip:

```bash
pip install speedylions
```

## Usage

### 1. Portfolio Statistics

Calculate annualized return and volatility:

```python
from speedylions.performance import portfolioStats
import pandas as pd

# Sample portfolio returns data
port_returns = pd.Series([...])  # Replace with your data

portfolio_summary = portfolioStats(port_returns)
print(portfolio_summary)
```

### 2. Rolling Cumulative Returns

To calculate rolling cumulative returns:

```python
from speedylions.performance import create_rolling_returns

# Sample return series
returns = pd.Series([...])  # Replace with your data

rolling_returns = create_rolling_returns(returns, window=12)
print(rolling_returns)
```

### 3. Value at Risk (VaR)

To calculate VaR using the historical method:

```python
from speedylions.performance import value_at_risk

returns = pd.Series([...])  # Replace with your data
var = value_at_risk(returns, type='historic', level=0.01)
print(f"Value at Risk: {var}")
```

### 4. Sharpe Ratio with Transaction Costs

Calculate Sharpe ratio while considering transaction costs:

```python
from speedylions.performance import calculate_sharpe_with_transaction

y_pred = [...]  # Model predictions or portfolio signals
y_index = [...]  # Corresponding index
market_returns = pd.Series([...])  # Historical market returns

sharpe_ratio = calculate_sharpe_with_transaction(y_pred, y_index, market_returns)
print(f"Sharpe Ratio with Transaction Cost: {sharpe_ratio}")
```

### 5. Risk Contributions

To calculate risk contributions of portfolio positions:

```python
from speedylions.performance import riskContribution

cov_matrix = [...]  # Covariance matrix
weights = [...]  # Portfolio weights

risk_contrib = riskContribution(cov_matrix, weights)
print(risk_contrib)
```

### 6. Ledoit-Wolf Covariance Shrinkage

Shrink sample covariance matrix for more stable estimates:

```python
from speedylions.performance import LedoitWolfShrink
import numpy as np

returns = np.array([...])  # Replace with returns data
cov_matrix_shrinked = LedoitWolfShrink(returns)
print(cov_matrix_shrinked)
```

## Documentation

For detailed documentation on each function, including parameters and return values, refer to the [official documentation](#).

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or new features, please open an issue or submit a pull request. 

### Steps to Contribute:

1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push to your fork and create a pull request.

## License

SpeedyLions is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

This library builds on contributions from the open-source community and aims to provide fast, efficient tools for quantitative finance. Special thanks to [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [SciPy](https://scipy.org/).
