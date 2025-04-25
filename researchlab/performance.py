"""
The performance module provides functions for calculating performance metrics
and measuring risks.
"""

from researchlab.core.date_funcs import *
import numpy as np
import pandas as pd
from typing import Union, List
from scipy import stats

pd_type = Union[pd.Series, pd.DataFrame]


def roll_cum_returns(series: pd.Series) -> float:
    """Rolling cumulative return function"""
    return (1 + series / 100).cumprod().iloc[-1] - 1


def create_rolling_returns(series: pd.Series, window: int) -> pd.Series:
    """Create rolling returns of different window lengths"""
    return_series = series.rolling(window).apply(roll_cum_returns)
    return return_series


def create_forward_returns(series: pd.Series, window: Union[int, List], lag: int = 0) -> Union[pd.Series, dict]:
    """Create rolling returns with lags"""
    if isinstance(window, int):
        lagged_fwd_returns = create_rolling_returns(series, window).shift(-window - lag)
        return lagged_fwd_returns.dropna()

    series_list = {}
    for horizon in window:
        series_list[horizon] = create_rolling_returns(series, window).shift(-window - lag).dropna()
    return series_list


# Ledoit-Wolf covariance shrinkage
def LedoitWolfShrink(returns: np.array) -> np.array:
    """
    Shrinks sample covariance matrix towards constant correlation unequal variance matrix.
    Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management, 30(2004),
    110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1 (constant
    sample average correlation unequal sample variance matrix).

    :param returns: t, n - returns of t observations of n shares.
    :return: Covariance matrix, sample average correlation, shrinkage.
    """
    t, n = returns.shape
    mean_returns = np.mean(returns, axis=0, keepdims=True)
    returns -= mean_returns

    sample_cov = returns.T @ returns / t

    # Sample average correlation
    var = np.diag(sample_cov).reshape(-1, 1)
    sqrt_var = var ** 0.5
    unit_cor_var = sqrt_var * sqrt_var.T
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var
    np.fill_diagonal(prior, var)

    # Pi-hat
    y = returns ** 2
    phi_mat = (y.T @ y) / t - sample_cov ** 2
    phi = phi_mat.sum()

    # Rho-hat
    theta_mat = ((returns ** 3).T @ returns) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0)
    rho = np.diag(phi_mat).sum() + average_cor * (1 / sqrt_var @ sqrt_var.T * theta_mat).sum()

    # Gamma-hat
    gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

    # Shrinkage constant
    kappa = (phi - rho) / gamma
    shrink = max(0, min(1, kappa / t))

    # Estimator
    sigma = shrink * prior + (1 - shrink) * sample_cov
    return sigma


def portVol(sigma: np.array, portwts: np.array) -> float:
    """Calculate volatility using weights and covariance matrix"""
    return np.dot(portwts.T, np.dot(sigma, portwts)) ** 0.5


def portVolmulti(weights: np.array, returns: pd.DataFrame, freq: str = 'M') -> np.array:
    """Calculate volatility on multiple simulated portfolios"""
    mtx = freqInt(freq)  # Integer adjustment for frequency i.e. 12 for monthly
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    covmat = LedoitWolfShrink(returns.values)
    vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, covmat, weights)) * np.sqrt(mtx)

    return vols


def maxDD(returns: pd.Series) -> float:
    """Calculate max draw-down"""
    wealth = (1 + returns).cumprod()
    cum_max = np.maximum.accumulate(wealth)
    dd = wealth / cum_max - 1
    return dd.min()


def annualise_return(returns: pd_type, freq: str = 'M') -> float:
    """Annualised return series"""
    log_ret = np.log(returns + 1)
    if freq is None:
        freq = returns.index.inferred_freq
    log_ann_return = log_ret.mean() * freqInt(freq)
    return np.exp(log_ann_return) - 1


def annualise_volatility(returns: pd_type, freq: str = 'M') -> float:
    """Annualised volatility"""
    if freq is None:
        freq = returns.index.inferred_freq
    return returns.std() * np.sqrt(freqInt(freq))


def annualiseStat(returns: pd_type, freq: str = 'M') -> tuple:
    """Get annualised return and volatility"""
    if freq is None:
        freq = returns.index.inferred_freq
    return annualise_return(returns, freq), annualise_volatility(returns, freq)


def value_at_risk(returns: pd.Series, type: str = 'historic', level: float = 0.01) -> float:
    """
    Returns the (1-level)% VaR using historical method of portfolio or series of holdings.
    :param returns: portfolio return series or holdings return series
    :param level: significance level, 1% by default, 0.01
    :param type: 'parametric' for parametric otherwise historic by default
    :return: Value at Risk
    """
    if type.lower() == 'parametric':
        vol = returns.std()
        alpha = stats.norm.ppf(1 - level / 2)
        return -vol * alpha
    return returns.quantile(level)


def expected_shortfall(returns: pd.Series, level: float = 0.01) -> float:
    """
    Computes the expected shortfall of a portfolio or series of holdings, 1% significance by default.
    :param returns: portfolio or holdings return series
    :param level: significance level, 1% by default
    :return: Expected Shortfall, also known as conditional VaR
    """
    breach = returns < value_at_risk(returns, level=level)
    return returns[breach].mean()


def portfolioStats(portreturns: pd.Series, freq: str = 'M', yields: pd.Series = None, name: str = 'portfolio') -> pd.DataFrame:
    """
    Produce portfolio statistics summary, includes performance and risk stats.
    :param portreturns: portfolio return series
    :param freq: frequency of data
    :param yields: portfolio historical yields, default is None
    :return: pd.DataFrame object, portfolio performance summary
    """
    AnnRet, AnnStd = annualiseStat(portreturns, freq)  # Return and volatility
    Sharpe = AnnRet / AnnStd  # Sharpe
    maxdd = maxDD(portreturns)  # Max DD
    skew = portreturns.skew()  # Skewness
    VaR = value_at_risk(portreturns)  # Value at Risk
    ES = expected_shortfall(portreturns)  # Expected Shortfall

    if yields is None:
        statlist = [AnnRet * 100, AnnStd * 100, Sharpe, maxdd * 100, VaR * 100, ES * 100, skew]
        colname = ['Return(Ann)', 'Volatility(Ann)', 'Sharpe', 'Max Drawdown',
                   'VaR-' + freq, 'Expected Shortfall-' + freq, 'Skew']
    else:
        TTMYld = yields
        RAY = TTMYld / AnnStd
        statlist = [AnnRet * 100, AnnStd * 100, Sharpe, maxdd * 100, VaR * 100, ES * 100, skew, TTMYld * 100, RAY * 100]
        colname = ['Return(Ann)', 'Volatility(Ann)', 'Sharpe', 'Max Drawdown',
                   'VaR-' + freq, 'Expected Shortfall-' + freq, 'Skew', 'TTM Yield', 'RAY']

    statlist = [round(i, 2) for i in statlist]
    return pd.DataFrame([statlist], index=[name], columns=colname)


def riskContribution(sigma: np.array, portwts: np.array) -> np.array:
    """
    Find risk contributions of positions.
    :param sigma: covariance matrix
    :param portwts: weight allocations
    :return: risk contributions by position, numpy array
    """
    port_var = np.dot(portwts.T, np.dot(sigma, portwts))
    mrc = np.dot(sigma, portwts)
    return (mrc * portwts) / port_var


def enc(weights: np.array) -> float:
    """
    Computes the Effective Number of Constituents (ENC) given an input
    vector of weights of a portfolio.
    """
    return (weights ** 2).sum() ** (-1)


def encb(risk_contrib: np.array) -> float:
    """
    Computes the Effective Number of Correlated Bets (ENBC) given an input
    vector of portfolio risk contributions.
    """
    return (risk_contrib ** 2).sum() ** (-1)


def tracking_error(r_a: pd.Series, r_b: pd.Series) -> float:
    """
    Returns the tracking error between two return series.
    This method is used in Sharpe Analysis minimization problem.
    """
    return np.sqrt(((r_a - r_b) ** 2).sum())


def strategy_sharpe(expected_returns: float, volatility: float, rf: float = 0.0) -> float:
    """Sharpe ratio analysis"""
    return (expected_returns - rf) / volatility


def sharpe_from_returns(returns: pd.DataFrame, freq: str = 'B', rf: float = 0.0) -> pd.DataFrame:
    """Calculate Sharpe ratio for each market"""
    returns = returns.astype(float)
    sim_ret, vol = annualiseStat(returns, freq)
    return strategy_sharpe(sim_ret, vol, rf)


def sharpe_from_weights(returns: pd.DataFrame, weights: np.array, freq: str = 'M', rf: float = 0.0) -> float:
    """Sharpe ratio analysis for portfolio from multiple assets"""
    sigma = LedoitWolfShrink(returns.values)
    if freq is None:
        freq = returns.index.inferred_freq
    volatility = portVol(sigma, weights) * np.sqrt(freqInt(freq))
    expected_returns = annualise_return(returns, freq)
    return (expected_returns - rf) / volatility


def calculate_sharpe_with_transaction(y_pred: np.array, y_index: pd.Index, market_returns: pd_type, transaction_cost: float = 0.001) -> float:
    """Calculate Sharpe ratio with transaction cost"""
    y_pred_series = pd.Series(y_pred, index=y_index)  # Convert y_pred to pandas Series
    regime_changes = y_pred_series != y_pred_series.shift(1)
    strategy_returns = y_pred_series * market_returns.loc[y_index]
    strategy_returns -= np.where(regime_changes, transaction_cost, 0)
    return strategy_returns.mean() / strategy_returns.std()


def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.
    The downside deviation is calculated as the square root of the exponentially
    weighted second moment of negative returns.
    """
    ret_ser_neg = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)
