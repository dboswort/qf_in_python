"""
Monte-Carlo mean-variance portfolio optimisation model based on Markowitz's Modern Portfolio Theory (MPT).

Historical log daily returns are calculated from stock price data taken from Yahoo Finance.
Random portfolios are generated and their annual returns and volatility are calculated.
Scipy optimisation methods are used to additionally find the tangency and global minimum variance portfolios.
The portfolios on the "efficient frontier" are visualised using matplotlib

Note: in this implementation, short-selling is not allowed, i.e. all asset weights must be between 0 and 1.
"""

import numpy as np
from numpy.typing import NDArray    
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from datetime import date

def download_data(assets: list, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download time series of closing prices via Yahoo Finance API.
    :param assets: list of stock tickers
    :param start_date: start date of time series
    :param end_date: end date of time series
    :return df: pd.DataFrame
    """

    stock_data = {}
    for stock in assets:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    df = pd.DataFrame(stock_data).dropna()  # drop rows with NaN values
    return df


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the one-day logarithmic returns of assets over time.
    :param data: pd.DataFrame
    :return log_returns: pd.DataFrame
    """
    log_returns = np.log(data/data.shift(1))
    return log_returns[1:]  # drop the first row containing NaN


def show_data(stock_data: pd.DataFrame, xlabel: str = None, ylabel: str = None) -> None:
    """
    Show stock price data.
    :param stock_data: pd.DataFrame
    :return None:
    """
    stock_data.plot(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def show_statistics(returns: pd.DataFrame, n_trading_days: int) -> None:
    """
    Show mean annual returns and covariance matrix of stock returns.
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :return None:
    """
    print("Mean annual returns")
    print(round(returns.mean() * n_trading_days, 2))
    print("Covariance matrix")
    print(round(returns.cov() * n_trading_days, 2))


def generate_portfolios(assets: list, returns: pd.DataFrame, n_portfolios: int, n_trading_days: int) -> tuple[NDArray[np.float64],
                                                                            NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate random portfolios and calculate their annualised expected returns and standard deviations.
    :param assets: list of asset tickers
    :param returns: pd.DataFrame
    :param n_portfolios: int
    :param n_trading_days: int
    :return: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    mu = returns.mean()         # returns vector
    cov_matrix = returns.cov()  # covariance matrix
    randlist = np.random.rand(n_portfolios , len(assets)-1)  # generate random weights for each asset in each portfolio
    pf_weights = np.zeros((randlist.shape[0], randlist.shape[1]+1)) 
    pf_returns = np.zeros(randlist.shape[0])
    pf_stdevs = np.zeros(randlist.shape[0])
    for idx, row in enumerate(randlist):
        pf_weights[idx,:-1] = row
        pf_weights[idx,-1] = 1-row.sum() # asset allocations must sum to 1 
        pf_returns[idx] = pf_weights[idx].T @ mu * n_trading_days
        pf_stdevs[idx] = np.sqrt(n_trading_days * pf_weights[idx].T @ cov_matrix @ pf_weights[idx])
    return pf_weights, pf_returns, pf_stdevs


def get_portfolio_statistics(weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float) -> NDArray[np.float64]:
    """
    Calculate expected return, volatility, and Sharpe ratio of a portfolio with given weights over the specified number of trading days.
    :param returns: pd.DataFrame
    :param weights: NDArray[np.float64]
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: NDArray[np.float64]
    """
    portfolio_return = np.sum(returns.mean() * weights) * n_trading_days
    portfolio_volatility = np.sqrt(np.dot(
        weights.T, np.dot(
            returns.cov() * n_trading_days, weights)
        )
    )
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility   

    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


def min_function_sharpe(weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float) -> float:
    """
    Returns the negative Sharpe ratio of a portfolio with given weights over the specified number of trading days.
    This is used as the objective function for optimization, as we want to maximize the Sharpe ratio: the maximum of f(x) is the minimum of -f(x).
    :param weights: NDArray[np.float64]
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: float
    """
    sharpe_ratio = get_portfolio_statistics(weights, returns, n_trading_days, risk_free_rate)[2]
    return -sharpe_ratio  # negative Sharpe ratio for minimization 


def min_function_variance(weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float) -> float:
    """
    Returns the variance of a portfolio with given weights over the specified number of trading days.
    This is used as the objective function for optimization, as we want to minimize the variance.
    :param weights: NDArray[np.float64]
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: float
    """
    variance = get_portfolio_statistics(weights, returns, n_trading_days, risk_free_rate)[1] ** 2
    return variance


def get_tangency_portfolio(assets: list, weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float):
    """
    Optimize the portfolio weights to maximize the Sharpe ratio using scipy's minimize function.
    Constraints and bounds are set to ensure that the sum of weights is equal to 1 and that each weight is between 0 and 1 (i.e. no short-selling).
    :param assets: list of asset tickers
    :param weights: NDArray[np.float64]
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: scipy.optimize.OptimizeResult
    """
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}    # budget equation - 100% of money is invested in the portfolio
    bounds = tuple((0,1) for _ in range(len(assets)))

    optimum = optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=(returns, n_trading_days, risk_free_rate), method='SLSQP',
                          constraints=constraints, bounds=bounds)
    if not optimum.success:
        raise BaseException("Optimization failed: " + optimum.message)
    return optimum


def get_gmv_portfolio(assets: list, weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float):
    """
    Optimize the portfolio weights to minimize the variance using scipy's minimize function.
    Constraints and bounds are set to ensure that the sum of weights is equal to 1 and that each weight is between 0 and 1 (i.e. no short-selling).
    :param assets: list of asset tickers
    :param weights: NDArray[np.float64]
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: scipy.optimize.OptimizeResult
    """
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}    # budget equation - 100% of money is invested in the portfolio
    bounds = tuple((0,1) for _ in range(len(assets)))

    optimum = optimization.minimize(fun=min_function_variance, x0=weights[0], args=(returns, n_trading_days, risk_free_rate), method='SLSQP',
                          constraints=constraints, bounds=bounds)
    if not optimum.success:
        raise BaseException("Optimization failed: " + optimum.message)
    return optimum


def plot_portfolios(assets, rets, n_portfolios, n_trading_days, risk_free_rate):
    """
    Determines and plots the portfolios with (i) the highest Sharpe ratio (ii) the minimum volatility on the efficient frontier plot.
    :param assets: list of asset tickers
    :param rets: pd.DataFrame
    :param n_portfolios: int
    :param n_trading_days: int
    :param risk_free_rate: float
    :return None:
    """
    pweights, pf_returns, pf_risks = generate_portfolios(assets, rets, n_portfolios, n_trading_days)
    pf_sharpes = (pf_returns - risk_free_rate) / pf_risks
    tangency_pf = get_tangency_portfolio(assets, pweights, rets, n_trading_days, risk_free_rate)
    gmv_pf = get_gmv_portfolio(assets, pweights, rets, n_trading_days, risk_free_rate)

    plt.figure(figsize=(10, 5))
    plt.scatter(pf_risks, pf_returns, c=pf_sharpes, marker='o')
    plt.grid(True)
    plt.xlabel('Standard Deviation (Volatility)')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    tangency_ret, tangency_vol, tangency_sharpe = get_portfolio_statistics(tangency_pf['x'], rets, n_trading_days, risk_free_rate)
    print("Tangency Portfolio Sharpe Ratio: ", np.round(tangency_sharpe,2))
    gmv_ret, gmv_vol, gmv_sharpe = get_portfolio_statistics(gmv_pf['x'], rets, n_trading_days, risk_free_rate)
    print("Global Minimum Variance Portfolio Sharpe Ratio: ", np.round(gmv_sharpe,2))
    plt.plot(tangency_vol, tangency_ret, 'g*', markersize=20, label='Tangency Portfolio (Max Sharpe Ratio)')
    plt.plot(gmv_vol, gmv_ret, 'r*', markersize=20, label='Global Minimum Variance Portfolio')
    plt.legend()
    plt.show()