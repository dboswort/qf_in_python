"""
In this script, we find the optimal weightings of a portfolio of assets.

1) Historical log daily returns are calculated from stock price data taken from Yahoo Finance via the yfinance package.
2) A number of random portfolios are generated and their annual returns and volatility are calculated.
3) Scipy optimisation methods are used to additionally find the portfolio with the highest Sharpe ratio:
    Sharpe Ratio = Annual Return / Annual Volatility
4) The portfolios on the "efficient frontier" are visualised using matplotlib

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

    df = pd.DataFrame(stock_data)
    df.index = df.index.date
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
    Generate random portfolios and calculate their expected returns and standard deviations.
    :param assets: list of asset tickers
    :param returns: pd.DataFrame
    :param n_portfolios: int
    :param n_trading_days: int
    :return: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(n_portfolios):
        w = np.random.random(len(assets))   
        w /= np.sum(w)                      # ensure weights sum to 1
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * n_trading_days)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * n_trading_days, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def show_portfolios(returns: NDArray[np.float64], volatilities: NDArray[np.float64]) -> None:
    """
    Plot the expected returns and volatilities of the generated portfolios.
    :param returns: NDArray[np.float64]
    :param volatilities: NDArray[np.float64]
    :return None:
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Standard Deviation (Volatility)')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weights: NDArray[np.float64], returns: NDArray[np.float64], n_trading_days: int, risk_free_rate: float = 0.02) -> NDArray[np.float64]:
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


def min_function_sharpe(weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float = 0.02) -> NDArray[np.float64]:
    """
    Returns the negative Sharpe ratio of a portfolio with given weights over the specified number of trading days.
    This is used as the objective function for optimization, as we want to maximize the Sharpe ratio: the maximum of f(x) is the minimum of -f(x).
    :param weights: NDArray[np.float64]
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: NDArray[np.float64]
    """
    return -statistics(weights, returns, n_trading_days, risk_free_rate)[2]


def optimize_portfolio(assets: list, weights: NDArray[np.float64], returns: pd.DataFrame, n_trading_days: int, risk_free_rate: float = 0.02):
    """
    Optimize the portfolio weights to maximize the Sharpe ratio using scipy's minimize function.
    :param assets: list of asset tickers
    :param weights: NDArray[np.float64]
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :param risk_free_rate: float
    :return: scipy.optimize.OptimizeResult
    """
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}    # budget equation - the sum of asset weights must be equal to 1, i.e. 100% of money is invested in the portfolio

    bounds = tuple((0,1) for _ in range(len(assets)))

    optimum = optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=(returns, n_trading_days, risk_free_rate), method='SLSQP',
                          constraints=constraints, bounds=bounds)

    return optimum


def print_optimal_portfolio(optimum, returns, n_trading_days, risk_free_rate: float = 0.02) -> None:
    """
    Print the optimal portfolio weights and its expected return, volatility, and Sharpe ratio.
    :param optimum: scipy.optimize.OptimizeResult
    :param returns: pd.DataFrame
    :param n_trading_days: int
    :return None:
    """
    print('Optimal portfolio: ', optimum['x'].round(3))
    print('Expected return, volatility and Sharpe ratio: ',
          statistics(optimum['x'].round(3), returns, n_trading_days, risk_free_rate).round(3))


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols, n_trading_days, risk_free_rate):
    """
    Show the optimal portfolio on the efficient frontier plot.
    :param opt: scipy.optimize.OptimizeResult
    :param rets: pd.DataFrame
    :param portfolio_rets: NDArray[np.float64]
    :param portfolio_vols: NDArray[np.float64]
    :param n_trading_days: int
    :param risk_free_rate: float
    :return None:
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Standard Deviation (Volatility)')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets, n_trading_days, risk_free_rate=risk_free_rate)[1], statistics(opt['x'], rets, n_trading_days, risk_free_rate=risk_free_rate)[0], 'g*', markersize=20)
    plt.show()


if __name__ == '__main__':
    n_trading_days = 252    # 252 trading days in a year
    n_portfolios = 10000    # number of random portfolios to generate

    assets = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

    start_date = date(2019, 1, 1)
    end_date = date.today()

    dataset = download_data(assets, start_date, end_date)
    show_data(dataset, 'Date', 'Closing price')

    log_daily_returns = calculate_returns(dataset)
    show_statistics(log_daily_returns, n_trading_days)

    show_data(log_daily_returns, 'Date', 'Log Return')

    # expected return and volatility with an equal-weighted portfolio
    equal_weighted_statistics = statistics(
        np.ones(len(assets)) / len(assets), log_daily_returns, n_trading_days
    )
    print("Expected return: %.2f" % equal_weighted_statistics[0])
    print("Expected standard deviation: %.2f" % equal_weighted_statistics[1])

    # generate random portfolios
    pweights, means, risks = generate_portfolios(assets, log_daily_returns, n_trading_days)
    show_portfolios(means, risks)

    # find portfolio with optimal Sharpe ratio
    optimum  = optimize_portfolio(assets, pweights, log_daily_returns, n_trading_days, risk_free_rate=0.02)
    print_optimal_portfolio(optimum, log_daily_returns, n_trading_days, risk_free_rate=0.02)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks, n_trading_days, risk_free_rate=0.02)