"""
In this script, we find the optimal weightings of a portfolio of stocks.

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

pd.set_option('display.max_columns', None)

NUM_TRADING_DAYS = 252    # on average 252 trading days in a year
NUM_PORTFOLIOS = 10000    # we will generate portfolios with random weightings

# stocks: Apple, Walmart, Tesla, General Electric, Amazon and Deutsche Bank
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
print(stocks)

# historical data - define START and END dates
start_date = date(2019, 1, 1)
end_date = date.today()

def download_data() -> pd.DataFrame:
    """
    Download stock price time series via Yahoo Finance API.
    :return df: pd.DataFrame
    """

    # name of stocks as keys - stock values (2010-2017) as values
    stock_data = {}

    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    # package into data frame and convert datetime to date
    df = pd.DataFrame(stock_data)
    df.index = df.index.date
    return df


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the one-day logarithmic returns of stocks over time.
    :param data: pd.DataFrame
    :return log_returns: pd.DataFrame
    """
    log_returns = np.log(data/data.shift(1))

    return log_returns[1:]  # drop the first row with NaN values


def show_data(stock_data: pd.DataFrame, xlabel: str = None, ylabel: str = None) -> None:
    """
    Show stock price data.
    :param stock_data:
    :return None:
    """
    stock_data.plot(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def show_statistics(returns: pd.DataFrame) -> None:
    print("Mean annual returns")
    print(round(returns.mean() * NUM_TRADING_DAYS, 2))
    print("Covariance matrix")
    print(round(returns.cov() * NUM_TRADING_DAYS, 2))


def show_mean_variance(returns: pd.DataFrame, weights: NDArray[np.float64]) -> None:
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS

    # multiply covariance matrix with weight vector
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    print("Expected return of portfolio (mean): %.2f" % portfolio_return)
    print("Expected volatility of portfolio (standard deviation): %.2f" % portfolio_volatility)


def generate_portfolios(returns: pd.DataFrame) -> tuple[NDArray[np.float64],
                                                            NDArray[np.float64], NDArray[np.float64]]:

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))   # generate a random 1D array
        w /= np.sum(w)  # normalise weights
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def show_portfolios(returns: NDArray[np.float64], volatilities: NDArray[np.float64]) -> None:
    plt.figure(figsize=(10, 5))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weights: NDArray[np.float64], returns: NDArray[np.float64]) -> NDArray[np.float64]:
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS # expected yearly return
    portfolio_volatility = np.sqrt(np.dot(
        weights.T, np.dot(
            returns.cov() * NUM_TRADING_DAYS, weights)
        )
    )

    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])


# scipy optimize module can find the minimum of a given function
# the maximum of f(x) is the minimum of -f(x)
def min_function_sharpe(weights: NDArray[np.float64], returns: pd.DataFrame) -> NDArray[np.float64]:
    return -statistics(weights, returns)[2]


# the sum of weights must be equal to 1
def optimize_portfolio(weights: NDArray[np.float64], returns: pd.DataFrame):
    # ensure that the sum of the weights is equal to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # weights can be at most 1, when 100% of money is invested in a single stock
    bounds = tuple((0,1) for _ in range(len(stocks)))

    optimum = optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP',
                          constraints=constraints, bounds=bounds)

    return optimum


def print_optimal_portfolio(optimum, returns):
    print('Optimal portfolio: ', optimum['x'].round(3))
    print('Expected return, volatility and Sharpe ratio: ',
          statistics(optimum['x'].round(3), returns))


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 5))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20)
    plt.show()


if __name__ == '__main__':
    dataset = download_data()
    #print(stock_data.head())

    # print stock price time series
    show_data(dataset, 'Date', 'Price')

    # get log daily returns
    print(calculate_returns(dataset))
    log_daily_returns = calculate_returns(dataset)
    show_statistics(log_daily_returns)

    # print returns time series
    show_data(log_daily_returns, 'Date', 'Log Return')

    # expected return and volatility with an equal-weighted portfolio
    show_mean_variance(log_daily_returns, weights=np.ones(len(stocks)) / len(stocks))

    # generate random portfolios
    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)

    # find portfolio with optimal Sharpe ratio
    optimum  = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)