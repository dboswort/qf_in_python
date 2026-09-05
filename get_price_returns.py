"""
This script downloads historical stock data from Yahoo Finance, calculates the log daily returns, 
and plots a histogram of the returns along with a fitted normal distribution curve.

Dependencies:   
- pandas
- matplotlib    
- numpy
- yfinance
- datetime
- scipy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime
import scipy


def download_financial_data(stock: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """
    Retrieve historical closing price data from Yahoo Finance for a given asset ticker symbol.
    """
    stock_data = {}
    ticker = yf.Ticker(stock)
    stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    df = pd.DataFrame(stock_data)
    df.index = df.index.date
    return df


def calculate_log_daily_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the log daily returns for the given price data P_t, where P_t is the price at time t. 
    The formula for log return is:
    log return = ln(P_t / P_(t-1))
    """
    log_returns = np.log(price_data / price_data.shift(1))

    return log_returns[1:]  # drop the first row which contains NaN


def plot_daily_returns_hist(stock: str, log_returns: pd.DataFrame):
    """
    Plot the histogram of log daily returns for the given stock.
    The histogram is fitted with a normal distribution curve based on the mean and standard deviation of the log returns.
    """
    plt.hist(log_returns[stock], bins=250, density=True)
    plt.xlabel('Log Return')
    plt.ylabel('Probability density')
    plt.grid(True)
    plt.title(log_returns.columns[0])

    sigma = log_returns[stock].std()
    mu = log_returns[stock].mean()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)    # grid for plotting the normal distribution
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
    plt.show()


if '__main__' == __name__:

    stock = 'GC=F' # 'AAPL', 'IBM', '^GSPC', 'GC=F', 'RHM.DE'

    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2026, 1, 1)

    stock_data = download_financial_data(stock)
    log_daily_returns = calculate_log_daily_returns(stock_data)
    print(log_daily_returns.head())
    plot_daily_returns_hist(stock, log_daily_returns)