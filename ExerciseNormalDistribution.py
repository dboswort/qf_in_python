import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime
import scipy

# get historical data
def download_financial_data(stock: str) -> pd.DataFrame:
    stock_data = {}
    ticker = yf.Ticker(stock)
    stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    # package into data frame and convert datetime to date
    df = pd.DataFrame(stock_data)
    df.index = df.index.date
    return df


# calculate daily returns
def calculate_log_daily_returns(stock_data: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(stock_data / stock_data.shift(1))

    return log_returns[1:]  # drop the first row with NaN values


# plot daily returns as a histogram
def plot_daily_returns_hist(stock: str, log_returns: pd.DataFrame):
    #hist, bins = np.histogram(log_returns, bins=100)
    plt.hist(log_returns[stock], bins=1000, density=True)
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    plt.grid(True)
    plt.title(log_returns.columns[0])

    sigma = log_returns[stock].std()
    mu = log_returns[stock].mean()
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
    plt.show()


if '__main__' == __name__:

    stock = 'RHM.DE' # Rheinmetall # 'AAPL', 'IBM', '^GSPC' (S&P 500), 'GC=F' (gold)

    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2026, 1, 1)

    stock_data = download_financial_data(stock)
    #print(stock_data)
    log_daily_returns = calculate_log_daily_returns(stock_data)
    print(log_daily_returns.head())
    plot_daily_returns_hist(stock, log_daily_returns)