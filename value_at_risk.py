"""
Value-at-Risk (VaR) is the maximum monetary loss over a defined future period to a specified confidence level.
In this script, the VaR is calculated using the parametric method, which assumes that returns are normally distributed.
The script downloads historical stock price data from Yahoo Finance, calculates the log daily returns, and then
calculates the VaR based on the position size, confidence level, mean and 
standard deviation of the log daily returns, and the margin period of risk (MPOR).
"""
import numpy as np
import pandas as pd
import datetime
from scipy.stats import norm
import yfinance as yf

def download_data(ticker_symbol: str, start_date: datetime.date, end_date: datetime.date):
    """
    Download historical stock price data from Yahoo Finance and return a Pandas DataFrame with the closing prices.
    :param ticker_symbol: str, stock ticker symbol
    :param start_date: datetime.date, start date for the data
    :param end_date: datetime.date, end date for the data
    :return: pd.DataFrame, Pandas DataFrame with the closing prices"""
    price_data = {}
    ticker = yf.Ticker(ticker_symbol)
    price_data[ticker_symbol] = ticker.history(start=start_date, end=end_date)['Close']
    df = pd.DataFrame(price_data)
    return df


def calculate_parametric_var_n(pos, cl, mu, sigma, n):
    """
    Calculate the parametric Value-at-Risk (VaR) for a given position size, confidence level, 
    mean and standard deviation of returns, and number of days.
    :param pos: float, position size (in monetary units)
    :param cl: float, confidence level (between 0 and 1)
    :param mu: float, mean of the log daily returns
    :param sigma: float, standard deviation of the log daily returns
    :param n: int, number of days (margin period of risk)
    :return: Value-at-Risk (VaR)
    """
    v = norm.ppf(1-cl) # inverse of the cumulative distribution function (CDF) for the standard normal distribution
    val = pos * (mu * n - sigma * np.sqrt(n) * v)
    return val


def get_log_daily_returns(ticker_symbol: str, data: pd.DataFrame) -> pd.Series:
    """"
    Calculate the log daily returns for a given stock from the historical price data.
    :param ticker_symbol: str, stock ticker symbol
    :param data: pd.DataFrame, Pandas DataFrame with the closing prices
    :return: pd.Series, log daily returns"""
    return np.log(data[ticker_symbol] / data[ticker_symbol].shift(1)).dropna()


if __name__ == '__main__':
    ticker_symbol = '^GSPC'  # S&P 500 index
    price_data = download_data(ticker_symbol, datetime.date(2001,1,1), datetime.date.today())
    log_daily_returns = get_log_daily_returns(ticker_symbol, price_data)

    position = 100e6
    confidence_level = 0.99
    mpor = 5                    # margin period of risk (MPOR)
    var = calculate_parametric_var_n(position, confidence_level, log_daily_returns.mean(), log_daily_returns.std(), mpor)
    print(f"To a {100*confidence_level:.2f}% confidence level, we will not lose more than ${var:,.2f} over the next {mpor} days.")
