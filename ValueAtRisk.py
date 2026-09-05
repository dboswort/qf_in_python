"""
Value-at-Risk quantifies the risk in monetary units over a defined future period based off historical returns.
Unlike my first project at d-fine, we assume that the log returns follow a normal distribution.
"""
import numpy as np
import polars as pl
import datetime
from scipy.stats import norm
import yfinance as yf

def download_data(stock: str, start_date: datetime.date, end_end: datetime.date):
    ticker = yf.download(stock, start=start_date, end=end_end)
    df = pl.from_pandas(ticker['Close'], include_index=True)
    return df


# assumes that returns are normally-distributed (!)
def calculate_var_n(pos, cl, mu, sigma, n):
    """
    cl: confidence level
    """
    v = norm.ppf(1-cl)
    val = pos * (mu * n - sigma * np.sqrt(n) * v)
    return val


def get_log_daily_returns(stock: str, data: pl.DataFrame):
    return (
        (data[stock] / data[stock].shift(1))
        .log()
        .drop_nulls()
    )


if __name__ == '__main__':
    #stock_data = download_data('^GSPC', datetime.date(2026,1,1), datetime.date.today())

    # Citigroup stock data
    stock_data = download_data('C', datetime.date(2018, 1, 1), datetime.date.today())
    print(stock_data)
    print(stock_data.select('C'))
    log_daily_returns = get_log_daily_returns('C', stock_data)
    print(log_daily_returns)

    position = 100e6
    confidence_level = 0.999
    mpor = 5 # margin period of risk (MPOR)
    var = calculate_var_n(position, confidence_level, log_daily_returns.mean(), log_daily_returns.std(), mpor)
    print(f"We can say with {100*confidence_level:.2f}% confidence that we will not lose more than ${var:,.2f} over the next {mpor} days.")