"""
Determines the beta of a stock using the Capital Asset Pricing Model (CAPM) 
and plots the regression line of the stock returns against the market returns.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

RISK_FREE_RATE = 0.05
MONTHS_IN_YEAR = 12     # consider monthly returns

class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):

        data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date, auto_adjust=True)
            data[stock] = ticker['Close'][stock]

        return pd.DataFrame(data)

    def initialize(self):

        stock_data = self.download_data()
        stock_data = stock_data.resample('ME').last()

        self.data = pd.DataFrame({'s_adjclose': stock_data[self.stocks[0]],
                                  'm_adjclose': stock_data[self.stocks[1]],})

        # logarithmic monthly returns
        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_adjclose', 'm_adjclose']] /
                                                       self.data[['s_adjclose', 'm_adjclose']].shift(1))

        # remove the NaN values
        self.data = self.data[1:]
        print(self.data)


    def calculate_beta(self):
        covariance_matrix = np.cov(self.data['s_returns'], self.data['m_returns'])
        beta = covariance_matrix[0,1] / covariance_matrix[1,1]

        return beta

    def regression(self):
        beta, alpha = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)

        expected_return = (RISK_FREE_RATE +
                           beta * (self.data['m_returns'].mean() * MONTHS_IN_YEAR - RISK_FREE_RATE))
        print('Expected Return: ', expected_return)
        self.plot_regression(alpha, beta)

        return beta

    def plot_regression(self, alpha, beta):
        fix, axis = plt.subplots(1, figsize=(10, 5))
        axis.scatter(self.data['m_returns'], self.data['s_returns'], label='Data points')
        axis.plot(self.data['m_returns'], beta * self.data['m_returns'] + alpha, color='red', label='CAPM Line')
        plt.xlabel('Market return $R_m$')
        plt.ylabel('Stock return $R_a$')

        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    stock = 'AAPL'
    market = '^GSPC'
    capm = CAPM([stock, market], '2001-01-01', '2026-01-01')
    capm.initialize()
    print('Beta from formula: ', capm.calculate_beta())
    print('Beta from regression: ', capm.regression())
