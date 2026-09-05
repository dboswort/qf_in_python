from scipy import stats
from numpy import log, exp, sqrt

def call_option_price(S, E, T, rf, sigma):
    """
    Solution from Black-Scholes equation for the price of a call (buy) option.
    S - stock price
    E - strike price
    T - maturity
    rf - risk-free rate
    sigma - volatility
    """

    # calculate the d1 and d2 parameters
    d1 = (log(S/E) + (rf + sigma*sigma/2.0)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    return S*stats.norm.cdf(d1) - E*exp(-rf*T)*stats.norm.cdf(d2)

def put_option_price(S, E, T, rf, sigma):
    """
    Solution from Black-Scholes equation for the price of a put (sell) option.
    S - stock price
    E - strike price
    T - maturity
    rf - risk-free rate
    sigma - volatility
    """

    # calculate the d1 and d2 parameters
    d1 = (log(S/E) + (rf + sigma*sigma/2.0)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    return -S*stats.norm.cdf(-d1) + E*exp(-rf*T)*stats.norm.cdf(-d2)

if __name__ == "__main__":
    S0 = 100
    E = 100
    T = 1
    rf = 0.05
    sigma = 0.2

    print("Call option price: $ {}".format(round(call_option_price(S0, E, T, rf, sigma), 2)))
    print("Put option price: $ {}".format(round(put_option_price(S0, E, T, rf, sigma), 2)))