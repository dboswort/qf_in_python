from scipy import stats
from numpy import log, exp, sqrt

def call_option_price(S, E, T, rf, sigma):
    """
    Black-Scholes equation for a European call option.
    S - stock spot price
    E - strike price
    T - time to expiry
    rf - risk-free rate
    sigma - volatility
    """

    d1 = (log(S/E) + (rf + sigma*sigma/2.0)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    return S*stats.norm.cdf(d1) - E*exp(-rf*T)*stats.norm.cdf(d2)

def put_option_price(S, E, T, rf, sigma):
    """
    Black-Scholes equation for a European put option.
    S - stock spot price
    E - strike price
    T - time to expiry
    rf - risk-free rate
    sigma - volatility
    """

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