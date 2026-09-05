"""
Monte Carlo Option Pricing
This module provides a class for pricing European call and put options using Monte Carlo simulation.
"""
import numpy as np

class OptionPricing:
    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        # payoff function max(0,S-E)
        option_data = np.zeros([self.iterations, 2])

        # mean = 0 and stadard deviation = 1
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for stock price S at time T (origin: solution to the geometric walk SDE using Ito's lemma)
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2)
                                       + self.sigma * np.sqrt(self.T) * rand)

        # calculate the call option payoff max(0, S-E)
        option_data[:, 1] = stock_price - self.E

        # average value for the Monte Carlo simulation
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # discount the future cash flow using the risk-free rate
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):
        # payoff function max(0,S-E)
        option_data = np.zeros([self.iterations, 2])

        # mean = 0 and stadard deviation = 1
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for stock price S at time T (origin: solution to the geometric walk SDE using Ito's lemma)
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2)
                                       + self.sigma * np.sqrt(self.T) * rand)

        # calculate the call option payoff max(0, S-E)
        option_data[:, 1] = stock_price - self.E

        # average value for the Monte Carlo simulation
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # discount the future cash flow using the risk-free rate
        return np.exp(-1.0 * self.rf * self.T) * average

if __name__ =="__main__":
    model = OptionPricing(S0=100, E=101, T=1, rf=0.01, sigma=0.5, iterations=5000000)
    print("The value of the call option is $%.2f" % model.call_option_simulation())
    print("The value of the put option is $%.2f" % model.put_option_simulation())