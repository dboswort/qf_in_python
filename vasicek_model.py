"""
This module simulates interest rate processes over time according to the Vasicek 
model via the Euler-Maruyama method and determines the bond price via Monte-Carlo simulation.
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


def price_bond_monte_carlo(principal, r0, kappa, theta, sigma, T=1, num_simulations=100, num_time_steps=252):
    """
    Simulate interest rate processes over time T (default value is one year)
    according to the Vasicek model and determine the bond price using Monte-Carlo simulation.
    :param principal: the principal amount of the bond
    :param r0: the initial interest rate
    :param kappa: the speed of mean reversion
    :param theta: the long-term mean interest rate
    :param sigma: the volatility of the interest rate
    :param T: the time horizon for the simulation (default is 1 year)
    :return: bond_price: the present value of the bond, rates: an array of simulated interest rate paths, time_grid: the time grid for the simulation
    """
    dt = T / num_time_steps
    time_grid = np.linspace(0, T, num_time_steps + 1)
    rates = simulate_vasicek_euler(r0, kappa, theta, sigma, dt, num_time_steps, num_simulations)
    bond_price = get_present_value(principal, rates, dt)
    return bond_price, rates, time_grid

@njit
def simulate_vasicek_euler(r0: float, kappa: float, theta: float, sigma: float, dt: float, num_steps: int, num_paths: int) -> np.ndarray:
    """
    Simulate interest rate processes according to the Vasicek model using the Euler-Maruyama method.
    njit is used to speed up the simulation by compiling the function to machine code.
    :param r0: the initial interest rate
    :param kappa: the speed of mean reversion
    :param theta: the long-term mean interest rate
    :param sigma: the volatility of the interest rate
    :param dt: the time step size
    :param num_steps: the number of time steps to simulate
    :param num_paths: the number of interest rate paths to simulate
    :return: an array of simulated interest rate paths
    """
    sqrt_dt = np.sqrt(dt)
    rates = np.empty((num_paths, num_steps + 1))
    rates[:, 0] = r0

    for i in range(num_paths):
        r = r0
        for t in range(num_steps):
            dr = kappa * (theta - r) * dt + sigma * sqrt_dt * np.random.normal()
            r = r + dr
            rates[i, t + 1] = r
    return rates


def get_present_value(principal: float, rates: np.ndarray, dt: float) -> float:
    """
    Calculate the present value of a bond given the principal and interest rate processes.
    :param principal: the principal amount of the bond
    :param rates: the simulated interest rate processes
    :param dt: the time step size
    :return: the present value of the bond
    """
    integral_sums = [rates[i].sum() * dt for i in range(rates.shape[0])]
    present_integral_sums = np.exp(-1 * np.array(integral_sums))
    bond_price = principal * np.mean(present_integral_sums)
    return bond_price


def plot_interest_rate_paths(rates: np.ndarray, time_grid: np.ndarray) -> None:
    """
    Plot the simulated interest rate paths.
    :param rates: an array of simulated interest rate paths
    :param time_grid: the time grid for the simulation
    :return: None
    """
    plt.plot(time_grid, rates.T)
    plt.title('Mean-reverting Ornstein-Uhlenbeck process (Vasicek model)')
    plt.xlabel('Time')
    plt.ylabel('Interest Rate')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    bond_price, rates, time_grid = price_bond_monte_carlo(
            principal=1000, 
            r0=0.1, 
            kappa=0.08, 
            theta=0.0225, 
            sigma=0.03)
    print('Bond price: $%.2f' % bond_price)



