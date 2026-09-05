"""
Also known as Brownian motion, the Wiener process is a continuous-time stochastic process that is widely used in finance and physics.
In this script, we simulate an arithmetic Wiener process with drift and plot the resulting path. 
Each step on the random walk is independent and drawn from the standard normal distribution N(0,1). 
"""

import numpy as np
import matplotlib.pyplot as plt


def wiener_process(mu=0.0, x0=0, T=1, n=1000):
    """
    Simulate an arithmetic Wiener process with drift.
    :param mu: float, drift term (mean of the process)
    :param x0: float, initial value of the process
    :param T: float, total time period
    :param n: int, number of steps in the process
    :return: tuple, time steps and the corresponding Wiener process values
    """
    # initialise W(0) with zeros
    W = np.zeros(n+1)

    #  create n+1 timesteps: t=0,1,2,3,...,n
    t = np.linspace(0, T, n+1)

    W[1:n+1] = np.cumsum(np.random.normal(mu, np.sqrt(T/n), n))

    return t, W


def plot_wiener_process(t, W):
    plt.plot(t, W)
    plt.xlabel('Time, t')
    plt.ylabel('Wiener process, W(t)')
    plt.title('Wiener process')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    t, W = wiener_process(mu=0.01, x0=0, T=1, n=100)
    plot_wiener_process(t, W)

    # compare changing the number of steps in the Wiener process 
    # for a fixed duration
    t1, W1 = wiener_process(mu=0.0005, x0=0, T=1, n=10)
    t2, W2 = wiener_process(mu=0.0005, x0=0, T=1, n=100)
    t3, W3 = wiener_process(mu=0.0005, x0=0, T=1, n=1000)
    t4, W4 = wiener_process(mu=0.0005, x0=0, T=1, n=10000)
    plt.plot(t1, W1, label='n=10 steps')
    plt.plot(t2, W2, label='n=100 steps')
    plt.plot(t3, W3, label='n=1000 steps')
    plt.plot(t4, W4, label='n=10000 steps')
    plt.xlabel('Time, t')
    plt.ylabel('Wiener process, W(t)')
    plt.title('Wiener process')
    plt.legend()
    plt.grid(True)
    plt.show()


