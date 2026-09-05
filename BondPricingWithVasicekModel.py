import matplotlib.pyplot as plt
import numpy as np
import polars as pl

NUM_OF_SIMULATIONS = 100   # number of interest rate processes
NUM_OF_POINTS = 50         # number of time steps for each process

def monte_carlo_simulation(principal, r0, kappa, theta, sigma, T=1):
    """
    Simulate interest rate processes over time T (default value is one year).
    This is the so-called Vasicek model.
    """
    dt = T / NUM_OF_POINTS
    result = []

    for _ in range(NUM_OF_SIMULATIONS):
        rates = [r0]
        for _ in range(NUM_OF_POINTS):
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)

        result.append(rates)

    sim_data = pl.DataFrame(result)
    #print(sim_data)

    plt.plot(sim_data)
    plt.title('Ornstein-Uhlenbeck process: mean reversion')
    plt.xlabel('Time')
    plt.ylabel('Interest Rate')
    plt.grid(True)
    plt.show()

    integral_sum = sim_data.sum() * dt
    print(integral_sum)
    present_integral_sum = np.exp(-1 *integral_sum)
    bond_price = principal * np.mean(present_integral_sum)

    print('Bond price based on Monte-Carlo simulation: $%.2f' % bond_price)


if __name__ == "__main__":
    monte_carlo_simulation(1000, 0.1, 0.3, 0.1, 0.03)



