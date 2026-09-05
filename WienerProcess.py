import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt


def wiener_process(mu=0.0, dt=0.1, x0=0, n=1000):

    # W(t=0)=0
    # initialise W(t) with zeros
    W = np.zeros(n+1)

    #  create n+1 timesteps: t=0,1,2,3,...,n
    t = np.linspace(x0,n,n+1)

    W[1:n+1] = np.cumsum(np.random.normal(mu, np.sqrt(dt), n))

    return t, W


def plot_wiener_process(t, W):
    plt.plot(t, W)
    plt.xlabel('Time, t')
    plt.ylabel('Wiener process, W(t)')
    plt.title('Wiener process')
    plt.show()


if __name__ == "__main__":
    t, W = wiener_process(0.01, 3, 0, 10000)
    plot_wiener_process(t, W)


