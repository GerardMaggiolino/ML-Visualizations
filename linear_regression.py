'''
Module displays application of gradient descent for linear regression.

Main trains and plots linear regression - function applies gradients.
'''

import numpy as np
from matplotlib import pyplot as plt
from random_data import random2d

def main():
    # Config options 
    delay = 0.2   # Lower delay is faster

    # Set up dynamic matplotlib graph
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Generate and store random linear data
    xcord = list()
    ycord = list()
    for x, y in random2d(50, v=True, noise=0):
        xcord.append(x)
        ycord.append(y)
    xcord = np.array(xcord)
    ycord = np.array(ycord)

    # Inital weights
    m = b = 0

    # Train for 20 epochs
    for _ in range(20):
        m, b = linear_reg(xcord, ycord, m, b, epoch=1, alpha=0.00001)
        # Plot datapoints
        ax.set_ylim([-150, 150])
        ax.set_xlim([-150, 150])
        ax.scatter(xcord, ycord, c='r')
        ax.plot([-150, 150], [m * -150 + b, m * 150 + b], c='b')
        fig.canvas.draw()
        plt.pause(delay)
        ax.clear()


def linear_reg(x, y, m=0, b=0, alpha=0.00001, epoch=15) -> tuple: 
    '''Returns trained m, b in tuple''' 
    oldb = b
    b = b - alpha * 1000 *  np.sum((m * x + b) - y)
    m = m - alpha * np.sum((m * x + oldb - y) * x)
    return m, b


if __name__ == "__main__":
    main()
