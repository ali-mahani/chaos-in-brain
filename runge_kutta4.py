#!/usr/bin/env python

""" modules to integrate using 4th order runge kutta method """

import numpy as np


def rk4(x_init, func, step, time):
    """ takes x and func as a vector of values and functions respectively """
    # find count
    count = int(time / step)
    length = len(x_init)

    # initialization
    ## store the record of x
    x = np.zeros(shape=(count, length))
    x[0] = x_init.copy()
    ## temporary values
    k1 = np.zeros(length)
    k2 = np.zeros(length)
    k3 = np.zeros(length)
    k4 = np.zeros(length)
    ## time
    time = np.zeros(count)

    # evolve
    for i in range(count - 1):
        # find k1
        for j in range(length):
            k1[j] = func[j](time[i], x[i])

        # find k2
        for j in range(length):
            k2[j] = func[j](time[i] + step / 2, x[i] + k1 * step / 2)

        # find k3
        for j in range(length):
            k3[j] = func[j](time[i] + step / 2, x[i] + k2 * step / 2)

        # find k4
        for j in range(length):
            k4[j] = func[j](time[i] + step, x[i] + step * k3)

        # update x
        x[i + 1] = x[i] + 1 / 6.0 * step * (k1 + 2 * k2 + 2 * k3 + k4)
        time[i + 1] = time[i] + step


    return time, x.T


def x_dot(time, x):
    """ function to test algorithm """
    return x[1]


def y_dot(time, x):
    """ function to test algorithm """
    return -x[0]


def main():
    """ main body with the sole purpose of tesing the algorithm"""
    x_init = [1, 0]
    funcs = [x_dot, y_dot]
    end = 50
    step = 0.001

    time, x = rk4(x_init, funcs, step, end)

    import matplotlib.pyplot as plt

    plt.plot(time, x[0], ls='--', label='position')
    plt.plot(time, x[1], ls='-.', label='speed')
    plt.xlabel('time')
    plt.legend()
    plt.show()

    plt.plot(x[0], x[1], ls='-.')
    plt.show()

if __name__ == "__main__":
    main()
