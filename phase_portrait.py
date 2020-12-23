#!/usr/bin/env python

""" Here we draw the 2-D phase portrait of the equations """

import numpy as np
import matplotlib.pyplot as plt
from runge_kutta4 import rk4


def x_dot(t, x):
    return x[1]


def u_dot(t, x):
    gamma1 = 0.1
    a1 = 3
    b1 = 2.2
    epsilon1 = 17.09
    omega1 = 2 * np.pi
    c1 = 2.2
    Omega = 15

    return -(gamma1 + a1 * x[0] ** 2 + b1 * x[2] ** 2) * x[1]\
           - omega1 ** 2 * (1 + epsilon1 * np.sin(2 * Omega * t)) * x[0]\
           - c1 * x[2] ** 2 * np.sin(Omega * t)


def y_dot(t, x):
    return x[3]


def v_dot(t, x):
    gamma2 = 0.1
    epsilon2 = 2.99
    omega2 = 2 * np.pi
    d2 = 10
    a2 = 9.8
    b2 = 2.2
    Omega = 15

    return -(gamma2 + a2 * x[2] ** 2 + b2 * x[0] ** 2) * x[3]\
           -omega2 ** 2 * (1 + epsilon2 * np.sin(2 * Omega * t)) * x[2]\
           - d2 * x[1]


def main():
    """ main body """
    dots = [x_dot, u_dot, y_dot, v_dot]
    x_init = [0.1, 0, 0.1, 0]
    step = 0.005
    end = 50

    time, data = rk4(x_init, dots, step, end)

    # plot y(t)
    plt.plot(time, data[2], lw=1, label='x(t)')
    plt.legend()
    plt.savefig('yplot.jpg', dpi=200, bbox_inches='tight')
    plt.show()

    # plot phase portrait for (x, u)
    plt.plot(data[0], data[1], lw=1, label='u(x)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('phase portrait u(x)')
    plt.savefig('portrait_xu.jpg', dpi=200, bbox_inches='tight')
    plt.show()

    # plot phase portrait for (y, v)
    plt.plot(data[2], data[3], lw=1, label='v(y)')
    plt.xlabel('y')
    plt.ylabel('v')
    plt.legend()
    plt.title('phase portrait u(x)')
    plt.savefig('portrait_yv.jpg', dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
