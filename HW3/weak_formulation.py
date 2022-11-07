from itertools import product as prod
import numpy as np
from pylab import mpl
from sklearn.linear_model import Ridge

mpl.use('macosx')


def generate_products_hw(d_max: int, x, y):
    """
    Generate power product basis.

    (x**0)(y**0)(z**0),
    (x**1)(y**0)(z**0), ....

    """

    power_mat = np.arange(0, d_max)
    products = np.array(list(prod(power_mat, power_mat)))

    xyz_products = {key: value for key, value in enumerate(products)}
    a_mat = np.zeros((len(x), len(xyz_products.keys())))
    symbols = []

    for item in xyz_products.values():
        symbols.append(r"$x^{} y^{}$".format(item[0], item[1]))

    for index, value in enumerate(xyz_products.keys()):
        a_mat[:, index] = x ** (xyz_products[value][0]) * y ** (xyz_products[value][1])

    return a_mat, symbols


def hw_detected_dynamical_system(t: int):
    """
    Compute detected dynamical system. For comparison.
    """
    dt, step_cnt = 0.1, t
    x = np.zeros((step_cnt + 1))
    y = np.zeros((step_cnt + 1))

    x_dot = np.empty((step_cnt + 1,))
    y_dot = np.empty((step_cnt + 1,))

    x[0], y[0] = 0, 2

    for t in range(step_cnt):

        x_dot[t], y_dot[t] = f1(x[t], y[t]), f2(x[t], y[t])

        x[t + 1] = x[t] + (x_dot[t] * dt)
        y[t + 1] = y[t] + (y_dot[t] * dt)

    return x, y

# Answers: \lambda was varied from .1 to 0.001 and minimum
# square difference with real answer was selected


def f1(x_i, y_i):
    return x_i - y_i + (-0.34040987) * x_i ** 3 + 0.5


def f2(x_i, y_i):
    return (-1.50964217e-01) * y_i + 1.91250794e-01 * x_i


if __name__ == "__main__":

    data = np.load('DataHw3Q1_x.npy')
    x1 = data[:, 0]
    x2 = data[:, 1]
    time = np.load('DataHw3Q1_t.npy')

    # derivative can be taken by approximating the function by polynomials
    # for this exercise simple derivative was taken.

    derivative1 = (x1[1:] - x1[:-1]) / 0.1
    derivative2 = (x2[1:] - x2[:-1]) / 0.1

    a, sym = generate_products_hw(4, x1[1:], x2[1:])

    y2 = f2(x1, x2)
    y1 = f1(x1, x2)

    clf3 = Ridge(alpha=0.01).fit(a, derivative1)
    c3 = clf3.coef_
    clf4 = Ridge(alpha=0.001)
    clf4.fit(a, derivative2)
    c4 = clf4.coef_

    pass
