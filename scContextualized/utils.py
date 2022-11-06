"""
Miscellaneous utility functions.
"""

import numpy as np


def sigmoid(x_in, slope=1):
    """

    :param x: float input value
    :param slope: float, slope (Default value = 1)

    returns 1/(1+exp(-slope*x))

    """
    return 1.0 / (1.0 + np.exp(-slope * x_in))


def inv_sigmoid(x_in, slope=1):
    """
    inv_sigmoid(sigmoid(x)) = x

    :param x: float input value
    :param slope:  (Default value = 1)

    """
    return (1.0 / slope) * np.log(x_in / (1 - x_in))


def to_lodds(x_in):
    """

    :param x: float probability

    return log-odds

    """
    return -np.log(1.0 / x_in - 1)


def prepend_zero(ar_in):
    """

    :param ar: np array

    returns ar with a zero at front

    """
    return np.vstack((np.zeros((1, ar_in.shape[-1])), ar_in))
