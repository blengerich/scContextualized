"""
Miscellaneous utility functions.
"""

import numpy as np


def sigmoid(x, slope=1):
    """

    :param x: float input value
    :param slope: float, slope (Default value = 1)

    returns 1/(1+exp(-slope*x))

    """
    return 1.0 / (1.0 + np.exp(-slope * x))


def inv_sigmoid(x, slope=1):
    """
    inv_sigmoid(sigmoid(x)) = x

    :param x: float input value
    :param slope:  (Default value = 1)

    """
    return (1.0 / slope) * np.log(x / (1 - x))


def to_lodds(x):
    """

    :param x: float probability

    return log-odds

    """
    return -np.log(1.0 / x - 1)


def prepend_zero(ar):
    """

    :param ar: np array

    returns ar with a zero at front

    """
    return np.vstack((np.zeros((1, ar.shape[-1])), ar))


def convert_to_one_hot(col):
    """

    :param col: np array with observations

    returns col converted to one-hot values, and list of one-hot values.

    """
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals
