import numpy as np

def sigmoid(x, slope=1):
    return 1. / (1. + np.exp(-slope*x))

def inv_sigmoid(x, slope=1):
    return (1./slope)*np.log(x / (1-x))

def to_lodds(x):
    return -np.log(1./x - 1)


def prepend_zero(ar):
    return np.vstack((np.zeros((1, ar.shape[-1])), ar))

def convert_to_one_hot(col):
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals
