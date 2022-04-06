import numpy as np

def convert_to_one_hot(col):
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals
