import numpy as np

def euclidean(x):
    return np.sqrt(np.sum(np.power(x,2)))

def distance(x,y):
    return euclidean(x-y)


def rescale(x, xmin, xmax, D):
    a = 2 * D / (xmax - xmin)
    b = - a * (xmax + xmin) / 2
    return a * x + b

def sigmoid(x, M):
    return M / (1 + np.exp(- x ))

    