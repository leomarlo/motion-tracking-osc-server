import numpy as np

def Euclidean(x,y):
    return np.sqrt((y[0]-x[0])**2 + (y[1] - x[1])**2)


def rescale(x, xmin, xmax, D):
    a = 2 * D / (xmax - xmin)
    b = - a * (xmax + xmin) / 2
    return a * x + b

def sigmoid(x, M):
    return M / (1 + np.exp(- x ))

    